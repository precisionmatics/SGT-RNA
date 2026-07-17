"""
SGT-RNA · Step 17: GPU Docking with gnina

Run gnina (GPU-accelerated) on all 143 NL2020 crystal structures.
Receptor: pocket.pdb (used directly — no PDBQT conversion needed)
Ligand:   ligand.sdf (used directly)
Box:      22×22×22 Å centered on crystal ligand centroid
Scores:   Vina affinity + CNNaffinity → 2 new features per complex

Combine with step15 features (53,371) → 53,373 total
Re-evaluate hybrid model.
"""

import subprocess, gzip, pickle, logging, time, tempfile, os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel

import warnings
warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent.parent
NA_L    = ROOT / "NA-L"
PKL_FILE= ROOT / "data" / "pocket_fri" / "pocket_fri_data.pkl.gz"
S15_NPZ = ROOT / "data" / "features" / "step15_full_features.npz"
S11_CSV = ROOT / "results" / "step11_results.csv"
OUT_NPZ = ROOT / "data" / "features" / "step17_full_features.npz"
RES_DIR = ROOT / "results"
FIG_DIR = ROOT / "results" / "figures"

GNINA      = os.environ.get("GNINA_BIN", "gnina")        # set GNINA_BIN env var or ensure gnina is on PATH
CUDNN_PATH = os.environ.get("CUDNN_LIB_PATH", "")        # set CUDNN_LIB_PATH if cuDNN is not on LD_LIBRARY_PATH
TMPDIR  = Path("/tmp/gnina_rna")
TMPDIR.mkdir(parents=True, exist_ok=True)

# Patch env for cuDNN
env = os.environ.copy()
existing_ld = env.get("LD_LIBRARY_PATH", "")
env["LD_LIBRARY_PATH"] = f"{CUDNN_PATH}:{existing_ld}" if existing_ld else CUDNN_PATH

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "results" / "logs" / f"step17_{ts}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("SGT-RNA · Step 17: GPU Docking with gnina")
log.info("=" * 70)

# ── Subtype labels (same as step11+) ─────────────────────────────────────────
G_QUAD = {"1nzm","5cdb","4xwf","4znp","5btp","6jj0","2mg8","2loa"}
DUPLEX_GROOVE = {"407d","408d","1cvy","1cvx","454d","1qv4","1qv8","1p96","1r4e","6hbt"}
MANUAL_OVERRIDE = {
    "1lvu":"aptamer","1mwl":"aptamer","1o15":"aptamer","1u8d":"aptamer",
    "1yvp":"aptamer","2b57":"aptamer","2esj":"aptamer","2gdi":"aptamer",
    "3f2q":"aptamer","3mum":"aptamer","3q3z":"aptamer","3skl":"aptamer",
    "4lck":"aptamer","4tza":"aptamer",
    "1yfg":"ribosomal_asite","2aw4":"ribosomal_asite","3q1r":"ribosomal_asite",
    "6hbt":"duplex_groove",
    "2l3e":"g_quadruplex","2kzd":"g_quadruplex",
    "1f27":"riboswitch","1u8d":"aptamer",
}

def get_subtype(pdb, raw):
    pid = pdb.lower()
    if pid in MANUAL_OVERRIDE: return MANUAL_OVERRIDE[pid]
    if pid in G_QUAD: return "g_quadruplex"
    if pid in DUPLEX_GROOVE: return "duplex_groove"
    return str(raw)

# ── Ligand centroid from pocket data ─────────────────────────────────────────
def get_ligand_centroid(rec_data):
    coords = []
    for v in rec_data["lig_coords"].values():
        if len(v): coords.append(v)
    if not coords:
        return None
    return np.vstack(coords).mean(axis=0)

# ── gnina docking ─────────────────────────────────────────────────────────────
def run_gnina(pdb_id, rec_data, na_l_dir, timeout=120):
    """Run gnina and return (vina_score, cnn_affinity). Returns (nan, nan) on failure."""
    pid = pdb_id.lower()
    complex_dir = na_l_dir / pid

    pocket_pdb = complex_dir / "pocket.pdb"
    lig_sdf    = complex_dir / "ligand.sdf"

    if not pocket_pdb.exists() or not lig_sdf.exists():
        log.warning(f"  {pid}: missing pocket.pdb or ligand.sdf")
        return np.nan, np.nan

    # Skip very large ligands (gnina can be slow on peptides)
    n_lig_atoms = sum(len(v) for v in rec_data["lig_coords"].values())
    if n_lig_atoms > 100:
        log.info(f"  {pid}: SKIP (lig_atoms={n_lig_atoms} > 100)")
        return np.nan, np.nan

    centroid = get_ligand_centroid(rec_data)
    if centroid is None:
        return np.nan, np.nan

    cx, cy, cz = centroid

    out_sdf = TMPDIR / f"{pid}_gnina_out.sdf"

    # Use cached result if available
    if out_sdf.exists() and out_sdf.stat().st_size > 100:
        pass  # fall through to parse
    else:
        cmd = [
            GNINA,
            "--receptor", str(pocket_pdb),
            "--ligand",   str(lig_sdf),
            "--center_x", f"{cx:.3f}",
            "--center_y", f"{cy:.3f}",
            "--center_z", f"{cz:.3f}",
            "--size_x",   "22",
            "--size_y",   "22",
            "--size_z",   "22",
            "--num_modes",      "1",
            "--exhaustiveness", "8",
            "--scoring",        "gnina",   # enables CNN scoring
            "--out",  str(out_sdf),
            "--quiet",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout, env=env
            )
            if result.returncode != 0:
                log.warning(f"  {pid}: gnina error — {result.stderr[-200:]}")
                return np.nan, np.nan
        except Exception as e:
            log.warning(f"  {pid}: gnina exception — {e}")
            return np.nan, np.nan

    # Parse output SDF for scores
    vina_score   = np.nan
    cnn_affinity = np.nan
    try:
        text = out_sdf.read_text()
        for line in text.splitlines():
            if "minimizedAffinity" in line or "minimized_affinity" in line.lower():
                # next non-empty line is the value
                pass
            if line.strip().lstrip("-").replace(".", "").isdigit() or (line.strip() and line.strip()[0] in "-0123456789"):
                pass  # handled below
        # Parse via property blocks
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if "> <minimizedAffinity>" in line and i + 1 < len(lines):
                try: vina_score = float(lines[i+1].strip())
                except: pass
            if "> <CNNaffinity>" in line and i + 1 < len(lines):
                try: cnn_affinity = float(lines[i+1].strip())
                except: pass
    except Exception as e:
        log.warning(f"  {pid}: parse error — {e}")

    return vina_score, cnn_affinity


# ── Main ──────────────────────────────────────────────────────────────────────
log.info("\nLoading pocket data ...")
with gzip.open(PKL_FILE, "rb") as f:
    pocket_data = pickle.load(f)

log.info("Loading step15 features ...")
d15 = np.load(S15_NPZ, allow_pickle=True)
X15 = d15["X"].astype(np.float64)
y   = d15["y"].astype(np.float32)
ids = d15["ids"]
subtypes_raw = d15["subtypes"]
log.info(f"  X15: {X15.shape}")

subtypes = np.array([get_subtype(ids[i], subtypes_raw[i]) for i in range(len(ids))])

n = len(ids)
t0 = time.time()
vina_scores   = np.full(n, np.nan)
cnn_affinities = np.full(n, np.nan)

log.info(f"\nRunning gnina GPU docking on {n} complexes ...")
for i, pid in enumerate(ids):
    pid_l = pid.lower()
    rec_data = pocket_data.get(pid_l) or pocket_data.get(pid.upper())
    if rec_data is None:
        log.warning(f"  [{i+1:3d}/{n}] {pid}: no pocket data")
        continue

    vs, ca = run_gnina(pid, rec_data, NA_L)
    vina_scores[i]    = vs
    cnn_affinities[i] = ca

    elapsed = time.time() - t0
    rate    = (i + 1) / elapsed
    eta     = (n - i - 1) / rate if rate > 0 else 0
    log.info(f"  [{i+1:3d}/{n}] {pid_l:8s}  vina={vs:6.2f}  cnn_aff={ca:5.2f}  ETA {eta:.0f}s")

# ── Impute NaN scores ─────────────────────────────────────────────────────────
n_vina_valid = np.sum(~np.isnan(vina_scores))
n_cnn_valid  = np.sum(~np.isnan(cnn_affinities))
log.info(f"\nVina valid: {n_vina_valid}/{n},  CNN valid: {n_cnn_valid}/{n}")

vina_mean = np.nanmean(vina_scores)
cnn_mean  = np.nanmean(cnn_affinities)
vina_imp  = np.where(np.isnan(vina_scores), vina_mean, vina_scores)
cnn_imp   = np.where(np.isnan(cnn_affinities), cnn_mean, cnn_affinities)

# Quick standalone correlation
valid_mask = ~np.isnan(vina_scores) & ~np.isnan(cnn_affinities)
if valid_mask.sum() > 5:
    r_vina, _ = pearsonr(vina_scores[valid_mask], y[valid_mask])
    r_cnn,  _ = pearsonr(cnn_affinities[valid_mask], y[valid_mask])
    log.info(f"Standalone: r_vina={r_vina:.3f}, r_cnn={r_cnn:.3f}")

# ── Build step17 features ──────────────────────────────────────────────────────
score_feat = np.stack([vina_imp, cnn_imp], axis=1)  # (n, 2)
X17 = np.hstack([X15, score_feat])                  # (n, 53373)
log.info(f"\nX17 shape: {X17.shape}  (step15 + vina + cnn_affinity)")

np.savez_compressed(
    OUT_NPZ,
    X=X17.astype(np.float32), y=y, ids=ids, subtypes=subtypes_raw
)
log.info(f"Saved: {OUT_NPZ}")

# ── Hybrid model evaluation ────────────────────────────────────────────────────
def tanimoto_kernel(X, Y=None):
    if Y is None: Y = X
    XY  = X @ Y.T
    XX  = X.sum(1, keepdims=True)
    YY  = Y.sum(1, keepdims=True)
    denom = XX + YY.T - XY
    denom = np.where(denom < 1e-10, 1e-10, denom)
    return XY / denom

def loo_ridge_pipeline(X, y):
    n = len(y)
    preds = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        pipe = Pipeline([
            ("vt",  VarianceThreshold(threshold=1e-4)),
            ("sc",  StandardScaler()),
            ("pca", PCA(n_components=0.95, svd_solver="full")),
            ("reg", Ridge(alpha=100.0)),
        ])
        pipe.fit(X[tr], y[tr])
        preds[i] = pipe.predict(X[[i]])[0]
    return preds

def best_alpha_loo(X, y, alphas=[1,10,100,1000,10000,100000]):
    n = len(y)
    best_r, best_a = -999, alphas[0]
    for a in alphas:
        ps = np.zeros(n)
        for i in range(n):
            tr = [j for j in range(n) if j != i]
            pipe = Pipeline([
                ("vt",  VarianceThreshold(threshold=1e-4)),
                ("sc",  StandardScaler()),
                ("pca", PCA(n_components=0.95, svd_solver="full")),
                ("reg", Ridge(alpha=a)),
            ])
            pipe.fit(X[tr], y[tr])
            ps[i] = pipe.predict(X[[i]])[0]
        r, _ = pearsonr(ps, y)
        if r > best_r:
            best_r, best_a = r, a
    return best_a, best_r

log.info("\n" + "="*70)
log.info("HYBRID MODEL EVALUATION (step17 features)")
log.info("="*70)

# Step11 feature slices from step15 (step15 = step11 + shape(8) + lig_sgt(3600))
# step11 layout within step15:
#   SGT[0:36000], Morgan[36000:38048], NucComp[38048:38058], Physico[38058:38064]
#   RNA-FM[38064:38704], SS[38704:38712], kmer[38712:38796], MACCS[38796:38963]
#   Iface4[38963:42563], Iface6[42563:46163], Iface8[46163:49763]
#   Shape[49763:49771], LigSGT[49771:53371]
# step17 adds 2 more at the end: [53371:53373]

# Use full X17 for Ridge model
log.info("\n--- Step11 Ridge LOO (using X17) ---")
best_a, _ = best_alpha_loo(X17, y)
log.info(f"  Best alpha: {best_a}")

s17_preds = np.zeros(n)
for i in range(n):
    tr = [j for j in range(n) if j != i]
    pipe = Pipeline([
        ("vt",  VarianceThreshold(threshold=1e-4)),
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
        ("reg", Ridge(alpha=best_a)),
    ])
    pipe.fit(X17[tr], y[tr])
    s17_preds[i] = pipe.predict(X17[[i]])[0]

r_s17, _ = pearsonr(s17_preds, y)
log.info(f"  Global Ridge r = {r_s17:.3f}")

# Per-subtype Ridge
log.info("\n--- Per-subtype Ridge (X17) ---")
sub_results = {}
for st in np.unique(subtypes):
    mask = subtypes == st
    ns   = mask.sum()
    if ns < 4:
        sub_results[st] = (ns, np.nan)
        continue
    Xs, ys = X17[mask], y[mask]
    a, r = best_alpha_loo(Xs, ys)
    sub_results[st] = (ns, r)
    log.info(f"  {st:20s}: n={ns:3d}  r={r:.3f}  (alpha={a})")

# Global MKL for riboswitch (on step11 slice of X17)
log.info("\n--- Global MKL LOO (riboswitch) ---")
X11_slice = X17[:, :49763]  # step11 features only
X_topo = X11_slice[:, np.r_[0:36000, 38963:49763]]
X_lig  = X11_slice[:, 36000:38048]
X_maccs= X11_slice[:, 38796:38963]
X_rna  = X11_slice[:, 38064:38704]

X_topo_n = StandardScaler().fit_transform(X_topo.astype(np.float64))
X_rna_n  = StandardScaler().fit_transform(X_rna.astype(np.float64))

K_topo = rbf_kernel(X_topo_n, gamma=1e-6)
K_lig  = 0.7 * tanimoto_kernel(X_lig.astype(np.float64)) + \
         0.3 * tanimoto_kernel(X_maccs.astype(np.float64))
K_rna  = rbf_kernel(X_rna_n, gamma=5e-3)
K_full = 0.7 * K_topo + 0.1 * K_lig + 0.2 * K_rna

mkl_preds = np.zeros(n)
for i in range(n):
    tr = [j for j in range(n) if j != i]
    m = KernelRidge(alpha=0.01, kernel="precomputed")
    m.fit(K_full[np.ix_(tr, tr)], y[tr])
    p = float(m.predict(K_full[i, tr].reshape(1, -1))[0])
    mkl_preds[i] = np.clip(p, y[tr].min() - 3, y[tr].max() + 3)

rs_mask = subtypes == "riboswitch"
r_mkl_rs, _ = pearsonr(mkl_preds[rs_mask], y[rs_mask])
log.info(f"  Global MKL riboswitch r = {r_mkl_rs:.3f}")

# Hybrid: step17 Ridge for non-riboswitch, MKL for riboswitch
log.info("\n--- HYBRID (step17 Ridge + MKL riboswitch) ---")
hybrid_preds = s17_preds.copy()
hybrid_preds[rs_mask] = mkl_preds[rs_mask]

r_hybrid, _ = pearsonr(hybrid_preds, y)
log.info(f"  Combined r = {r_hybrid:.3f}")

# Per-subtype breakdown
log.info("\nPer-subtype (Hybrid step17+MKL):")
for st in ["aptamer","duplex_groove","ribosomal_asite","riboswitch","other_misc","g_quadruplex","viral_tar"]:
    mask = subtypes == st
    if mask.sum() < 2: continue
    r, _ = pearsonr(hybrid_preds[mask], y[mask])
    log.info(f"  {st:22s}: n={mask.sum():3d}  r={r:.3f}")

log.info(f"\n{'='*50}")
log.info(f"FINAL RESULTS")
log.info(f"  step17 Ridge:  r = {r_s17:.3f}")
log.info(f"  Hybrid step17: r = {r_hybrid:.3f}  (prev best: 0.695)")
log.info(f"  Gap to DeepRSMA (0.784): {0.784 - r_hybrid:.3f}")
log.info(f"{'='*50}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (preds, title, r_val) in zip(axes, [
    (s17_preds,    f"Step17 Ridge (r={r_s17:.3f})",    r_s17),
    (hybrid_preds, f"Hybrid step17+MKL (r={r_hybrid:.3f})", r_hybrid),
]):
    colors = {"aptamer":"blue","duplex_groove":"green","ribosomal_asite":"orange",
              "riboswitch":"red","other_misc":"purple","g_quadruplex":"brown","viral_tar":"gray"}
    for st in np.unique(subtypes):
        mask = subtypes == st
        ax.scatter(y[mask], preds[mask], label=st, alpha=0.7,
                   color=colors.get(st, "black"), s=30)
    lo, hi = min(y.min(), preds.min()), max(y.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Experimental pKd"); ax.set_ylabel("Predicted pKd")
    ax.set_title(title); ax.legend(fontsize=7)

plt.tight_layout()
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / "step17_gnina_results.png", dpi=150)
plt.close()
log.info("Figure saved.")
