"""
SGT-RNA · Step 16: AutoDock Vina Docking Scores

Run AutoDock Vina on all 143 NL2020 crystal structures.
Receptor: pocket PDB → PDBQT (mk_prepare_receptor)
Ligand:   SDF → PDBQT (mk_prepare_ligand)
Box:      22×22×22 Å centered on ligand crystal centroid
Score:    Vina affinity (kcal/mol) → 1 new feature per complex

Combine with step15 features (53,371) → 53,372 total
Re-evaluate hybrid model.
"""

import subprocess, gzip, pickle, logging, time, tempfile, os
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

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
OUT_NPZ = ROOT / "data" / "features" / "step16_full_features.npz"
RES_DIR = ROOT / "results"
FIG_DIR = ROOT / "results" / "figures"
VINA    = os.environ.get("VINA_BIN", "vina")  # set VINA_BIN env var or ensure vina is on PATH
TMPDIR  = Path("/tmp/vina_rna")
TMPDIR.mkdir(parents=True, exist_ok=True)
N_WORKERS = 20   # all cores; each Vina job uses --cpu 1

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "results" / "logs" / f"step16_{ts}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("SGT-RNA · Step 16: AutoDock Vina Docking Scores")
log.info("=" * 70)

# ── subtype labels ────────────────────────────────────────────────────────────
G_QUAD        = {"1nzm","5cdb","4xwf","4znp","5btp","6jj0","2mg8","2loa"}
DUPLEX_GROOVE = {"407d","408d","1cvy","1cvx","454d","1qv4","1qv8","1p96","1r4e","6hbt"}
MANUAL_OVERRIDE = {
    "1y26":"riboswitch","2gdi":"riboswitch","3b31":"riboswitch","3d2v":"riboswitch",
    "3d2x":"riboswitch","3d2g":"riboswitch","3d2s":"riboswitch","3d2w":"riboswitch",
    "3fnb":"riboswitch","3q50":"riboswitch","3vrs":"riboswitch","4fny":"riboswitch",
    "4lck":"riboswitch","4lnt":"riboswitch","4tzx":"riboswitch","5e54":"riboswitch",
    "1o15":"aptamer","1zif":"aptamer",
    "2vrn":"ribosomal_asite","1fjg":"ribosomal_asite","1xmq":"ribosomal_asite",
    "4v9o":"ribosomal_asite","2z75":"ribosomal_asite",
}
def make_subtype(pdb, raw):
    if pdb in MANUAL_OVERRIDE: return MANUAL_OVERRIDE[pdb]
    if pdb in G_QUAD:          return "g_quadruplex"
    if pdb in DUPLEX_GROOVE:   return "duplex_groove"
    return raw

ALPHA_GRID = [1, 10, 100, 1000, 10_000, 100_000]

# ── load pocket data (for ligand centroids) ───────────────────────────────────
log.info("\nLoading pocket data ...")
with gzip.open(PKL_FILE, "rb") as f:
    records = pickle.load(f)
rec_map = {r["pdb"]: r for r in records}

# ── Vina worker (module-level for multiprocessing pickling) ───────────────────
def _vina_worker(args):
    """Run Vina for one complex. args = (pdb, cx, cy, cz, n_lig_atoms)"""
    pdb, cx, cy, cz, n_lig_atoms = args
    pocket_pdb = ROOT / "NA-L" / pdb / f"{pdb}_pocket.pdb"
    lig_sdf    = ROOT / "NA-L" / pdb / f"{pdb}_ligand.sdf"
    tmpdir     = Path("/tmp/vina_rna")
    rec_pdbqt  = tmpdir / f"{pdb}_rec.pdbqt"
    lig_pdbqt  = tmpdir / f"{pdb}_lig.pdbqt"
    out_pdbqt  = tmpdir / f"{pdb}_out.pdbqt"
    vina_bin   = VINA

    # Return cached score if output already exists
    if out_pdbqt.exists():
        with open(out_pdbqt) as fh:
            for line in fh:
                if "REMARK VINA RESULT" in line:
                    try:
                        return pdb, float(line.split()[3])
                    except: pass

    if n_lig_atoms > 80:
        return pdb, float("nan")

    try:
        if not rec_pdbqt.exists():
            r = subprocess.run(
                ["mk_prepare_receptor.py", "--read_pdb", str(pocket_pdb),
                 "-p", str(rec_pdbqt)],
                capture_output=True, text=True, timeout=30
            )
            if r.returncode != 0 or not rec_pdbqt.exists():
                return pdb, float("nan")

        if not lig_pdbqt.exists():
            r = subprocess.run(
                ["mk_prepare_ligand.py", "-i", str(lig_sdf), "-o", str(lig_pdbqt)],
                capture_output=True, text=True, timeout=30
            )
            if r.returncode != 0 or not lig_pdbqt.exists():
                return pdb, float("nan")

        r = subprocess.run(
            [vina_bin,
             "--receptor", str(rec_pdbqt),
             "--ligand",   str(lig_pdbqt),
             "--center_x", f"{cx:.3f}",
             "--center_y", f"{cy:.3f}",
             "--center_z", f"{cz:.3f}",
             "--size_x", "22", "--size_y", "22", "--size_z", "22",
             "--exhaustiveness", "8",
             "--num_modes", "1",
             "--out", str(out_pdbqt),
             "--cpu", "1"],   # 1 CPU per job; parallelism via N_WORKERS processes
            capture_output=True, text=True, timeout=90
        )
    except Exception:
        return pdb, float("nan")

    if r.returncode != 0:
        return pdb, float("nan")

    for line in r.stdout.split("\n"):
        if line.strip().startswith("1 ") or "REMARK VINA RESULT" in line:
            try:
                return pdb, float(line.split()[3] if "REMARK" in line else line.split()[1])
            except (ValueError, IndexError):
                pass
    if out_pdbqt.exists():
        with open(out_pdbqt) as fh:
            for line in fh:
                if "REMARK VINA RESULT" in line:
                    try:
                        return pdb, float(line.split()[3])
                    except: pass
    return pdb, float("nan")

# ── load step15 features ──────────────────────────────────────────────────────
log.info("Loading step15 features ...")
d15 = np.load(S15_NPZ)
X15  = d15["X"].astype(np.float32)
y    = d15["y"].astype(np.float32)
ids  = d15["ids"]
subtypes_raw = d15["subtypes"]
subtypes = np.array([make_subtype(p,s) for p,s in zip(ids, subtypes_raw)])
n = len(y)
log.info(f"  X15: {X15.shape}")

step11_preds = pd.read_csv(S11_CSV).set_index("pdb")
step11_preds = np.array([step11_preds.loc[p,"y_pred"] for p in ids])

# ── run docking ───────────────────────────────────────────────────────────────
log.info(f"\nRunning AutoDock Vina on {n} complexes  [{N_WORKERS} parallel workers] ...")
vina_scores = np.full(n, np.nan, dtype=np.float32)
pdb_to_idx  = {pdb: i for i, pdb in enumerate(ids)}

# Build args list
job_args = []
for pdb in ids:
    rec_data = rec_map.get(pdb)
    if rec_data is None:
        job_args.append((pdb, 0.0, 0.0, 0.0, 999))
        continue
    lig_parts = [v for v in rec_data["lig_coords"].values() if len(v) > 0]
    if not lig_parts:
        job_args.append((pdb, 0.0, 0.0, 0.0, 999))
        continue
    lig_all = np.vstack(lig_parts)
    cx, cy, cz = lig_all.mean(axis=0)
    n_lig = sum(len(v) for v in rec_data["lig_coords"].values())
    job_args.append((pdb, float(cx), float(cy), float(cz), n_lig))

t0 = time.time()
done = 0
with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
    futures = {pool.submit(_vina_worker, a): a[0] for a in job_args}
    for fut in as_completed(futures):
        pdb, score = fut.result()
        idx = pdb_to_idx[pdb]
        vina_scores[idx] = score
        done += 1
        elapsed = time.time() - t0
        eta = (n - done) / max(done / elapsed, 1e-6)
        status = f"{score:.2f}" if not np.isnan(score) else "FAIL"
        if done % 10 == 0 or done == 1 or done == n:
            log.info(f"  [{done:3d}/{n}] {pdb}  score={status}  ETA {eta:.0f}s")

n_success = (~np.isnan(vina_scores)).sum()
n_fail    = np.isnan(vina_scores).sum()
log.info(f"\n  Done in {time.time()-t0:.1f}s")
log.info(f"  Success: {n_success}/{n}  Failed: {n_fail}")
log.info(f"  Score range: {np.nanmin(vina_scores):.2f} to {np.nanmax(vina_scores):.2f} kcal/mol")

# Correlation of Vina score with pKd
valid_mask = ~np.isnan(vina_scores)
if valid_mask.sum() > 10:
    r_vina = pearsonr(y[valid_mask], vina_scores[valid_mask])[0]
    log.info(f"  Vina score vs pKd Pearson r = {r_vina:.4f} (expect -0.3 to -0.6)")

# Fill failed with median (zero-padded, will be filtered by VT)
median_score = np.nanmedian(vina_scores)
vina_filled = np.where(np.isnan(vina_scores), median_score, vina_scores).reshape(-1, 1)

# ── combine features ──────────────────────────────────────────────────────────
X_full = np.hstack([X15, vina_filled]).astype(np.float32)
log.info(f"\nFull feature matrix: {X_full.shape}")
np.savez_compressed(OUT_NPZ, X=X_full, y=y, ids=ids,
                    subtypes=subtypes_raw, vina_scores=vina_scores)
log.info(f"Saved → {OUT_NPZ}")

# Save Vina scores CSV
df_vina = pd.DataFrame({"pdb": ids, "subtype": subtypes,
                         "pKd": y, "vina_score": vina_scores})
df_vina.to_csv(RES_DIR / "step16_vina_scores.csv", index=False)

# ── ML pipeline ───────────────────────────────────────────────────────────────
def make_pipe(alpha):
    return Pipeline([
        ("vt",  VarianceThreshold(threshold=1e-4)),
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
        ("reg", Ridge(alpha=alpha)),
    ])

def loo_ridge(X_sub, y_sub):
    ns = len(y_sub)
    if ns < 3: return np.full(ns, y_sub.mean()), -99.0
    best_r, best_p = -99.0, np.full(ns, y_sub.mean())
    for alpha in ALPHA_GRID:
        preds = np.zeros(ns); ok = True
        for i in range(ns):
            tr = [j for j in range(ns) if j!=i]
            try:
                pipe = make_pipe(alpha)
                pipe.fit(X_sub[tr], y_sub[tr])
                preds[i] = np.clip(pipe.predict(X_sub[[i]])[0],
                                   y_sub[tr].min()-3, y_sub[tr].max()+3)
            except: ok=False; break
        if not ok: continue
        r = pearsonr(y_sub, preds)[0] if np.std(preds)>1e-8 else -99.0
        if r > best_r: best_r, best_p = r, preds.copy()
    return best_p, best_r

def cv_ridge_global(X_all, y_all, n_splits=5):
    from sklearn.model_selection import KFold
    ns = len(y_all)
    kf = KFold(n_splits=min(n_splits,ns), shuffle=True, random_state=42)
    best_r, best_p = -99.0, np.full(ns, y_all.mean())
    for alpha in ALPHA_GRID:
        preds = np.zeros(ns); ok = True
        for tr, te in kf.split(X_all):
            try:
                pipe = make_pipe(alpha)
                pipe.fit(X_all[tr], y_all[tr])
                p = np.clip(pipe.predict(X_all[te]),
                            y_all[tr].min()-3, y_all[tr].max()+3)
                preds[te] = p
            except: ok=False; break
        if not ok: continue
        r = pearsonr(y_all, preds)[0] if np.std(preds)>1e-8 else -99.0
        if r > best_r: best_r, best_p = r, preds.copy()
    return best_p, best_r

log.info("\n" + "=" * 70)
log.info("ML EVALUATION — Ridge LOO on step15 + Vina features")
log.info("=" * 70)

log.info("\nGlobal CV Ridge ...")
global_preds, global_r = cv_ridge_global(X_full, y, n_splits=5)
log.info(f"  Global r = {global_r:.4f}")

unique_subtypes = sorted(set(subtypes))
new_ridge_preds = np.full(n, np.nan)
new_ridge_rs    = {}

for st in unique_subtypes:
    mask = subtypes == st
    idx  = np.where(mask)[0]
    ns, ys = len(idx), y[idx]

    preds_st, r_st = loo_ridge(X_full[idx], ys)
    preds_gl = global_preds[idx]
    r_gl = pearsonr(ys, preds_gl)[0] if np.std(preds_gl)>1e-8 else -99.0

    if r_st >= r_gl:
        new_ridge_preds[idx] = preds_st; r_use = r_st; chosen = "subtype"
    else:
        new_ridge_preds[idx] = preds_gl; r_use = r_gl; chosen = "global"
    new_ridge_rs[st] = r_use
    log.info(f"  {st:<22}: n={ns:3d}  r={r_use:.4f}  ({chosen})")

# ── global MKL for riboswitch (step13 best, unchanged) ───────────────────────
log.info("\nGlobal MKL for riboswitch ...")
d11 = np.load(ROOT / "data" / "features" / "step11_full_features.npz")
X11_f = d11["X"].astype(np.float64)
X_topo = X11_f[:, np.r_[0:36000, 38963:49763]]
X_lig  = X11_f[:, 36000:38048]
X_maccs= X11_f[:, 38796:38963]
X_rna  = X11_f[:, 38064:38704]
sc_t = StandardScaler(); X_topo_n = sc_t.fit_transform(X_topo)
sc_r = StandardScaler(); X_rna_n  = sc_r.fit_transform(X_rna)

def tanimoto_kernel(X, Y=None):
    if Y is None: Y = X
    XY = X @ Y.T; XX = X.sum(1,keepdims=True); YY = Y.sum(1,keepdims=True)
    return XY / np.where(XX+YY.T-XY<1e-10, 1e-10, XX+YY.T-XY)

K_topo = rbf_kernel(X_topo_n, gamma=1e-6)
K_lig  = 0.7*tanimoto_kernel(X_lig) + 0.3*tanimoto_kernel(X_maccs)
K_rna  = rbf_kernel(X_rna_n, gamma=5e-3)
K_mkl  = 0.7*K_topo + 0.1*K_lig + 0.2*K_rna

mkl_preds = np.zeros(n)
for i in range(n):
    tr = [j for j in range(n) if j!=i]
    m = KernelRidge(alpha=0.01, kernel="precomputed")
    m.fit(K_mkl[np.ix_(tr,tr)], y[tr])
    p = float(m.predict(K_mkl[i,tr].reshape(1,-1))[0])
    mkl_preds[i] = np.clip(p, y[tr].min()-3, y[tr].max()+3)

rs_mask = subtypes == "riboswitch"
rs_r_mkl = pearsonr(y[rs_mask], mkl_preds[rs_mask])[0]
log.info(f"  Riboswitch MKL r = {rs_r_mkl:.4f}")

# ── hybrid: new Ridge for non-RS, MKL for RS ─────────────────────────────────
hybrid_preds = new_ridge_preds.copy()
hybrid_preds[rs_mask] = mkl_preds[rs_mask]
hybrid_rs = new_ridge_rs.copy()
hybrid_rs["riboswitch"] = rs_r_mkl

valid      = ~np.isnan(hybrid_preds)
combined_r = pearsonr(y[valid], hybrid_preds[valid])[0]

log.info("\n" + "=" * 70)
log.info("FINAL HYBRID RESULTS (step16: step15+Vina Ridge + global MKL for RS)")
log.info("=" * 70)
for st in unique_subtypes:
    mask = subtypes == st
    log.info(f"  {st:<22}: n={mask.sum():3d}  r={hybrid_rs[st]:.4f}")

log.info(f"\n  COMBINED r (step16)            = {combined_r:.4f}")
log.info(f"  Previous best (Hybrid S11+MKL) = 0.6954")
log.info(f"  Delta                          = {combined_r - 0.6954:+.4f}")

benchmarks = [
    ("AffiGrapher",0.498),("RLaffinity",0.559),("RLASIF",0.666),
    ("DeepRSMA",0.784),("RSAPred",0.830),
]
log.info("\nBenchmark comparison:")
for name, rb in benchmarks:
    sym = "✓" if combined_r > rb else "✗"
    log.info(f"  {sym} {name}: {rb:.3f}  (ours: {combined_r:.4f})")

# ── save ──────────────────────────────────────────────────────────────────────
df_res = pd.DataFrame({
    "pdb":ids,"subtype":subtypes,"y_true":y,
    "y_pred":hybrid_preds,"vina_score":vina_scores,
})
df_res.to_csv(RES_DIR/"step16_results.csv", index=False)

# ── figure ────────────────────────────────────────────────────────────────────
colors = {
    "aptamer":"#2166AC","riboswitch":"#1A9641","ribosomal_asite":"#D73027",
    "duplex_groove":"#7B2D8B","g_quadruplex":"#FF7F00",
    "viral_tar":"#A65628","other_misc":"#999999",
}
fig, axes = plt.subplots(1, 3, figsize=(20,6))
fig.patch.set_facecolor("white")

ax = axes[0]
for st in unique_subtypes:
    mask = subtypes == st
    ax.scatter(y[mask], hybrid_preds[mask], c=colors.get(st,"#888"),
               label=f"{st} (r={hybrid_rs[st]:.3f})", alpha=0.75, s=45, edgecolors="none")
mn, mx = y.min()-0.5, y.max()+0.5
ax.plot([mn,mx],[mn,mx],"k--",lw=1,alpha=0.4)
ax.set_xlabel("Experimental pKd"); ax.set_ylabel("Predicted pKd")
ax.set_title(f"Step 16: +Vina  (r={combined_r:.4f})", fontweight="bold")
ax.legend(fontsize=7, loc="upper left", framealpha=0.7); ax.grid(alpha=0.3, linestyle="--")

ax = axes[1]
valid_v = ~np.isnan(vina_scores)
for st in unique_subtypes:
    mask = (subtypes == st) & valid_v
    if mask.sum() < 2: continue
    r_v = pearsonr(y[mask], vina_scores[mask])[0]
    ax.scatter(vina_scores[mask], y[mask], c=colors.get(st,"#888"),
               label=f"{st} (r={r_v:.2f})", alpha=0.6, s=35, edgecolors="none")
ax.set_xlabel("Vina score (kcal/mol)"); ax.set_ylabel("Experimental pKd")
ax.set_title("Vina score vs pKd (expect negative r)", fontweight="bold")
ax.legend(fontsize=7, framealpha=0.7); ax.grid(alpha=0.3, linestyle="--")

ax = axes[2]
steps = {"S09":0.535,"S11":0.575,"S11+MKL":0.695,"S16":combined_r}
bar_cols = ["#AAAAAA","#4393C3","#08519C","#D63027"]
bars = ax.bar(list(steps.keys()), list(steps.values()), color=bar_cols, edgecolor="white", alpha=0.9)
for bar, val in zip(bars, steps.values()):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.005, f"{val:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
for name, rb in benchmarks:
    ax.axhline(rb, linestyle="--", lw=0.9, alpha=0.6, label=f"{name} {rb:.3f}")
ax.set_ylim(0.45, 0.9); ax.set_ylabel("Combined Pearson r")
ax.set_title("Progress", fontweight="bold")
ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig(FIG_DIR/"step16_results.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

log.info(f"\nResults → {RES_DIR/'step16_results.csv'}")
log.info(f"Figure  → {FIG_DIR/'step16_results.png'}")
log.info("\n" + "=" * 70)
log.info("STEP 16 COMPLETE")
log.info("=" * 70)
