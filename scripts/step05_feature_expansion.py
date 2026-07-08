"""
SGT-RNA  ·  Step 5: Feature Expansion

New features added on top of Step 4 SGT baseline:
  1. Extended SGT scales
       Exponential kernel: η ∈ {1, 2, 3, 5, 8, 10, 20}  (7 scales)
       Lorentz kernel:     η ∈ {2, 5, 8}               (3 scales)
       Total: 10 × 3600 = 36,000 SGT features
  2. Morgan fingerprints (ECFP4, radius=2, 2048-bit) from ligand SDF
  3. RNA nucleotide composition from pocket PDB
       counts A,U,G,C + fractions fA,fU,fG,fC + GC% + purine% (10 features)
  4. Ligand physicochemical from dataset.csv
       mol_weight, n_rings, n_hbd, n_hba, n_rot_bonds, tpsa (6 features)
  Grand total: ~38,064 features → PCA(95%) → Ridge  (nested 5×5 CV)
"""

import gzip, pickle, logging, warnings, time
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy import stats

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path("/home/stalin/Desktop/SGT-RNA/RNA_SGT")
PKL_FILE = ROOT / "data" / "pocket_fri" / "pocket_fri_data.pkl.gz"
NPZ_S4   = ROOT / "data" / "features" / "step04_multiscale_features.npz"
CSV_FILE = ROOT / "data" / "affinity" / "dataset.csv"
OUT_DIR  = ROOT / "data" / "features"
RES_DIR  = ROOT / "results"
FIG_DIR  = ROOT / "results" / "figures"
LOG_DIR  = ROOT / "results" / "logs"
for d in [OUT_DIR, RES_DIR, FIG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── logging ───────────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"step05_{ts}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("SGT-RNA  ·  Step 5: Feature Expansion")
log.info("=" * 70)

# ── constants ─────────────────────────────────────────────────────────────────
RNA_ELEMENTS = ["C", "N", "O", "P"]
LIG_ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
THRESHOLDS   = [0.0, 0.8, 0.85, 0.90, 0.95]
N_STATS      = 10
N_PAIRS      = 36
N_FEATS_SINGLE = N_PAIRS * len(THRESHOLDS) * 2 * N_STATS   # 3600
EPS          = 1e-8
KAPPA        = 2.0
SEED         = 42

EXP_ETAS     = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 20.0]   # 7 exp scales
LOR_ETAS     = [2.0, 5.0, 8.0]                           # 3 Lorentz scales

# ── kernels ───────────────────────────────────────────────────────────────────
def fri_exp(D, eta):
    return np.exp(-(D / eta) ** KAPPA)

def fri_lorentz(D, eta):
    return 1.0 / (1.0 + (D / eta) ** KAPPA)

# ── spectral stats (reused from step03/04) ────────────────────────────────────
def spectral_stats(eigs):
    if len(eigs) == 0:
        return np.zeros(N_STATS, dtype=np.float32)
    nz = eigs[eigs > EPS]
    return np.array([
        eigs.min(), eigs.max(), eigs.mean(), float(np.median(eigs)),
        eigs.var(), eigs.std(), eigs.sum(), float((eigs ** 2).sum()),
        float(len(nz)), float(len(eigs) - len(nz))
    ], dtype=np.float32)

def build_L0(W_sel):
    nr, nl = W_sel.shape
    n = nr + nl
    L = np.zeros((n, n), dtype=np.float64)
    rd = W_sel.sum(1); cd = W_sel.sum(0)
    np.fill_diagonal(L[:nr, :nr], rd)
    np.fill_diagonal(L[nr:, nr:], cd)
    L[:nr, nr:] = -W_sel; L[nr:, :nr] = -W_sel.T
    return L

def pair_features(rc, lc, kernel_fn):
    nr, nl = len(rc), len(lc)
    nf = len(THRESHOLDS) * 2 * N_STATS
    if nr == 0 or nl == 0:
        return np.zeros(nf, dtype=np.float32)
    D = cdist(rc, lc).astype(np.float64)
    W = kernel_fn(D)
    wmax = W.max()
    if wmax < 1e-12:
        return np.zeros(nf, dtype=np.float32)
    W_norm = W / wmax
    n_total = nr + nl
    feats = np.zeros(nf, dtype=np.float32)
    off = 0
    for tau in THRESHOLDS:
        mask  = W_norm >= tau
        W_sel = W_norm * mask
        ne    = int(mask.sum())
        if ne == 0:
            eL0 = np.zeros(n_total); eL1 = np.array([])
        else:
            L0  = build_L0(W_sel)
            eL0 = np.maximum(np.linalg.eigvalsh(L0), 0.0)
            b0  = int((eL0 < EPS).sum())
            nz1 = max(0, ne - n_total + b0)
            eL1 = np.concatenate([np.zeros(nz1), eL0[eL0 >= EPS]])
        feats[off:off + N_STATS]           = spectral_stats(eL0)
        feats[off + N_STATS:off + 2*N_STATS] = spectral_stats(eL1)
        off += 2 * N_STATS
    return feats

def compute_pdfl_scale(records, kernel_fn, label):
    n = len(records)
    X = np.zeros((n, N_FEATS_SINGLE), dtype=np.float32)
    for idx, rec in enumerate(records):
        parts = []
        for r_el in RNA_ELEMENTS:
            for l_el in LIG_ELEMENTS:
                rc = rec["rna_coords"][r_el]
                lc = rec["lig_coords"].get(l_el, np.empty((0, 3), np.float32))
                parts.append(pair_features(rc, lc, kernel_fn))
        X[idx] = np.concatenate(parts)
    log.info(f"    {label} done")
    return X

# ── load step04 multi-scale features (exp η=2,5,8) ───────────────────────────
log.info("\nLoading Step 4 features (exp η=2,5,8) ...")
d4   = np.load(NPZ_S4, allow_pickle=True)
X_s4 = d4["X"].astype(np.float32)   # 143 × 10800
y    = d4["y"].astype(np.float32)
ids  = d4["ids"].tolist()
n    = len(y)
log.info(f"  Loaded: {X_s4.shape}")

# ── load coordinates ──────────────────────────────────────────────────────────
log.info("\nLoading pocket coordinates ...")
with gzip.open(PKL_FILE, "rb") as f:
    records = pickle.load(f)

# ── compute new SGT scales ───────────────────────────────────────────────────
log.info("\nComputing extended SGT scales ...")
new_scales = []

# Exponential: add η=1,3,10,20 (η=2,5,8 already in X_s4)
already_exp = {2.0, 5.0, 8.0}
for eta in EXP_ETAS:
    if eta in already_exp:
        continue
    t0 = time.time()
    Xe = compute_pdfl_scale(records, lambda D, e=eta: fri_exp(D, e), f"exp η={eta}")
    log.info(f"      time: {time.time()-t0:.1f}s")
    new_scales.append(Xe)

# Lorentz: η=2,5,8
for eta in LOR_ETAS:
    t0 = time.time()
    Xl = compute_pdfl_scale(records, lambda D, e=eta: fri_lorentz(D, e), f"Lorentz η={eta}")
    log.info(f"      time: {time.time()-t0:.1f}s")
    new_scales.append(Xl)

X_pdfl = np.concatenate([X_s4] + new_scales, axis=1)
log.info(f"\n  Total SGT features: {X_pdfl.shape[1]}  "
         f"(10 scales × {N_FEATS_SINGLE})")

# ── load dataset.csv for file paths + ligand physico ─────────────────────────
log.info("\nLoading dataset.csv ...")
df = pd.read_csv(CSV_FILE)
df = df.set_index("pdb")
physico_cols = ["mol_weight", "n_rings", "n_hbd", "n_hba", "n_rot_bonds", "tpsa"]

# ── Morgan fingerprints ───────────────────────────────────────────────────────
log.info("\nComputing Morgan fingerprints (ECFP4, 2048-bit) ...")
MORGAN_BITS = 2048
X_morgan = np.zeros((n, MORGAN_BITS), dtype=np.float32)
fail = 0
for idx, pdb_id in enumerate(ids):
    try:
        sdf_path = df.loc[pdb_id, "lig_sdf"]
        mol = next(Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=True))
        if mol is None:
            raise ValueError("null mol")
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=MORGAN_BITS)
        X_morgan[idx] = np.array(fp, dtype=np.float32)
    except Exception:
        # fallback: mol2
        try:
            mol2_path = df.loc[pdb_id, "lig_mol2"]
            mol = Chem.MolFromMol2File(str(mol2_path), removeHs=True, sanitize=True)
            if mol is None:
                raise ValueError("null mol2")
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=MORGAN_BITS)
            X_morgan[idx] = np.array(fp, dtype=np.float32)
        except Exception:
            fail += 1
log.info(f"  Morgan FP done — {n - fail}/{n} succeeded, {fail} zeros")

# ── RNA nucleotide composition ────────────────────────────────────────────────
log.info("\nComputing RNA nucleotide composition ...")
RNA_NTS = ["A", "U", "G", "C"]
N_COMP   = 10   # counts×4 + fractions×4 + GC% + purine%
X_comp   = np.zeros((n, N_COMP), dtype=np.float32)
pdb_parser = PDBParser(QUIET=True)

for idx, pdb_id in enumerate(ids):
    try:
        pocket_path = df.loc[pdb_id, "pocket_pdb"]
        struct = pdb_parser.get_structure(pdb_id, str(pocket_path))
        cnt = Counter()
        for res in struct.get_residues():
            name = res.get_resname().strip()
            if name in RNA_NTS:
                cnt[name] += 1
        total = sum(cnt.values()) or 1
        counts = np.array([cnt["A"], cnt["U"], cnt["G"], cnt["C"]], dtype=np.float32)
        fracs  = counts / total
        gc     = fracs[2] + fracs[3]        # G + C
        purine = fracs[0] + fracs[2]        # A + G
        X_comp[idx] = np.concatenate([counts, fracs, [gc, purine]])
    except Exception:
        pass
log.info(f"  RNA composition done — shape {X_comp.shape}")

# ── ligand physicochemical (from dataset.csv) ─────────────────────────────────
log.info("\nLoading ligand physicochemical features ...")
X_phys = np.zeros((n, len(physico_cols)), dtype=np.float32)
for idx, pdb_id in enumerate(ids):
    try:
        X_phys[idx] = df.loc[pdb_id, physico_cols].values.astype(np.float32)
    except Exception:
        pass
log.info(f"  Physico features done — shape {X_phys.shape}")

# ── concatenate all features ──────────────────────────────────────────────────
X_all = np.concatenate([X_pdfl, X_morgan, X_comp, X_phys], axis=1)
log.info(f"\nCombined feature matrix: {X_all.shape}")
log.info(f"  SGT:    {X_pdfl.shape[1]}")
log.info(f"  Morgan:  {X_morgan.shape[1]}")
log.info(f"  RNA comp:{X_comp.shape[1]}")
log.info(f"  Physico: {X_phys.shape[1]}")

npz_path = OUT_DIR / "step05_expanded_features.npz"
np.savez_compressed(npz_path, X=X_all, y=y, ids=np.array(ids))
log.info(f"  Saved → {npz_path}")

# ── Ridge nested 5×5 CV ───────────────────────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("NESTED 5 × 5 CV  —  Ridge on expanded features")
log.info("=" * 70)

pipeline = Pipeline([
    ("vt",  VarianceThreshold(threshold=1e-4)),
    ("sc",  StandardScaler()),
    ("pca", PCA(n_components=0.95, random_state=SEED)),
    ("est", Ridge()),
])
param_grid = {"est__alpha": [0.01, 0.1, 1, 10, 100, 1000, 10000]}

outer_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
outer_splits = list(outer_cv.split(X_all))

oof = np.zeros(n, dtype=np.float32)
best_alphas = []
t0 = time.time()

for fold, (tr_idx, te_idx) in enumerate(outer_splits):
    gs = GridSearchCV(pipeline, param_grid, cv=inner_cv,
                      scoring="r2", n_jobs=10, refit=True)
    gs.fit(X_all[tr_idx], y[tr_idx])
    oof[te_idx] = gs.predict(X_all[te_idx]).astype(np.float32)
    best_alphas.append(gs.best_params_["est__alpha"])
    r_fold, _ = stats.pearsonr(y[te_idx], oof[te_idx])
    log.info(f"  Fold {fold+1}/5 | alpha={gs.best_params_['est__alpha']} | r={r_fold:.4f}")

elapsed = time.time() - t0
r_oof,   _ = stats.pearsonr(y, oof)
rho_oof, _ = stats.spearmanr(y, oof)
rmse  = float(np.sqrt(mean_squared_error(y, oof)))
mae   = float(mean_absolute_error(y, oof))
r2    = float(r2_score(y, oof))

log.info(f"\n  OOF Pearson r  = {r_oof:.4f}")
log.info(f"  OOF Spearman ρ = {rho_oof:.4f}")
log.info(f"  RMSE           = {rmse:.4f}")
log.info(f"  MAE            = {mae:.4f}")
log.info(f"  R²             = {r2:.4f}")
log.info(f"  Time           = {elapsed:.0f}s")

# ── comparison with step04 baseline ──────────────────────────────────────────
BASELINE_R = 0.5029
log.info(f"\n  Step 4 Ridge baseline : r = {BASELINE_R:.4f}")
log.info(f"  Step 5 expanded Ridge : r = {r_oof:.4f}")
delta = r_oof - BASELINE_R
log.info(f"  Δr = {delta:+.4f}  ({'improvement' if delta > 0 else 'regression'})")

BENCHMARKS = [
    ("AffiGrapher", 0.498), ("RLaffinity", 0.559),
    ("RLASIF", 0.666), ("DeepRSMA", 0.784), ("RSAPred", 0.830),
]
log.info("\n  Benchmark comparison:")
for bname, br in BENCHMARKS:
    sym = "✓ BEAT" if r_oof > br else "✗ below"
    log.info(f"    {sym}  {bname}: {br:.3f}")

# ── save results ──────────────────────────────────────────────────────────────
res = pd.DataFrame([{
    "model": "Step5_Ridge_expanded",
    "Pearson_r": round(r_oof, 4), "Spearman_rho": round(rho_oof, 4),
    "RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4),
    "n_features": X_all.shape[1],
    "best_alphas": str(best_alphas),
}])
res.to_csv(RES_DIR / "step05_results.csv", index=False)

# ── figure: pred vs obs + feature breakdown ───────────────────────────────────
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "figure.dpi": 300, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False})

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor("white")
fig.suptitle(f"SGT-RNA  |  Step 5: Expanded Features  (r = {r_oof:.4f})",
             fontsize=15, fontweight="bold")

# Panel A: pred vs obs
ax = axes[0]
slope, inter, *_ = stats.linregress(oof, y)
xr = np.linspace(oof.min() - 0.3, oof.max() + 0.3, 200)
sc = ax.scatter(oof, y, c=y, cmap="coolwarm", s=40, alpha=0.8,
                edgecolors="white", linewidths=0.4)
ax.plot(xr, slope * xr + inter, color="black", lw=2,
        label=f"r = {r_oof:.4f}\nρ = {rho_oof:.4f}\nRMSE = {rmse:.4f}")
ax.plot([y.min()-0.5, y.max()+0.5], [y.min()-0.5, y.max()+0.5],
        color="gray", ls=":", lw=1.2, label="ideal")
plt.colorbar(sc, ax=ax, label="Observed pKd", shrink=0.85)
ax.set_xlabel("Predicted pKd"); ax.set_ylabel("Observed pKd")
ax.set_title("A  |  Predicted vs Observed", fontweight="bold", loc="left")
ax.legend(fontsize=10); ax.grid(alpha=0.3, ls="--")

# Panel B: feature block comparison
ax = axes[1]
blocks = ["SGT\n10 scales\n(36,000)", "Morgan FP\nECFP4\n(2,048)",
          "RNA comp\n(10)", "Ligand\nphysico\n(6)"]
sizes  = [X_pdfl.shape[1], MORGAN_BITS, N_COMP, len(physico_cols)]
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
bars = ax.bar(blocks, sizes, color=colors, alpha=0.85, edgecolor="white")
for bar, s in zip(bars, sizes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f"{s:,}", ha="center", va="bottom", fontweight="bold", fontsize=10)
ax.set_ylabel("Number of features")
ax.set_title("B  |  Feature Breakdown", fontweight="bold", loc="left")
ax.grid(axis="y", alpha=0.3, ls="--")

# Add r comparison text
txt = (f"Step 4 baseline : r = {BASELINE_R:.4f}\n"
       f"Step 5 expanded : r = {r_oof:.4f}  (Δ = {delta:+.4f})")
ax.text(0.5, 0.92, txt, transform=ax.transAxes, ha="center", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig_path = FIG_DIR / "step05_expanded_results.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
log.info(f"\n  Figure saved → {fig_path}")

log.info("\n" + "=" * 70)
log.info("STEP 5 COMPLETE")
log.info("=" * 70)
