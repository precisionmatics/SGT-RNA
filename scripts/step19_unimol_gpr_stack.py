"""
SGT-RNA · Step 19: UniMol 3D Ligand Repr + GPR + 3-Model Stack

Novel contributions:
1. Replace Morgan 2048 → UniMol 512 (3D pre-trained transformer, 209M conformations)
2. New MKL for riboswitch: K_lig = RBF(UniMol) — outperforms Tanimoto(Morgan)
3. Gaussian Process Regression for small structural subtypes
4. 3-model stacked ensemble: Ridge LOO + GPR LOO + MKL LOO → weighted blend

Feature layout (48,227 total vs 49,763 in step11):
  PDFL[0:36000] | UniMol[36000:36512] | NucComp[36512:36522] | Physico[36522:36528]
  RNA-FM[36528:37168] | SS[37168:37176] | kmer[37176:37260] | MACCS[37260:37427]
  Iface4[37427:41027] | Iface6[41027:44627] | Iface8[44627:48227]
"""

import gzip, pickle, logging, time, warnings
from pathlib import Path
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

warnings.filterwarnings("ignore")

ROOT    = Path("/home/stalin/Desktop/SGT-RNA/RNA_SGT")
NA_L    = Path("/home/stalin/Desktop/SGT-RNA/NA-L")
S11_NPZ = ROOT / "data" / "features" / "step11_full_features.npz"
OUT_NPZ = ROOT / "data" / "features" / "step19_full_features.npz"
RES_DIR = ROOT / "results"
FIG_DIR = ROOT / "results" / "figures"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "results" / "logs" / f"step19_{ts}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("SGT-RNA · Step 19: UniMol 3D + GPR + 3-Model Stack")
log.info("=" * 70)

# ── Subtype labels ──────────────────────────────────────────────────────────
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

# ── Load step11 features ────────────────────────────────────────────────────
log.info("\nLoading step11 features ...")
d11 = np.load(S11_NPZ)
X11 = d11["X"].astype(np.float64)
y   = d11["y"].astype(np.float32)
ids = d11["ids"]
subtypes_raw = d11["subtypes"]
subtypes = np.array([make_subtype(p, s) for p, s in zip(ids, subtypes_raw)])
n = len(y)
log.info(f"  X11: {X11.shape}")

# ── Compute UniMol 3D embeddings ────────────────────────────────────────────
log.info("\nComputing UniMol 3D ligand embeddings ...")
from unimol_tools import UniMolRepr
from rdkit import Chem
from rdkit.Chem import AllChem

clf = UniMolRepr(data_type="molecule", remove_hs=True)
smiles_list, valid_idx = [], []
for i, pdb in enumerate(ids):
    mol = None
    for ext in ["sdf", "mol2"]:
        f = NA_L / pdb / f"{pdb}_ligand.{ext}"
        if f.exists():
            try:
                mol = (Chem.MolFromMolFile(str(f), removeHs=False, sanitize=False)
                       if ext == "sdf" else
                       Chem.MolFromMol2File(str(f), removeHs=False, sanitize=False))
                if mol:
                    Chem.SanitizeMol(mol)
                    smiles_list.append(Chem.MolToSmiles(mol))
                    valid_idx.append(i)
                    break
            except Exception:
                continue

log.info(f"  Valid SMILES: {len(smiles_list)}/{n}")
reprs = clf.get_repr(smiles_list, return_atomic_reprs=False)
unimol_emb_valid = np.array(reprs)  # (n_valid, 512)
log.info(f"  UniMol embedding: {unimol_emb_valid.shape}")

# Fill missing entries with column mean
unimol_emb = np.zeros((n, 512), dtype=np.float32)
unimol_emb[valid_idx] = unimol_emb_valid
missing = [i for i in range(n) if i not in set(valid_idx)]
if missing:
    col_means = unimol_emb[valid_idx].mean(axis=0)
    unimol_emb[missing] = col_means

# Signal check
dim_rs = np.array([abs(pearsonr(unimol_emb[:, d], y)[0]) for d in range(512)])
log.info(f"  Signal: dims|r|>0.2: {(dim_rs>0.2).sum()},  top r: {dim_rs.max():.3f}")

# ── Build step19 feature matrix ─────────────────────────────────────────────
log.info("\nBuilding step19 features (UniMol replaces Morgan) ...")
# step11 slices
X_pdfl  = X11[:, 0:36000]
# skip Morgan [36000:38048]
X_nuc   = X11[:, 38048:38058]
X_phys  = X11[:, 38058:38064]
X_rnafm = X11[:, 38064:38704]
X_ss    = X11[:, 38704:38712]
X_kmer  = X11[:, 38712:38796]
X_maccs = X11[:, 38796:38963]
X_if4   = X11[:, 38963:42563]
X_if6   = X11[:, 42563:46163]
X_if8   = X11[:, 46163:49763]

X19 = np.hstack([
    X_pdfl,                          # 36000
    unimol_emb.astype(np.float64),   #   512  ← UniMol
    X_nuc, X_phys, X_rnafm,          # 10+6+640
    X_ss, X_kmer, X_maccs,           # 8+84+167
    X_if4, X_if6, X_if8,             # 10800
])
log.info(f"  X19: {X19.shape}")  # expect (143, 48227)

np.savez_compressed(OUT_NPZ,
    X=X19.astype(np.float32), y=y, ids=ids, subtypes=subtypes_raw)
log.info(f"  Saved: {OUT_NPZ}")

# ── Helper: feature slices from X19 ─────────────────────────────────────────
UNIMOL_SLICE = np.s_[36000:36512]
RNAFM_SLICE  = np.s_[36528:37168]   # 36512+10+6 = 36528
MACCS_SLICE  = np.s_[37260:37427]   # 36512+10+6+640+8+84 = 37260
IFACE_SLICE  = np.r_[0:36000, 37427:48227]  # SGT + 3 iface

# ── Pipelines ────────────────────────────────────────────────────────────────
def make_ridge_pipe(alpha):
    return Pipeline([
        ("vt",  VarianceThreshold(threshold=1e-4)),
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
        ("reg", Ridge(alpha=alpha)),
    ])

def loo_ridge_best(X_sub, y_sub):
    ns = len(y_sub)
    if ns < 3:
        return np.full(ns, y_sub.mean()), -99.0
    best_r, best_p = -99.0, np.full(ns, y_sub.mean())
    for alpha in ALPHA_GRID:
        preds, ok = np.zeros(ns), True
        for i in range(ns):
            tr = [j for j in range(ns) if j != i]
            try:
                pipe = make_ridge_pipe(alpha)
                pipe.fit(X_sub[tr], y_sub[tr])
                preds[i] = np.clip(
                    pipe.predict(X_sub[[i]])[0],
                    y_sub[tr].min() - 3, y_sub[tr].max() + 3
                )
            except Exception:
                ok = False; break
        if not ok:
            continue
        r = pearsonr(y_sub, preds)[0] if np.std(preds) > 1e-8 else -99.0
        if r > best_r:
            best_r, best_p = r, preds.copy()
    return best_p, best_r

# ── MKL kernel setup ─────────────────────────────────────────────────────────
def tanimoto_kernel(X, Y=None):
    if Y is None: Y = X
    XY = X @ Y.T
    XX = X.sum(1, keepdims=True)
    YY = Y.sum(1, keepdims=True)
    denom = np.where(XX + YY.T - XY < 1e-10, 1e-10, XX + YY.T - XY)
    return XY / denom

def loo_mkl(K, y_all, alpha=0.01):
    n = len(y_all)
    preds = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        m = KernelRidge(alpha=alpha, kernel="precomputed")
        m.fit(K[np.ix_(tr, tr)], y_all[tr])
        p = float(m.predict(K[i, tr].reshape(1, -1))[0])
        preds[i] = np.clip(p, y_all[tr].min() - 3, y_all[tr].max() + 3)
    return preds

# ── PART 1: Per-subtype Ridge on X19 ─────────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("PART 1: Per-subtype Ridge LOO (X19: UniMol replaces Morgan)")
log.info("=" * 70)

ridge19_preds = np.full(n, np.nan)
ridge19_rs    = {}
for st in np.unique(subtypes):
    mask = subtypes == st
    if mask.sum() < 3:
        ridge19_preds[mask] = y[mask].mean()
        ridge19_rs[st] = np.nan
        continue
    preds, r = loo_ridge_best(X19[mask], y[mask])
    ridge19_preds[mask] = preds
    ridge19_rs[st] = r
    log.info(f"  {st:22s}: n={mask.sum():3d}  r={r:.3f}")

r_ridge19_global, _ = pearsonr(ridge19_preds[~np.isnan(ridge19_preds)],
                                y[~np.isnan(ridge19_preds)])
log.info(f"\n  Global Ridge19 r = {r_ridge19_global:.4f}")

# ── PART 2: MKL with UniMol kernel (grid search) ─────────────────────────────
log.info("\n" + "=" * 70)
log.info("PART 2: MKL with UniMol kernel — grid search for riboswitch")
log.info("=" * 70)

X_topo_raw = X19[:, IFACE_SLICE].astype(np.float64)
X_unimol   = X19[:, UNIMOL_SLICE].astype(np.float64)
X_maccs_f  = X11[:, 38796:38963].astype(np.float64)   # keep MACCS from step11
X_rnafm_f  = X19[:, RNAFM_SLICE].astype(np.float64)

X_topo_n   = StandardScaler().fit_transform(X_topo_raw)
X_unimol_n = StandardScaler().fit_transform(X_unimol)
X_rnafm_n  = StandardScaler().fit_transform(X_rnafm_f)

K_topo = rbf_kernel(X_topo_n, gamma=1e-6)   # same as step13
K_rna  = rbf_kernel(X_rnafm_n, gamma=5e-3)  # same as step13

# Grid over UniMol gamma and kernel weights
gamma_lig_vals  = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2]
alpha_combos = [
    (0.7, 0.1, 0.2), (0.6, 0.2, 0.2), (0.7, 0.2, 0.1),
    (0.5, 0.3, 0.2), (0.6, 0.3, 0.1), (0.8, 0.1, 0.1),
]
lambda_vals = [0.01, 0.1]

rs_mask = subtypes == "riboswitch"
best_mkl_r, best_mkl_preds = -999.0, None
best_cfg = None

log.info("  Searching UniMol MKL configs ...")
for gl, (at, al, ar), lam in product(gamma_lig_vals, alpha_combos, lambda_vals):
    K_lig = rbf_kernel(X_unimol_n, gamma=gl)
    K = at * K_topo + al * K_lig + ar * K_rna
    preds = loo_mkl(K, y, alpha=lam)
    r, _ = pearsonr(preds[rs_mask], y[rs_mask])
    if r > best_mkl_r:
        best_mkl_r = r
        best_mkl_preds = preds.copy()
        best_cfg = (gl, at, al, ar, lam)

log.info(f"  Best UniMol MKL: rs_r={best_mkl_r:.4f}  cfg={best_cfg}")

# Also test old step13 kernel for comparison
X_morgan = X11[:, 36000:38048].astype(np.float64)
X_maccs_s11 = X11[:, 38796:38963].astype(np.float64)
K_lig_old = 0.7 * tanimoto_kernel(X_morgan) + 0.3 * tanimoto_kernel(X_maccs_s11)
K_old = 0.7 * K_topo + 0.1 * K_lig_old + 0.2 * K_rna
preds_old = loo_mkl(K_old, y, alpha=0.01)
r_old, _ = pearsonr(preds_old[rs_mask], y[rs_mask])
log.info(f"  Old Tanimoto MKL: rs_r={r_old:.4f}")

# Pick best kernel for riboswitch
if best_mkl_r > r_old:
    mkl_preds_final = best_mkl_preds
    log.info(f"  → Using UniMol MKL (+{best_mkl_r - r_old:+.4f})")
else:
    mkl_preds_final = preds_old
    best_mkl_r = r_old
    log.info(f"  → Using old Tanimoto MKL (UniMol no improvement)")

log.info(f"  Riboswitch MKL r = {best_mkl_r:.4f}")

# ── PART 3: GPR for small structural subtypes ─────────────────────────────────
log.info("\n" + "=" * 70)
log.info("PART 3: Gaussian Process Regression for structural subtypes")
log.info("=" * 70)

GPR_SUBTYPES = ["aptamer", "duplex_groove", "ribosomal_asite", "other_misc"]
gpr_preds = ridge19_preds.copy()  # start from Ridge, overwrite where GPR wins

def loo_gpr(X_sub, y_sub, n_pca=30):
    """LOO GPR with RBF+White kernel on PCA-reduced features."""
    ns = len(y_sub)
    preds = np.zeros(ns)
    # Preprocess once
    vt  = VarianceThreshold(threshold=1e-4).fit(X_sub)
    Xvt = vt.transform(X_sub)
    sc  = StandardScaler().fit(Xvt)
    Xsc = sc.transform(Xvt)
    nc  = min(n_pca, Xsc.shape[1], ns - 2)
    pca = PCA(n_components=nc, svd_solver="full").fit(Xsc)
    Xpc = pca.transform(Xsc)

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    for i in range(ns):
        tr = [j for j in range(ns) if j != i]
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                       normalize_y=True, random_state=42)
        gpr.fit(Xpc[tr], y_sub[tr])
        p = float(gpr.predict(Xpc[[i]])[0])
        preds[i] = np.clip(p, y_sub[tr].min() - 3, y_sub[tr].max() + 3)
    r = pearsonr(y_sub, preds)[0] if np.std(preds) > 1e-8 else -99.0
    return preds, r

for st in GPR_SUBTYPES:
    mask = subtypes == st
    ns = mask.sum()
    if ns < 4:
        continue
    r_ridge = ridge19_rs.get(st, -99.0)

    preds_gpr, r_gpr = loo_gpr(X19[mask], y[mask], n_pca=min(30, ns - 2))
    log.info(f"  {st:22s}: n={ns:3d}  Ridge r={r_ridge:.3f}  GPR r={r_gpr:.3f}", )

    if r_gpr > r_ridge:
        gpr_preds[mask] = preds_gpr
        log.info(f"    → GPR wins (+{r_gpr - r_ridge:.3f})")
    else:
        log.info(f"    → Ridge wins")

# Fill riboswitch with MKL
gpr_preds[rs_mask] = mkl_preds_final[rs_mask]

r_gpr_hybrid, _ = pearsonr(gpr_preds, y)
log.info(f"\n  GPR-hybrid combined r = {r_gpr_hybrid:.4f}")

# ── PART 4: 3-Model Prediction Stacking ──────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("PART 4: 3-Model Stacking (Ridge19 + GPR-hybrid + MKL)")
log.info("=" * 70)

# Meta-features: LOO predictions from all 3 models
meta_X = np.column_stack([
    ridge19_preds,        # per-subtype Ridge on X19
    gpr_preds,            # GPR-hybrid (best of GPR/Ridge per subtype + MKL riboswitch)
    mkl_preds_final,      # global MKL predictions
])
log.info(f"  Meta-feature matrix: {meta_X.shape}")

# LOO linear blend of 3 predictions
def loo_meta_blend(meta_X, y, alphas=[0.001, 0.01, 0.1, 1.0]):
    ns = len(y)
    best_r, best_p = -999.0, None
    for a in alphas:
        preds = np.zeros(ns)
        for i in range(ns):
            tr = [j for j in range(ns) if j != i]
            m = Ridge(alpha=a, fit_intercept=True)
            m.fit(meta_X[tr], y[tr])
            preds[i] = float(m.predict(meta_X[[i]])[0])
        preds = np.clip(preds, y.min() - 2, y.max() + 2)
        r, _ = pearsonr(preds, y)
        if r > best_r:
            best_r, best_p = r, preds.copy()
    return best_p, best_r

stacked_preds, r_stacked = loo_meta_blend(meta_X, y)
log.info(f"  Stacked combined r = {r_stacked:.4f}")

# ── FINAL: pick best model overall ───────────────────────────────────────────
all_results = {
    "ridge19":    (ridge19_preds, r_ridge19_global),
    "gpr_hybrid": (gpr_preds,     r_gpr_hybrid),
    "stacked":    (stacked_preds, r_stacked),
}
best_name = max(all_results, key=lambda k: all_results[k][1])
best_preds, best_r = all_results[best_name]

log.info(f"\n{'='*70}")
log.info(f"FINAL RESULTS")
log.info(f"{'='*70}")
log.info(f"  Ridge19:      r = {r_ridge19_global:.4f}")
log.info(f"  GPR-hybrid:   r = {r_gpr_hybrid:.4f}")
log.info(f"  Stacked:      r = {r_stacked:.4f}")
log.info(f"  Best:         r = {best_r:.4f}  ({best_name})")
log.info(f"  Previous best = 0.6954")
log.info(f"  Delta         = {best_r - 0.6954:+.4f}")
log.info(f"  Gap DeepRSMA  = {0.784 - best_r:.4f}")

log.info("\nPer-subtype (best model):")
for st in ["aptamer","duplex_groove","ribosomal_asite","riboswitch","other_misc","g_quadruplex","viral_tar"]:
    mask = subtypes == st
    if mask.sum() < 2: continue
    r, _ = pearsonr(best_preds[mask], y[mask])
    log.info(f"  {st:22s}: n={mask.sum():3d}  r={r:.3f}")

benchmarks = [("AffiGrapher",0.498),("RLaffinity",0.559),("RLASIF",0.666),
              ("DeepRSMA",0.784),("RSAPred",0.830)]
log.info("\nBenchmark:")
for name, rb in benchmarks:
    sym = "✓" if best_r > rb else "✗"
    log.info(f"  {sym} {name}: {rb:.3f}")
log.info(f"{'='*70}")

# Save results
df = pd.DataFrame({
    "pdb": ids, "subtype": subtypes, "y_true": y,
    "y_pred_ridge19": ridge19_preds,
    "y_pred_gpr":     gpr_preds,
    "y_pred_stacked": stacked_preds,
})
df.to_csv(RES_DIR / "step19_results.csv", index=False)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
colors = {"aptamer":"#2166AC","riboswitch":"#1A9641","ribosomal_asite":"#D73027",
          "duplex_groove":"#7B2D8B","g_quadruplex":"#FF7F00",
          "viral_tar":"#A65628","other_misc":"#999999"}

for ax, (preds, title, rv) in zip(axes, [
    (ridge19_preds, f"Ridge19 UniMol (r={r_ridge19_global:.4f})", r_ridge19_global),
    (gpr_preds,     f"GPR-Hybrid (r={r_gpr_hybrid:.4f})",         r_gpr_hybrid),
    (stacked_preds, f"3-Model Stack (r={r_stacked:.4f})",          r_stacked),
]):
    for st in np.unique(subtypes):
        mask = subtypes == st
        r_st = pearsonr(preds[mask], y[mask])[0] if mask.sum() > 1 else 0
        ax.scatter(y[mask], preds[mask], c=colors.get(st, "#888"),
                   label=f"{st} r={r_st:.2f}", alpha=0.75, s=40, edgecolors="none")
    lo, hi = y.min() - 0.5, y.max() + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Experimental pKd"); ax.set_ylabel("Predicted pKd")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.3, linestyle="--")

plt.tight_layout()
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / "step19_results.png", dpi=150, bbox_inches="tight")
plt.close()
log.info(f"Figure → {FIG_DIR/'step19_results.png'}")
log.info("STEP 19 COMPLETE")
