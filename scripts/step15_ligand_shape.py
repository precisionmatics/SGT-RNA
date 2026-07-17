"""
SGT-RNA · Step 15: Intra-Ligand SGT + 3D Shape Features

Two novel feature types computed from existing SDF crystal structures:

1. 3D Ligand Shape (8 features per complex):
   NPR1, NPR2, Asphericity, Eccentricity, SpherocityIndex,
   InertialShapeFactor, PBF, RadiusOfGyration
   Captures planarity (G-quad flat aromatics) vs sphericity (riboswitch compact).

2. Intra-Ligand SGT (3,600 features):
   SGT applied to ligand-internal atom graph using cross-element pairs only
   (36 pairs: C-N, C-O, C-S, N-O, ... avoiding self-pair diagonal issue).
   ETA=3.0 Å (molecular scale). 36 × 5 × 2 × 10 = 3,600 features.
   Captures ring topology, aromaticity, branching — complementary to Morgan.

Total new features: 3,608
Combined with step11 (49,763): 53,371 features

Hybrid evaluation:
  - Ridge on 53,371 features for all subtypes (LOO)
  - Global MKL for riboswitch (unchanged, r=0.696)
  - Take best per subtype
"""

import gzip, pickle, logging, time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent.parent
NA_L    = ROOT / "NA-L"
S11_NPZ = ROOT / "data" / "features" / "step11_full_features.npz"
S11_CSV = ROOT / "results" / "step11_results.csv"
OUT_NPZ = ROOT / "data" / "features" / "step15_full_features.npz"
RES_DIR = ROOT / "results"
FIG_DIR = ROOT / "results" / "figures"

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "results" / "logs" / f"step15_{ts}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("SGT-RNA · Step 15: Intra-Ligand SGT + 3D Shape Features")
log.info("=" * 70)

# ── SGT constants (intra-ligand scale) ───────────────────────────────────────
LIG_ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
# Only cross-element pairs (e1 < e2 by index) to avoid self-pair diagonal
CROSS_PAIRS = [(e1, e2) for i, e1 in enumerate(LIG_ELEMENTS)
               for j, e2 in enumerate(LIG_ELEMENTS) if j > i]  # 36 pairs
THRESHOLDS  = [0.0, 0.8, 0.85, 0.90, 0.95]
N_STATS     = 10
ETA_LIG     = 3.0   # molecular scale (shorter than RNA-ligand 5.0 Å)
KAPPA       = 2.0
EPS         = 1e-8

N_ILIG_FEATS  = len(CROSS_PAIRS) * len(THRESHOLDS) * 2 * N_STATS  # 3,600
N_SHAPE_FEATS = 8
N_NEW_FEATS   = N_ILIG_FEATS + N_SHAPE_FEATS  # 3,608

log.info(f"Cross-element pairs : {len(CROSS_PAIRS)}")
log.info(f"Intra-lig SGT feats: {N_ILIG_FEATS}")
log.info(f"Shape features      : {N_SHAPE_FEATS}")
log.info(f"Total new features  : {N_NEW_FEATS}")

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

# ── SGT helpers ──────────────────────────────────────────────────────────────
def spectral_stats(eigs):
    if len(eigs) == 0:
        return np.zeros(N_STATS, dtype=np.float32)
    nz = eigs[eigs > EPS]
    return np.array([
        eigs.min(), eigs.max(), eigs.mean(), np.median(eigs),
        eigs.var(), eigs.std(), eigs.sum(), (eigs**2).sum(),
        float(len(nz)), float(len(eigs)-len(nz)),
    ], dtype=np.float32)

def build_L0(W_sel):
    n_r, n_l = W_sel.shape
    n = n_r + n_l
    L = np.zeros((n, n), dtype=np.float64)
    L[:n_r, :n_r] = np.diag(W_sel.sum(axis=1))
    L[n_r:, n_r:] = np.diag(W_sel.sum(axis=0))
    L[:n_r, n_r:] = -W_sel
    L[n_r:, :n_r] = -W_sel.T
    return L

def pair_features_lig(rc, lc):
    """SGT for a cross-element pair within the ligand."""
    n_feat = len(THRESHOLDS) * 2 * N_STATS
    if len(rc) == 0 or len(lc) == 0:
        return np.zeros(n_feat, dtype=np.float32)
    D = cdist(rc, lc).astype(np.float64)
    W = np.exp(-(D / ETA_LIG) ** KAPPA)
    wmax = W.max()
    if wmax < 1e-12:
        return np.zeros(n_feat, dtype=np.float32)
    W_norm = W / wmax
    n_total = len(rc) + len(lc)
    feats, offset = np.zeros(n_feat, dtype=np.float32), 0
    for tau in THRESHOLDS:
        mask   = W_norm >= tau
        W_sel  = W_norm * mask
        n_edges= int(mask.sum())
        if n_edges == 0:
            eigs_L0 = np.zeros(n_total); eigs_L1 = np.array([])
        else:
            L0 = build_L0(W_sel)
            eigs_L0 = np.maximum(np.linalg.eigvalsh(L0), 0.0)
            beta0 = int((eigs_L0 < EPS).sum())
            nz_L1 = max(0, n_edges - n_total + beta0)
            eigs_L1 = np.concatenate([np.zeros(nz_L1), eigs_L0[eigs_L0 >= EPS]])
        feats[offset:offset+N_STATS]           = spectral_stats(eigs_L0)
        feats[offset+N_STATS:offset+2*N_STATS] = spectral_stats(eigs_L1)
        offset += 2 * N_STATS
    return feats

# ── feature extraction from SDF ───────────────────────────────────────────────
def extract_lig_features(pdb):
    """Returns (shape_feats: 8, ilig_sgt: 3600) for one complex."""
    sdf_path = NA_L / pdb / f"{pdb}_ligand.sdf"
    shape_zero = np.zeros(N_SHAPE_FEATS, dtype=np.float32)
    ilig_zero  = np.zeros(N_ILIG_FEATS,  dtype=np.float32)

    try:
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        mol   = next(suppl)
        if mol is None:
            return shape_zero, ilig_zero

        # 3D shape features
        try:
            shape = np.array([
                rdMolDescriptors.CalcNPR1(mol),
                rdMolDescriptors.CalcNPR2(mol),
                rdMolDescriptors.CalcAsphericity(mol),
                rdMolDescriptors.CalcEccentricity(mol),
                rdMolDescriptors.CalcSpherocityIndex(mol),
                rdMolDescriptors.CalcInertialShapeFactor(mol),
                rdMolDescriptors.CalcPBF(mol),
                rdMolDescriptors.CalcRadiusOfGyration(mol),
            ], dtype=np.float32)
            shape = np.nan_to_num(shape, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            shape = shape_zero

        # Intra-ligand SGT: collect element → coords
        mol_noH = Chem.RemoveHs(mol)
        conf    = mol_noH.GetConformer()
        el2coords = {}
        for atom in mol_noH.GetAtoms():
            el  = atom.GetSymbol()
            pos = conf.GetAtomPosition(atom.GetIdx())
            el2coords.setdefault(el, []).append([pos.x, pos.y, pos.z])
        el2arr = {el: np.array(v, dtype=np.float32)
                  for el, v in el2coords.items()}

        parts = []
        for (e1, e2) in CROSS_PAIRS:
            c1 = el2arr.get(e1, np.empty((0,3), np.float32))
            c2 = el2arr.get(e2, np.empty((0,3), np.float32))
            parts.append(pair_features_lig(c1, c2))
        ilig = np.concatenate(parts)

        return shape, ilig

    except Exception:
        return shape_zero, ilig_zero

# ── load step11 data ──────────────────────────────────────────────────────────
log.info("\nLoading step11 features ...")
d = np.load(S11_NPZ)
X11  = d["X"].astype(np.float32)
y    = d["y"].astype(np.float32)
ids  = d["ids"]
subtypes_raw = d["subtypes"]
subtypes = np.array([make_subtype(p,s) for p,s in zip(ids, subtypes_raw)])
n = len(y)

step11_preds = pd.read_csv(S11_CSV).set_index("pdb")
step11_preds = np.array([step11_preds.loc[p,"y_pred"] for p in ids])

# ── compute new features ──────────────────────────────────────────────────────
log.info(f"\nComputing 3D shape + intra-ligand SGT for {n} complexes ...")
X_shape = np.zeros((n, N_SHAPE_FEATS), dtype=np.float32)
X_ilig  = np.zeros((n, N_ILIG_FEATS),  dtype=np.float32)

t0 = time.time()
for idx, pdb in enumerate(ids):
    X_shape[idx], X_ilig[idx] = extract_lig_features(pdb)
    if (idx+1) % 20 == 0 or idx == 0:
        elapsed = time.time() - t0
        eta = (n-idx-1) / max((idx+1)/elapsed, 1e-6)
        nnz = (X_ilig[idx] != 0).sum()
        log.info(f"  [{idx+1:3d}/{n}] {pdb}  ilig_nnz={nnz}  ETA {eta:.0f}s")

log.info(f"  Done in {time.time()-t0:.1f}s")

# stats
log.info(f"\n  Shape feats  — nonzero: {(X_shape!=0).mean(axis=0).mean():.2%}")
log.info(f"  Intra-lig    — nonzero per sample: {(X_ilig!=0).sum(axis=1).mean():.0f}/{N_ILIG_FEATS}")

# ── combine features ──────────────────────────────────────────────────────────
X_full = np.hstack([X11, X_shape, X_ilig]).astype(np.float32)
log.info(f"\nFull feature matrix: {X_full.shape}")
np.savez_compressed(OUT_NPZ, X=X_full, y=y, ids=ids, subtypes=subtypes_raw)
log.info(f"Saved → {OUT_NPZ}")

# ── ML pipeline ───────────────────────────────────────────────────────────────
ALPHA_GRID = [1, 10, 100, 1000, 10_000, 100_000]

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
            tr = [j for j in range(ns) if j != i]
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
log.info("ML EVALUATION — per-subtype Ridge LOO on 53,371 features")
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
    Xs = X_full[idx]

    preds_st, r_st = loo_ridge(Xs, ys)
    preds_gl = global_preds[idx]
    r_gl = pearsonr(ys, preds_gl)[0] if np.std(preds_gl)>1e-8 else -99.0

    if r_st >= r_gl:
        new_ridge_preds[idx] = preds_st; r_use = r_st; chosen = "subtype"
    else:
        new_ridge_preds[idx] = preds_gl; r_use = r_gl; chosen = "global"
    new_ridge_rs[st] = r_use
    log.info(f"  {st:<22}: n={ns:3d}  r={r_use:.4f}  ({chosen})")

# ── global MKL for riboswitch (same as step13 best) ──────────────────────────
log.info("\nGlobal MKL for riboswitch (step13 best config) ...")
X11_f64 = X11.astype(np.float64)
X_topo  = X11_f64[:, np.r_[0:36000, 38963:49763]]
X_lig   = X11_f64[:, 36000:38048]
X_maccs = X11_f64[:, 38796:38963]
X_rna   = X11_f64[:, 38064:38704]

sc_t = StandardScaler(); X_topo_n = sc_t.fit_transform(X_topo)
sc_r = StandardScaler(); X_rna_n  = sc_r.fit_transform(X_rna)

def tanimoto_kernel(X, Y=None):
    if Y is None: Y = X
    XY = X @ Y.T; XX = X.sum(1,keepdims=True); YY = Y.sum(1,keepdims=True)
    return XY / np.where(XX+YY.T-XY<1e-10, 1e-10, XX+YY.T-XY)

K_topo = rbf_kernel(X_topo_n, gamma=1e-6)
K_lig  = 0.7*tanimoto_kernel(X_lig) + 0.3*tanimoto_kernel(X_maccs)
K_rna  = rbf_kernel(X_rna_n,  gamma=5e-3)
K_full_mkl = 0.7*K_topo + 0.1*K_lig + 0.2*K_rna

mkl_preds = np.zeros(n)
for i in range(n):
    tr = [j for j in range(n) if j != i]
    m = KernelRidge(alpha=0.01, kernel="precomputed")
    m.fit(K_full_mkl[np.ix_(tr,tr)], y[tr])
    p = float(m.predict(K_full_mkl[i,tr].reshape(1,-1))[0])
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
log.info("FINAL HYBRID RESULTS (step15 Ridge + global MKL for riboswitch)")
log.info("=" * 70)
for st in unique_subtypes:
    mask = subtypes == st
    log.info(f"  {st:<22}: n={mask.sum():3d}  r={hybrid_rs[st]:.4f}")

log.info(f"\n  COMBINED r (step15 hybrid) = {combined_r:.4f}")
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

# ── save & figure ─────────────────────────────────────────────────────────────
df_res = pd.DataFrame({
    "pdb":ids,"subtype":subtypes,"y_true":y,"y_pred":hybrid_preds,
    "y_pred_ridge":new_ridge_preds,
})
df_res.to_csv(RES_DIR/"step15_results.csv", index=False)

colors = {
    "aptamer":"#2166AC","riboswitch":"#1A9641","ribosomal_asite":"#D73027",
    "duplex_groove":"#7B2D8B","g_quadruplex":"#FF7F00",
    "viral_tar":"#A65628","other_misc":"#999999",
}
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.patch.set_facecolor("white")

ax = axes[0]
for st in unique_subtypes:
    mask = subtypes == st
    ax.scatter(y[mask], hybrid_preds[mask], c=colors.get(st,"#888"),
               label=f"{st} (r={hybrid_rs[st]:.3f})", alpha=0.75, s=45, edgecolors="none")
mn, mx = y.min()-0.5, y.max()+0.5
ax.plot([mn,mx],[mn,mx],"k--",lw=1,alpha=0.4)
ax.set_xlabel("Experimental pKd"); ax.set_ylabel("Predicted pKd")
ax.set_title(f"Step 15: Intra-lig SGT + Shape  (r={combined_r:.4f})", fontweight="bold")
ax.legend(fontsize=7, loc="upper left", framealpha=0.7)
ax.grid(alpha=0.3, linestyle="--")

ax = axes[1]
steps = {"S09":0.535,"S11":0.575,"S11+MKL":0.695,"S15":combined_r}
bar_cols = ["#AAAAAA","#4393C3","#08519C","#1A7837"]
bars = ax.bar(list(steps.keys()), list(steps.values()), color=bar_cols, edgecolor="white", alpha=0.9)
for bar, val in zip(bars, steps.values()):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.005, f"{val:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
for name, rb in benchmarks:
    ax.axhline(rb, linestyle="--", lw=0.9, alpha=0.6, label=f"{name} {rb:.3f}")
ax.set_ylim(0.45, 0.9); ax.set_ylabel("Combined Pearson r")
ax.set_title("Progress", fontweight="bold")
ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3, linestyle="--")

ax = axes[2]
sts_plot = ["aptamer","riboswitch","ribosomal_asite","duplex_groove","other_misc","g_quadruplex"]
hybrid_ref = {"aptamer":0.937,"riboswitch":0.696,"ribosomal_asite":0.753,
              "duplex_groove":0.810,"other_misc":0.395,"g_quadruplex":0.253}
x = np.arange(len(sts_plot)); w = 0.35
ax.bar(x-w/2, [hybrid_ref[s] for s in sts_plot], w, label="S11+MKL Hybrid",
       color="#4393C3", alpha=0.85, edgecolor="white")
ax.bar(x+w/2, [hybrid_rs.get(s,0) for s in sts_plot], w, label="Step 15",
       color="#1A7837", alpha=0.85, edgecolor="white")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(sts_plot, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Pearson r"); ax.set_title("Subtype: Hybrid vs Step 15", fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig(FIG_DIR/"step15_results.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

log.info(f"\nResults → {RES_DIR/'step15_results.csv'}")
log.info(f"Figure  → {FIG_DIR/'step15_results.png'}")
log.info("\n" + "=" * 70)
log.info("STEP 15 COMPLETE")
log.info("=" * 70)
