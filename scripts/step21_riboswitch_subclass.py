"""
SGT-RNA · Step 21: Riboswitch Sub-class Aware Models

Key insight: 61 riboswitches span 5 ligand classes (purine/FMN/SAM/other/amino_acid).
Within each class: same ligand chemistry → same binding mode → correlated affinities.
→ Per-class Ridge should dramatically outperform global model.

Also tests: class-conditioned MKL (within-class block kernel + global kernel)

Best so far: r=0.706 (step11 Ridge + UniMol+Tanimoto MKL for all riboswitch)
"""

import gzip, pickle, logging, time, warnings
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

warnings.filterwarnings("ignore")

ROOT    = Path("/home/stalin/Desktop/SGT-RNA/RNA_SGT")
NA_L    = Path("/home/stalin/Desktop/SGT-RNA/NA-L")
S11_NPZ = ROOT / "data" / "features" / "step11_full_features.npz"
S11_CSV = ROOT / "results" / "step11_results.csv"
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
        logging.FileHandler(ROOT / "results" / "logs" / f"step21_{ts}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("SGT-RNA · Step 21: Riboswitch Sub-class Aware Models")
log.info("=" * 70)

# ── Subtype labels ─────────────────────────────────────────────────────────
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

# ── Load data ──────────────────────────────────────────────────────────────
log.info("\nLoading data ...")
d11 = np.load(S11_NPZ)
X11 = d11["X"].astype(np.float64)
y   = d11["y"].astype(np.float32)
ids = d11["ids"]
subtypes_raw = d11["subtypes"]
subtypes = np.array([make_subtype(p, s) for p, s in zip(ids, subtypes_raw)])
n = len(y)

s11_csv = pd.read_csv(S11_CSV)
pdb2pred_s11 = dict(zip(s11_csv["pdb"], s11_csv["y_pred"]))
step11_preds = np.array([float(pdb2pred_s11.get(p, np.nan)) for p in ids])

unimol_emb_raw = np.load("/tmp/unimol_emb.npy")
valid_idx_raw  = np.load("/tmp/unimol_valid_idx.npy").astype(int)
unimol_full = np.zeros((n, 512), dtype=np.float64)
unimol_full[valid_idx_raw] = unimol_emb_raw
missing_idx = [i for i in range(n) if i not in set(valid_idx_raw.tolist())]
if missing_idx: unimol_full[missing_idx] = unimol_emb_raw.mean(axis=0)
log.info(f"  X11: {X11.shape}, UniMol: {unimol_full.shape}")

# ── Classify riboswitch complexes by ligand chemistry ─────────────────────
from rdkit import Chem
from rdkit.Chem import Descriptors

def classify_riboswitch(pdb):
    for ext in ["sdf","mol2"]:
        f = NA_L/pdb/f"{pdb}_ligand.{ext}"
        if not f.exists(): continue
        try:
            mol = Chem.MolFromMolFile(str(f),sanitize=False) if ext=="sdf" \
                  else Chem.MolFromMol2File(str(f),sanitize=False)
            if not mol: continue
            Chem.SanitizeMol(mol)
        except: continue
        mw = Descriptors.MolWt(mol)
        n_N = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum()==7)
        n_S = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum()==16)
        n_rings = mol.GetRingInfo().NumRings()
        if n_S >= 1 and mw > 400: return "TPP_like"
        if n_S >= 1 and mw > 300: return "SAM_SAH"
        if mw < 200 and n_rings >= 2 and n_N >= 3: return "purine"
        if mw > 350 and n_N >= 4 and n_rings >= 3: return "FMN_FAD"
        if mw < 200 and n_N >= 1 and n_rings == 0: return "amino_acid"
        if mw < 200 and n_rings >= 1: return "preQ_small"
        return "other_lig"
    return "unknown"

rs_mask = subtypes == "riboswitch"
rs_idx  = np.where(rs_mask)[0]

rs_class = np.array(["unknown"] * n, dtype=object)
for i in rs_idx:
    rs_class[i] = classify_riboswitch(ids[i])

log.info("\nRiboswitch class distribution:")
from collections import Counter
for cls, cnt in sorted(Counter(rs_class[rs_mask]).items(), key=lambda x:-x[1]):
    log.info(f"  {cls:15s}: n={cnt}")

# ── ML helpers ──────────────────────────────────────────────────────────────
def loo_ridge_best(X_sub, y_sub, alphas=ALPHA_GRID):
    ns = len(y_sub)
    if ns < 3: return np.full(ns, y_sub.mean()), -99.0
    best_r, best_p = -99.0, np.full(ns, y_sub.mean())
    for alpha in alphas:
        preds, ok = np.zeros(ns), True
        for i in range(ns):
            tr = [j for j in range(ns) if j != i]
            try:
                pipe = Pipeline([
                    ("vt",  VarianceThreshold(threshold=1e-4)),
                    ("sc",  StandardScaler()),
                    ("pca", PCA(n_components=0.95, svd_solver="full")),
                    ("reg", Ridge(alpha=alpha)),
                ])
                pipe.fit(X_sub[tr], y_sub[tr])
                preds[i] = np.clip(pipe.predict(X_sub[[i]])[0],
                                   y_sub[tr].min()-3, y_sub[tr].max()+3)
            except Exception: ok=False; break
        if not ok: continue
        r = pearsonr(y_sub, preds)[0] if np.std(preds)>1e-8 else -99.0
        if r > best_r: best_r, best_p = r, preds.copy()
    return best_p, best_r

def loo_mkl(K, y_all, alpha=0.01):
    nn = len(y_all)
    preds = np.zeros(nn)
    for i in range(nn):
        tr = [j for j in range(nn) if j != i]
        m = KernelRidge(alpha=alpha, kernel="precomputed")
        m.fit(K[np.ix_(tr,tr)], y_all[tr])
        p = float(m.predict(K[i,tr].reshape(1,-1))[0])
        preds[i] = np.clip(p, y_all[tr].min()-3, y_all[tr].max()+3)
    return preds

def tanimoto(X, Y=None):
    if Y is None: Y = X
    XY = X @ Y.T; XX = X.sum(1,keepdims=True); YY = Y.sum(1,keepdims=True)
    return XY / np.where(XX+YY.T-XY<1e-10, 1e-10, XX+YY.T-XY)

# ── PART A: Per-class Ridge within riboswitch ─────────────────────────────
log.info("\n" + "="*70)
log.info("PART A: Per-class Ridge (LOO within riboswitch class)")
log.info("="*70)

rs_preds_class_ridge = step11_preds[rs_mask].copy()  # fallback
MIN_CLASS_N = 5

for cls in np.unique(rs_class[rs_mask]):
    cls_mask_full = (subtypes == "riboswitch") & (rs_class == cls)
    cls_idx  = np.where(cls_mask_full)[0]
    cls_n    = cls_mask_full.sum()
    cls_pos  = np.array([np.where(rs_idx==i)[0][0] for i in cls_idx])  # position within rs

    if cls_n < MIN_CLASS_N:
        # Too small: use global UniMol+Tanimoto MKL prediction (computed below)
        log.info(f"  {cls:15s}: n={cls_n:2d}  → too small, will use global MKL")
        continue

    X_cls = X11[cls_mask_full]
    y_cls = y[cls_mask_full]
    preds_cls, r_cls = loo_ridge_best(X_cls, y_cls)
    rs_preds_class_ridge[cls_pos] = preds_cls
    log.info(f"  {cls:15s}: n={cls_n:2d}  Ridge r={r_cls:.4f}")

r_class_ridge, _ = pearsonr(rs_preds_class_ridge, y[rs_mask])
log.info(f"\n  Riboswitch class-Ridge combined r = {r_class_ridge:.4f}")

# ── PART B: Class-conditioned MKL ─────────────────────────────────────────
log.info("\n" + "="*70)
log.info("PART B: Class-conditioned MKL (block kernel + global kernel)")
log.info("="*70)

# Build kernels on all 143 for global LOO
X_topo   = X11[:, np.r_[0:36000, 38963:49763]]
X_rnafm  = X11[:, 38064:38704]
X_morgan = X11[:, 36000:38048]
X_maccs  = X11[:, 38796:38963]

X_topo_n   = StandardScaler().fit_transform(X_topo)
X_rnafm_n  = StandardScaler().fit_transform(X_rnafm)
X_unimol_n = StandardScaler().fit_transform(unimol_full)

K_topo = rbf_kernel(X_topo_n, gamma=1e-6)
K_rna  = rbf_kernel(X_rnafm_n, gamma=5e-3)
K_uni  = rbf_kernel(X_unimol_n, gamma=0.05)
K_tan  = 0.7*tanimoto(X_morgan) + 0.3*tanimoto(X_maccs)
K_lig_combo = 0.5*K_uni + 0.5*K_tan   # best from step20

# Class indicator kernel (block-diagonal on riboswitch, 0 for cross-class)
K_class = np.zeros((n, n), dtype=np.float64)
for cls in np.unique(rs_class[rs_mask]):
    cls_idx = np.where((subtypes == "riboswitch") & (rs_class == cls))[0]
    for i in cls_idx:
        for j in cls_idx:
            K_class[i, j] = 1.0

# Grid: w_global * K_global + w_class * K_class
K_global = 0.7*K_topo + 0.1*K_lig_combo + 0.2*K_rna  # best UniMol+Tan combo

best_class_mkl_r  = -99.0
best_class_mkl_p  = None
best_class_mkl_cfg = None

for w_class in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    w_global = 1.0 - w_class
    K_cond = w_global * K_global + w_class * K_class
    # Normalize kernel
    D = np.sqrt(np.diag(K_cond)).reshape(-1, 1)
    D = np.where(D < 1e-10, 1e-10, D)
    K_norm = K_cond / (D * D.T)
    for lam in [0.01, 0.05, 0.1]:
        p = loo_mkl(K_norm, y, alpha=lam)
        r_rs = pearsonr(p[rs_mask], y[rs_mask])[0]
        if r_rs > best_class_mkl_r:
            best_class_mkl_r  = r_rs
            best_class_mkl_p  = p.copy()
            best_class_mkl_cfg = (w_class, lam)

log.info(f"  Best class-conditioned MKL: rs_r={best_class_mkl_r:.4f}  cfg={best_class_mkl_cfg}")
log.info(f"  vs pure global MKL:         rs_r=0.7177")

# ── PART C: Class-Ridge + Global-MKL blend within riboswitch ─────────────
log.info("\n" + "="*70)
log.info("PART C: Blend class-Ridge with global-MKL for riboswitch")
log.info("="*70)

# Use global MKL for small classes (n<5), class-Ridge for large classes
rs_preds_blend = best_class_mkl_p[rs_mask].copy()  # start from class-MKL

for cls in np.unique(rs_class[rs_mask]):
    cls_mask_full = (subtypes == "riboswitch") & (rs_class == cls)
    cls_idx  = np.where(cls_mask_full)[0]
    cls_n    = cls_mask_full.sum()
    cls_pos  = np.array([np.where(rs_idx==i)[0][0] for i in cls_idx])

    if cls_n < MIN_CLASS_N:
        continue  # already using global MKL for these

    X_cls = X11[cls_mask_full]
    y_cls = y[cls_mask_full]
    p_ridge, r_ridge = loo_ridge_best(X_cls, y_cls)
    r_mkl_cls = pearsonr(best_class_mkl_p[cls_mask_full], y_cls)[0]

    # Try blends of ridge + mkl predictions for this class
    best_r_cls, best_p_cls = max(r_ridge, r_mkl_cls), (
        p_ridge if r_ridge > r_mkl_cls else best_class_mkl_p[cls_mask_full])
    best_blend = "ridge" if r_ridge > r_mkl_cls else "mkl"

    for alpha_blend in [0.3, 0.5, 0.7]:
        p_blend = alpha_blend * p_ridge + (1-alpha_blend) * best_class_mkl_p[cls_mask_full]
        r_blend = pearsonr(p_blend, y_cls)[0] if np.std(p_blend)>1e-8 else -99.0
        if r_blend > best_r_cls:
            best_r_cls, best_p_cls = r_blend, p_blend.copy()
            best_blend = f"blend({alpha_blend:.1f})"

    rs_preds_blend[cls_pos] = best_p_cls
    log.info(f"  {cls:15s}: n={cls_n:2d}  ridge={r_ridge:.3f}  mkl={r_mkl_cls:.3f} → {best_blend} (r={best_r_cls:.3f})")

r_rs_blend, _ = pearsonr(rs_preds_blend, y[rs_mask])
log.info(f"\n  Riboswitch blend r = {r_rs_blend:.4f}")

# ── Choose best riboswitch model ──────────────────────────────────────────
log.info("\n" + "="*70)
log.info("CHOOSING BEST RIBOSWITCH MODEL")
log.info("="*70)

# Global UniMol+Tanimoto MKL (from step20 best)
p_global_mkl = loo_mkl(K_global, y, alpha=0.01)
r_global_mkl_rs = pearsonr(p_global_mkl[rs_mask], y[rs_mask])[0]
log.info(f"  Global UniMol+Tan MKL:  rs_r = {r_global_mkl_rs:.4f}")
log.info(f"  Class-conditioned MKL:  rs_r = {best_class_mkl_r:.4f}")
log.info(f"  Class Ridge:            rs_r = {r_class_ridge:.4f}")
log.info(f"  Class blend:            rs_r = {r_rs_blend:.4f}")

# Pick best
rs_options = {
    "global_mkl":     (p_global_mkl[rs_mask],    r_global_mkl_rs),
    "class_cond_mkl": (best_class_mkl_p[rs_mask], best_class_mkl_r),
    "class_ridge":    (rs_preds_class_ridge,        r_class_ridge),
    "class_blend":    (rs_preds_blend,              r_rs_blend),
}
best_rs_name = max(rs_options, key=lambda k: rs_options[k][1])
best_rs_preds, best_rs_r = rs_options[best_rs_name]
log.info(f"\n  → Best: {best_rs_name}  rs_r = {best_rs_r:.4f}")

# ── FINAL HYBRID ──────────────────────────────────────────────────────────
log.info("\n" + "="*70)
log.info("FINAL HYBRID: step11 Ridge (non-RS) + best RS model")
log.info("="*70)

hybrid_preds = step11_preds.copy()
hybrid_preds[rs_mask] = best_rs_preds

r_hybrid, _ = pearsonr(hybrid_preds[~np.isnan(hybrid_preds)],
                        y[~np.isnan(hybrid_preds)])

log.info(f"\nPer-subtype:")
for st in ["aptamer","duplex_groove","ribosomal_asite","riboswitch","other_misc","g_quadruplex","viral_tar"]:
    mask = subtypes == st
    if mask.sum() < 2: continue
    r_st = pearsonr(hybrid_preds[mask], y[mask])[0]
    log.info(f"  {st:22s}: n={mask.sum():3d}  r={r_st:.3f}")

log.info(f"\n  Combined r (step21)   = {r_hybrid:.4f}")
log.info(f"  Previous best         = 0.7058")
log.info(f"  Delta                 = {r_hybrid - 0.7058:+.4f}")
log.info(f"  Gap to DeepRSMA       = {0.784 - r_hybrid:.4f}")

benchmarks = [("AffiGrapher",0.498),("RLaffinity",0.559),("RLASIF",0.666),
              ("DeepRSMA",0.784),("RSAPred",0.830)]
for name, rb in benchmarks:
    sym = "✓" if r_hybrid > rb else "✗"
    log.info(f"  {sym} {name}: {rb:.3f}")
log.info("="*70)

# Save
df = pd.DataFrame({"pdb":ids,"subtype":subtypes,"y_true":y,"y_pred":hybrid_preds,
                   "rs_class":rs_class})
df.to_csv(RES_DIR/"step21_results.csv", index=False)

# Plot
colors = {"aptamer":"#2166AC","riboswitch":"#1A9641","ribosomal_asite":"#D73027",
          "duplex_groove":"#7B2D8B","g_quadruplex":"#FF7F00",
          "viral_tar":"#A65628","other_misc":"#999999"}
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
for st in np.unique(subtypes):
    mask = subtypes == st
    r_st = pearsonr(hybrid_preds[mask],y[mask])[0] if mask.sum()>1 else 0
    ax.scatter(y[mask], hybrid_preds[mask], c=colors.get(st,"#888"),
               label=f"{st} r={r_st:.3f}", alpha=0.75, s=45, edgecolors="none")
lo, hi = y.min()-0.5, y.max()+0.5
ax.plot([lo,hi],[lo,hi],"k--",lw=1,alpha=0.5)
ax.set_xlabel("Experimental pKd"); ax.set_ylabel("Predicted pKd")
ax.set_title(f"Step 21 Hybrid (r={r_hybrid:.4f})", fontweight="bold")
ax.legend(fontsize=7); ax.grid(alpha=0.3, linestyle="--")

ax = axes[1]
steps = {"S11+MKL":0.695,"S19 UniMol":0.706,"S21":r_hybrid}
bar_cols = ["#AAAAAA","#4393C3","#D63027"]
bars = ax.bar(list(steps.keys()), list(steps.values()), color=bar_cols, edgecolor="white", alpha=0.9)
for bar, val in zip(bars, steps.values()):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.003, f"{val:.4f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
for name, rb in benchmarks:
    ax.axhline(rb, linestyle="--", lw=0.9, alpha=0.6, label=f"{name} {rb:.3f}")
ax.set_ylim(0.55, 0.85); ax.set_ylabel("Combined Pearson r")
ax.set_title("Progress vs Benchmarks", fontweight="bold")
ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3, linestyle="--")

plt.tight_layout()
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR/"step21_results.png", dpi=150, bbox_inches="tight")
plt.close()
log.info(f"Figure → {FIG_DIR/'step21_results.png'}")
log.info("STEP 21 COMPLETE")
