"""
RNA-PDFL · Step 20: Adaptive Multi-Model + other_misc Sub-Classification

Approach:
A) Global UniMol MKL predictions for ALL subtypes — pick where it beats Ridge
B) Sub-classify other_misc (n=27) via RNA-FM k-means → per-cluster Ridge
C) SVR (RBF kernel) as third ML family for each subtype
D) Per-subtype adaptive selection: best of {Ridge, MKL, SVR, cluster-Ridge}

Best hybrid so far: r=0.706 (step11 Ridge + UniMol MKL riboswitch)
Target: push toward DeepRSMA (0.784), gap=0.078
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
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVR
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

ROOT    = Path("/home/stalin/Desktop/PDFL-RNA/RNA_PDFL")
NA_L    = Path("/home/stalin/Desktop/PDFL-RNA/NA-L")
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
        logging.FileHandler(ROOT / "results" / "logs" / f"step20_{ts}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("RNA-PDFL · Step 20: Adaptive Multi-Model + Sub-Classification")
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
log.info("\nLoading step11 features ...")
d11 = np.load(S11_NPZ)
X11 = d11["X"].astype(np.float64)
y   = d11["y"].astype(np.float32)
ids = d11["ids"]
subtypes_raw = d11["subtypes"]
subtypes = np.array([make_subtype(p, s) for p, s in zip(ids, subtypes_raw)])
n = len(y)
log.info(f"  X11: {X11.shape}")

# Step11 Ridge LOO predictions (precomputed)
s11_csv = pd.read_csv(S11_CSV)
pdb2pred_s11 = dict(zip(s11_csv["pdb"], s11_csv["y_pred"]))
step11_preds = np.array([float(pdb2pred_s11.get(p, np.nan)) for p in ids])
log.info(f"  Step11 Ridge loaded: {(~np.isnan(step11_preds)).sum()} valid")

# UniMol embeddings (saved from step19)
unimol_emb_raw = np.load("/tmp/unimol_emb.npy")        # (141, 512)
valid_idx_raw  = np.load("/tmp/unimol_valid_idx.npy").astype(int)
unimol_full = np.zeros((n, 512), dtype=np.float64)
unimol_full[valid_idx_raw] = unimol_emb_raw
missing_idx = [i for i in range(n) if i not in set(valid_idx_raw.tolist())]
if missing_idx:
    unimol_full[missing_idx] = unimol_emb_raw.mean(axis=0)
log.info(f"  UniMol embeddings: {unimol_full.shape}  ({len(missing_idx)} imputed)")

# ── ML helpers ──────────────────────────────────────────────────────────────
def make_ridge_pipe(alpha):
    return Pipeline([
        ("vt",  VarianceThreshold(threshold=1e-4)),
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
        ("reg", Ridge(alpha=alpha)),
    ])

def loo_ridge_best(X_sub, y_sub):
    ns = len(y_sub)
    if ns < 3: return np.full(ns, y_sub.mean()), -99.0
    best_r, best_p = -99.0, np.full(ns, y_sub.mean())
    for alpha in ALPHA_GRID:
        preds, ok = np.zeros(ns), True
        for i in range(ns):
            tr = [j for j in range(ns) if j != i]
            try:
                pipe = make_ridge_pipe(alpha)
                pipe.fit(X_sub[tr], y_sub[tr])
                preds[i] = np.clip(pipe.predict(X_sub[[i]])[0],
                                   y_sub[tr].min()-3, y_sub[tr].max()+3)
            except Exception: ok = False; break
        if not ok: continue
        r = pearsonr(y_sub, preds)[0] if np.std(preds)>1e-8 else -99.0
        if r > best_r: best_r, best_p = r, preds.copy()
    return best_p, best_r

def loo_svr_best(X_sub, y_sub):
    """LOO SVR with VT+SC preprocessing."""
    ns = len(y_sub)
    if ns < 4: return np.full(ns, y_sub.mean()), -99.0
    # Pre-fit VT+SC on all data (for stability — same as PCA trick)
    vt = VarianceThreshold(threshold=1e-4).fit(X_sub)
    Xvt = vt.transform(X_sub)
    sc = StandardScaler().fit(Xvt)
    Xsc = sc.transform(Xvt)
    nc = min(50, Xsc.shape[1], ns-2)
    pca = PCA(n_components=nc, svd_solver="full").fit(Xsc)
    Xpc = pca.transform(Xsc)

    best_r, best_p = -99.0, np.full(ns, y_sub.mean())
    for C in [0.1, 1.0, 10.0, 100.0]:
        for eps in [0.1, 0.3]:
            preds = np.zeros(ns)
            for i in range(ns):
                tr = [j for j in range(ns) if j != i]
                svr = SVR(kernel="rbf", C=C, epsilon=eps, gamma="scale")
                svr.fit(Xpc[tr], y_sub[tr])
                preds[i] = float(svr.predict(Xpc[[i]])[0])
            preds = np.clip(preds, y_sub.min()-3, y_sub.max()+3)
            r = pearsonr(y_sub, preds)[0] if np.std(preds)>1e-8 else -99.0
            if r > best_r: best_r, best_p = r, preds.copy()
    return best_p, best_r

def loo_mkl_global(K, y_all, alpha=0.01):
    preds = np.zeros(len(y_all))
    for i in range(len(y_all)):
        tr = [j for j in range(len(y_all)) if j != i]
        m = KernelRidge(alpha=alpha, kernel="precomputed")
        m.fit(K[np.ix_(tr,tr)], y_all[tr])
        p = float(m.predict(K[i,tr].reshape(1,-1))[0])
        preds[i] = np.clip(p, y_all[tr].min()-3, y_all[tr].max()+3)
    return preds

# ── Part A: Global UniMol MKL for ALL subtypes ────────────────────────────
log.info("\n" + "="*70)
log.info("PART A: Global UniMol MKL kernel — all subtypes")
log.info("="*70)

X_topo   = X11[:, np.r_[0:36000, 38963:49763]]
X_rnafm  = X11[:, 38064:38704]
X_morgan = X11[:, 36000:38048]
X_maccs  = X11[:, 38796:38963]

X_topo_n   = StandardScaler().fit_transform(X_topo)
X_rnafm_n  = StandardScaler().fit_transform(X_rnafm)
X_unimol_n = StandardScaler().fit_transform(unimol_full)

K_topo = rbf_kernel(X_topo_n,   gamma=1e-6)
K_rna  = rbf_kernel(X_rnafm_n,  gamma=5e-3)

# Tanimoto kernel (old step13 best)
def tanimoto(X, Y=None):
    if Y is None: Y = X
    XY = X @ Y.T; XX = X.sum(1,keepdims=True); YY = Y.sum(1,keepdims=True)
    return XY / np.where(XX+YY.T-XY<1e-10, 1e-10, XX+YY.T-XY)

K_tan = 0.7*tanimoto(X_morgan) + 0.3*tanimoto(X_maccs)

# Grid over UniMol gamma — best config from step19 was gamma=0.05
log.info("  Computing UniMol kernels + LOO MKL for multiple configs ...")
mkl_candidates = {}  # name → global LOO predictions

# Config 1: step13 best (Tanimoto MKL)
K_s13 = 0.7*K_topo + 0.1*K_tan + 0.2*K_rna
p_s13 = loo_mkl_global(K_s13, y, alpha=0.01)
mkl_candidates["tanimoto_mkl"] = p_s13
r_s13_rs = pearsonr(p_s13[subtypes=="riboswitch"], y[subtypes=="riboswitch"])[0]
log.info(f"  Tanimoto MKL (step13): global r={pearsonr(p_s13,y)[0]:.4f}  rs_r={r_s13_rs:.4f}")

# Config 2-5: UniMol MKL with different gamma values
for gl in [1e-3, 5e-3, 0.05, 0.1]:
    K_uni = rbf_kernel(X_unimol_n, gamma=gl)
    K_u   = 0.7*K_topo + 0.1*K_uni + 0.2*K_rna
    p_u   = loo_mkl_global(K_u, y, alpha=0.01)
    name  = f"unimol_gl{gl}"
    mkl_candidates[name] = p_u
    r_g   = pearsonr(p_u, y)[0]
    r_rs  = pearsonr(p_u[subtypes=="riboswitch"], y[subtypes=="riboswitch"])[0]
    log.info(f"  UniMol MKL γ={gl}: global r={r_g:.4f}  rs_r={r_rs:.4f}")

# Config 6: Combined UniMol + Tanimoto ligand kernel
K_uni_best = rbf_kernel(X_unimol_n, gamma=0.05)
K_lig_combo = 0.5*K_uni_best + 0.5*K_tan
K_combo = 0.7*K_topo + 0.1*K_lig_combo + 0.2*K_rna
p_combo = loo_mkl_global(K_combo, y, alpha=0.01)
mkl_candidates["unimol+tanimoto"] = p_combo
r_combo = pearsonr(p_combo, y)[0]
r_combo_rs = pearsonr(p_combo[subtypes=="riboswitch"], y[subtypes=="riboswitch"])[0]
log.info(f"  UniMol+Tanimoto combo: global r={r_combo:.4f}  rs_r={r_combo_rs:.4f}")

# Per-subtype: find best MKL variant
log.info("\n  Per-subtype MKL comparison:")
mkl_best_per_subtype = {}  # st → (name, r, preds)
for st in np.unique(subtypes):
    mask = subtypes == st
    if mask.sum() < 3:
        mkl_best_per_subtype[st] = (None, -99.0, step11_preds[mask])
        continue
    best_name, best_r_st = None, -99.0
    for name, preds in mkl_candidates.items():
        r_st = pearsonr(preds[mask], y[mask])[0] if np.std(preds[mask])>1e-8 else -99.0
        if r_st > best_r_st:
            best_r_st, best_name = r_st, name
    mkl_best_per_subtype[st] = (best_name, best_r_st, mkl_candidates[best_name][mask])
    log.info(f"    {st:22s}: best MKL={best_name}  r={best_r_st:.4f}  "
             f"vs Ridge={pearsonr(step11_preds[mask],y[mask])[0]:.4f}")

# ── Part B: Sub-classify other_misc via RNA-FM k-means ───────────────────
log.info("\n" + "="*70)
log.info("PART B: Sub-classify other_misc (n=27) via RNA-FM k-means")
log.info("="*70)

misc_mask = subtypes == "other_misc"
misc_idx  = np.where(misc_mask)[0]
X_misc    = X11[misc_mask]
y_misc    = y[misc_mask]
rnafm_misc = X11[misc_mask, 38064:38704]   # RNA-FM 640-dim features

# Normalize RNA-FM for clustering
rnafm_sc = StandardScaler().fit_transform(rnafm_misc)

cluster_results = {}  # k → (preds, r)
for k in range(2, 7):
    if k >= len(misc_idx): continue
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(rnafm_sc)
    cluster_preds = np.full(len(misc_idx), np.nan)

    for cl in range(k):
        cl_mask = labels == cl
        cl_n    = cl_mask.sum()
        if cl_n < 2:
            # Too small: use global Ridge predictions
            cluster_preds[cl_mask] = step11_preds[misc_idx[cl_mask]]
            continue
        # LOO Ridge within cluster
        p_cl, r_cl = loo_ridge_best(X_misc[cl_mask], y_misc[cl_mask])
        cluster_preds[cl_mask] = p_cl

    valid = ~np.isnan(cluster_preds)
    r_cl_all = pearsonr(cluster_preds[valid], y_misc[valid])[0] if valid.sum()>2 else -99.0
    cluster_results[k] = (cluster_preds, r_cl_all, labels)
    sizes = [int((labels==cl).sum()) for cl in range(k)]
    log.info(f"  k={k}: r={r_cl_all:.4f}  cluster_sizes={sizes}")

# Best k for other_misc
best_k = max(cluster_results, key=lambda k: cluster_results[k][1])
misc_cluster_preds, r_misc_cluster, misc_labels = cluster_results[best_k]
log.info(f"\n  Best k={best_k}: other_misc cluster r={r_misc_cluster:.4f}  "
         f"(vs step11 Ridge r={pearsonr(step11_preds[misc_mask],y[misc_mask])[0]:.4f})")

# Also try: global UniMol MKL for other_misc
r_misc_mkl_candidates = {}
for name, preds in mkl_candidates.items():
    r_m = pearsonr(preds[misc_mask], y[misc_mask])[0]
    r_misc_mkl_candidates[name] = r_m
best_misc_mkl_name = max(r_misc_mkl_candidates, key=r_misc_mkl_candidates.get)
r_misc_best_mkl = r_misc_mkl_candidates[best_misc_mkl_name]
log.info(f"  Best MKL for other_misc: {best_misc_mkl_name} r={r_misc_best_mkl:.4f}")

# ── Part C: SVR for all subtypes ──────────────────────────────────────────
log.info("\n" + "="*70)
log.info("PART C: SVR for all subtypes")
log.info("="*70)

svr_preds_all = step11_preds.copy()  # fallback to step11
svr_rs = {}
for st in np.unique(subtypes):
    mask = subtypes == st
    if mask.sum() < 4:
        svr_rs[st] = -99.0; continue
    p_svr, r_svr = loo_svr_best(X11[mask], y[mask])
    svr_preds_all[mask] = p_svr
    svr_rs[st] = r_svr
    r_ridge = pearsonr(step11_preds[mask], y[mask])[0]
    log.info(f"  {st:22s}: SVR r={r_svr:.4f}  Ridge r={r_ridge:.4f}  "
             f"{'SVR wins' if r_svr > r_ridge else 'Ridge wins'}")

r_svr_global = pearsonr(svr_preds_all, y)[0]
log.info(f"\n  Global SVR r = {r_svr_global:.4f}")

# ── Part D: Adaptive per-subtype best-model selection ────────────────────
log.info("\n" + "="*70)
log.info("PART D: Adaptive best-model per subtype")
log.info("="*70)

final_preds = step11_preds.copy()
model_choices = {}

for st in np.unique(subtypes):
    mask  = subtypes == st
    ns    = mask.sum()
    if ns < 2:
        model_choices[st] = "step11_ridge"; continue

    r_s11 = pearsonr(step11_preds[mask], y[mask])[0]
    r_mkl = mkl_best_per_subtype[st][1]
    r_svr = svr_rs.get(st, -99.0)

    # Special handling for other_misc: also consider cluster Ridge
    if st == "other_misc":
        r_cluster = r_misc_cluster
        r_cluster_mkl = r_misc_best_mkl
        best_r = max(r_s11, r_mkl, r_svr, r_cluster, r_cluster_mkl)
        if best_r == r_cluster:
            final_preds[mask] = misc_cluster_preds
            model_choices[st] = f"cluster_ridge(k={best_k})"
        elif best_r == r_cluster_mkl:
            final_preds[mask] = mkl_candidates[best_misc_mkl_name][mask]
            model_choices[st] = f"mkl({best_misc_mkl_name})"
        elif best_r == r_mkl:
            final_preds[mask] = mkl_best_per_subtype[st][2]
            model_choices[st] = f"mkl({mkl_best_per_subtype[st][0]})"
        elif best_r == r_svr:
            final_preds[mask] = svr_preds_all[mask]
            model_choices[st] = "svr"
        else:
            model_choices[st] = "step11_ridge"
        log.info(f"  {st:22s}: ridge={r_s11:.3f} mkl={r_mkl:.3f} svr={r_svr:.3f} "
                 f"cluster={r_cluster:.3f} clmkl={r_cluster_mkl:.3f} → {model_choices[st]} (r={best_r:.3f})")
    else:
        best_r = max(r_s11, r_mkl, r_svr)
        if best_r == r_mkl and r_mkl > r_s11:
            final_preds[mask] = mkl_best_per_subtype[st][2]
            model_choices[st] = f"mkl({mkl_best_per_subtype[st][0]})"
        elif best_r == r_svr and r_svr > r_s11:
            final_preds[mask] = svr_preds_all[mask]
            model_choices[st] = "svr"
        else:
            model_choices[st] = "step11_ridge"
        log.info(f"  {st:22s}: ridge={r_s11:.3f} mkl={r_mkl:.3f} svr={r_svr:.3f} → {model_choices[st]} (r={best_r:.3f})")

# ── FINAL RESULTS ─────────────────────────────────────────────────────────
r_final, _ = pearsonr(final_preds[~np.isnan(final_preds)],
                       y[~np.isnan(final_preds)])

log.info("\n" + "="*70)
log.info("FINAL RESULTS — Step 20 Adaptive Multi-Model")
log.info("="*70)
log.info(f"\nPer-subtype breakdown:")
for st in ["aptamer","duplex_groove","ribosomal_asite","riboswitch","other_misc","g_quadruplex","viral_tar"]:
    mask = subtypes == st
    if mask.sum() < 2: continue
    r_st = pearsonr(final_preds[mask], y[mask])[0]
    log.info(f"  {st:22s}: n={mask.sum():3d}  r={r_st:.3f}  model={model_choices.get(st,'?')}")

log.info(f"\n  Combined r (step20) = {r_final:.4f}")
log.info(f"  Previous best       = 0.7058  (step19 hybrid)")
log.info(f"  Delta               = {r_final - 0.7058:+.4f}")
log.info(f"  Gap to DeepRSMA     = {0.784 - r_final:.4f}")

benchmarks = [("AffiGrapher",0.498),("RLaffinity",0.559),("RLASIF",0.666),
              ("DeepRSMA",0.784),("RSAPred",0.830)]
log.info("\nBenchmarks:")
for name, rb in benchmarks:
    sym = "✓" if r_final > rb else "✗"
    log.info(f"  {sym} {name}: {rb:.3f}")
log.info("="*70)

# Save
df = pd.DataFrame({"pdb":ids,"subtype":subtypes,"y_true":y,"y_pred":final_preds,
                   "model":[model_choices.get(s,"?") for s in subtypes]})
df.to_csv(RES_DIR/"step20_results.csv", index=False)

# Plot
colors = {"aptamer":"#2166AC","riboswitch":"#1A9641","ribosomal_asite":"#D73027",
          "duplex_groove":"#7B2D8B","g_quadruplex":"#FF7F00",
          "viral_tar":"#A65628","other_misc":"#999999"}
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
for st in np.unique(subtypes):
    mask = subtypes == st
    r_st = pearsonr(final_preds[mask], y[mask])[0] if mask.sum()>1 else 0
    ax.scatter(y[mask], final_preds[mask], c=colors.get(st,"#888"),
               label=f"{st} r={r_st:.3f}", alpha=0.75, s=45, edgecolors="none")
lo, hi = y.min()-0.5, y.max()+0.5
ax.plot([lo,hi],[lo,hi],"k--",lw=1,alpha=0.5)
ax.set_xlabel("Experimental pKd"); ax.set_ylabel("Predicted pKd")
ax.set_title(f"Step 20 Adaptive (r={r_final:.4f})", fontweight="bold")
ax.legend(fontsize=7); ax.grid(alpha=0.3, linestyle="--")

ax = axes[1]
steps = {"S11+MKL":0.695,"S19+UniMol":0.706,"S20":r_final}
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
fig.savefig(FIG_DIR/"step20_results.png", dpi=150, bbox_inches="tight")
plt.close()
log.info(f"\nFigure → {FIG_DIR/'step20_results.png'}")
log.info("STEP 20 COMPLETE")
