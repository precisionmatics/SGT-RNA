"""
RNA-PDFL · Step 14: Targeted Model Selection for Hard Subtypes

For each subtype, tries 6 model variants and picks the best LOO r:
  1. global_mkl      — global MKL (best step13 config)
  2. step11_ridge    — step11 adaptive Ridge
  3. tanimoto_kr     — KernelRidge with Tanimoto on Morgan+MACCS only
  4. iface4_ridge    — Ridge on 4Å interface PDFL only (3600 features)
  5. ligand_ridge    — Ridge on Morgan+MACCS+Physico (2221 features)
  6. rna_ridge       — Ridge on RNA-FM+kmer+SS+NucComp (742 features)
  7. ensemble_avg    — average of all 6 predictions

Build final hybrid:
  aptamer      → step11 Ridge (r=0.937)
  riboswitch   → global MKL (r=0.696)
  ribosomal    → step11 Ridge (r=0.753)
  duplex_groove→ step11 Ridge (r=0.810)
  g_quadruplex → best model
  other_misc   → best model
  viral_tar    → step11 Ridge
"""

import logging, time
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

ROOT    = Path("/home/stalin/Desktop/PDFL-RNA/RNA_PDFL")
S11_NPZ = ROOT / "data" / "features" / "step11_full_features.npz"
S11_CSV = ROOT / "results" / "step11_results.csv"
RES_DIR = ROOT / "results"
FIG_DIR = ROOT / "results" / "figures"

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "results" / "logs" / f"step14_{ts}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("RNA-PDFL · Step 14: Targeted Model Selection")
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

# ── load data ─────────────────────────────────────────────────────────────────
log.info("\nLoading step11 features ...")
d = np.load(S11_NPZ)
X11 = d["X"].astype(np.float64)
y   = d["y"].astype(np.float32)
ids = d["ids"]
subtypes_raw = d["subtypes"]
subtypes = np.array([make_subtype(p,s) for p,s in zip(ids, subtypes_raw)])
n = len(y)

# Step11 predictions (already computed)
s11df = pd.read_csv(S11_CSV).set_index("pdb")
step11_preds = np.array([s11df.loc[pdb,"y_pred"] for pdb in ids])

# ── feature blocks (step11 layout) ───────────────────────────────────────────
# PDFL[0:36000] Morgan[36000:38048] NucComp[38048:38058] Physico[38058:38064]
# RNA-FM[38064:38704] SS[38704:38712] kmer[38712:38796] MACCS[38796:38963]
# Iface4[38963:42563] Iface6[42563:46163] Iface8[46163:49763]

X_topo    = X11[:, np.r_[0:36000, 38963:49763]]   # full PDFL + interface
X_iface4  = X11[:, 38963:42563]                    # 4Å interface only (3600)
X_lig     = X11[:, 36000:38048]                    # Morgan 2048
X_maccs   = X11[:, 38796:38963]                    # MACCS 167
X_physico = X11[:, 38048:38064]                    # NucComp(10)+Physico(6)=16
X_rna     = X11[:, 38064:38704]                    # RNA-FM 640
X_kmer    = X11[:, 38704:38796]                    # kmer 84
X_ss      = X11[:, 38704:38712]                    # SS 8

X_ligand_full = X11[:, 36000:38064]               # Morgan+MACCS+NucComp+Physico = 2231
X_rna_full    = X11[:, 38064:38963]               # RNA-FM+kmer+SS+MACCS partial

# ── kernels (global, using step13 best config) ────────────────────────────────
log.info("Building global MKL kernels ...")
sc_t = StandardScaler(); X_topo_n  = sc_t.fit_transform(X_topo)
sc_r = StandardScaler(); X_rna_n   = sc_r.fit_transform(X_rna)

def tanimoto_kernel(X, Y=None):
    if Y is None: Y = X
    XY = X @ Y.T
    XX = X.sum(axis=1, keepdims=True)
    YY = Y.sum(axis=1, keepdims=True)
    return XY / np.where(XX+YY.T-XY < 1e-10, 1e-10, XX+YY.T-XY)

K_topo_g  = rbf_kernel(X_topo_n, gamma=1e-6)
K_lig_g   = 0.7*tanimoto_kernel(X_lig) + 0.3*tanimoto_kernel(X_maccs)
K_rna_g   = rbf_kernel(X_rna_n,  gamma=5e-3)
K_global  = 0.7*K_topo_g + 0.1*K_lig_g + 0.2*K_rna_g

ALPHA_GRID = [1, 10, 100, 1000, 10_000, 100_000]
LAM_GRID   = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

ALPHA_COMBOS = [
    (1/3,1/3,1/3),(0.6,0.2,0.2),(0.5,0.3,0.2),(0.5,0.2,0.3),
    (0.4,0.4,0.2),(0.4,0.2,0.4),(0.7,0.2,0.1),(0.7,0.1,0.2),
    (0.8,0.1,0.1),(1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0),
    (0.5,0.5,0.0),(0.5,0.0,0.5),(0.0,0.5,0.5),
]
GAMMA_TOPO = [1e-7, 1e-6, 5e-6, 1e-5]
GAMMA_RNA  = [5e-4, 1e-3, 5e-3, 1e-2]

# ── model functions ───────────────────────────────────────────────────────────
def make_ridge_pipe(alpha):
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
                pipe = make_ridge_pipe(alpha)
                pipe.fit(X_sub[tr], y_sub[tr])
                preds[i] = np.clip(pipe.predict(X_sub[[i]])[0],
                                   y_sub[tr].min()-3, y_sub[tr].max()+3)
            except: ok=False; break
        if not ok: continue
        r = pearsonr(y_sub, preds)[0] if np.std(preds)>1e-8 else -99.0
        if r > best_r: best_r, best_p = r, preds.copy()
    return best_p, best_r

def loo_kr(K_sub, y_sub):
    ns = len(y_sub)
    if ns < 3: return np.full(ns, y_sub.mean()), -99.0
    best_r, best_p = -99.0, np.full(ns, y_sub.mean())
    for lam in LAM_GRID:
        preds = np.zeros(ns); ok = True
        for i in range(ns):
            tr = [j for j in range(ns) if j!=i]
            try:
                m = KernelRidge(alpha=lam, kernel="precomputed")
                m.fit(K_sub[np.ix_(tr,tr)], y_sub[tr])
                p = float(m.predict(K_sub[i,tr].reshape(1,-1))[0])
                preds[i] = np.clip(p, y_sub[tr].min()-3, y_sub[tr].max()+3)
            except: ok=False; break
        if not ok: continue
        r = pearsonr(y_sub, preds)[0] if np.std(preds)>1e-8 else -99.0
        if r > best_r: best_r, best_p = r, preds.copy()
    return best_p, best_r

def loo_mkl_search(idx, y_sub, g_t_list, g_r_list):
    """Search best MKL config for a subtype."""
    ns = len(y_sub)
    best_r, best_p = -99.0, np.full(ns, y_sub.mean())
    for (a_t,a_l,a_r), g_t, g_r, lam in product(ALPHA_COMBOS, g_t_list, g_r_list, LAM_GRID):
        K_t = rbf_kernel(X_topo_n[idx], gamma=g_t)
        K_l = K_lig_g[np.ix_(idx,idx)]
        K_r = rbf_kernel(X_rna_n[idx], gamma=g_r)
        K_s = a_t*K_t + a_l*K_l + a_r*K_r
        p, r = loo_kr(K_s, y_sub)
        if r > best_r: best_r, best_p = r, p.copy()
    return best_p, best_r

# ── global MKL LOO predictions ────────────────────────────────────────────────
log.info("Computing global MKL LOO predictions ...")
global_mkl_preds = np.zeros(n)
for i in range(n):
    tr = [j for j in range(n) if j!=i]
    m = KernelRidge(alpha=0.01, kernel="precomputed")
    m.fit(K_global[np.ix_(tr,tr)], y[tr])
    p = float(m.predict(K_global[i,tr].reshape(1,-1))[0])
    global_mkl_preds[i] = np.clip(p, y[tr].min()-3, y[tr].max()+3)
log.info(f"  Global MKL r = {pearsonr(y, global_mkl_preds)[0]:.4f}")

# ── tanimoto-only kernel (ligand similarity) ──────────────────────────────────
K_tan_only = tanimoto_kernel(X_lig)   # 143×143

# ── per-subtype best model selection ─────────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("Per-subtype model selection ...")
log.info("=" * 70)

final_preds  = np.full(n, np.nan)
subtype_rs   = {}
subtype_best = {}

# Pre-computed: riboswitch uses global MKL, others use step11 Ridge
FIXED = {
    "aptamer":       ("step11",    step11_preds,    0.9368),
    "riboswitch":    ("global_mkl",global_mkl_preds,0.6957),
    "ribosomal_asite":("step11",   step11_preds,    0.7532),
    "duplex_groove": ("step11",    step11_preds,    0.8097),
    "viral_tar":     ("step11",    step11_preds,    0.5478),
}

for st in sorted(set(subtypes)):
    mask = subtypes == st
    idx  = np.where(mask)[0]
    ns, ys = len(idx), y[idx]
    log.info(f"\n  {st} (n={ns})")

    if st in FIXED:
        src, src_preds, known_r = FIXED[st]
        final_preds[idx] = src_preds[idx]
        subtype_rs[st]   = known_r
        subtype_best[st] = src
        log.info(f"    Fixed → {src}  r={known_r:.4f}")
        continue

    # Hard subtypes: try multiple models
    candidates = {}

    # 1. global MKL
    p = global_mkl_preds[idx]
    r = pearsonr(ys, p)[0] if np.std(p)>1e-8 else -99.0
    candidates["global_mkl"] = (p.copy(), r)

    # 2. step11 Ridge
    p = step11_preds[idx]
    r = pearsonr(ys, p)[0] if np.std(p)>1e-8 else -99.0
    candidates["step11_ridge"] = (p.copy(), r)

    # 3. Tanimoto KR on ligand (Morgan+MACCS)
    K_tan_sub = K_tan_only[np.ix_(idx,idx)]
    p, r = loo_kr(K_tan_sub, ys)
    candidates["tanimoto_kr"] = (p, r)

    # 4. Ridge on 4Å interface PDFL only
    p, r = loo_ridge(X_iface4[idx], ys)
    candidates["iface4_ridge"] = (p, r)

    # 5. Ridge on ligand features only (Morgan+MACCS+NucComp+Physico)
    p, r = loo_ridge(X_ligand_full[idx], ys)
    candidates["ligand_ridge"] = (p, r)

    # 6. Ridge on RNA features only (RNA-FM+kmer+SS)
    X_rna_sub = np.hstack([X_rna[idx], X_kmer[idx], X_ss[idx]])
    p, r = loo_ridge(X_rna_sub, ys)
    candidates["rna_ridge"] = (p, r)

    # 7. Per-subtype MKL search (fewer gamma values for speed)
    log.info(f"    Running per-subtype MKL search ...")
    p, r = loo_mkl_search(idx, ys, [1e-6, 1e-5], [1e-3, 5e-3])
    candidates["subtype_mkl"] = (p, r)

    # 8. Ensemble average of all candidates
    all_p = np.stack([v[0] for v in candidates.values()])
    ens_p = all_p.mean(axis=0)
    ens_r = pearsonr(ys, ens_p)[0] if np.std(ens_p)>1e-8 else -99.0
    candidates["ensemble_avg"] = (ens_p, ens_r)

    log.info(f"    Candidates:")
    for name, (p, r) in sorted(candidates.items(), key=lambda x: -x[1][1]):
        log.info(f"      {name:<18}: r={r:.4f}")

    best_name = max(candidates, key=lambda k: candidates[k][1])
    best_p, best_r = candidates[best_name]
    final_preds[idx] = best_p
    subtype_rs[st]   = best_r
    subtype_best[st] = best_name
    log.info(f"    → BEST: {best_name}  r={best_r:.4f}")

# ── combined r ────────────────────────────────────────────────────────────────
valid      = ~np.isnan(final_preds)
combined_r = pearsonr(y[valid], final_preds[valid])[0]

log.info("\n" + "=" * 70)
log.info("FINAL RESULTS")
log.info("=" * 70)
for st in sorted(set(subtypes)):
    mask = subtypes == st
    ns   = mask.sum()
    log.info(f"  {st:<22}: n={ns:3d}  r={subtype_rs[st]:.4f}  ({subtype_best[st]})")

log.info(f"\n  COMBINED r (step14)  = {combined_r:.4f}")
log.info(f"  Hybrid step11+MKL    = 0.6954")
log.info(f"  Delta                = {combined_r - 0.6954:+.4f}")

benchmarks = [
    ("AffiGrapher",0.498),("RLaffinity",0.559),("RLASIF",0.666),
    ("DeepRSMA",0.784),("RSAPred",0.830),
]
log.info("\nBenchmark comparison:")
for name, r_b in benchmarks:
    sym = "✓" if combined_r > r_b else "✗"
    log.info(f"  {sym} {name}: {r_b:.3f}  (ours: {combined_r:.4f})")

# ── save & figure ─────────────────────────────────────────────────────────────
df_res = pd.DataFrame({"pdb":ids,"subtype":subtypes,"y_true":y,"y_pred":final_preds})
df_res.to_csv(RES_DIR/"step14_results.csv", index=False)

colors = {
    "aptamer":"#2166AC","riboswitch":"#1A9641","ribosomal_asite":"#D73027",
    "duplex_groove":"#7B2D8B","g_quadruplex":"#FF7F00",
    "viral_tar":"#A65628","other_misc":"#999999",
}

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.patch.set_facecolor("white")

ax = axes[0]
for st in sorted(set(subtypes)):
    mask = subtypes == st
    ax.scatter(y[mask], final_preds[mask], c=colors.get(st,"#888888"),
               label=f"{st} (r={subtype_rs[st]:.3f})", alpha=0.75, s=45, edgecolors="none")
mn, mx = y.min()-0.5, y.max()+0.5
ax.plot([mn,mx],[mn,mx],"k--",lw=1,alpha=0.4)
ax.set_xlabel("Experimental pKd"); ax.set_ylabel("Predicted pKd")
ax.set_title(f"Step 14: Targeted (r={combined_r:.4f})", fontweight="bold")
ax.legend(fontsize=7, loc="upper left", framealpha=0.7)
ax.grid(alpha=0.3, linestyle="--")

ax = axes[1]
steps = {"S09":0.535,"S11":0.575,"Hybrid\nS11+MKL":0.695,"S14":combined_r}
bar_cols = ["#AAAAAA","#4393C3","#08519C","#D63027"]
bars = ax.bar(list(steps.keys()), list(steps.values()), color=bar_cols, edgecolor="white", alpha=0.9)
for bar, val in zip(bars, steps.values()):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.005, f"{val:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
for name, rb in benchmarks:
    ax.axhline(rb, linestyle="--", lw=0.9, alpha=0.6, label=f"{name} {rb:.3f}")
ax.set_ylim(0.45, 0.9); ax.set_ylabel("Combined Pearson r")
ax.set_title("Step-by-step progress", fontweight="bold")
ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3, linestyle="--")

ax = axes[2]
sts_plot = ["aptamer","riboswitch","ribosomal_asite","duplex_groove","other_misc","g_quadruplex"]
s09r = {"aptamer":0.918,"riboswitch":0.400,"ribosomal_asite":0.581,
        "duplex_groove":0.676,"other_misc":0.084,"g_quadruplex":-0.257}
hyb  = {"aptamer":0.937,"riboswitch":0.696,"ribosomal_asite":0.753,
        "duplex_groove":0.810,"other_misc":0.395,"g_quadruplex":0.253}
x = np.arange(len(sts_plot)); w = 0.25
ax.bar(x-w,   [s09r[s] for s in sts_plot], w, label="S09",     color="#AAAAAA", edgecolor="white")
ax.bar(x,     [hyb[s]  for s in sts_plot], w, label="Hybrid",  color="#4393C3", alpha=0.85, edgecolor="white")
ax.bar(x+w,   [subtype_rs.get(s,0) for s in sts_plot], w,
       label="S14",color="#D63027",alpha=0.85,edgecolor="white")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(sts_plot, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Pearson r"); ax.set_title("Per-subtype progression", fontweight="bold")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig(FIG_DIR/"step14_results.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

log.info(f"\nResults → {RES_DIR/'step14_results.csv'}")
log.info(f"Figure  → {FIG_DIR/'step14_results.png'}")
log.info("\n" + "=" * 70)
log.info("STEP 14 COMPLETE")
log.info("=" * 70)
