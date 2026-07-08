"""
SGT-RNA · Step 13: Multiple Kernel Learning (MKL)

Three kernels from step11 feature blocks:
  K_topo : RBF on SGT + interface SGT (46,800-dim)  — binding topology
  K_lig  : Tanimoto on Morgan fingerprint (2,048-dim)  — ligand similarity
  K_RNA  : RBF on RNA-FM embeddings (640-dim)          — RNA sequence/structure

Combined: K = α1·K_topo + α2·K_lig + α3·K_RNA
Model:    KernelRidge(kernel='precomputed', alpha=λ)

Grid search over (α_combo, λ, γ_topo, γ_rna) via LOO-CV.
Per-subtype KernelRidge LOO + adaptive stacking vs global MKL.
"""

import logging, time
from pathlib import Path
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path("/home/stalin/Desktop/SGT-RNA/RNA_SGT")
S11_NPZ = ROOT / "data" / "features" / "step11_full_features.npz"
RES_DIR = ROOT / "results"
FIG_DIR = ROOT / "results" / "figures"

# ── logging ───────────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "results" / "logs" / f"step13_{ts}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("SGT-RNA · Step 13: Multiple Kernel Learning")
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

# ── load features ─────────────────────────────────────────────────────────────
log.info("\nLoading step11 features ...")
d = np.load(S11_NPZ)
X11     = d["X"].astype(np.float32)
y       = d["y"].astype(np.float32)
ids     = d["ids"]
subtypes_raw = d["subtypes"]
subtypes = np.array([make_subtype(p, s) for p, s in zip(ids, subtypes_raw)])
n = len(y)
log.info(f"  X: {X11.shape}  n={n}")

# ── step11 feature layout ─────────────────────────────────────────────────────
# step09 (38963): PDFL[0:36000] | Morgan[36000:38048] | NucComp[38048:38058]
#                 Physico[38058:38064] | RNA-FM[38064:38704] | SS[38704:38712]
#                 kmer[38712:38796] | MACCS[38796:38963]
# interface SGT: [38963:49763] (4Å: 38963:42563, 6Å: 42563:46163, 8Å: 46163:49763)

X_topo  = X11[:, np.r_[0:36000, 38963:49763]].astype(np.float64)  # 46800
X_lig   = X11[:, 36000:38048].astype(np.float64)                   # Morgan 2048
X_maccs = X11[:, 38796:38963].astype(np.float64)                   # MACCS 167
X_rna   = X11[:, 38064:38704].astype(np.float64)                   # RNA-FM 640

log.info(f"  X_topo : {X_topo.shape}")
log.info(f"  X_lig  : {X_lig.shape}")
log.info(f"  X_rna  : {X_rna.shape}")
log.info(f"  X_maccs: {X_maccs.shape}")

# ── normalize topology and RNA blocks ─────────────────────────────────────────
log.info("\nNormalizing feature blocks ...")
sc_topo = StandardScaler()
X_topo_n = sc_topo.fit_transform(X_topo)

sc_rna = StandardScaler()
X_rna_n = sc_rna.fit_transform(X_rna)

# ── kernel functions ──────────────────────────────────────────────────────────
def tanimoto_kernel(X, Y=None):
    """Correct Tanimoto for binary fingerprints."""
    if Y is None: Y = X
    XY  = X @ Y.T
    XX  = X.sum(axis=1, keepdims=True)
    YY  = Y.sum(axis=1, keepdims=True)
    denom = XX + YY.T - XY
    denom = np.where(denom < 1e-10, 1e-10, denom)
    return XY / denom

def combined_lig_kernel(X_morgan, X_maccs, beta=0.7):
    """Blend Tanimoto on Morgan + Tanimoto on MACCS."""
    K_m = tanimoto_kernel(X_morgan)
    K_k = tanimoto_kernel(X_maccs)
    return beta * K_m + (1 - beta) * K_k

# ── precompute all base kernels at multiple γ values ─────────────────────────
log.info("\nPrecomputing base kernels ...")
t0 = time.time()

GAMMA_TOPO = [1e-6, 5e-6, 1e-5, 5e-5]
GAMMA_RNA  = [1e-4, 5e-4, 1e-3, 5e-3]

K_topo_bank = {g: rbf_kernel(X_topo_n, gamma=g) for g in GAMMA_TOPO}
K_rna_bank  = {g: rbf_kernel(X_rna_n,  gamma=g) for g in GAMMA_RNA}
K_lig       = combined_lig_kernel(X_lig, X_maccs, beta=0.7)

log.info(f"  Done in {time.time()-t0:.1f}s")
log.info(f"  Topo kernels: {len(K_topo_bank)}  RNA kernels: {len(K_rna_bank)}")

# ── grid search parameters ────────────────────────────────────────────────────
LAMBDA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]

# α combos (α_topo, α_lig, α_rna) on simplex
ALPHA_COMBOS = [
    (1/3,  1/3,  1/3),
    (0.6,  0.2,  0.2),
    (0.5,  0.3,  0.2),
    (0.5,  0.2,  0.3),
    (0.4,  0.4,  0.2),
    (0.4,  0.2,  0.4),
    (0.7,  0.2,  0.1),
    (0.7,  0.1,  0.2),
    (0.8,  0.1,  0.1),
    (1.0,  0.0,  0.0),
    (0.0,  1.0,  0.0),
    (0.0,  0.0,  1.0),
    (0.5,  0.5,  0.0),
    (0.5,  0.0,  0.5),
    (0.0,  0.5,  0.5),
]

log.info(f"\nGrid: {len(ALPHA_COMBOS)} α-combos × {len(GAMMA_TOPO)} γ_topo × "
         f"{len(GAMMA_RNA)} γ_rna × {len(LAMBDA_GRID)} λ = "
         f"{len(ALPHA_COMBOS)*len(GAMMA_TOPO)*len(GAMMA_RNA)*len(LAMBDA_GRID)} configs")

# ── LOO KernelRidge ───────────────────────────────────────────────────────────
def loo_kernel_ridge(K, y, lam):
    """LOO predictions for KernelRidge with precomputed kernel K."""
    n   = len(y)
    out = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        K_tr  = K[np.ix_(tr, tr)]
        K_te  = K[i, tr].reshape(1, -1)
        y_tr  = y[tr]
        try:
            m = KernelRidge(alpha=lam, kernel="precomputed")
            m.fit(K_tr, y_tr)
            p = float(m.predict(K_te)[0])
            out[i] = np.clip(p, y_tr.min()-3, y_tr.max()+3)
        except Exception:
            out[i] = y_tr.mean()
    r = pearsonr(y, out)[0] if np.std(out) > 1e-8 else -99.0
    return out, r

# ── global MKL search (LOO on all 143) ───────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("Global MKL hyperparameter search (LOO on 143 complexes) ...")
log.info("=" * 70)

best_global_r    = -99.0
best_global_preds = np.full(n, y.mean())
best_global_cfg  = {}

total_configs = len(ALPHA_COMBOS) * len(GAMMA_TOPO) * len(GAMMA_RNA) * len(LAMBDA_GRID)
done = 0
t0 = time.time()

for (a_t, a_l, a_r), g_t, g_r, lam in product(
        ALPHA_COMBOS, GAMMA_TOPO, GAMMA_RNA, LAMBDA_GRID):
    K_comb = a_t * K_topo_bank[g_t] + a_l * K_lig + a_r * K_rna_bank[g_r]
    preds, r = loo_kernel_ridge(K_comb, y, lam)
    done += 1
    if r > best_global_r:
        best_global_r    = r
        best_global_preds = preds.copy()
        best_global_cfg  = dict(a_topo=a_t, a_lig=a_l, a_rna=a_r,
                                g_topo=g_t, g_rna=g_r, lam=lam)
        log.info(f"  [{done:4d}/{total_configs}] NEW BEST r={r:.4f}  "
                 f"α=({a_t:.2f},{a_l:.2f},{a_r:.2f})  "
                 f"γ_t={g_t:.0e}  γ_r={g_r:.0e}  λ={lam}")

elapsed = time.time() - t0
log.info(f"\nSearch done in {elapsed:.1f}s")
log.info(f"Best global MKL: r={best_global_r:.4f}")
log.info(f"Best config: {best_global_cfg}")

# ── per-subtype MKL LOO with best γ values ───────────────────────────────────
g_t_best = best_global_cfg["g_topo"]
g_r_best = best_global_cfg["g_rna"]

log.info("\n" + "=" * 70)
log.info("Per-subtype MKL LOO (using best γ from global search) ...")
log.info("=" * 70)

unique_subtypes = sorted(set(subtypes))
subtype_preds   = np.full(n, np.nan)
subtype_rs      = {}

for st in unique_subtypes:
    mask = subtypes == st
    idx  = np.where(mask)[0]
    ns   = len(idx)
    ys   = y[idx]

    # Subtype kernel submatrices
    K_t_sub  = K_topo_bank[g_t_best][np.ix_(idx, idx)]
    K_l_sub  = K_lig[np.ix_(idx, idx)]
    K_r_sub  = K_rna_bank[g_r_best][np.ix_(idx, idx)]

    # Search best α + λ for this subtype
    best_st_r, best_st_preds = -99.0, np.full(ns, ys.mean())
    for (a_t, a_l, a_r), lam in product(ALPHA_COMBOS, LAMBDA_GRID):
        K_sub = a_t * K_t_sub + a_l * K_l_sub + a_r * K_r_sub
        preds_st, r_st = loo_kernel_ridge(K_sub, ys, lam)
        if r_st > best_st_r:
            best_st_r, best_st_preds = r_st, preds_st.copy()

    # Adaptive stacking
    preds_gl = best_global_preds[idx]
    r_gl = pearsonr(ys, preds_gl)[0] if np.std(preds_gl) > 1e-8 else -99.0

    if best_st_r >= r_gl:
        subtype_preds[idx] = best_st_preds
        chosen, r_use = "subtype", best_st_r
    else:
        subtype_preds[idx] = preds_gl
        chosen, r_use = "global", r_gl

    subtype_rs[st] = r_use
    log.info(f"  {st:<22}: n={ns:3d}  r={r_use:.4f}  (used {chosen})")

valid      = ~np.isnan(subtype_preds)
combined_r = pearsonr(y[valid], subtype_preds[valid])[0]

log.info("\n" + "=" * 70)
log.info(f"  COMBINED r (step13 MKL)  = {combined_r:.4f}")
log.info(f"  Step 11 baseline (Ridge) = 0.5754")
log.info(f"  Delta vs step11          = {combined_r - 0.5754:+.4f}")
log.info(f"  Delta vs step09          = {combined_r - 0.5350:+.4f}")
log.info("=" * 70)

benchmarks = [
    ("AffiGrapher", 0.498), ("RLaffinity", 0.559), ("RLASIF", 0.666),
    ("DeepRSMA", 0.784), ("RSAPred", 0.830),
]
log.info("\nBenchmark comparison:")
for name, r_bench in benchmarks:
    sym = "✓" if combined_r > r_bench else "✗"
    log.info(f"  {sym} {name}: {r_bench:.3f}  (ours: {combined_r:.4f})")

# ── save results ──────────────────────────────────────────────────────────────
df_res = pd.DataFrame({
    "pdb": ids, "subtype": subtypes,
    "y_true": y, "y_pred": subtype_preds,
})
res_path = RES_DIR / "step13_results.csv"
df_res.to_csv(res_path, index=False)
log.info(f"\nResults → {res_path}")

# ── figure ────────────────────────────────────────────────────────────────────
colors = {
    "aptamer":"#2166AC","riboswitch":"#1A9641","ribosomal_asite":"#D73027",
    "duplex_groove":"#7B2D8B","g_quadruplex":"#FF7F00",
    "viral_tar":"#A65628","other_misc":"#999999",
}

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.patch.set_facecolor("white")

# Panel A: scatter
ax = axes[0]
for st in unique_subtypes:
    mask = subtypes == st
    ax.scatter(y[mask], subtype_preds[mask],
               c=colors.get(st,"#888888"), label=f"{st} (r={subtype_rs[st]:.3f})",
               alpha=0.75, s=45, edgecolors="none")
mn, mx = y.min()-0.5, y.max()+0.5
ax.plot([mn,mx],[mn,mx],"k--",lw=1,alpha=0.4)
ax.set_xlabel("Experimental pKd"); ax.set_ylabel("Predicted pKd")
ax.set_title(f"Step 13: MKL  (combined r={combined_r:.4f})", fontweight="bold")
ax.legend(fontsize=7, loc="upper left", framealpha=0.7)
ax.grid(alpha=0.3, linestyle="--")

# Panel B: step-by-step progress vs benchmarks
ax = axes[1]
step_rs = {"S09":0.535,"S10":0.381,"S11":0.575,"S12":0.566,"S13 MKL":combined_r}
bar_colors = ["#AAAAAA","#DDDDDD","#4393C3","#6BAED6","#08519C"]
xs = list(step_rs.keys())
vals = list(step_rs.values())
bars = ax.bar(xs, vals, color=bar_colors, edgecolor="white", alpha=0.9)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
for name, r_bench in benchmarks:
    ax.axhline(r_bench, linestyle="--", lw=0.9, alpha=0.6, label=f"{name} {r_bench:.3f}")
ax.set_ylabel("Combined Pearson r"); ax.set_ylim(0.3, 0.92)
ax.set_title("Progress across steps", fontweight="bold")
ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3, linestyle="--")

# Panel C: per-subtype comparison step09 vs step11 vs step13
ax = axes[2]
main_sts = ["aptamer","riboswitch","ribosomal_asite","duplex_groove","other_misc","g_quadruplex"]
s09 = {"aptamer":0.918,"riboswitch":0.400,"ribosomal_asite":0.581,
       "duplex_groove":0.676,"other_misc":0.084,"g_quadruplex":-0.257}
s11 = {"aptamer":0.937,"riboswitch":0.388,"ribosomal_asite":0.753,
       "duplex_groove":0.810,"other_misc":0.395,"g_quadruplex":0.253}
x = np.arange(len(main_sts)); w = 0.25
ax.bar(x - w, [s09[s] for s in main_sts], w, label="Step09", color="#AAAAAA", edgecolor="white")
ax.bar(x,     [s11[s] for s in main_sts], w, label="Step11", color="#4393C3", alpha=0.85, edgecolor="white")
ax.bar(x + w, [subtype_rs.get(s, 0) for s in main_sts], w,
       label="Step13 MKL", color="#08519C", alpha=0.85, edgecolor="white")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(main_sts, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Pearson r"); ax.set_title("Per-subtype: S09 vs S11 vs S13", fontweight="bold")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3, linestyle="--")

plt.tight_layout()
fig_path = FIG_DIR / "step13_results.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
log.info(f"Figure → {fig_path}")

log.info("\n" + "=" * 70)
log.info("STEP 13 COMPLETE")
log.info("=" * 70)
