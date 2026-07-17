"""
SGT-RNA  ·  Step 3: Spectral Graph Topology Feature Extraction

For each complex and each of the 36 element pairs (RNA {C,N,O,P} × Lig {C,N,O,S,P,F,Cl,Br,I}):
  1. Compute pairwise distance matrix D_{αβ}
  2. Apply Exponential FRI kernel: W = exp(-(D/η)^κ), normalize per-pair to [0,1]
  3. At each of 5 filtration thresholds τ ∈ {0.0, 0.8, 0.85, 0.90, 0.95}:
       a. Retain edges where W̃ ≥ τ  (weighted bipartite directed graph)
       b. Build weighted vertex Laplacian L_0 of the bipartite graph
       c. Compute eigenvalues of L_0  (numpy.linalg.eigvalsh — symmetric PSD)
       d. Derive L_1 (edge Laplacian) eigenvalues via algebraic relationship:
             non-zero eigenvalues of L_1 = non-zero eigenvalues of L_0
             zero eigenvalues  of L_1 = m − (n − β_0)  where β_0 = dim ker L_0
       e. Extract 10 spectral statistics from each:
             min, max, mean, median, var, std, sum, sum², #nonzero, #zero (Betti)
  4. Concatenate → 36 × 5 × 2 × 10 = 3 600-dimensional feature vector per complex

Output:
  data/features/step03_sgt_features.npz  — X (n×3600) and y (n,) arrays
  results/figures/step03_*.png            — journal-quality figures
"""

import gzip, pickle, logging, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
PKL_FILE = ROOT / "data" / "pocket_fri" / "pocket_fri_data.pkl.gz"
OUT_DIR  = ROOT / "data" / "features"
FIG_DIR  = ROOT / "results" / "figures"
LOG_DIR  = ROOT / "results" / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── logging ───────────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = LOG_DIR / f"step03_{ts}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("SGT-RNA  ·  Step 3: SGT Feature Extraction")
log.info("=" * 70)

# ── constants ─────────────────────────────────────────────────────────────────
RNA_ELEMENTS = ["C", "N", "O", "P"]
LIG_ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
N_PAIRS      = len(RNA_ELEMENTS) * len(LIG_ELEMENTS)   # 36
THRESHOLDS   = [0.0, 0.8, 0.85, 0.90, 0.95]           # 5 filtration levels
N_STATS      = 10                                       # per eigenspectrum
N_FEATS      = N_PAIRS * len(THRESHOLDS) * 2 * N_STATS # 3 600

ETA   = 5.0
KAPPA = 2.0
EPS   = 1e-8

log.info(f"  Element pairs : {N_PAIRS}  (RNA × Ligand)")
log.info(f"  Filtration τ  : {THRESHOLDS}")
log.info(f"  FRI kernel    : Exponential  η={ETA}  κ={KAPPA}")
log.info(f"  Feature dim   : {N_FEATS}")

# ── FRI kernel ────────────────────────────────────────────────────────────────
def fri_exp(D):
    return np.exp(-(D / ETA) ** KAPPA)

# ── 10 spectral statistics ────────────────────────────────────────────────────
def spectral_stats(eigs: np.ndarray) -> np.ndarray:
    if len(eigs) == 0:
        return np.zeros(N_STATS, dtype=np.float32)
    nz   = eigs[eigs > EPS]
    out  = np.array([
        float(eigs.min()),          # 0  min eigenvalue
        float(eigs.max()),          # 1  max eigenvalue
        float(eigs.mean()),         # 2  mean
        float(np.median(eigs)),     # 3  median
        float(eigs.var()),          # 4  variance
        float(eigs.std()),          # 5  std dev
        float(eigs.sum()),          # 6  sum
        float((eigs**2).sum()),     # 7  sum of squares
        float(len(nz)),             # 8  # non-zero (rank)
        float(len(eigs) - len(nz)), # 9  # zero  =  Betti number
    ], dtype=np.float32)
    return out

# ── weighted vertex Laplacian of bipartite graph ──────────────────────────────
def build_L0(W_sel: np.ndarray) -> np.ndarray:
    """
    W_sel : (n_r, n_l) selected weight matrix (zeros for absent edges).
    Returns L_0 of shape (n_r+n_l, n_r+n_l).
    """
    n_r, n_l = W_sel.shape
    n = n_r + n_l
    L = np.zeros((n, n), dtype=np.float64)
    row_deg = W_sel.sum(axis=1)
    col_deg = W_sel.sum(axis=0)
    for i in range(n_r):
        L[i, i] = row_deg[i]
    for j in range(n_l):
        L[n_r + j, n_r + j] = col_deg[j]
    L[:n_r, n_r:] = -W_sel
    L[n_r:, :n_r] = -W_sel.T
    return L

# ── feature extraction for one element pair ───────────────────────────────────
def pair_features(rc: np.ndarray, lc: np.ndarray) -> np.ndarray:
    """
    rc : (n_r, 3)  RNA-α atom coords
    lc : (n_l, 3)  Lig-β atom coords
    Returns (len(THRESHOLDS) * 2 * N_STATS,) float32 features.
    """
    n_tau  = len(THRESHOLDS)
    n_feat = n_tau * 2 * N_STATS
    n_r, n_l = len(rc), len(lc)

    if n_r == 0 or n_l == 0:
        return np.zeros(n_feat, dtype=np.float32)

    D     = cdist(rc, lc, metric="euclidean").astype(np.float64)
    W     = fri_exp(D)
    wmax  = W.max()
    if wmax < 1e-12:
        return np.zeros(n_feat, dtype=np.float32)
    W_norm = W / wmax            # normalise to [0, 1] per-pair

    n_total = n_r + n_l
    feats   = np.zeros(n_feat, dtype=np.float32)
    offset  = 0

    for tau in THRESHOLDS:
        mask   = W_norm >= tau
        W_sel  = W_norm * mask   # zero out sub-threshold edges
        n_edges = int(mask.sum())

        if n_edges == 0:
            # No edges: L_0 = 0, all eigenvalues = 0
            eigs_L0 = np.zeros(n_total)
            eigs_L1 = np.array([])
        else:
            L0      = build_L0(W_sel)
            eigs_L0 = np.linalg.eigvalsh(L0)
            eigs_L0 = np.maximum(eigs_L0, 0.0)   # clip small negatives

            # L_1 eigenvalues via algebraic identity (avoids m×m matrix)
            beta0      = int((eigs_L0 < EPS).sum())
            n_zeros_L1 = max(0, n_edges - n_total + beta0)   # = β_1
            nonzero_L0 = eigs_L0[eigs_L0 >= EPS]
            eigs_L1    = np.concatenate([np.zeros(n_zeros_L1), nonzero_L0])

        s0 = spectral_stats(eigs_L0)
        s1 = spectral_stats(eigs_L1)
        feats[offset: offset + N_STATS]           = s0
        feats[offset + N_STATS: offset + 2*N_STATS] = s1
        offset += 2 * N_STATS

    return feats

# ── load pocket data ──────────────────────────────────────────────────────────
log.info(f"\nLoading pocket data from {PKL_FILE} ...")
with gzip.open(PKL_FILE, "rb") as f:
    records = pickle.load(f)
log.info(f"  Loaded {len(records)} complexes")

# ── main feature extraction loop ──────────────────────────────────────────────
log.info(f"\nExtracting {N_FEATS}-dim SGT features ...")
n = len(records)
X   = np.zeros((n, N_FEATS), dtype=np.float32)
y   = np.zeros(n, dtype=np.float32)
ids = []

# build column names for reference
col_names = []
for r_el in RNA_ELEMENTS:
    for l_el in LIG_ELEMENTS:
        for tau in THRESHOLDS:
            for lap in ["L0", "L1"]:
                for stat in ["min","max","mean","med","var","std","sum","sum2","rank","betti"]:
                    col_names.append(f"{r_el}-{l_el}|τ={tau}|{lap}|{stat}")

import time
t0 = time.time()

for idx, rec in enumerate(records):
    pdb = rec["pdb"]
    ids.append(pdb)
    y[idx] = rec["pkd"]

    feat_parts = []
    for r_el in RNA_ELEMENTS:
        for l_el in LIG_ELEMENTS:
            rc = rec["rna_coords"][r_el]
            lc = rec["lig_coords"].get(l_el, np.empty((0, 3), np.float32))
            feat_parts.append(pair_features(rc, lc))

    X[idx] = np.concatenate(feat_parts)

    if (idx + 1) % 20 == 0 or idx == 0:
        elapsed = time.time() - t0
        rate    = (idx + 1) / elapsed
        eta_s   = (n - idx - 1) / rate if rate > 0 else 0
        nnz     = np.count_nonzero(X[idx])
        log.info(f"  [{idx+1:3d}/{n}] {pdb}  nonzero={nnz}/{N_FEATS} "
                 f"({100*nnz/N_FEATS:.1f}%)  ETA {eta_s:.0f}s")

elapsed = time.time() - t0
log.info(f"\nFeature extraction complete in {elapsed:.1f}s  ({elapsed/n:.2f}s per complex)")

# ── basic feature stats ───────────────────────────────────────────────────────
nnz_per_sample   = (X != 0).sum(axis=1)
nnz_per_feat     = (X != 0).sum(axis=0)
var_per_feat     = X.var(axis=0)
zero_feat_mask   = (nnz_per_feat == 0)
n_zero_feats     = zero_feat_mask.sum()

log.info(f"\n  Non-zero features per sample : {nnz_per_sample.mean():.0f} ± {nnz_per_sample.std():.0f}")
log.info(f"  Zero features (all samples) : {n_zero_feats}/{N_FEATS}")
log.info(f"  Feature variance (mean)     : {var_per_feat.mean():.4f}")
log.info(f"  pKd range                   : {y.min():.3f} – {y.max():.3f}")

# ── save features ─────────────────────────────────────────────────────────────
out_npz = OUT_DIR / "step03_sgt_features.npz"
np.savez_compressed(out_npz, X=X, y=y, ids=np.array(ids), col_names=np.array(col_names))
log.info(f"\nSaved features → {out_npz}")

# Save CSV for inspection
df_feat = pd.DataFrame(X, columns=col_names)
df_feat.insert(0, "pdb", ids)
df_feat.insert(1, "pKd", y)
feat_csv = OUT_DIR / "step03_sgt_features_head.csv"
df_feat.head(10).to_csv(feat_csv, index=False)
log.info(f"Sample CSV (head 10) → {feat_csv}")

# ── journal-quality figures ───────────────────────────────────────────────────
log.info("\nGenerating figures ...")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.linewidth": 1.2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
})

PAIR_LABELS = [f"{r}-{l}" for r in RNA_ELEMENTS for l in LIG_ELEMENTS]

# ── Figure 1: Feature overview (6 panels) ────────────────────────────────────
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor("white")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)
fig.suptitle("SGT-RNA  |  Step 3: 3600-dim SGT Feature Matrix Overview",
             fontsize=16, fontweight="bold", y=0.98)

# Panel A: Feature variance per dimension
ax = fig.add_subplot(gs[0, 0])
sorted_var = np.sort(var_per_feat)[::-1]
ax.semilogy(sorted_var, color="#4C72B0", linewidth=1.2, alpha=0.85)
ax.axhline(np.median(sorted_var), color="crimson", linestyle="--", linewidth=1.2,
           label=f"Median var = {np.median(sorted_var):.4f}")
ax.set_xlabel("Feature rank (by variance)", fontsize=11)
ax.set_ylabel("Variance (log scale)", fontsize=11)
ax.set_title("A  |  Feature Variance Spectrum", fontsize=12, fontweight="bold", loc="left")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, linestyle="--")

# Panel B: % nonzero per sample
ax = fig.add_subplot(gs[0, 1])
pct_nz = 100 * nnz_per_sample / N_FEATS
ax.hist(pct_nz, bins=25, color="#55A868", edgecolor="white", linewidth=0.8)
ax.axvline(pct_nz.mean(), color="crimson", linestyle="--", linewidth=1.5,
           label=f"Mean = {pct_nz.mean():.1f}%")
ax.set_xlabel("% Non-zero Features per Complex", fontsize=11)
ax.set_ylabel("# Complexes", fontsize=11)
ax.set_title("B  |  Feature Density per Sample", fontsize=12, fontweight="bold", loc="left")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3, linestyle="--")

# Panel C: Per-pair feature variance heatmap
ax = fig.add_subplot(gs[0, 2])
var_per_pair = np.zeros(N_PAIRS)
for pi, (r_el, l_el) in enumerate([(r, l) for r in RNA_ELEMENTS for l in LIG_ELEMENTS]):
    start = pi * len(THRESHOLDS) * 2 * N_STATS
    end   = start + len(THRESHOLDS) * 2 * N_STATS
    var_per_pair[pi] = var_per_feat[start:end].mean()
var_mat = var_per_pair.reshape(4, 9)
cmap = LinearSegmentedColormap.from_list("var", ["#EFF3FF", "#2171B5", "#08306B"])
im = ax.imshow(var_mat, aspect="auto", cmap=cmap)
for ri in range(4):
    for li in range(9):
        ax.text(li, ri, f"{var_mat[ri,li]:.3f}", ha="center", va="center",
                fontsize=7.5, color="white" if var_mat[ri,li] > var_mat.max()*0.5 else "black")
ax.set_xticks(range(9)); ax.set_xticklabels(LIG_ELEMENTS, fontsize=9)
ax.set_yticks(range(4)); ax.set_yticklabels(RNA_ELEMENTS, fontsize=10)
ax.set_xlabel("Ligand Element", fontsize=11)
ax.set_ylabel("RNA Pocket Element", fontsize=11)
ax.set_title("C  |  Mean Feature Variance per Pair", fontsize=12, fontweight="bold", loc="left")
plt.colorbar(im, ax=ax, shrink=0.85, label="Mean variance")

# Panel D: Betti-0 statistics (zero eigs of L0) for top pairs at τ=0
ax = fig.add_subplot(gs[1, 0])
# Extract β_0 (zero eigenvalue count of L0) at τ=0.0 for each pair
betti0_per_pair = np.zeros((n, 4))  # 4 RNA elements, C-C, N-C, O-C, P-C
top4 = [0, 9, 18, 27]  # pairs {C-C, N-C, O-C, P-C}
top4_labels = ["C-C", "N-C", "O-C", "P-C"]
for k, pi in enumerate(top4):
    # Feature offset: pair_idx=pi, threshold=0 (τ=0.0), L0, stat_idx=9 (betti)
    feat_idx = pi * len(THRESHOLDS) * 2 * N_STATS + 0 * 2 * N_STATS + N_STATS - 1
    betti0_per_pair[:, k] = X[:, feat_idx]
bp = ax.boxplot([betti0_per_pair[:, k] for k in range(4)],
                patch_artist=True, widths=0.55,
                medianprops=dict(color="black", linewidth=2))
colors4 = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
for patch, c in zip(bp["boxes"], colors4):
    patch.set_facecolor(c); patch.set_alpha(0.8)
ax.set_xticks(range(1,5)); ax.set_xticklabels(top4_labels, fontsize=11)
ax.set_xlabel("Element Pair", fontsize=11)
ax.set_ylabel("β₀ (# Connected Components)", fontsize=11)
ax.set_title("D  |  Betti-0 from L₀  (τ=0.0)", fontsize=12, fontweight="bold", loc="left")
ax.grid(axis="y", alpha=0.3, linestyle="--")

# Panel E: How eigenvalue stats change across filtration levels (C-C pair, L0)
ax = fig.add_subplot(gs[1, 1])
stat_names = ["min","max","mean","median","var","std","sum","sum²","rank","β₀"]
cc_pi = 0  # C-C pair index
for si, sname in enumerate(["max", "mean", "rank", "β₀"]):
    si_idx = {"max":1,"mean":2,"rank":8,"β₀":9}[sname]
    vals = []
    for ti, tau in enumerate(THRESHOLDS):
        feat_idx = cc_pi * len(THRESHOLDS) * 2 * N_STATS + ti * 2 * N_STATS + si_idx
        vals.append(X[:, feat_idx].mean())
    ax.plot(THRESHOLDS, vals, marker="o", linewidth=2, label=sname, markersize=5)
ax.set_xlabel("Filtration Threshold τ", fontsize=11)
ax.set_ylabel("Mean value across complexes", fontsize=11)
ax.set_title("E  |  C-C Pair L₀ Stats vs Filtration", fontsize=12, fontweight="bold", loc="left")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, linestyle="--")

# Panel F: Feature correlation with pKd (per pair)
ax = fig.add_subplot(gs[1, 2])
corr_with_pkd = np.abs(np.corrcoef(X.T, y)[:-1, -1])
corr_with_pkd = np.nan_to_num(corr_with_pkd, nan=0.0)
corr_per_pair = np.zeros(N_PAIRS)
for pi in range(N_PAIRS):
    start = pi * len(THRESHOLDS) * 2 * N_STATS
    end   = start + len(THRESHOLDS) * 2 * N_STATS
    corr_per_pair[pi] = corr_with_pkd[start:end].max()
corr_mat = corr_per_pair.reshape(4, 9)
cmap2 = LinearSegmentedColormap.from_list("corr", ["#FFF7BC", "#FD8D3C", "#BD0026"])
im2 = ax.imshow(corr_mat, aspect="auto", cmap=cmap2, vmin=0, vmax=0.4)
for ri in range(4):
    for li in range(9):
        ax.text(li, ri, f"{corr_mat[ri,li]:.2f}", ha="center", va="center",
                fontsize=7.5, color="white" if corr_mat[ri,li]>0.2 else "black")
ax.set_xticks(range(9)); ax.set_xticklabels(LIG_ELEMENTS, fontsize=9)
ax.set_yticks(range(4)); ax.set_yticklabels(RNA_ELEMENTS, fontsize=10)
ax.set_xlabel("Ligand Element", fontsize=11)
ax.set_ylabel("RNA Pocket Element", fontsize=11)
ax.set_title("F  |  Max |Pearson r| with pKd per Pair", fontsize=12, fontweight="bold", loc="left")
plt.colorbar(im2, ax=ax, shrink=0.85, label="|r| with pKd")

fig_path = FIG_DIR / "step03_feature_overview.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
log.info(f"  Feature overview saved → {fig_path}")

# ── Figure 2: Top feature correlations with pKd ───────────────────────────────
top_n = 20
top_idx = np.argsort(corr_with_pkd)[::-1][:top_n]
top_corr = corr_with_pkd[top_idx]
top_lbls = [col_names[i] for i in top_idx]
# Shorten labels for readability
top_lbls_short = [l.replace("τ=","τ").replace("|"," | ") for l in top_lbls]

fig2, ax2 = plt.subplots(figsize=(14, 8))
fig2.patch.set_facecolor("white")
bars = ax2.barh(range(top_n)[::-1], top_corr, color="#2171B5", alpha=0.85, edgecolor="white")
ax2.set_yticks(range(top_n)[::-1])
ax2.set_yticklabels(top_lbls_short, fontsize=8.5)
ax2.set_xlabel("|Pearson r| with pKd", fontsize=12)
ax2.set_title("SGT-RNA  |  Step 3: Top 20 Features Correlated with pKd",
              fontsize=14, fontweight="bold")
ax2.set_xlim(0, top_corr.max() * 1.2)
for i, v in enumerate(top_corr):
    ax2.text(v + 0.002, top_n - 1 - i, f"{v:.3f}", va="center", fontsize=8.5)
ax2.grid(axis="x", alpha=0.3, linestyle="--")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
fig2.tight_layout()
fig2_path = FIG_DIR / "step03_top_feature_correlations.png"
plt.savefig(fig2_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
log.info(f"  Top-feature correlations saved → {fig2_path}")

# ── Figure 3: Filtration progression for best pair ───────────────────────────
best_pair_idx = np.argmax(corr_per_pair)
best_r_el = RNA_ELEMENTS[best_pair_idx // 9]
best_l_el = LIG_ELEMENTS[best_pair_idx % 9]
log.info(f"  Best pair by pKd correlation: RNA-{best_r_el} × Lig-{best_l_el}")

fig3, axes3 = plt.subplots(2, 5, figsize=(22, 9))
fig3.patch.set_facecolor("white")
fig3.suptitle(
    f"SGT-RNA  |  Eigenvalue Distributions — Best Pair: RNA-{best_r_el}×Lig-{best_l_el}\n"
    f"(Top row: L₀ vertex Laplacian  ·  Bottom row: L₁ edge Laplacian)",
    fontsize=14, fontweight="bold"
)

for ti, tau in enumerate(THRESHOLDS):
    for li_row, (lap_name, offset_extra) in enumerate([("L₀", 0), ("L₁", N_STATS)]):
        ax = axes3[li_row, ti]
        base = best_pair_idx * len(THRESHOLDS) * 2 * N_STATS + ti * 2 * N_STATS + offset_extra
        feat_slice = X[:, base: base + N_STATS]

        # Plot distribution of mean eigenvalue across complexes
        mean_eig_vals = feat_slice[:, 2]  # "mean" statistic
        rank_vals     = feat_slice[:, 8]  # "rank" = n_nonzero
        betti_vals    = feat_slice[:, 9]  # "betti"

        color = "#4C72B0" if li_row == 0 else "#C44E52"
        ax.hist(mean_eig_vals, bins=20, color=color, alpha=0.78, edgecolor="white")
        ax.set_title(f"τ = {tau}\n{lap_name}  β = {betti_vals.mean():.1f}  rank = {rank_vals.mean():.0f}",
                     fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Mean eigenvalue", fontsize=9)
        if ti == 0:
            ax.set_ylabel(f"{lap_name}\n# Complexes", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3, linestyle="--")

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig3_path = FIG_DIR / "step03_filtration_progression.png"
plt.savefig(fig3_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
log.info(f"  Filtration progression saved → {fig3_path}")

# ── final summary ─────────────────────────────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("STEP 3 COMPLETE")
log.info(f"  Complexes      : {n}")
log.info(f"  Feature dim    : {N_FEATS}")
log.info(f"  Zero feats     : {n_zero_feats} ({100*n_zero_feats/N_FEATS:.1f}%)")
log.info(f"  Best pKd corr  : RNA-{best_r_el}×Lig-{best_l_el}  "
         f"|r|={corr_per_pair[best_pair_idx]:.3f}")
log.info(f"  NPZ file       : {out_npz}")
log.info(f"  Log file       : {log_path}")
log.info("=" * 70)
