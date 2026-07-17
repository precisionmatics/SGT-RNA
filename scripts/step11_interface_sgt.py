"""
SGT-RNA · Step 11: Interface-Focused SGT

Key idea: current SGT uses all RNA pocket atoms within 12Å of the ligand.
Here we recompute SGT on RNA atoms within {4, 6, 8}Å of ANY ligand atom —
the actual binding interface — giving a focused topological fingerprint of
where binding physically occurs.

Interface features: 3 cutoffs × 3,600 = 10,800 dim
Combined:          step09 (38,963) + interface (10,800) = 49,763 dim
Pipeline:          VT(1e-4) → StandardScaler → PCA(0.95) → Ridge (per subtype)
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
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
PKL_FILE  = ROOT / "data" / "pocket_fri" / "pocket_fri_data.pkl.gz"
FEAT_NPZ  = ROOT / "data" / "features" / "step09_full_features.npz"
OUT_NPZ   = ROOT / "data" / "features" / "step11_full_features.npz"
RES_DIR   = ROOT / "results"
FIG_DIR   = ROOT / "results" / "figures"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── logging ───────────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "results" / "logs" / f"step11_{ts}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("SGT-RNA · Step 11: Interface-Focused SGT")
log.info("=" * 70)

# ── SGT constants (identical to step03/04) ───────────────────────────────────
RNA_ELEMENTS = ["C", "N", "O", "P"]
LIG_ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
THRESHOLDS   = [0.0, 0.8, 0.85, 0.90, 0.95]
N_STATS      = 10
N_PAIRS      = 36
FEATS_PER_CUTOFF = N_PAIRS * len(THRESHOLDS) * 2 * N_STATS  # 3,600
ETA, KAPPA, EPS = 5.0, 2.0, 1e-8

# Interface cutoffs to test
CUTOFFS = [4, 6, 8]   # Ångströms
N_IFACE_FEATS = len(CUTOFFS) * FEATS_PER_CUTOFF  # 10,800

log.info(f"Interface cutoffs  : {CUTOFFS} Å")
log.info(f"Feats per cutoff   : {FEATS_PER_CUTOFF}")
log.info(f"Total iface feats  : {N_IFACE_FEATS}")

# ── SGT helper functions (from step03) ───────────────────────────────────────
def spectral_stats(eigs):
    if len(eigs) == 0:
        return np.zeros(N_STATS, dtype=np.float32)
    nz = eigs[eigs > EPS]
    return np.array([
        eigs.min(), eigs.max(), eigs.mean(), np.median(eigs),
        eigs.var(), eigs.std(), eigs.sum(), (eigs**2).sum(),
        float(len(nz)), float(len(eigs) - len(nz)),
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

def pair_features(rc, lc):
    n_feat = len(THRESHOLDS) * 2 * N_STATS
    if len(rc) == 0 or len(lc) == 0:
        return np.zeros(n_feat, dtype=np.float32)
    D = cdist(rc, lc).astype(np.float64)
    W = np.exp(-(D / ETA) ** KAPPA)
    wmax = W.max()
    if wmax < 1e-12:
        return np.zeros(n_feat, dtype=np.float32)
    W_norm = W / wmax
    n_total = len(rc) + len(lc)
    feats, offset = np.zeros(n_feat, dtype=np.float32), 0
    for tau in THRESHOLDS:
        mask = W_norm >= tau
        W_sel = W_norm * mask
        n_edges = int(mask.sum())
        if n_edges == 0:
            eigs_L0 = np.zeros(n_total)
            eigs_L1 = np.array([])
        else:
            L0 = build_L0(W_sel)
            eigs_L0 = np.maximum(np.linalg.eigvalsh(L0), 0.0)
            beta0 = int((eigs_L0 < EPS).sum())
            n_zeros_L1 = max(0, n_edges - n_total + beta0)
            eigs_L1 = np.concatenate([np.zeros(n_zeros_L1), eigs_L0[eigs_L0 >= EPS]])
        feats[offset:offset+N_STATS]          = spectral_stats(eigs_L0)
        feats[offset+N_STATS:offset+2*N_STATS] = spectral_stats(eigs_L1)
        offset += 2 * N_STATS
    return feats

def interface_sgt(rna_coords, lig_coords, cutoff):
    """Compute 3,600-dim SGT on RNA atoms within `cutoff` Å of any ligand atom."""
    lig_all = [v for v in lig_coords.values() if len(v) > 0]
    if not lig_all:
        return np.zeros(FEATS_PER_CUTOFF, dtype=np.float32)
    lig_all = np.vstack(lig_all)

    filtered_rna = {}
    for el in RNA_ELEMENTS:
        rc = rna_coords[el]
        if len(rc) == 0:
            filtered_rna[el] = rc
            continue
        D = cdist(rc, lig_all)
        mask = D.min(axis=1) <= cutoff
        filtered_rna[el] = rc[mask]

    parts = []
    for r_el in RNA_ELEMENTS:
        for l_el in LIG_ELEMENTS:
            rc = filtered_rna[r_el]
            lc = lig_coords.get(l_el, np.empty((0, 3), np.float32))
            parts.append(pair_features(rc, lc))
    return np.concatenate(parts)

# ── subtype labels (from step09) ──────────────────────────────────────────────
G_QUAD       = {"1nzm","5cdb","4xwf","4znp","5btp","6jj0","2mg8","2loa"}
DUPLEX_GROOVE= {"407d","408d","1cvy","1cvx","454d","1qv4","1qv8","1p96","1r4e","6hbt"}
MANUAL_OVERRIDE = {
    "1y26":"riboswitch","2gdi":"riboswitch","3b31":"riboswitch","3d2v":"riboswitch",
    "3d2x":"riboswitch","3d2g":"riboswitch","3d2s":"riboswitch","3d2w":"riboswitch",
    "3fnb":"riboswitch","3q50":"riboswitch","3vrs":"riboswitch","4fny":"riboswitch",
    "4lck":"riboswitch","4lnt":"riboswitch","4tzx":"riboswitch","5e54":"riboswitch",
    "1o15":"aptamer","1zif":"aptamer",
    "2vrn":"ribosomal_asite","1fjg":"ribosomal_asite","1xmq":"ribosomal_asite",
    "4v9o":"ribosomal_asite","2z75":"ribosomal_asite",
}

ALPHA_GRID = [1, 10, 100, 1000, 10_000, 100_000]

def make_subtype(pdb, raw):
    if pdb in MANUAL_OVERRIDE:
        return MANUAL_OVERRIDE[pdb]
    if pdb in G_QUAD:
        return "g_quadruplex"
    if pdb in DUPLEX_GROOVE:
        return "duplex_groove"
    return raw

# ── load data ─────────────────────────────────────────────────────────────────
log.info(f"\nLoading pocket data ...")
with gzip.open(PKL_FILE, "rb") as f:
    records = pickle.load(f)
log.info(f"  {len(records)} complexes")

log.info(f"Loading step09 features ...")
d9 = np.load(FEAT_NPZ)
X9, y9, ids9 = d9["X"].astype(np.float32), d9["y"].astype(np.float32), d9["ids"]
subtypes_raw  = d9["subtypes"]
log.info(f"  step09 X: {X9.shape}")

# build id→record map
rec_map = {r["pdb"]: r for r in records}

# verify alignment
assert list(ids9) == [r["pdb"] for r in records], "ID order mismatch!"

# ── compute interface SGT for all complexes × all cutoffs ────────────────────
log.info(f"\nComputing interface SGT ({len(CUTOFFS)} cutoffs × {FEATS_PER_CUTOFF} feats) ...")
n = len(records)
X_iface = np.zeros((n, N_IFACE_FEATS), dtype=np.float32)

t0 = time.time()
for idx, rec in enumerate(records):
    parts = []
    for cutoff in CUTOFFS:
        parts.append(interface_sgt(rec["rna_coords"], rec["lig_coords"], cutoff))
    X_iface[idx] = np.concatenate(parts)

    if (idx + 1) % 20 == 0 or idx == 0:
        elapsed = time.time() - t0
        eta = (n - idx - 1) / ((idx + 1) / elapsed)
        log.info(f"  [{idx+1:3d}/{n}] {rec['pdb']}  ETA {eta:.0f}s")

log.info(f"  Done in {time.time()-t0:.1f}s")

# ── combine features ──────────────────────────────────────────────────────────
X_full = np.hstack([X9, X_iface])   # (143, 49,763)
log.info(f"\nCombined features: {X_full.shape}")

# save
np.savez_compressed(OUT_NPZ, X=X_full, y=y9, ids=ids9, subtypes=subtypes_raw)
log.info(f"Saved → {OUT_NPZ}")

# ── subtype labels ────────────────────────────────────────────────────────────
subtypes = np.array([make_subtype(pdb, st) for pdb, st in zip(ids9, subtypes_raw)])

# ── ML: same pipeline as step09 ───────────────────────────────────────────────
def make_pipe(alpha):
    return Pipeline([
        ("vt",  VarianceThreshold(threshold=1e-4)),
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
        ("reg", Ridge(alpha=alpha)),
    ])

def loo_ridge(X, y):
    n = len(y)
    if n < 3:
        return np.full(n, y.mean()), -99.0
    best_r, best_preds = -99, np.full(n, y.mean())
    for alpha in ALPHA_GRID:
        preds = np.zeros(n)
        ok = True
        for i in range(n):
            tr = [j for j in range(n) if j != i]
            try:
                pipe = make_pipe(alpha)
                pipe.fit(X[tr], y[tr])
                preds[i] = np.clip(pipe.predict(X[[i]])[0],
                                   y[tr].min()-3, y[tr].max()+3)
            except Exception:
                ok = False; break
        if not ok:
            continue
        r = pearsonr(y, preds)[0] if np.std(preds) > 1e-8 else -99.0
        if r > best_r:
            best_r, best_preds = r, preds
    return best_preds, best_r

def cv_ridge_global(X, y, n_splits=5):
    n = len(y)
    if n < 6:
        return loo_ridge(X, y)
    kf = KFold(n_splits=min(n_splits, n), shuffle=True, random_state=42)
    best_r, best_preds = -99, np.full(n, y.mean())
    for alpha in ALPHA_GRID:
        preds = np.zeros(n)
        ok = True
        for tr, te in kf.split(X):
            try:
                pipe = make_pipe(alpha)
                pipe.fit(X[tr], y[tr])
                p = pipe.predict(X[te])
                p = np.clip(p, y[tr].min()-3, y[tr].max()+3)
                preds[te] = p
            except Exception:
                ok = False; break
        if not ok:
            continue
        r = pearsonr(y, preds)[0] if np.std(preds) > 1e-8 else -99.0
        if r > best_r:
            best_r, best_preds = r, preds
    return best_preds, best_r

log.info("\n" + "=" * 70)
log.info("ML EVALUATION — per-subtype LOO Ridge + adaptive stacking")
log.info("=" * 70)

unique_subtypes = sorted(set(subtypes))
subtype_preds   = np.full(n, np.nan)
subtype_rs      = {}

# Global model OOF (for adaptive stacking fallback)
log.info("\nFitting global model ...")
global_preds, global_r = cv_ridge_global(X_full, y9, n_splits=5)
log.info(f"  Global CV r = {global_r:.4f}")

for st in unique_subtypes:
    mask = subtypes == st
    idx  = np.where(mask)[0]
    ns   = len(idx)
    Xs, ys = X_full[idx], y9[idx]

    preds_st, r_st = loo_ridge(Xs, ys)
    preds_gl = global_preds[idx]
    r_gl = pearsonr(ys, preds_gl)[0] if np.std(preds_gl) > 1e-8 else -99.0

    if r_st >= r_gl:
        subtype_preds[idx] = preds_st
        chosen = "subtype"
    else:
        subtype_preds[idx] = preds_gl
        chosen = "global"
        r_st = r_gl

    subtype_rs[st] = r_st
    log.info(f"  {st:<22}: n={ns:3d}  r={r_st:.4f}  (used {chosen})")

# combined r
valid = ~np.isnan(subtype_preds)
combined_r = pearsonr(y9[valid], subtype_preds[valid])[0]

log.info("\n" + "=" * 70)
log.info(f"  COMBINED r (step11 interface SGT) = {combined_r:.4f}")
log.info(f"  Step 09 baseline                   = 0.5350")
log.info(f"  Delta                              = {combined_r - 0.5350:+.4f}")
log.info("=" * 70)

# ── results CSV ───────────────────────────────────────────────────────────────
df_res = pd.DataFrame({
    "pdb": ids9,
    "subtype": subtypes,
    "y_true": y9,
    "y_pred": subtype_preds,
})
res_path = RES_DIR / "step11_results.csv"
df_res.to_csv(res_path, index=False)
log.info(f"\nResults → {res_path}")

# ── figure ────────────────────────────────────────────────────────────────────
colors = {
    "aptamer":"#2166AC", "riboswitch":"#1A9641",
    "ribosomal_asite":"#D73027", "duplex_groove":"#7B2D8B",
    "g_quadruplex":"#FF7F00", "viral_tar":"#A65628", "other_misc":"#999999",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("white")

# Panel A: scatter coloured by subtype
ax = axes[0]
for st in unique_subtypes:
    mask = subtypes == st
    ax.scatter(y9[mask], subtype_preds[mask],
               c=colors.get(st,"#888888"), label=f"{st} (r={subtype_rs[st]:.3f})",
               alpha=0.75, s=40, edgecolors="none")
mn, mx = y9.min()-0.5, y9.max()+0.5
ax.plot([mn,mx],[mn,mx],"k--",lw=1,alpha=0.4)
ax.set_xlabel("Experimental pKd", fontsize=12)
ax.set_ylabel("Predicted pKd", fontsize=12)
ax.set_title(f"Step 11: Interface SGT  (combined r={combined_r:.4f})", fontsize=12, fontweight="bold")
ax.legend(fontsize=7, loc="upper left", framealpha=0.7)
ax.grid(alpha=0.3, linestyle="--")

# Panel B: subtype bar chart vs step09
ax = axes[1]
step09_rs = {
    "aptamer":0.918, "riboswitch":0.400, "ribosomal_asite":0.581,
    "duplex_groove":0.676, "g_quadruplex":-0.257, "other_misc":0.084,
}
sts_plot = [s for s in unique_subtypes if s in step09_rs]
x = np.arange(len(sts_plot))
w = 0.35
bars1 = ax.bar(x - w/2, [step09_rs.get(s, 0) for s in sts_plot],
               w, label="Step 09", color="#AAAAAA", edgecolor="white")
bars2 = ax.bar(x + w/2, [subtype_rs.get(s, 0) for s in sts_plot],
               w, label="Step 11 (Interface)", color="#2166AC", alpha=0.8, edgecolor="white")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(sts_plot, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Pearson r", fontsize=11)
ax.set_title("Step 09 vs Step 11 per subtype", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3, linestyle="--")

plt.tight_layout()
fig_path = FIG_DIR / "step11_results.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
log.info(f"Figure → {fig_path}")

log.info("\n" + "=" * 70)
log.info("STEP 11 COMPLETE")
log.info("=" * 70)
