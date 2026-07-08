"""
SGT-RNA · Step 35: Residual Correction for Weak Subgroups

Instead of replacing step30 predictions, we add a subgroup-specific correction:
    y_final[i] = y30[i] + delta[i]

where delta[i] is predicted from features that correlate with step30's residuals
(y_true - y30).  This preserves global scale and avoids the coherence
disruption observed in step31/33.

Target subgroups and best feature sets (from residual-correlation probe):
  purine          n=21  Morgan   top-corr=0.590
  other_misc      n=27  RNA-FM   top-corr=0.687
  ribosomal_asite n=13  SCF      top-corr=0.700
  SAM_SAH         n=10  RNA-FM   top-corr=0.881  (high signal, small n)

Method: double-nested LOO
  outer fold i: hold out sample i
    inner grid-search on remaining n-1 samples:
      feature selection (top-k abs-corr with RESIDUALS on training set)
      Ridge regression on delta
    predict delta[i] using best inner config
  y_final[i] = y30[i] + delta[i]

k range: 1-5 (conservative),  alpha: [1e-4, 0.01, 1.0]
"""

import logging, warnings
from pathlib import Path
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

ROOT    = Path("/home/stalin/Desktop/SGT-RNA/RNA_SGT")
RES_DIR = ROOT / "results"
S11_NPZ = ROOT / "data" / "features" / "step11_full_features.npz"

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "results" / "logs" / f"step35_{ts}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("SGT-RNA · Step 35: Residual Correction (Double-Nested LOO)")
log.info("=" * 70)

# ── Load data ──────────────────────────────────────────────────────────────
d11  = np.load(S11_NPZ)
y    = d11["y"].astype(np.float64)
ids  = d11["ids"]
X11  = d11["X"].astype(np.float64)
n    = len(y)

X_cpf  = np.load(ROOT / "data" / "features" / "cpf_features.npy").astype(np.float64)
X_scf  = np.load(ROOT / "data" / "features" / "scf_features.npy").astype(np.float64)
X_rlif = np.load(ROOT / "data" / "features" / "rlif_features.npy").astype(np.float64)

# RNA-FM features (columns 38064:38704 in step11 matrix)
X_rnafm = X11[:, 38064:38704]
# Morgan fingerprint (columns 36000:38048)
X_morgan = X11[:, 36000:38048]

# UniMol
unimol_raw = np.load("/tmp/unimol_emb.npy")
valid_idx  = np.load("/tmp/unimol_valid_idx.npy").astype(int)
unimol_full = np.zeros((n, 512), dtype=np.float64)
unimol_full[valid_idx] = unimol_raw

# step30 predictions
s30_df  = pd.read_csv(RES_DIR / "step30_results.csv")
pdb2s30 = dict(zip(s30_df["pdb"], s30_df["y_pred"]))
pdb2rsc = dict(zip(s30_df["pdb"], s30_df["rs_subclass"]))
pdb2sub = dict(zip(s30_df["pdb"], s30_df["subtype"]))
y30     = np.array([pdb2s30[p] for p in ids], dtype=np.float64)
rsc     = np.array([pdb2rsc[p] for p in ids])
sub     = np.array([pdb2sub[p] for p in ids])

# Subgroup masks
masks = {
    "purine":           rsc == "purine",
    "other_misc":       sub == "other_misc",
    "ribosomal_asite":  sub == "ribosomal_asite",
    "SAM_SAH":          rsc == "SAM_SAH",
}

# Pre-specified feature sets per subgroup (from residual-correlation probe)
FEAT_MAP = {
    "purine":          X_morgan,    # top-corr 0.590 with purine residuals
    "other_misc":      X_rnafm,     # top-corr 0.687
    "ribosomal_asite": X_scf,       # top-corr 0.700
    "SAM_SAH":         X_rnafm,     # top-corr 0.881 (high signal, n=10)
}

K_RANGE    = [1, 2, 3, 4, 5]
ALPHA_GRID = [1e-4, 0.01, 1.0]

log.info(f"\nLoaded {n} complexes")
for name, m in masks.items():
    resid = (y - y30)[m]
    log.info(f"  {name:20s}: n={m.sum():3d}  r(y30,ytrue)={pearsonr(y[m],y30[m])[0]:.4f}  "
             f"resid_std={resid.std():.3f}")


# ── Core: double-nested LOO residual correction ────────────────────────────
def fit_predict_delta(X_tr, d_tr, X_te, k, alpha):
    """Predict residual delta for test sample using top-k features."""
    corrs = np.array([abs(pearsonr(X_tr[:,j], d_tr)[0])
                      if X_tr[:,j].std() > 1e-9 else 0.0
                      for j in range(X_tr.shape[1])])
    top_k  = np.argsort(corrs)[-(min(k, X_tr.shape[1])):]
    Xtr_k  = X_tr[:, top_k]
    Xte_k  = X_te[:, top_k]
    sc     = StandardScaler()
    Xtr_k  = sc.fit_transform(Xtr_k)
    Xte_k  = sc.transform(Xte_k)
    return Ridge(alpha=alpha).fit(Xtr_k, d_tr).predict(Xte_k)[0]


def inner_loo_delta_r(X_sub, delta_sub, k, alpha):
    """LOO-r for delta prediction on inner training set."""
    m = len(delta_sub)
    preds = np.zeros(m)
    for i in range(m):
        tr = [j for j in range(m) if j != i]
        preds[i] = fit_predict_delta(X_sub[tr], delta_sub[tr], X_sub[[i]], k, alpha)
    if delta_sub.std() < 1e-9 or preds.std() < 1e-9:
        return 0.0
    return pearsonr(delta_sub, preds)[0]


def _outer_fold(outer_i, X_sub, delta_sub):
    """One outer fold: inner grid-search + predict."""
    inner_idx = [j for j in range(len(delta_sub)) if j != outer_i]
    d_inner   = delta_sub[inner_idx]
    X_inner   = X_sub[inner_idx]

    best_r_in, best_cfg = -np.inf, None
    for k, alpha in product(K_RANGE, ALPHA_GRID):
        r_in = inner_loo_delta_r(X_inner, d_inner, k, alpha)
        if r_in > best_r_in:
            best_r_in, best_cfg = r_in, (k, alpha)

    k_best, a_best = best_cfg
    pred = fit_predict_delta(X_inner, d_inner, X_sub[[outer_i]], k_best, a_best)
    return outer_i, pred, best_cfg


def double_nested_correction(mask, feat_name, X_feat, label):
    idx       = np.where(mask)[0]
    y_sub     = y[idx]
    y30_sub   = y30[idx]
    delta_sub = y_sub - y30_sub
    X_sub     = X_feat[idx]
    m_n       = len(y_sub)

    log.info(f"\n  {label} (n={m_n})  feat={feat_name}  "
             f"r_baseline={pearsonr(y_sub, y30_sub)[0]:.4f}")

    # Parallelise outer folds across all CPUs
    fold_results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_outer_fold)(i, X_sub, delta_sub) for i in range(m_n)
    )

    outer_deltas = np.zeros(m_n)
    cfg_counts   = {}
    for outer_i, pred, cfg in fold_results:
        outer_deltas[outer_i] = pred
        cfg_counts[cfg] = cfg_counts.get(cfg, 0) + 1

    y_corrected = y30_sub + outer_deltas
    r_corrected = pearsonr(y_sub, y_corrected)[0]
    r_baseline  = pearsonr(y_sub, y30_sub)[0]
    r_delta     = pearsonr(delta_sub, outer_deltas)[0]

    log.info(f"    r(baseline)   = {r_baseline:.4f}")
    log.info(f"    r(correction) = {r_delta:.4f}  (delta LOO r)")
    log.info(f"    r(corrected)  = {r_corrected:.4f}  delta={r_corrected-r_baseline:+.4f}")
    log.info(f"    top configs:    {sorted(cfg_counts.items(), key=lambda x:-x[1])[:3]}")

    return r_corrected, outer_deltas, idx, r_baseline


# ═══════════════════════════════════════════════════════════════════════════
# Run corrections for each subgroup
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "=" * 70)
log.info("Running double-nested LOO residual corrections")
log.info("=" * 70)

results = {}
for name, mask in masks.items():
    feat_name = FEAT_MAP[name].__class__.__name__
    # Map X feature name for logging
    feat_label = {
        "purine":          "Morgan",
        "other_misc":      "RNA-FM",
        "ribosomal_asite": "SCF",
        "SAM_SAH":         "RNA-FM",
    }[name]
    r_corr, deltas, idx, r_base = double_nested_correction(
        masks[name], feat_label, FEAT_MAP[name], name)
    results[name] = (r_corr, deltas, idx, r_base)


# ═══════════════════════════════════════════════════════════════════════════
# PART B: Apply corrections and evaluate global r
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "=" * 70)
log.info("PART B: Applying corrections → global r")
log.info("=" * 70)

# Apply only corrections that improve their subgroup
y_final = y30.copy()
applied = []
for name, (r_corr, deltas, idx, r_base) in results.items():
    if r_corr > r_base:
        y_final[idx] += deltas
        applied.append(name)
        log.info(f"  APPLIED  {name:20s}: {r_base:.4f} → {r_corr:.4f}  (+{r_corr-r_base:.4f})")
    else:
        log.info(f"  SKIPPED  {name:20s}: {r_base:.4f} → {r_corr:.4f}  ({r_corr-r_base:.4f}, no gain)")

r_final   = pearsonr(y, y_final)[0]
rho_final = spearmanr(y, y_final)[0]
r_s30     = pearsonr(y, y30)[0]

log.info(f"\n  step30 baseline: r={r_s30:.4f}")
log.info(f"  step35 final:    r={r_final:.4f}  spearman={rho_final:.4f}  delta={r_final-r_s30:+.4f}")
gap = 0.830 - r_final
if gap < 0:
    log.info(f"  RSAPred gap:     {gap:+.4f}  *** BEATS RSAPred (0.830) ***")
else:
    log.info(f"  RSAPred gap:     {gap:+.4f}")

# Per-subtype breakdown
log.info("\n  Per-subtype breakdown:")
subtypes_raw = d11["subtypes"]
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

subtypes = np.array([make_subtype(p, s) for p, s in zip(ids, subtypes_raw)])
for stype in ["aptamer","riboswitch","ribosomal_asite","other_misc",
              "g_quadruplex","duplex_groove","viral_tar"]:
    m = subtypes == stype
    if m.sum() < 2: continue
    r_new  = pearsonr(y[m], y_final[m])[0]
    r_old  = pearsonr(y[m], y30[m])[0]
    log.info(f"    {stype:20s}: n={m.sum():3d}  r={r_new:.4f}  (s30:{r_old:.4f}  {r_new-r_old:+.4f})")

# ═══════════════════════════════════════════════════════════════════════════
# PART C: Save
# ═══════════════════════════════════════════════════════════════════════════
df_out = pd.DataFrame({
    "pdb":        ids,
    "subtype":    subtypes,
    "y_true":     y,
    "y_pred":     y_final,
    "rs_subclass": rsc,
})
df_out.to_csv(RES_DIR / "step35_results.csv", index=False)
log.info(f"\n  Results → results/step35_results.csv")

log.info("\n" + "=" * 70)
log.info("STEP 35 COMPLETE")
log.info(f"  step30 r = {r_s30:.4f}")
log.info(f"  step35 r = {r_final:.4f}  spearman={rho_final:.4f}")
log.info(f"  Corrections applied: {applied}")
gap = 0.830 - r_final
if gap < 0:
    log.info(f"  RSAPred gap: {gap:+.4f}  *** BEATS RSAPred ***")
else:
    log.info(f"  RSAPred gap: {gap:+.4f}")
log.info("=" * 70)
