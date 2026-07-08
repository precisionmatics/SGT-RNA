"""
SGT-RNA · Step 34: Proper Double-Nested LOO Validation

Fixes the audit flag from step33: hyperparameter config was selected from 250
candidates using the same LOO-r that was reported (meta-level selection bias).

Fix: double-nested LOO
  - Outer loop: leave one sample out (produces final prediction)
  - Inner loop: for each outer training set, grid-search config by inner LOO
    (feature selection and alpha chosen on training data only)
  - Final prediction uses the config chosen by inner loop

Feature sets: CPF (72d), SCF (72d), RLIF (21d), CPF+SCF (144d), CPF+RLIF (93d)
k range: 1–10,  alpha: [1e-6, 1e-4, 0.01, 1, 100]

Applied to TPP (n=10) and FMN_FAD (n=8).
Then substitute into step30 hybrid and compare to step33.
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

warnings.filterwarnings("ignore")

ROOT    = Path("/home/stalin/Desktop/SGT-RNA/RNA_SGT")
RES_DIR = ROOT / "results"
S11_NPZ = ROOT / "data" / "features" / "step11_full_features.npz"

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "results" / "logs" / f"step34_{ts}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("SGT-RNA · Step 34: Double-Nested LOO Validation")
log.info("=" * 70)

# ── Load data ──────────────────────────────────────────────────────────────
d11   = np.load(S11_NPZ)
y     = d11["y"].astype(np.float64)
ids   = d11["ids"]
n     = len(y)

X_cpf  = np.load(ROOT / "data" / "features" / "cpf_features.npy").astype(np.float64)
X_scf  = np.load(ROOT / "data" / "features" / "scf_features.npy").astype(np.float64)
X_rlif = np.load(ROOT / "data" / "features" / "rlif_features.npy").astype(np.float64)

FEAT_SETS = {
    "CPF":      X_cpf,
    "SCF":      X_scf,
    "RLIF":     X_rlif,
    "CPF+SCF":  np.hstack([X_cpf, X_scf]),
    "CPF+RLIF": np.hstack([X_cpf, X_rlif]),
}

s30_df  = pd.read_csv(RES_DIR / "step30_results.csv")
pdb2s30 = dict(zip(s30_df["pdb"], s30_df["y_pred"]))
pdb2rsc = dict(zip(s30_df["pdb"], s30_df["rs_subclass"]))
y30     = np.array([pdb2s30[p] for p in ids], dtype=np.float64)
rs_sub  = np.array([pdb2rsc[p] for p in ids])

tpp_mask = rs_sub == "TPP"
fmn_mask = rs_sub == "FMN_FAD"

ALPHA_GRID = [1e-6, 1e-4, 0.01, 1.0, 100.0]
K_MAX      = 10


# ── Helpers ────────────────────────────────────────────────────────────────
def fit_predict_one(X_tr, y_tr, X_te, k, alpha):
    """Top-k feature selection + Ridge on one train/test split."""
    corrs = np.array([abs(pearsonr(X_tr[:, j], y_tr)[0])
                      if X_tr[:, j].std() > 1e-9 else 0.0
                      for j in range(X_tr.shape[1])])
    top_k  = np.argsort(corrs)[-(min(k, X_tr.shape[1])):]
    Xtr_k  = X_tr[:, top_k]
    Xte_k  = X_te[:, top_k]
    sc     = StandardScaler()
    Xtr_k  = sc.fit_transform(Xtr_k)
    Xte_k  = sc.transform(Xte_k)
    return Ridge(alpha=alpha).fit(Xtr_k, y_tr).predict(Xte_k)[0]


def inner_loo_r(X_sub, y_sub, k, alpha):
    """LOO-r on a subgroup (used for inner loop config selection)."""
    m = len(y_sub)
    preds = np.zeros(m)
    for i in range(m):
        tr = [j for j in range(m) if j != i]
        preds[i] = fit_predict_one(X_sub[tr], y_sub[tr], X_sub[[i]], k, alpha)
    if y_sub.std() < 1e-9 or preds.std() < 1e-9:
        return 0.0
    return pearsonr(y_sub, preds)[0]


def double_nested_loo(mask, label):
    """
    Outer LOO: for each left-out sample i,
      - inner grid search on remaining n-1 samples → best (fname, k, alpha)
      - predict sample i using that config
    Returns final LOO predictions and config usage counts.
    """
    idx   = np.where(mask)[0]
    y_sub = y[idx]
    m     = len(y_sub)
    preds = np.zeros(m)
    cfg_counts = {}

    log.info(f"\n  {label} (n={m}): running {m} outer folds …")

    for outer_i in range(m):
        inner_idx = [j for j in range(m) if j != outer_i]
        y_inner   = y_sub[inner_idx]

        # Grid search on inner (n-1) samples
        best_r_inner, best_cfg = -np.inf, None
        for fname, X_all in FEAT_SETS.items():
            X_inner = X_all[idx][inner_idx]
            k_max   = min(K_MAX, X_inner.shape[1])
            for k, alpha in product(range(1, k_max + 1), ALPHA_GRID):
                r_in = inner_loo_r(X_inner, y_inner, k, alpha)
                if r_in > best_r_inner:
                    best_r_inner, best_cfg = r_in, (fname, k, alpha)

        # Predict outer sample using best config
        fname, k, alpha = best_cfg
        X_all   = FEAT_SETS[fname]
        X_inner = X_all[idx][inner_idx]
        X_outer = X_all[idx][[outer_i]]
        preds[outer_i] = fit_predict_one(X_inner, y_inner, X_outer, k, alpha)

        cfg_counts[best_cfg] = cfg_counts.get(best_cfg, 0) + 1
        log.info(f"    fold {outer_i+1:2d}/{m}: best_inner_cfg={best_cfg}  "
                 f"inner_r={best_r_inner:.4f}  pred={preds[outer_i]:.3f}  true={y_sub[outer_i]:.3f}")

    r_outer = pearsonr(y_sub, preds)[0] if y_sub.std() > 1e-9 and preds.std() > 1e-9 else 0.0
    log.info(f"\n  {label} double-nested LOO r = {r_outer:.4f}")
    log.info(f"  Config usage: {sorted(cfg_counts.items(), key=lambda x:-x[1])[:5]}")
    return r_outer, preds, idx


# ═══════════════════════════════════════════════════════════════════════════
# PART A: Double-nested LOO for TPP
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "=" * 70)
log.info("PART A: TPP — double-nested LOO")
log.info("=" * 70)

tpp_r_nested, tpp_preds_nested, tpp_idx = double_nested_loo(tpp_mask, "TPP")

# ═══════════════════════════════════════════════════════════════════════════
# PART B: Double-nested LOO for FMN_FAD
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "=" * 70)
log.info("PART B: FMN_FAD — double-nested LOO")
log.info("=" * 70)

fmn_r_nested, fmn_preds_nested, fmn_idx = double_nested_loo(fmn_mask, "FMN_FAD")

# ═══════════════════════════════════════════════════════════════════════════
# PART C: Compare biased (step33) vs unbiased (step34)
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "=" * 70)
log.info("PART C: Biased (step33) vs Unbiased (step34) comparison")
log.info("=" * 70)

s33_df  = pd.read_csv(RES_DIR / "step33_results.csv")
pdb2s33 = dict(zip(s33_df["pdb"], s33_df["y_pred"]))
y33     = np.array([pdb2s33[p] for p in ids], dtype=np.float64)

# step33 per-subgroup r (biased)
tpp_r_s33 = pearsonr(y[tpp_idx], y33[tpp_idx])[0]
fmn_r_s33 = pearsonr(y[fmn_idx], y33[fmn_idx])[0]

log.info(f"\n  TPP:     biased(step33)={tpp_r_s33:.4f}  unbiased(step34)={tpp_r_nested:.4f}  "
         f"bias={tpp_r_s33 - tpp_r_nested:+.4f}")
log.info(f"  FMN_FAD: biased(step33)={fmn_r_s33:.4f}  unbiased(step34)={fmn_r_nested:.4f}  "
         f"bias={fmn_r_s33 - fmn_r_nested:+.4f}")

# Build step34 global predictions
y34 = y30.copy()
y34[tpp_idx] = tpp_preds_nested
y34[fmn_idx] = fmn_preds_nested

r34  = pearsonr(y, y34)[0]
rho34 = spearmanr(y, y34)[0]
r33  = pearsonr(y, y33)[0]
r30  = pearsonr(y, y30)[0]

log.info(f"\n  Global r comparison:")
log.info(f"    step30 (no subclass override): r={r30:.4f}")
log.info(f"    step33 (biased config select): r={r33:.4f}")
log.info(f"    step34 (proper nested LOO):    r={r34:.4f}  spearman={rho34:.4f}")
log.info(f"    Bias in step33 global r:       {r33 - r34:+.4f}")

gap34 = 0.830 - r34
if gap34 < 0:
    log.info(f"  Gap to RSAPred (0.830): {gap34:+.4f}  *** BEATS RSAPred ***")
else:
    log.info(f"  Gap to RSAPred (0.830): {gap34:+.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# PART D: Save step34 results
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "=" * 70)
log.info("PART D: Save results")
log.info("=" * 70)

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

df_out = pd.DataFrame({
    "pdb":        ids,
    "subtype":    subtypes,
    "y_true":     y,
    "y_pred":     y34,
    "rs_subclass": rs_sub,
})
out_path = RES_DIR / "step34_results.csv"
df_out.to_csv(out_path, index=False)
log.info(f"  Results → {out_path}")

log.info("\n" + "=" * 70)
log.info("STEP 34 COMPLETE")
log.info(f"  step33 (biased):   r={r33:.4f}")
log.info(f"  step34 (unbiased): r={r34:.4f}  spearman={rho34:.4f}")
log.info(f"  Bias magnitude:    {r33 - r34:+.4f}")
gap = 0.830 - r34
if gap < 0:
    log.info(f"  RSAPred gap:       {gap:+.4f}  *** BEATS RSAPred (0.830) ***")
else:
    log.info(f"  RSAPred gap:       {gap:+.4f}")
log.info("=" * 70)
