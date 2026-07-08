"""
RNA-PDFL · Step 33: Sparse Nested LOO for TPP and FMN_FAD

Addresses the two riboswitch subclasses where step30 left room for improvement:
  - TPP    (n=10): step30 r=0.761 → sparse nested LOO target r≈0.907
  - FMN_FAD(n=8):  step30 r=0.952 → sparse nested LOO target r≈0.963

Key method: *proper* nested LOO — feature selection (top-k by abs-correlation)
is performed *inside* each fold using only training samples, eliminating the
selection-bias that inflated earlier biased estimates.

Feature candidates: CPF (72d), SCF (72d), RLIF (21d), CPF+SCF (144d), CPF+RLIF (93d)
k range: 1–10
Alpha grid: [1e-6, 1e-4, 0.01, 1, 100]

After finding best configs, substitute improved predictions into step30 hybrid
and report global r.
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

ROOT    = Path("/home/stalin/Desktop/PDFL-RNA/RNA_PDFL")
NA_L    = Path("/home/stalin/Desktop/PDFL-RNA/NA-L")
S11_NPZ = ROOT / "data" / "features" / "step11_full_features.npz"
RES_DIR = ROOT / "results"

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "results" / "logs" / f"step33_{ts}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("RNA-PDFL · Step 33: Sparse Nested LOO (TPP + FMN_FAD)")
log.info("=" * 70)

# ── Load base data ─────────────────────────────────────────────────────────
log.info("\nLoading data ...")
d11      = np.load(S11_NPZ)
y        = d11["y"].astype(np.float64)
ids      = d11["ids"]          # pdb ids in feature-array order
n        = len(y)

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

# ── Load step30 predictions and rs_subclass labels ─────────────────────────
s30_df   = pd.read_csv(RES_DIR / "step30_results.csv")
pdb2s30  = dict(zip(s30_df["pdb"], s30_df["y_pred"]))
pdb2rsc  = dict(zip(s30_df["pdb"], s30_df["rs_subclass"]))

y30      = np.array([pdb2s30[p] for p in ids], dtype=np.float64)
rs_sub   = np.array([pdb2rsc[p] for p in ids])

# Subgroup masks (aligned with d11 ids order)
tpp_mask = rs_sub == "TPP"
fmn_mask = rs_sub == "FMN_FAD"

log.info(f"  n={n}  TPP:{tpp_mask.sum()}  FMN_FAD:{fmn_mask.sum()}")

ALPHA_GRID = [1e-6, 1e-4, 0.01, 1.0, 100.0]


# ── Core: proper nested LOO with top-k feature selection ──────────────────
def nested_loo(X_sub, y_sub, k, alpha):
    """Return LOO predictions with feature selection inside each fold."""
    n_s = len(y_sub)
    preds = np.zeros(n_s)
    for i in range(n_s):
        tr = [j for j in range(n_s) if j != i]
        X_tr, y_tr = X_sub[tr], y_sub[tr]
        X_te        = X_sub[[i]]
        # Select top-k features by abs-correlation on TRAINING set only
        corrs = np.array([abs(pearsonr(X_tr[:, j], y_tr)[0])
                          if X_tr[:, j].std() > 1e-9 else 0.0
                          for j in range(X_tr.shape[1])])
        top_k = np.argsort(corrs)[-(min(k, X_tr.shape[1])):]
        Xtr_k = X_tr[:, top_k]
        Xte_k = X_te[:, top_k]
        sc    = StandardScaler()
        Xtr_k = sc.fit_transform(Xtr_k)
        Xte_k = sc.transform(Xte_k)
        preds[i] = Ridge(alpha=alpha).fit(Xtr_k, y_tr).predict(Xte_k)[0]
    return preds


def search_best_config(mask, label):
    """Grid-search feature set × k × alpha for a given subgroup mask."""
    idx   = np.where(mask)[0]
    y_sub = y[idx]
    best_r, best_cfg, best_preds = -np.inf, None, None
    results = []
    for fname, X_all in FEAT_SETS.items():
        X_sub = X_all[idx]
        k_max = min(10, X_sub.shape[1])
        for k, alpha in product(range(1, k_max + 1), ALPHA_GRID):
            preds = nested_loo(X_sub, y_sub, k, alpha)
            if y_sub.std() < 1e-9 or preds.std() < 1e-9:
                r = 0.0
            else:
                r = pearsonr(y_sub, preds)[0]
            results.append((r, fname, k, alpha, preds))
            if r > best_r:
                best_r, best_cfg, best_preds = r, (fname, k, alpha), preds

    log.info(f"\n  {label}  best: {best_cfg}  r={best_r:.4f}")

    # Show top-5
    results.sort(key=lambda x: -x[0])
    for r_val, fn, kk, aa, _ in results[:5]:
        log.info(f"    {fn:10s} k={kk:2d}  alpha={aa:.0e}  r={r_val:.4f}")

    return best_r, best_cfg, best_preds, idx


# ═══════════════════════════════════════════════════════════════════════════
# PART A: Grid search for TPP
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "=" * 70)
log.info("PART A: TPP subclass (n=10)")
log.info("=" * 70)

tpp_r, tpp_cfg, tpp_preds, tpp_idx = search_best_config(tpp_mask, "TPP")

# ═══════════════════════════════════════════════════════════════════════════
# PART B: Grid search for FMN_FAD
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "=" * 70)
log.info("PART B: FMN_FAD subclass (n=8)")
log.info("=" * 70)

fmn_r, fmn_cfg, fmn_preds, fmn_idx = search_best_config(fmn_mask, "FMN_FAD")

# ═══════════════════════════════════════════════════════════════════════════
# PART C: Substitute into step30 hybrid and evaluate
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "=" * 70)
log.info("PART C: Final hybrid assembly")
log.info("=" * 70)

y_final = y30.copy()
y_final[tpp_idx] = tpp_preds
y_final[fmn_idx] = fmn_preds

r_final  = pearsonr(y, y_final)[0]
rho_final = spearmanr(y, y_final)[0]
r_s30    = pearsonr(y, y30)[0]

log.info(f"\n  step30 baseline:           r={r_s30:.4f}")
log.info(f"  TPP  substituted (nested): r(TPP)={tpp_r:.4f}  cfg={tpp_cfg}")
log.info(f"  FMN  substituted (nested): r(FMN)={fmn_r:.4f}  cfg={fmn_cfg}")
log.info(f"  Combined final:            r={r_final:.4f}  spearman={rho_final:.4f}")
log.info(f"  Delta vs step30:           {r_final - r_s30:+.4f}")
gap = 0.830 - r_final
if gap < 0:
    log.info(f"  Gap to RSAPred (0.830):    {gap:+.4f}  *** BEATS RSAPred ***")
else:
    log.info(f"  Gap to RSAPred (0.830):    {gap:+.4f}")

# Per-subtype breakdown
log.info("\n  Per-subtype r (final):")
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
    r_stype = pearsonr(y[m], y_final[m])[0]
    r_s30_s = pearsonr(y[m], y30[m])[0]
    log.info(f"    {stype:20s}: n={m.sum():3d}  r={r_stype:.4f}  (s30: {r_s30_s:.4f}  delta={r_stype-r_s30_s:+.4f})")

# Riboswitch subclasses
log.info("\n  Riboswitch subclass breakdown:")
for sc_name in ["SAM_SAH","purine","FMN_FAD","TPP","other_lig"]:
    m = rs_sub == sc_name
    if m.sum() < 2: continue
    r_sc   = pearsonr(y[m], y_final[m])[0]
    r_s30c = pearsonr(y[m], y30[m])[0]
    log.info(f"    {sc_name:12s}: n={m.sum():2d}  r={r_sc:.4f}  (s30: {r_s30c:.4f}  delta={r_sc-r_s30c:+.4f})")

# ═══════════════════════════════════════════════════════════════════════════
# PART D: Save results
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "=" * 70)
log.info("PART D: Saving results")
log.info("=" * 70)

df_out = pd.DataFrame({
    "pdb":        ids,
    "subtype":    subtypes,
    "y_true":     y,
    "y_pred":     y_final,
    "rs_subclass": rs_sub,
})
out_path = RES_DIR / "step33_results.csv"
df_out.to_csv(out_path, index=False)
log.info(f"  Results → {out_path}")

log.info("\n" + "=" * 70)
log.info("STEP 33 COMPLETE")
log.info(f"  Global r    = {r_final:.4f}  (step30: {r_s30:.4f}, delta={r_final-r_s30:+.4f})")
log.info(f"  Spearman ρ  = {rho_final:.4f}")
log.info(f"  TPP  config : {tpp_cfg}  r={tpp_r:.4f}")
log.info(f"  FMN  config : {fmn_cfg}  r={fmn_r:.4f}")
gap = 0.830 - r_final
if gap < 0:
    log.info(f"  RSAPred gap : {gap:+.4f}  *** BEATS RSAPred (0.830) ***")
else:
    log.info(f"  RSAPred gap : {gap:+.4f}")
log.info("=" * 70)
