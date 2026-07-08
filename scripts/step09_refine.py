"""
SGT-RNA  ·  Step 9: Refined Subclassification + MACCS Keys

Two improvements over Step 8:
  1. Split "other" (n=45) into sub-types from NL2020 index analysis:
       - g_quadruplex   (n= 8): G-rich sequences, telomere, RHPS4, AMZ, BMVC
       - duplex_groove  (n=10): polyamides/intercalators on short duplex RNA
       - other_misc     (n=27): remaining heterogeneous entries
  2. MACCS keys (167-bit) added to feature set for all complexes.
     MACCS keys are rule-based substructure fingerprints that better
     differentiate nucleotide analogs (SAM/preQ1/c-di-GMP) than ECFP4.

Features: Step 7 NPZ (38,796) + MACCS (167) = 38,963 total
"""

import logging, warnings, time
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, LeaveOneOut, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path("/home/stalin/Desktop/SGT-RNA/RNA_SGT")
NPZ_S7   = ROOT / "data" / "features" / "step07_full_features.npz"
LABELS   = ROOT / "results" / "step06_subtype_labels.csv"
DATA_DIR = Path("/home/stalin/Desktop/SGT-RNA/NA-L")
OUT_NPZ  = ROOT / "data" / "features" / "step09_full_features.npz"
RES_DIR  = ROOT / "results"
FIG_DIR  = ROOT / "results" / "figures"
LOG_DIR  = ROOT / "results" / "logs"
for d in [RES_DIR, FIG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── logging ───────────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"step09_{ts}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("SGT-RNA  ·  Step 9: Refined Subtypes + MACCS Keys")
log.info("=" * 70)

SEED = 42
np.random.seed(SEED)

# ── subtype override map ───────────────────────────────────────────────────────
# Step 8 manual overrides (misclassified by keyword parser)
STEP8_OVERRIDES = {
    "4jf2": "riboswitch", "3q3z": "riboswitch", "4erj": "riboswitch",
    "4erl": "riboswitch", "4lvz": "riboswitch", "4lvy": "riboswitch",
    "4lvx": "riboswitch", "4lw0": "riboswitch", "4lx5": "riboswitch",
    "4lx6": "riboswitch", "4nyb": "riboswitch", "4nyc": "riboswitch",
    "4nya": "riboswitch", "4nyd": "riboswitch", "6dmc": "riboswitch",
    "6ck4": "riboswitch",
    "6e84": "aptamer", "6e8s": "aptamer",
    "2bee": "ribosomal_asite", "2be0": "ribosomal_asite",
    "2f4s": "ribosomal_asite", "2o3w": "ribosomal_asite",
    "2o3x": "ribosomal_asite",
}

# Step 9: split "other" into three sub-types based on NL2020 index analysis
# G-quadruplex: bind G-rich/telomere sequences; flat aromatic ligands stack on G-quartets
G_QUAD = {
    "1nzm",  # RHPS4 + d(TTAGGGT)4 telomere G-quad
    "5cdb",  # NAX053 + human telomeric G-quad
    "4xwf",  # AMZ + G-quad RNA
    "4znp",  # AMZ + G-quad RNA
    "5btp",  # AMZ + G-quad RNA
    "6jj0",  # BMVC carbazole + G-quad
    "2mg8",  # XR5944 + G-quad (FID assay)
    "2loa",  # Zn(cy4q) porphyrin + G-quad
}

# Duplex/groove binders: polyamides or intercalators binding short duplex RNA
DUPLEX_GROOVE = {
    "407d",  # ImHpPyPy-b-Dp + 5'-AGTACT-3' duplex
    "408d",  # ImPyPyPy-b-Dp + 5'-AGTACT-3' duplex
    "1cvy",  # ImPyPyPybDp + 5'-CCAGATCTGG-3' duplex
    "1cvx",  # ImPyHpPybDp + 5'-CCAGATCTGG-3' duplex
    "454d",  # RH[ME2TRIEN]PHI + short duplex (intercalator)
    "1qv4",  # methylproamine + duplex RNA
    "1qv8",  # proamine + duplex RNA
    "1p96",  # DDI intercalator + duplex
    "1r4e",  # ent-DDI intercalator + duplex
    "6hbt",  # diguanidine-C4 groove binder
}

STEP9_OVERRIDES = {}
for pid in G_QUAD:
    STEP9_OVERRIDES[pid] = "g_quadruplex"
for pid in DUPLEX_GROOVE:
    STEP9_OVERRIDES[pid] = "duplex_groove"

# ── load features ─────────────────────────────────────────────────────────────
log.info("Loading Step 7 features ...")
d7 = np.load(NPZ_S7, allow_pickle=True)
X7 = d7["X"].astype(np.float32)
y  = d7["y"].astype(np.float32)
ids = [str(i) for i in d7["ids"]]
n   = len(y)
log.info(f"  X shape: {X7.shape}")

df_labels = pd.read_csv(LABELS).set_index("pdb")

def assign_subtype(pid):
    if pid in STEP9_OVERRIDES:
        return STEP9_OVERRIDES[pid]
    if pid in STEP8_OVERRIDES:
        return STEP8_OVERRIDES[pid]
    return df_labels.loc[pid, "subtype"] if pid in df_labels.index else "other_misc"

subtypes = np.array([assign_subtype(pid) for pid in ids])
# rename plain "other" → "other_misc" for clarity
subtypes = np.where(subtypes == "other", "other_misc", subtypes)

dist = Counter(subtypes)
log.info("\nRefined subtype distribution:")
for k, v in sorted(dist.items(), key=lambda x: -x[1]):
    log.info(f"  {k:<20}: {v:3d} complexes")

# ── MACCS keys (167-bit) ──────────────────────────────────────────────────────
log.info("\nComputing MACCS keys (167-bit) for all ligands ...")

def get_maccs(pdb_id: str) -> np.ndarray:
    sdf = DATA_DIR / pdb_id / f"{pdb_id}_ligand.sdf"
    mol = Chem.MolFromMolFile(str(sdf), sanitize=True)
    if mol is None:
        mol = Chem.MolFromMolFile(str(sdf), sanitize=False)
    if mol is None:
        return np.zeros(167, dtype=np.float32)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp, dtype=np.float32)

maccs_feats = np.array([get_maccs(pid) for pid in ids], dtype=np.float32)
n_nonzero = (maccs_feats.sum(axis=1) > 0).sum()
log.info(f"  MACCS shape: {maccs_feats.shape}, nonzero rows: {n_nonzero}/{n}")

# ── combined feature matrix ───────────────────────────────────────────────────
X_full = np.concatenate([X7, maccs_feats], axis=1).astype(np.float32)
X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
log.info(f"\nCombined features: {X_full.shape} (Step7 + MACCS)")

np.savez_compressed(OUT_NPZ, X=X_full, y=y, ids=np.array(ids), subtypes=subtypes)
log.info(f"  Saved → {OUT_NPZ}")

# ── pipeline ──────────────────────────────────────────────────────────────────
ALPHA_GRID = {"reg__alpha": [1, 10, 100, 1000, 10_000, 100_000]}
MLP_PARAMS = {
    "reg__hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32)],
    "reg__alpha": [1e-4, 1e-3, 1e-2],
}

def make_ridge_pipe():
    return Pipeline([
        ("vt",  VarianceThreshold(threshold=1e-4)),
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=SEED)),
        ("reg", Ridge()),
    ])

def make_mlp_pipe():
    return Pipeline([
        ("vt",  VarianceThreshold(threshold=1e-4)),
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=SEED)),
        ("reg", MLPRegressor(
            activation="relu", solver="adam", max_iter=1000,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=30, random_state=SEED,
        )),
    ])

def run_nested_cv(Xs, ys, pipe, param_grid, ns, n_jobs=10):
    cv_outer = LeaveOneOut() if ns < 15 else KFold(5, shuffle=True, random_state=SEED)
    outer_splits = list(cv_outer.split(Xs))
    oof = np.full(ns, np.nan)
    for tr, te in outer_splits:
        ni = min(5, len(tr))
        gs = GridSearchCV(pipe, param_grid, cv=ni, scoring="r2",
                          n_jobs=n_jobs, refit=True)
        gs.fit(Xs[tr], ys[tr])
        preds = gs.best_estimator_.predict(Xs[te])
        oof[te] = np.clip(preds, ys[tr].min() - 3, ys[tr].max() + 3)
    return oof

def compute_metrics(yt, yp):
    r   = float(np.corrcoef(yt, yp)[0, 1])
    rho = float(stats.spearmanr(yt, yp).statistic)
    rms = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2  = float(r2_score(yt, yp))
    return r, rho, rms, mae, r2

# ── per-subtype nested CV ─────────────────────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("PER-SUBTYPE NESTED CV")
log.info("=" * 70)

MIN_N   = 8    # skip subtypes smaller than this
MLP_MIN = 15   # use MLP for subtypes >= this

all_oof = np.full(n, np.nan)
results = []
t0 = time.time()

for sub in sorted(set(subtypes)):
    mask = subtypes == sub
    Xs, ys = X_full[mask], y[mask]
    ns = int(mask.sum())
    cv_name = "LOO" if ns < 15 else "5-fold"

    if ns < MIN_N:
        log.info(f"  [{sub}] n={ns} — too small, skipping")
        continue

    log.info(f"  [{sub}] n={ns:3d} | {cv_name}")

    # Ridge OOF
    oof_ridge = run_nested_cv(Xs, ys, make_ridge_pipe(), ALPHA_GRID, ns, n_jobs=10)
    r_ridge = float(np.corrcoef(ys, oof_ridge)[0, 1])
    log.info(f"    Ridge   r = {r_ridge:.4f}")

    oof_best = oof_ridge

    # MLP ensemble for n >= MLP_MIN
    if ns >= MLP_MIN:
        try:
            oof_mlp = run_nested_cv(Xs, ys, make_mlp_pipe(), MLP_PARAMS, ns, n_jobs=4)
            r_mlp = float(np.corrcoef(ys, oof_mlp)[0, 1])
            log.info(f"    MLP     r = {r_mlp:.4f}")
            w_r = max(0.0, r_ridge)
            w_m = max(0.0, r_mlp)
            if w_r + w_m > 0:
                oof_ens = (w_r * oof_ridge + w_m * oof_mlp) / (w_r + w_m)
                r_ens = float(np.corrcoef(ys, oof_ens)[0, 1])
                log.info(f"    Ensemb  r = {r_ens:.4f}")
                if r_ens > r_ridge:
                    oof_best = oof_ens
        except Exception as e:
            log.info(f"    MLP failed: {e}")

    all_oof[mask] = oof_best
    r, rho, rms, mae, r2 = compute_metrics(ys, oof_best)
    log.info(f"    FINAL   r={r:.4f} | ρ={rho:.4f} | RMSE={rms:.4f} | MAE={mae:.4f}")
    results.append({
        "subtype": sub, "n": ns, "Pearson_r": round(r, 4),
        "Spearman_rho": round(rho, 4), "RMSE": round(rms, 4),
        "MAE": round(mae, 4), "R2": round(r2, 4)
    })

# ── global Ridge ──────────────────────────────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("GLOBAL RIDGE (adaptive stacking signal)")
log.info("=" * 70)

oof_global = np.full(n, np.nan)
for tr, te in KFold(5, shuffle=True, random_state=SEED).split(X_full):
    gs = GridSearchCV(make_ridge_pipe(), ALPHA_GRID,
                      cv=KFold(5, shuffle=True, random_state=SEED),
                      scoring="r2", n_jobs=10, refit=True)
    gs.fit(X_full[tr], y[tr])
    preds = gs.best_estimator_.predict(X_full[te])
    oof_global[te] = np.clip(preds, y[tr].min() - 3, y[tr].max() + 3)

r_global = float(np.corrcoef(y, oof_global)[0, 1])
log.info(f"  Global Ridge  r = {r_global:.4f}")

# ── adaptive ensemble ─────────────────────────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("ADAPTIVE ENSEMBLE")
log.info("=" * 70)

oof_stack = np.copy(oof_global)
for sub in sorted(set(subtypes)):
    mask = subtypes == sub
    has_sub = ~np.isnan(all_oof[mask])
    if has_sub.sum() < 4:
        log.info(f"  {sub:<20}: global only")
        continue
    r_sub  = float(np.corrcoef(y[mask][has_sub], all_oof[mask][has_sub])[0, 1])
    r_glob = float(np.corrcoef(y[mask], oof_global[mask])[0, 1])
    if r_sub > r_glob:
        oof_stack[mask] = all_oof[mask]
        log.info(f"  {sub:<20}: SUBTYPE  r={r_sub:.4f} > global r={r_glob:.4f}  ✓")
    else:
        log.info(f"  {sub:<20}: GLOBAL   r={r_glob:.4f} ≥ subtype r={r_sub:.4f}")

r_s, rho_s, rm_s, ma_s, r2_s = compute_metrics(y, oof_stack)
log.info(f"\n  n evaluated  : {n}/{n}")
log.info(f"  Pearson r    : {r_s:.4f}")
log.info(f"  Spearman ρ   : {rho_s:.4f}")
log.info(f"  RMSE         : {rm_s:.4f}")
log.info(f"  MAE          : {ma_s:.4f}")
log.info(f"  R²           : {r2_s:.4f}")
log.info(f"  Time         : {int(time.time()-t0)}s")

# ── benchmarks ────────────────────────────────────────────────────────────────
BENCHMARKS = {
    "AffiGrapher": 0.498, "RLaffinity": 0.559,
    "RLASIF": 0.666, "DeepRSMA": 0.784, "RSAPred": 0.830
}
log.info("\n  Benchmark comparison:")
for bm, bv in BENCHMARKS.items():
    sym = "✓" if r_s >= bv else "✗"
    log.info(f"    {sym} {'above' if r_s >= bv else 'below'}  {bm}: {bv}")

prev = {"Step 5": 0.5165, "Step 7": 0.4876, "Step 8": 0.5342}
log.info("\n  Δr vs previous steps:")
for k, v in prev.items():
    log.info(f"    {k}: {v:.4f}  Δr = {r_s-v:+.4f}")

# ── save ──────────────────────────────────────────────────────────────────────
df_res = pd.DataFrame(results)
df_res.to_csv(RES_DIR / "step09_results.csv", index=False)
log.info(f"\n  Results saved → {RES_DIR / 'step09_results.csv'}")

# ── figure ────────────────────────────────────────────────────────────────────
COLORS = {
    "riboswitch": "#4C72B0", "aptamer": "#55A868",
    "ribosomal_asite": "#C44E52", "viral_tar": "#DD8452",
    "g_quadruplex": "#E377C2", "duplex_groove": "#7F7F7F",
    "other_misc": "#8172B2",
}

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(
    f"Step 9 — Refined Subtypes + MACCS Keys   r = {r_s:.4f}",
    fontsize=14, fontweight="bold"
)

ax = axes[0]
for sub in sorted(set(subtypes)):
    mask = subtypes == sub
    ax.scatter(y[mask], oof_stack[mask], label=sub, alpha=0.75, s=45,
               color=COLORS.get(sub, "gray"), edgecolors="none")
lims = [y.min() - 0.5, y.max() + 0.5]
ax.plot(lims, lims, "k--", lw=1)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Experimental pKd"); ax.set_ylabel("Predicted pKd")
ax.set_title("Parity plot (adaptive ensemble)")
ax.legend(fontsize=7)

ax2 = axes[1]
df_bar = df_res.sort_values("Pearson_r", ascending=True)
ax2.barh(df_bar["subtype"], df_bar["Pearson_r"],
         color=[COLORS.get(s, "gray") for s in df_bar["subtype"]])
ax2.axvline(0, color="k", lw=0.8)
ax2.axvline(r_s, color="purple", lw=1.5, ls="--", label=f"Step9 r={r_s:.3f}")
ax2.axvline(0.5342, color="green", lw=1.2, ls=":", label="Step8 r=0.534")
for bm, bv in BENCHMARKS.items():
    ax2.axvline(bv, color="gray", lw=0.8, ls=":")
    ax2.text(bv + 0.005, 0.05, bm, fontsize=6, rotation=90, va="bottom",
             transform=ax2.get_xaxis_transform())
ax2.set_xlabel("Pearson r")
ax2.set_title("Per-subtype Pearson r (nested OOF)")
ax2.legend(fontsize=7)

plt.tight_layout()
plt.savefig(FIG_DIR / "step09_results.png", dpi=150, bbox_inches="tight")
plt.close()
log.info(f"  Figure saved → {FIG_DIR / 'step09_results.png'}")

log.info("\n" + "=" * 70)
log.info("STEP 9 COMPLETE")
log.info("=" * 70)
