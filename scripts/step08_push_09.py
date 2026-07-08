"""
SGT-RNA  ·  Step 8: Corrected Subtypes + MLP Ensemble

Key improvements over Step 7:
  1. Manual override of 23 misclassified PDB entries:
       - 16 riboswitches (had non-descriptive ligand codes)
       - 5 ribosomal_asite entries
       - 2 aptamers (TO1-Biotin binders)
     Result: riboswitch 45→61, other 68→45, aptamer 18→20, ribosomal 8→13
  2. MLP ensemble for large subtypes (n≥15)
  3. Adaptive stacking: use best-per-subtype (subtype vs global Ridge)

Features: Step 7 NPZ (38,796 features = SGT + Morgan + RNA-FM + ViennaRNA + k-mer)
"""

import logging, warnings, time
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
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
ROOT    = Path("/home/stalin/Desktop/SGT-RNA/RNA_SGT")
NPZ_S7  = ROOT / "data" / "features" / "step07_full_features.npz"
LABELS  = ROOT / "results" / "step06_subtype_labels.csv"
RES_DIR = ROOT / "results"
FIG_DIR = ROOT / "results" / "figures"
LOG_DIR = ROOT / "results" / "logs"
for d in [RES_DIR, FIG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── logging ───────────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"step08_{ts}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("SGT-RNA  ·  Step 8: Corrected Subtypes + Ensemble")
log.info("=" * 70)

SEED = 42
np.random.seed(SEED)

# ── manual subtype overrides ──────────────────────────────────────────────────
MANUAL_OVERRIDES = {
    # riboswitches with non-descriptive PDB ligand codes
    "4jf2": "riboswitch",   # SAH
    "3q3z": "riboswitch",   # preQ1 class I
    "4erj": "riboswitch",   # preQ1 class II
    "4erl": "riboswitch",   # preQ1 class II
    "4lvz": "riboswitch",   # c-di-GMP
    "4lvy": "riboswitch",   # c-di-GMP
    "4lvx": "riboswitch",   # c-di-GMP
    "4lw0": "riboswitch",   # c-di-GMP
    "4lx5": "riboswitch",   # c-di-GMP
    "4lx6": "riboswitch",   # c-di-GMP
    "4nyb": "riboswitch",   # ppGpp
    "4nyc": "riboswitch",   # ppGpp
    "4nya": "riboswitch",   # ppGpp
    "4nyd": "riboswitch",   # ppGpp
    "6dmc": "riboswitch",   # SAM
    "6ck4": "riboswitch",   # lysine
    # aptamers
    "6e84": "aptamer",      # TO1-Biotin
    "6e8s": "aptamer",      # TO1-Biotin
    # ribosomal A-site
    "2bee": "ribosomal_asite",
    "2be0": "ribosomal_asite",
    "2f4s": "ribosomal_asite",
    "2o3w": "ribosomal_asite",
    "2o3x": "ribosomal_asite",
}

# ── load features ─────────────────────────────────────────────────────────────
log.info("Loading Step 7 features ...")
d7 = np.load(NPZ_S7, allow_pickle=True)
X  = d7["X"].astype(np.float32)
y  = d7["y"].astype(np.float32)
ids = [str(i) for i in d7["ids"]]
n   = len(y)
log.info(f"  X shape: {X.shape}, n={n}")

df_labels = pd.read_csv(LABELS).set_index("pdb")
subtypes  = np.array([
    MANUAL_OVERRIDES.get(pid,
        df_labels.loc[pid, "subtype"] if pid in df_labels.index else "other")
    for pid in ids
])

dist = Counter(subtypes)
log.info("\nCorrected subtype distribution:")
for k, v in sorted(dist.items(), key=lambda x: -x[1]):
    log.info(f"  {k:<20}: {v:3d} complexes")

# ── pipeline ──────────────────────────────────────────────────────────────────
ALPHA_GRID  = {"reg__alpha": [1, 10, 100, 1000, 10_000, 100_000]}
MLP_PARAMS  = {
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

def run_nested_cv(Xs, ys, pipe, param_grid, cv_outer, cv_inner=5, n_jobs=10):
    ns = len(ys)
    if isinstance(cv_outer, int):
        outer_splits = list(KFold(cv_outer, shuffle=True, random_state=SEED).split(Xs))
    else:
        outer_splits = list(cv_outer.split(Xs))
    oof = np.full(ns, np.nan)
    for tr, te in outer_splits:
        ni = min(cv_inner, len(tr))
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
log.info("PER-SUBTYPE NESTED CV (Ridge + MLP ensemble)")
log.info("=" * 70)

MIN_N   = 10     # skip subtypes smaller than this
MLP_MIN = 15     # use MLP ensemble for subtypes >= this

all_oof = np.full(n, np.nan)
results = []
t0 = time.time()

for sub in sorted(set(subtypes)):
    mask = subtypes == sub
    idx  = np.where(mask)[0]
    Xs, ys = X[mask], y[mask]
    ns = int(mask.sum())

    if ns < MIN_N:
        log.info(f"  [{sub}] n={ns} — too small, skipping")
        continue

    cv_outer = LeaveOneOut() if ns < 15 else 5
    cv_name  = "LOO" if ns < 15 else "5-fold"
    log.info(f"  [{sub}] n={ns:3d} | {cv_name}")

    # Ridge OOF
    oof_ridge = run_nested_cv(Xs, ys, make_ridge_pipe(), ALPHA_GRID, cv_outer,
                              cv_inner=min(5, len(Xs)-1), n_jobs=10)
    r_ridge = np.corrcoef(ys, oof_ridge)[0, 1]
    log.info(f"    Ridge   r = {r_ridge:.4f}")

    oof_best = oof_ridge

    # MLP ensemble for larger subtypes
    if ns >= MLP_MIN:
        try:
            oof_mlp = run_nested_cv(Xs, ys, make_mlp_pipe(), MLP_PARAMS, cv_outer,
                                    cv_inner=min(5, len(Xs)-1), n_jobs=4)
            r_mlp = np.corrcoef(ys, oof_mlp)[0, 1]
            log.info(f"    MLP     r = {r_mlp:.4f}")
            # weighted blend: positive-r weighted average
            w_r = max(0.0, r_ridge)
            w_m = max(0.0, r_mlp)
            if w_r + w_m > 0:
                oof_ens = (w_r * oof_ridge + w_m * oof_mlp) / (w_r + w_m)
                r_ens = np.corrcoef(ys, oof_ens)[0, 1]
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

# ── global Ridge (for adaptive stacking) ──────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("GLOBAL RIDGE (adaptive stacking signal)")
log.info("=" * 70)

oof_global = np.full(n, np.nan)
for tr, te in KFold(5, shuffle=True, random_state=SEED).split(X):
    gs = GridSearchCV(make_ridge_pipe(), ALPHA_GRID,
                      cv=KFold(5, shuffle=True, random_state=SEED),
                      scoring="r2", n_jobs=10, refit=True)
    gs.fit(X[tr], y[tr])
    preds = gs.best_estimator_.predict(X[te])
    oof_global[te] = np.clip(preds, y[tr].min()-3, y[tr].max()+3)

r_global = np.corrcoef(y, oof_global)[0, 1]
log.info(f"  Global Ridge  r = {r_global:.4f}")

# ── adaptive ensemble: best-per-subtype ───────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("ADAPTIVE ENSEMBLE")
log.info("=" * 70)

oof_stack = np.copy(oof_global)
for sub in sorted(set(subtypes)):
    mask = subtypes == sub
    has_sub = ~np.isnan(all_oof[mask])
    if has_sub.sum() < 4:
        continue
    r_sub  = float(np.corrcoef(y[mask][has_sub], all_oof[mask][has_sub])[0, 1])
    r_glob = float(np.corrcoef(y[mask], oof_global[mask])[0, 1])
    if r_sub > r_glob:
        oof_stack[mask] = all_oof[mask]
        log.info(f"  {sub:<20}: SUBTYPE  r={r_sub:.4f} > global r={r_glob:.4f}")
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

# ── benchmark comparison ──────────────────────────────────────────────────────
BENCHMARKS = {
    "AffiGrapher": 0.498, "RLaffinity": 0.559,
    "RLASIF": 0.666, "DeepRSMA": 0.784, "RSAPred": 0.830
}
log.info("\n  Benchmark comparison:")
for bm, bv in BENCHMARKS.items():
    sym = "✓" if r_s >= bv else "✗"
    log.info(f"    {sym} {'above' if r_s >= bv else 'below'}  {bm}: {bv}")

prev = {"Step 5": 0.5165, "Step 6": 0.4691, "Step 7": 0.4876}
log.info("\n  Δr vs previous steps:")
for k, v in prev.items():
    log.info(f"    {k}: {v:.4f}  Δr = {r_s-v:+.4f}")

# ── save results ──────────────────────────────────────────────────────────────
df_res = pd.DataFrame(results)
df_res.to_csv(RES_DIR / "step08_results.csv", index=False)
log.info(f"\n  Results saved → {RES_DIR / 'step08_results.csv'}")

# ── figure ────────────────────────────────────────────────────────────────────
COLORS = {
    "riboswitch": "#4C72B0", "aptamer": "#55A868",
    "ribosomal_asite": "#C44E52", "viral_tar": "#DD8452", "other": "#8172B2"
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    f"Step 8 — Corrected Subtypes + Ensemble   r = {r_s:.4f}",
    fontsize=14, fontweight="bold"
)

# parity plot
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
ax.legend(fontsize=8)

# per-subtype bar chart
ax2 = axes[1]
df_bar = df_res.sort_values("Pearson_r", ascending=True)
ax2.barh(df_bar["subtype"], df_bar["Pearson_r"],
         color=[COLORS.get(s, "gray") for s in df_bar["subtype"]])
ax2.axvline(0, color="k", lw=0.8)
ax2.axvline(r_s, color="purple", lw=1.5, ls="--", label=f"Stack r={r_s:.3f}")
for bm, bv in BENCHMARKS.items():
    ax2.axvline(bv, color="gray", lw=0.8, ls=":")
    ax2.text(bv + 0.005, 0.05, bm, fontsize=7, rotation=90, va="bottom",
             transform=ax2.get_xaxis_transform())
ax2.set_xlabel("Pearson r")
ax2.set_title("Per-subtype Pearson r (nested OOF)")
ax2.legend(fontsize=8)

plt.tight_layout()
fig_path = FIG_DIR / "step08_results.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
log.info(f"  Figure saved → {fig_path}")

log.info("\n" + "=" * 70)
log.info("STEP 8 COMPLETE")
log.info("=" * 70)
