"""
RNA-PDFL  ·  Step 4: Multi-Scale Feature Expansion + Advanced ML Training

Advanced strategies:
  1. Multi-scale FRI: η ∈ {2, 5, 8} Å  (Exponential kernel)
     → 3 × 3 600 = 10 800-dim combined feature vector
  2. Six models with nested 5-outer × 5-inner CV:
       Ridge, ElasticNet, SVR(RBF), Gradient Boosting, LightGBM, XGBoost
  3. Preprocessing:
       Linear/kernel → VarianceThreshold → StandardScaler → PCA(95 %)
       Tree-based   → VarianceThreshold only
  4. Stacking meta-ensemble: Ridge meta-learner on 6 OOF prediction stacks
  5. Metrics: Pearson r, Spearman ρ, RMSE, MAE, R²
  6. Journal-quality result figures
"""

import gzip, pickle, logging, warnings, time
from pathlib import Path
from datetime import datetime
from itertools import product
import subprocess

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy import stats

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

# ── temperature logging (display only, no throttling) ─────────────────────────
def get_cpu_temp():
    try:
        import psutil
        sensors = psutil.sensors_temperatures()
        for key in ("coretemp", "k10temp", "zenpower", "acpitz"):
            if key in sensors:
                return max(t.current for t in sensors[key])
    except Exception:
        pass
    for p in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
        try:
            return int(p.read_text().strip()) / 1000.0
        except Exception:
            pass
    return 0.0

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path("/home/stalin/Desktop/PDFL-RNA/RNA_PDFL")
PKL_FILE = ROOT / "data" / "pocket_fri" / "pocket_fri_data.pkl.gz"
NPZ_ETA5 = ROOT / "data" / "features" / "step03_pdfl_features.npz"
OUT_DIR  = ROOT / "data" / "features"
RES_DIR  = ROOT / "results"
FIG_DIR  = ROOT / "results" / "figures"
LOG_DIR  = ROOT / "results" / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── logging ───────────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = LOG_DIR / f"step04_{ts}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("RNA-PDFL  ·  Step 4: Multi-Scale ML Training")
log.info("=" * 70)

# ── constants ─────────────────────────────────────────────────────────────────
RNA_ELEMENTS = ["C","N","O","P"]
LIG_ELEMENTS = ["C","N","O","S","P","F","Cl","Br","I"]
THRESHOLDS   = [0.0, 0.8, 0.85, 0.90, 0.95]
N_STATS      = 10
N_PAIRS      = 36
N_FEATS_SINGLE = N_PAIRS * len(THRESHOLDS) * 2 * N_STATS  # 3 600
ETA_LIST     = [2.0, 5.0, 8.0]   # multi-scale FRI
KAPPA        = 2.0
EPS          = 1e-8
SEED         = 42

# ── FRI & Laplacian helpers ───────────────────────────────────────────────────
def fri_exp(D, eta):
    return np.exp(-(D / eta) ** KAPPA)

def spectral_stats(eigs):
    if len(eigs) == 0:
        return np.zeros(N_STATS, dtype=np.float32)
    nz = eigs[eigs > EPS]
    return np.array([
        eigs.min(), eigs.max(), eigs.mean(), float(np.median(eigs)),
        eigs.var(), eigs.std(), eigs.sum(), float((eigs**2).sum()),
        float(len(nz)), float(len(eigs)-len(nz))
    ], dtype=np.float32)

def build_L0(W_sel):
    nr, nl = W_sel.shape
    n = nr + nl
    L = np.zeros((n, n), dtype=np.float64)
    rd = W_sel.sum(1); cd = W_sel.sum(0)
    np.fill_diagonal(L[:nr,:nr], rd)
    np.fill_diagonal(L[nr:,nr:], cd)
    L[:nr,nr:] = -W_sel; L[nr:,:nr] = -W_sel.T
    return L

def pair_features_eta(rc, lc, eta):
    nr, nl = len(rc), len(lc)
    nf = len(THRESHOLDS) * 2 * N_STATS
    if nr == 0 or nl == 0:
        return np.zeros(nf, dtype=np.float32)
    D = cdist(rc, lc).astype(np.float64)
    W = fri_exp(D, eta)
    wmax = W.max()
    if wmax < 1e-12:
        return np.zeros(nf, dtype=np.float32)
    W_norm = W / wmax
    n_total = nr + nl
    feats = np.zeros(nf, dtype=np.float32)
    off = 0
    for tau in THRESHOLDS:
        mask   = W_norm >= tau
        W_sel  = W_norm * mask
        ne     = int(mask.sum())
        if ne == 0:
            eL0 = np.zeros(n_total); eL1 = np.array([])
        else:
            L0   = build_L0(W_sel)
            eL0  = np.maximum(np.linalg.eigvalsh(L0), 0.0)
            b0   = int((eL0 < EPS).sum())
            nz1  = max(0, ne - n_total + b0)
            eL1  = np.concatenate([np.zeros(nz1), eL0[eL0 >= EPS]])
        feats[off:off+N_STATS]            = spectral_stats(eL0)
        feats[off+N_STATS:off+2*N_STATS]  = spectral_stats(eL1)
        off += 2 * N_STATS
    return feats

# ── load η=5 features (already computed in Step 3) ───────────────────────────
log.info(f"\nLoading η=5 features from {NPZ_ETA5} ...")
d5   = np.load(NPZ_ETA5, allow_pickle=True)
X5   = d5["X"].astype(np.float32)
y    = d5["y"].astype(np.float32)
ids  = d5["ids"].tolist()
n    = len(y)
log.info(f"  η=5: X shape {X5.shape}")

# ── compute η=2 and η=8 features on-the-fly ──────────────────────────────────
log.info(f"\nLoading pocket coordinates ...")
with gzip.open(PKL_FILE, "rb") as f:
    records = pickle.load(f)

Xeta = {5.0: X5}
for eta in [2.0, 8.0]:
    log.info(f"  Computing η={eta} features ...")
    t0 = time.time()
    Xe = np.zeros((n, N_FEATS_SINGLE), dtype=np.float32)
    for idx, rec in enumerate(records):
        parts = []
        for r_el in RNA_ELEMENTS:
            for l_el in LIG_ELEMENTS:
                rc = rec["rna_coords"][r_el]
                lc = rec["lig_coords"].get(l_el, np.empty((0,3), np.float32))
                parts.append(pair_features_eta(rc, lc, eta))
        Xe[idx] = np.concatenate(parts)
    Xeta[eta] = Xe
    log.info(f"    η={eta} done in {time.time()-t0:.1f}s  shape={Xe.shape}")

# ── concatenate multi-scale features: η=2 | η=5 | η=8 ───────────────────────
X_multi = np.concatenate([Xeta[2.0], Xeta[5.0], Xeta[8.0]], axis=1)
log.info(f"\nMulti-scale feature matrix: {X_multi.shape}  "
         f"({X_multi.shape[1]} = 3 × {N_FEATS_SINGLE})")

# Save multi-scale features
msf_path = OUT_DIR / "step04_multiscale_features.npz"
np.savez_compressed(msf_path, X=X_multi, y=y, ids=np.array(ids))
log.info(f"Multi-scale features saved → {msf_path}")

# ── metrics helper ────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, label=""):
    r, _  = stats.pearsonr(y_true, y_pred)
    rho,_ = stats.spearmanr(y_true, y_pred)
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae   = float(mean_absolute_error(y_true, y_pred))
    r2    = float(r2_score(y_true, y_pred))
    return {"label": label, "Pearson_r": round(r,4), "Spearman_rho": round(rho,4),
            "RMSE": round(rmse,4), "MAE": round(mae,4), "R2": round(r2,4)}

# ── cross-validation setup ────────────────────────────────────────────────────
outer_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
outer_splits = list(outer_cv.split(X_multi))

log.info("\n" + "=" * 70)
log.info("NESTED 5 × 5 CROSS-VALIDATION")
log.info("=" * 70)

# ── model definitions ─────────────────────────────────────────────────────────
def make_linear_pipeline(estimator):
    return Pipeline([
        ("vt",  VarianceThreshold(threshold=1e-4)),
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=SEED)),
        ("est", estimator),
    ])

def make_tree_pipeline(estimator):
    return Pipeline([
        ("vt",  VarianceThreshold(threshold=1e-4)),
        ("est", estimator),
    ])

# MODELS: (pipeline, param_dist, gs_n_jobs, n_iter)
# n_iter=0  → GridSearchCV (exhaustive); n_iter>0 → RandomizedSearchCV
MODELS = {
    "Ridge": (
        make_linear_pipeline(Ridge()),
        {"est__alpha": [0.01, 0.1, 1, 10, 100, 1000, 10000]},
        2, 0
    ),
    "ElasticNet": (
        make_linear_pipeline(ElasticNet(max_iter=5000)),
        {"est__alpha": [0.001, 0.01, 0.1, 1.0],
         "est__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
        2, 0
    ),
    "SVR": (
        make_linear_pipeline(SVR(kernel="rbf", max_iter=5000)),
        {"est__C": [0.1, 1, 10, 100],
         "est__gamma": ["scale", 0.001, 0.01]},
        2, 0
    ),
    "GBR": (
        make_tree_pipeline(GradientBoostingRegressor(random_state=SEED)),
        {"est__n_estimators": [100, 200, 300],
         "est__max_depth":    [3, 5],
         "est__learning_rate":[0.05, 0.1],
         "est__subsample":    [0.8, 1.0]},
        10, 0
    ),
    "LightGBM": (
        make_tree_pipeline(lgb.LGBMRegressor(
            device="gpu", verbose=-1, random_state=SEED, n_jobs=1)),
        {"est__n_estimators":  [100, 300, 500, 800],
         "est__num_leaves":    [31, 63, 127, 255],
         "est__learning_rate": [0.01, 0.05, 0.1, 0.2],
         "est__min_child_samples": [5, 10, 20],
         "est__subsample":    [0.6, 0.8, 1.0]},
        4, 20
    ),
    "XGBoost": (
        make_tree_pipeline(xgb.XGBRegressor(
            device="cuda", verbosity=0, random_state=SEED, n_jobs=1)),
        {"est__n_estimators":  [100, 200, 300, 500],
         "est__max_depth":     [3, 5, 7],
         "est__learning_rate": [0.01, 0.05, 0.1, 0.2],
         "est__subsample":     [0.6, 0.8, 1.0],
         "est__colsample_bytree": [0.6, 0.8, 1.0]},
        4, 20
    ),
}

# ── checkpoint helpers ────────────────────────────────────────────────────────
CKPT_PATH = RES_DIR / "step04_checkpoint.pkl"

def save_checkpoint(oof_preds, all_metrics, best_params_all):
    with open(CKPT_PATH, "wb") as f:
        pickle.dump({"oof_preds": oof_preds, "all_metrics": all_metrics,
                     "best_params_all": best_params_all}, f)
    log.info(f"  [CKPT] checkpoint saved → {CKPT_PATH}")

def load_checkpoint():
    if CKPT_PATH.exists():
        with open(CKPT_PATH, "rb") as f:
            d = pickle.load(f)
        log.info(f"  [CKPT] loaded checkpoint — completed: {list(d['oof_preds'].keys())}")
        return d["oof_preds"], d["all_metrics"], d["best_params_all"]
    return {}, [], {}

# ── nested CV loop ────────────────────────────────────────────────────────────
oof_preds, all_metrics, best_params_all = load_checkpoint()

for mname, (pipeline, param_dist, gs_njobs, n_iter) in MODELS.items():
    if mname in oof_preds:
        log.info(f"\n── {mname} — skipped (already in checkpoint)")
        continue

    log.info(f"\n── {mname} ──────────────────────────────────")
    t0 = time.time()
    oof = np.zeros(n, dtype=np.float32)
    best_params_folds = []

    for fold, (tr_idx, te_idx) in enumerate(outer_splits):
        Xtr, Xte = X_multi[tr_idx], X_multi[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]

        if n_iter > 0:
            gs = RandomizedSearchCV(pipeline, param_dist, n_iter=n_iter,
                                    cv=inner_cv, scoring="r2",
                                    n_jobs=gs_njobs, refit=True,
                                    random_state=SEED + fold)
        else:
            gs = GridSearchCV(pipeline, param_dist, cv=inner_cv,
                              scoring="r2", n_jobs=gs_njobs, refit=True)
        gs.fit(Xtr, ytr)
        oof[te_idx] = gs.predict(Xte).astype(np.float32)
        best_params_folds.append(gs.best_params_)

        r_fold, _ = stats.pearsonr(yte, oof[te_idx])
        log.info(f"  Fold {fold+1}/5 | best_params={gs.best_params_} | r={r_fold:.4f}  "
                 f"[cpu {get_cpu_temp():.1f}°C]")

    oof_preds[mname] = oof
    met = compute_metrics(y, oof, label=mname)
    all_metrics.append(met)
    best_params_all[mname] = best_params_folds
    save_checkpoint(oof_preds, all_metrics, best_params_all)

    elapsed = time.time() - t0
    log.info(f"  OOF Pearson r = {met['Pearson_r']:.4f}  "
             f"Spearman ρ = {met['Spearman_rho']:.4f}  "
             f"RMSE = {met['RMSE']:.4f}  R² = {met['R2']:.4f}  "
             f"[{elapsed:.0f}s]")

# ── stacking meta-ensemble ────────────────────────────────────────────────────
log.info("\n── Stacking Meta-Ensemble ──────────────────")
meta_X = np.column_stack([oof_preds[m] for m in MODELS]).astype(np.float32)

# Evaluate meta-learner via leave-one-out CV for honest estimate
from sklearn.model_selection import cross_val_predict
meta_learner = Ridge(alpha=1.0)
meta_oof = cross_val_predict(
    meta_learner, meta_X, y,
    cv=KFold(n_splits=10, shuffle=True, random_state=SEED)
)
meta_met = compute_metrics(y, meta_oof, label="Stack (Ridge meta)")
all_metrics.append(meta_met)
log.info(f"  Stack OOF Pearson r = {meta_met['Pearson_r']:.4f}  "
         f"Spearman ρ = {meta_met['Spearman_rho']:.4f}  "
         f"RMSE = {meta_met['RMSE']:.4f}  R² = {meta_met['R2']:.4f}")

# ── results table ─────────────────────────────────────────────────────────────
df_res = pd.DataFrame(all_metrics)
df_res = df_res.sort_values("Pearson_r", ascending=False).reset_index(drop=True)

log.info("\n" + "=" * 70)
log.info("RESULTS SUMMARY (sorted by Pearson r)")
log.info("=" * 70)
log.info("\n" + df_res.to_string(index=False))

res_csv = RES_DIR / "step04_results.csv"
df_res.to_csv(res_csv, index=False)
log.info(f"\nResults saved → {res_csv}")

best_row   = df_res.iloc[0]
best_model = best_row["label"]
best_r     = best_row["Pearson_r"]
best_oof   = oof_preds.get(best_model, meta_oof if best_model == "Stack (Ridge meta)" else None)
if best_oof is None:
    best_oof = meta_oof

log.info(f"\nBest model : {best_model}  Pearson r = {best_r:.4f}")

# ── benchmark comparison ──────────────────────────────────────────────────────
BENCHMARKS = [
    ("AffiGrapher",   0.498),
    ("RLaffinity",    0.559),
    ("RLASIF",        0.666),
    ("DeepRSMA",      0.784),
    ("RSAPred",       0.830),
]
log.info("\n── Benchmark Comparison ──────────────────────────────────")
for bname, br in BENCHMARKS:
    status = "✓ BEAT" if best_r > br else "✗ below"
    log.info(f"  {status}  {bname}: {br:.3f}  (our best: {best_r:.4f})")

# ── journal-quality figures ───────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.linewidth": 1.2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
})

MODEL_COLORS = {
    "Ridge":    "#4C72B0", "ElasticNet": "#55A868", "SVR":  "#C44E52",
    "GBR":      "#8172B2", "LightGBM":  "#CCB974", "XGBoost": "#DD8452",
    "Stack (Ridge meta)": "#2D2D2D",
}
BENCH_COLOR = "#AAAAAA"

# ── Figure 1: Pred vs Obs + residual for best model ──────────────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(16, 7))
fig1.patch.set_facecolor("white")
fig1.suptitle(f"RNA-PDFL  |  Step 4: Best Model — {best_model}  (r = {best_r:.4f})",
              fontsize=15, fontweight="bold")

ax = axes1[0]
slope, inter, r_val, p_val, _ = stats.linregress(best_oof, y)
xr = np.linspace(best_oof.min() - 0.3, best_oof.max() + 0.3, 200)
sc = ax.scatter(best_oof, y, c=y, cmap="coolwarm", s=35, alpha=0.78,
                edgecolors="white", linewidths=0.4, zorder=3)
ax.plot(xr, slope*xr+inter, color="black", linewidth=2, zorder=4,
        label=f"Pearson r = {best_r:.4f}\nSpearman ρ = {best_row['Spearman_rho']:.4f}\n"
              f"RMSE = {best_row['RMSE']:.4f}\nR² = {best_row['R2']:.4f}")
ax.plot([y.min()-0.5, y.max()+0.5], [y.min()-0.5, y.max()+0.5],
        color="gray", linestyle=":", linewidth=1.2, label="y = x (ideal)")
plt.colorbar(sc, ax=ax, label="Observed pKd", shrink=0.85)
ax.set_xlabel("Predicted pKd", fontsize=12)
ax.set_ylabel("Observed pKd", fontsize=12)
ax.set_title("A  |  Predicted vs Observed", fontsize=13, fontweight="bold", loc="left")
ax.legend(fontsize=10, framealpha=0.85)
ax.grid(alpha=0.3, linestyle="--")

ax = axes1[1]
residuals = y - best_oof
ax.scatter(best_oof, residuals, c=np.abs(residuals), cmap="RdYlGn_r",
           s=35, alpha=0.78, edgecolors="white", linewidths=0.4)
ax.axhline(0, color="black", linewidth=1.5)
ax.axhline( residuals.std(), color="crimson", linestyle="--", linewidth=1.2,
            label=f"+1σ = {residuals.std():.3f}")
ax.axhline(-residuals.std(), color="crimson", linestyle="--", linewidth=1.2,
            label=f"−1σ")
ax.set_xlabel("Predicted pKd", fontsize=12)
ax.set_ylabel("Residual (obs − pred)", fontsize=12)
ax.set_title("B  |  Residual Plot", fontsize=13, fontweight="bold", loc="left")
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle="--")

plt.tight_layout(rect=[0, 0, 1, 0.95])
p1 = FIG_DIR / "step04_best_model_scatter.png"
plt.savefig(p1, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
log.info(f"  Figure saved → {p1}")

# ── Figure 2: All-model comparison + benchmarks ───────────────────────────────
fig2 = plt.figure(figsize=(20, 12))
fig2.patch.set_facecolor("white")
gs2  = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.45, wspace=0.38)
fig2.suptitle("RNA-PDFL  |  Step 4: Model Performance vs Benchmarks",
              fontsize=16, fontweight="bold", y=0.98)

# Panel A: Pearson r comparison bar chart
ax = fig2.add_subplot(gs2[0, :])
labels_all  = [r["label"] for r in all_metrics]
pearson_all = [r["Pearson_r"] for r in all_metrics]
sort_order  = np.argsort(pearson_all)[::-1]
labels_s    = [labels_all[i] for i in sort_order]
pearson_s   = [pearson_all[i] for i in sort_order]
colors_s    = [MODEL_COLORS.get(labels_s[i], "#888") for i in range(len(labels_s))]

x_pos = np.arange(len(labels_s))
bars  = ax.bar(x_pos, pearson_s, color=colors_s, alpha=0.88,
               edgecolor="white", linewidth=0.8, width=0.6)

# Benchmark horizontal lines
bench_styles = ["-.", "--", "-", ":", "-."]
bench_cols   = ["#4D4D4D","#888","#333","#AAA","#666"]
for (bname, br), ls, bc in zip(BENCHMARKS, bench_styles, bench_cols):
    ax.axhline(br, linestyle=ls, color=bc, linewidth=1.6, alpha=0.85,
               label=f"{bname} ({br:.3f})")

ax.set_xticks(x_pos)
ax.set_xticklabels(labels_s, fontsize=11)
ax.set_ylabel("Pearson r (OOF)", fontsize=12)
ax.set_title("A  |  RNA-PDFL Model Performance vs Published Benchmarks",
             fontsize=13, fontweight="bold", loc="left")
ax.set_ylim(0, max(max(pearson_all), 0.84) * 1.08)
for i, v in enumerate(pearson_s):
    ax.text(i, v + 0.008, f"{v:.4f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold")
ax.legend(fontsize=9.5, ncol=3, loc="upper right", framealpha=0.9)
ax.grid(axis="y", alpha=0.3, linestyle="--")

# Panel B: All metrics heatmap
ax = fig2.add_subplot(gs2[1, 0])
metrics_mat = np.zeros((len(all_metrics), 5))
for i, r in enumerate(all_metrics):
    metrics_mat[i] = [r["Pearson_r"], r["Spearman_rho"],
                      1-r["RMSE"]/2.0,   # normalise for display
                      1-r["MAE"]/2.0,
                      max(0, r["R2"])]
cmap_m = LinearSegmentedColormap.from_list("met", ["#FFF5F0","#FC8D59","#7F0000"])
im = ax.imshow(metrics_mat, aspect="auto", cmap=cmap_m, vmin=0, vmax=1)
for i in range(len(all_metrics)):
    for j,v in enumerate(metrics_mat[i]):
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                fontsize=8.5, color="white" if v > 0.6 else "black", fontweight="bold")
ax.set_xticks(range(5))
ax.set_xticklabels(["Pearson r","Spearman ρ","1-RMSE/2","1-MAE/2","R²"], fontsize=9, rotation=30)
ax.set_yticks(range(len(all_metrics)))
ax.set_yticklabels([r["label"] for r in all_metrics], fontsize=9)
ax.set_title("B  |  Multi-Metric Comparison", fontsize=12, fontweight="bold", loc="left")
plt.colorbar(im, ax=ax, shrink=0.85)

# Panel C: Pred vs obs for all models (small multiples)
ax = fig2.add_subplot(gs2[1, 1])
for mname, oof in oof_preds.items():
    r_m, _ = stats.pearsonr(y, oof)
    ax.scatter(oof, y, alpha=0.25, s=12,
               color=MODEL_COLORS.get(mname, "#888"), label=f"{mname} r={r_m:.3f}")
# Stack
r_s, _ = stats.pearsonr(y, meta_oof)
ax.scatter(meta_oof, y, alpha=0.45, s=16, marker="D",
           color=MODEL_COLORS["Stack (Ridge meta)"],
           label=f"Stack r={r_s:.3f}")
ax.plot([y.min()-0.5, y.max()+0.5],[y.min()-0.5, y.max()+0.5],
        color="gray", linestyle=":", linewidth=1.2)
ax.set_xlabel("Predicted pKd", fontsize=11)
ax.set_ylabel("Observed pKd", fontsize=11)
ax.set_title("C  |  Pred vs Obs — All Models", fontsize=12, fontweight="bold", loc="left")
ax.legend(fontsize=7.5, ncol=2, framealpha=0.85)
ax.grid(alpha=0.3, linestyle="--")

plt.tight_layout(rect=[0, 0, 1, 0.96])
p2 = FIG_DIR / "step04_all_models_comparison.png"
plt.savefig(p2, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
log.info(f"  Figure saved → {p2}")

# ── Figure 3: LightGBM feature importance (top 30) ───────────────────────────
# Retrain LightGBM on full dataset with best params to get feature importance
log.info("\nRetraining LightGBM on full dataset for feature importance ...")
lgb_vt   = VarianceThreshold(threshold=1e-4).fit(X_multi)
X_vt     = lgb_vt.transform(X_multi)
# Use median best params across folds
lgb_bp   = best_params_all["LightGBM"]
from collections import Counter
best_ne  = Counter([p["est__n_estimators"]  for p in lgb_bp]).most_common(1)[0][0]
best_nl  = Counter([p["est__num_leaves"]     for p in lgb_bp]).most_common(1)[0][0]
best_lr  = Counter([p["est__learning_rate"]  for p in lgb_bp]).most_common(1)[0][0]
best_mc  = Counter([p["est__min_child_samples"] for p in lgb_bp]).most_common(1)[0][0]
best_ss  = Counter([p["est__subsample"]      for p in lgb_bp]).most_common(1)[0][0]

lgb_full = lgb.LGBMRegressor(
    n_estimators=best_ne, num_leaves=best_nl, learning_rate=best_lr,
    min_child_samples=best_mc, subsample=best_ss,
    device="gpu", verbose=-1, random_state=SEED, n_jobs=1
)
lgb_full.fit(X_vt, y)
fi = lgb_full.feature_importances_

# Map back to feature names (multi-scale: eta_label + pair + tau + stat)
eta_names = ["η=2","η=5","η=8"]
rna_e = RNA_ELEMENTS; lig_e = LIG_ELEMENTS
stat_names = ["min","max","mean","med","var","std","sum","sum²","rank","β"]
lap_names  = ["L₀","L₁"]
feat_names = []
for eta_lbl in eta_names:
    for r_el in rna_e:
        for l_el in lig_e:
            for tau in THRESHOLDS:
                for lap in lap_names:
                    for sn in stat_names:
                        feat_names.append(f"{eta_lbl}|{r_el}-{l_el}|τ={tau}|{lap}|{sn}")

vt_feat_names = np.array(feat_names)[lgb_vt.get_support()]
top30_idx   = np.argsort(fi)[::-1][:30]
top30_imp   = fi[top30_idx]
top30_names = [vt_feat_names[i] if i < len(vt_feat_names) else f"feat_{i}"
               for i in top30_idx]

fig3, ax3 = plt.subplots(figsize=(14, 10))
fig3.patch.set_facecolor("white")
# Colour by η level
bar_cols = []
for nm in top30_names:
    if "η=2" in nm:   bar_cols.append("#D6604D")
    elif "η=5" in nm: bar_cols.append("#4393C3")
    else:             bar_cols.append("#4DAC26")

ax3.barh(range(30)[::-1], top30_imp, color=bar_cols, alpha=0.85, edgecolor="white")
ax3.set_yticks(range(30)[::-1])
ax3.set_yticklabels([n.replace("|"," | ") for n in top30_names], fontsize=8.5)
ax3.set_xlabel("LightGBM Feature Importance (gain)", fontsize=12)
ax3.set_title("RNA-PDFL  |  Step 4: Top 30 LightGBM Feature Importances\n"
              "(red=η=2 Å  ·  blue=η=5 Å  ·  green=η=8 Å)",
              fontsize=13, fontweight="bold")
ax3.grid(axis="x", alpha=0.3, linestyle="--")
ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
fig3.tight_layout()
p3 = FIG_DIR / "step04_lgbm_feature_importance.png"
plt.savefig(p3, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
log.info(f"  Figure saved → {p3}")

# ── final summary ─────────────────────────────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("STEP 4 COMPLETE — FINAL RESULTS")
log.info("=" * 70)
log.info(f"  Feature dim    : {X_multi.shape[1]} (3 × 3600 multi-scale)")
log.info(f"  Dataset        : n = {n}")
log.info(f"  Best model     : {best_model}")
log.info(f"  Pearson r      : {best_row['Pearson_r']:.4f}")
log.info(f"  Spearman ρ     : {best_row['Spearman_rho']:.4f}")
log.info(f"  RMSE           : {best_row['RMSE']:.4f}")
log.info(f"  MAE            : {best_row['MAE']:.4f}")
log.info(f"  R²             : {best_row['R2']:.4f}")
log.info("\n  Benchmark comparison:")
for bname, br in BENCHMARKS:
    sym = "✓ BEAT" if best_r > br else "✗ below"
    log.info(f"    {sym}  {bname}: {br:.3f}")
log.info(f"\n  Log: {log_path}")
log.info("=" * 70)
