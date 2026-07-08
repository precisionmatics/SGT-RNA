"""
RNA-PDFL  ·  Step 6: Subtype-Specific Ridge Models

Strategy:
  1. Classify 143 complexes into RNA subtypes using NL2020 index descriptions
     Categories: riboswitch, aptamer, ribosomal_asite, viral_tar, other
  2. For each subtype ≥ 8 samples: nested CV Ridge (5-fold or LOO)
  3. Combine OOF predictions → overall Pearson r
  4. Compare vs Step 5 global Ridge (r = 0.5165)
"""

import re, logging, warnings, time
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
from sklearn.model_selection import KFold, LeaveOneOut, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path("/home/stalin/Desktop/PDFL-RNA/RNA_PDFL")
NPZ_S5   = ROOT / "data" / "features" / "step05_expanded_features.npz"
INDEX_FILE = Path("/run/media/stalin/PortableSSD/ML_Projects/CAML_RNA/data/raw/NL/index/INDEX_general_NL.2020")
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
        logging.FileHandler(LOG_DIR / f"step06_{ts}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("RNA-PDFL  ·  Step 6: Subtype-Specific Ridge Models")
log.info("=" * 70)

SEED = 42

# ── subtype classification ────────────────────────────────────────────────────
RIBOSWITCH_KW = [
    "riboswitch", "Riboswitch", "SAM", "SAH", "preQ1", "PreQ1",
    "c-di-GMP", "c-di-AMP", "glycine riboswitch", "folinic",
    "guanine riboswitch", "6-chloroguanine", "xanthine", "2-aminopurine",
    "c-di-dAMP", "cGAMP", "ppGpp", "PRPP", "L-glutamine", "azacytosine",
    "hypoxanthine", "tetrahydrofolate", "SAM-I", "7-deazaguanine",
    "2-fluoroadenine", "lysine", "thiamine", "6-O-methylguanine",
    "2-aminopurine", "GR RNA", "GR(C74U)", "mRNA", "riboswitch"
]
APTAMER_KW = [
    "aptamer", "APTAMER", "DFHBI", "TO1-Biotin", "TO3-Biotin",
    "Spinach", "Corn", "Mango", "MG-RNA", "GTP aptamer",
    "fluorophore", "TO1", "TO3", "ThT", "thiazole orange",
    "Biotin-binding", "5HTP", "5GR-II", "OTA aptamer",
]
VIRAL_KW = [
    "TAR RNA", "HIV-1 TAR", "TAR", "HIV", "Tat protein",
    "Argininamide binding by TAR"
]
RIBOSOMAL_KW = [
    "A-site", "A site", "A-SITE", "GENTAMICIN", "ribosomal",
    "aminoglycoside", "18S", "paromomycin", "eukaryotic decoding",
    "neomycin", "ribostamycin", "decoding region"
]
TRNA_KW   = ["tRNA"]
RIBOZYME_KW = ["ribozyme", "Ribozyme", "Diels-Alder"]
GQUAD_KW  = ["telomere", "telomerase", "G4", "quadruplex", "TTAGGGT",
              "G-quadruplex", "carbazole", "BMVC", "RHPS4"]

def classify_rna(desc):
    d = desc.lower()
    if any(k.lower() in d for k in VIRAL_KW):
        return "viral_tar"
    if any(k.lower() in d for k in RIBOSOMAL_KW):
        return "ribosomal_asite"
    if any(k.lower() in d for k in TRNA_KW):
        return "ribosomal_asite"   # group tRNA with ribosomal
    if any(k.lower() in d for k in RIBOZYME_KW):
        return "other"
    if any(k.lower() in d for k in GQUAD_KW):
        return "other"
    if any(k.lower() in d for k in RIBOSWITCH_KW):
        return "riboswitch"
    if any(k.lower() in d for k in APTAMER_KW):
        return "aptamer"
    return "other"

# ── parse NL2020 index ────────────────────────────────────────────────────────
log.info("\nParsing NL2020 index for RNA subtype classification ...")
pdb_desc = {}
with open(INDEX_FILE) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("//", 1)
        pdb_id = parts[0].split()[0].strip().lower()
        desc = parts[1].strip() if len(parts) > 1 else ""
        pdb_desc[pdb_id] = desc

# ── load features ─────────────────────────────────────────────────────────────
log.info("Loading Step 5 expanded features ...")
d5  = np.load(NPZ_S5, allow_pickle=True)
X   = d5["X"].astype(np.float32)
y   = d5["y"].astype(np.float32)
ids = [str(i) for i in d5["ids"]]
n   = len(y)
log.info(f"  X shape: {X.shape}")

# ── assign subtypes ───────────────────────────────────────────────────────────
subtypes = []
for pdb_id in ids:
    desc = pdb_desc.get(pdb_id.lower(), "")
    subtypes.append(classify_rna(desc))

subtypes = np.array(subtypes)
subtype_counts = Counter(subtypes)

log.info("\nSubtype distribution:")
for st, cnt in sorted(subtype_counts.items(), key=lambda x: -x[1]):
    log.info(f"  {st:20s}: {cnt:3d} complexes")

# Save subtype labels
df_labels = pd.DataFrame({"pdb": ids, "subtype": subtypes, "pKd": y})
df_labels.to_csv(RES_DIR / "step06_subtype_labels.csv", index=False)

# ── Ridge pipeline ────────────────────────────────────────────────────────────
ALPHA_GRID = {"est__alpha": [1, 10, 100, 1000, 10000, 100000]}

def make_pipeline():
    return Pipeline([
        ("vt",  VarianceThreshold(threshold=1e-4)),
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=SEED)),
        ("est", Ridge()),
    ])

def cv_ridge(X_sub, y_sub, label, n_jobs=10):
    n_sub = len(y_sub)
    if n_sub < 5:
        log.info(f"  [{label}] n={n_sub} — too small, skipping")
        return None, None

    if n_sub < 15:
        # LOO-CV
        loo = LeaveOneOut()
        oof = np.zeros(n_sub, dtype=np.float32)
        for tr_idx, te_idx in loo.split(X_sub):
            # simple inner grid search via 3-fold (or LOO if very small)
            cv_inner = min(3, len(tr_idx))
            gs = GridSearchCV(make_pipeline(), ALPHA_GRID,
                              cv=cv_inner, scoring="r2",
                              n_jobs=n_jobs, refit=True)
            gs.fit(X_sub[tr_idx], y_sub[tr_idx])
            oof[te_idx] = gs.predict(X_sub[te_idx])
        cv_name = "LOO"
    else:
        # 5-fold CV
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
        oof = np.zeros(n_sub, dtype=np.float32)
        for tr_idx, te_idx in kf.split(X_sub):
            gs = GridSearchCV(make_pipeline(), ALPHA_GRID,
                              cv=inner_cv, scoring="r2",
                              n_jobs=n_jobs, refit=True)
            gs.fit(X_sub[tr_idx], y_sub[tr_idx])
            oof[te_idx] = gs.predict(X_sub[te_idx])
        cv_name = "5-fold"

    if len(np.unique(y_sub)) < 2 or len(oof) < 3:
        return oof, None

    r, _   = stats.pearsonr(y_sub, oof)
    rho, _ = stats.spearmanr(y_sub, oof)
    rmse   = float(np.sqrt(mean_squared_error(y_sub, oof)))
    mae    = float(mean_absolute_error(y_sub, oof))
    r2     = float(r2_score(y_sub, oof))
    log.info(f"  [{label}] n={n_sub:3d} | {cv_name} | r={r:.4f} | ρ={rho:.4f} | "
             f"RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}")
    return oof, {"subtype": label, "n": n_sub, "cv": cv_name,
                 "Pearson_r": round(r, 4), "Spearman_rho": round(rho, 4),
                 "RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}

# ── run subtype-specific CV ───────────────────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("SUBTYPE-SPECIFIC NESTED CV")
log.info("=" * 70)

all_oof   = np.zeros(n, dtype=np.float32)
all_valid = np.zeros(n, dtype=bool)
results   = []
t0 = time.time()

for st in sorted(subtype_counts.keys()):
    mask  = subtypes == st
    idx   = np.where(mask)[0]
    X_sub = X[idx]
    y_sub = y[idx]
    oof_sub, met = cv_ridge(X_sub, y_sub, st)
    if oof_sub is not None and met is not None:
        all_oof[idx]   = oof_sub
        all_valid[idx] = True
        results.append(met)

# ── overall OOF metrics ───────────────────────────────────────────────────────
y_valid   = y[all_valid]
oof_valid = all_oof[all_valid]
n_valid   = all_valid.sum()

r_all,   _ = stats.pearsonr(y_valid, oof_valid)
rho_all, _ = stats.spearmanr(y_valid, oof_valid)
rmse_all   = float(np.sqrt(mean_squared_error(y_valid, oof_valid)))
mae_all    = float(mean_absolute_error(y_valid, oof_valid))
r2_all     = float(r2_score(y_valid, oof_valid))

log.info(f"\n{'='*70}")
log.info("COMBINED OOF (all subtypes)")
log.info(f"{'='*70}")
log.info(f"  n evaluated    : {n_valid}/{n}")
log.info(f"  Pearson r      : {r_all:.4f}")
log.info(f"  Spearman ρ     : {rho_all:.4f}")
log.info(f"  RMSE           : {rmse_all:.4f}")
log.info(f"  MAE            : {mae_all:.4f}")
log.info(f"  R²             : {r2_all:.4f}")
log.info(f"  Time           : {time.time()-t0:.0f}s")

STEP5_R = 0.5165
log.info(f"\n  Step 5 global Ridge    : r = {STEP5_R:.4f}")
log.info(f"  Step 6 subtype Ridge   : r = {r_all:.4f}")
log.info(f"  Δr = {r_all - STEP5_R:+.4f}  ({'improvement' if r_all > STEP5_R else 'regression'})")

BENCHMARKS = [
    ("AffiGrapher", 0.498), ("RLaffinity", 0.559),
    ("RLASIF", 0.666), ("DeepRSMA", 0.784), ("RSAPred", 0.830),
]
log.info("\n  Benchmark comparison:")
for bname, br in BENCHMARKS:
    sym = "✓ BEAT" if r_all > br else "✗ below"
    log.info(f"    {sym}  {bname}: {br:.3f}")

# ── save results ──────────────────────────────────────────────────────────────
df_res = pd.DataFrame(results)
df_res.to_csv(RES_DIR / "step06_results.csv", index=False)
log.info(f"\n  Results saved → {RES_DIR / 'step06_results.csv'}")

# ── figures ───────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "figure.dpi": 300, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False})

SUBTYPE_COLORS = {
    "riboswitch":     "#4C72B0",
    "aptamer":        "#55A868",
    "ribosomal_asite":"#C44E52",
    "viral_tar":      "#DD8452",
    "other":          "#8172B2",
}

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.patch.set_facecolor("white")
fig.suptitle(f"RNA-PDFL  |  Step 6: Subtype-Specific Ridge Models  "
             f"(overall r = {r_all:.4f})",
             fontsize=15, fontweight="bold")

# Panel A: overall pred vs obs coloured by subtype
ax = axes[0]
for st in sorted(subtype_counts.keys()):
    mask = subtypes == st
    idx  = np.where(mask & all_valid)[0]
    if len(idx) == 0:
        continue
    ax.scatter(all_oof[idx], y[idx], label=f"{st} (n={mask.sum()})",
               color=SUBTYPE_COLORS.get(st, "#888"),
               s=40, alpha=0.8, edgecolors="white", linewidths=0.3)
slope, inter, *_ = stats.linregress(oof_valid, y_valid)
xr = np.linspace(oof_valid.min()-0.3, oof_valid.max()+0.3, 200)
ax.plot(xr, slope*xr+inter, color="black", lw=2,
        label=f"r = {r_all:.4f}")
ax.plot([y.min()-0.5, y.max()+0.5], [y.min()-0.5, y.max()+0.5],
        color="gray", ls=":", lw=1.2)
ax.set_xlabel("Predicted pKd"); ax.set_ylabel("Observed pKd")
ax.set_title("A  |  Pred vs Obs (by subtype)", fontweight="bold", loc="left")
ax.legend(fontsize=8.5, framealpha=0.85)
ax.grid(alpha=0.3, ls="--")

# Panel B: per-subtype Pearson r bar chart
ax = axes[1]
if results:
    df_plot = pd.DataFrame(results).sort_values("Pearson_r", ascending=False)
    colors  = [SUBTYPE_COLORS.get(r, "#888") for r in df_plot["subtype"]]
    bars = ax.bar(df_plot["subtype"], df_plot["Pearson_r"],
                  color=colors, alpha=0.85, edgecolor="white")
    for bar, r_val in zip(bars, df_plot["Pearson_r"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{r_val:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.axhline(STEP5_R, color="black", ls="--", lw=1.5,
               label=f"Step 5 global ({STEP5_R:.4f})")
    ax.axhline(r_all, color="red", ls="-.", lw=1.5,
               label=f"Step 6 combined ({r_all:.4f})")
    ax.set_ylabel("Pearson r (OOF)")
    ax.set_title("B  |  Per-Subtype Performance", fontweight="bold", loc="left")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3, ls="--")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

# Panel C: subtype distribution pie
ax = axes[2]
labels_pie = [f"{st}\n(n={cnt})" for st, cnt in
              sorted(subtype_counts.items(), key=lambda x: -x[1])]
sizes_pie  = [cnt for _, cnt in sorted(subtype_counts.items(), key=lambda x: -x[1])]
colors_pie = [SUBTYPE_COLORS.get(st, "#888") for st, _ in
              sorted(subtype_counts.items(), key=lambda x: -x[1])]
wedges, texts, autotexts = ax.pie(
    sizes_pie, labels=labels_pie, colors=colors_pie,
    autopct="%1.0f%%", startangle=140,
    wedgeprops=dict(edgecolor="white", linewidth=1.5),
    textprops=dict(fontsize=9)
)
for at in autotexts:
    at.set_fontsize(8); at.set_fontweight("bold")
ax.set_title("C  |  Dataset Composition", fontweight="bold", loc="left")

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig_path = FIG_DIR / "step06_subtype_results.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
log.info(f"  Figure saved → {fig_path}")

log.info("\n" + "=" * 70)
log.info("STEP 6 COMPLETE")
log.info("=" * 70)
