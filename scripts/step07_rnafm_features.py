"""
SGT-RNA  ·  Step 7: RNA-FM Embeddings + Secondary Structure Features

For each complex:
  1. RNA-FM embeddings  → mean-pooled 640-dim vector per complex
  2. ViennaRNA secondary structure features:
       MFE, base-pair count, unpaired fraction, stem fraction,
       mean base-pair probability, ensemble diversity (8 features)
  3. k-mer composition (1/2/3-mer): 4 + 16 + 64 = 84 features

Combined with Step 5 SGT features (38,064) → retrain per-subtype Ridge
"""

import logging, warnings, time
from pathlib import Path
from datetime import datetime
from itertools import product
from collections import Counter

import numpy as np
import pandas as pd
import torch
from multimolecule import RnaFmModel, RnaTokenizer
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, LeaveOneOut, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import RNA   # ViennaRNA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
NPZ_S5    = ROOT / "data" / "features" / "step05_expanded_features.npz"
SEQ_CSV   = ROOT / "data" / "affinity" / "rna_sequences.csv"
LABELS_CSV = ROOT / "results" / "step06_subtype_labels.csv"
OUT_DIR   = ROOT / "data" / "features"
RES_DIR   = ROOT / "results"
FIG_DIR   = ROOT / "results" / "figures"
LOG_DIR   = ROOT / "results" / "logs"
for d in [OUT_DIR, RES_DIR, FIG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── logging ───────────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"step07_{ts}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("SGT-RNA  ·  Step 7: RNA-FM + Secondary Structure + k-mer Features")
log.info("=" * 70)

SEED   = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"  Device: {DEVICE}")

# ── load sequences and subtype labels ─────────────────────────────────────────
df_seq    = pd.read_csv(SEQ_CSV).set_index("pdb")
df_labels = pd.read_csv(LABELS_CSV).set_index("pdb")
d5        = np.load(NPZ_S5, allow_pickle=True)
X_sgt    = d5["X"].astype(np.float32)
y         = d5["y"].astype(np.float32)
ids       = [str(i) for i in d5["ids"]]
n         = len(y)
subtypes  = np.array([df_labels.loc[pid, "subtype"] if pid in df_labels.index
                      else "other" for pid in ids])

# ── 1. RNA-FM embeddings (via multimolecule) ──────────────────────────────────
log.info("\nLoading RNA-FM via multimolecule ...")
RNA_FM_DIM = 640
X_rnafm    = np.zeros((n, RNA_FM_DIM), dtype=np.float32)

try:
    tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnafm")
    rnafm_model = RnaFmModel.from_pretrained("multimolecule/rnafm")
    rnafm_model = rnafm_model.to(DEVICE).eval()
    log.info("  RNA-FM loaded OK")

    with torch.no_grad():
        for idx, pdb_id in enumerate(ids):
            seq = df_seq.loc[pdb_id, "rna_sequence"] if pdb_id in df_seq.index else ""
            seq = str(seq).strip().upper().replace("T", "U")
            seq = "".join(c for c in seq if c in "AUGC")
            if len(seq) < 3:
                continue
            seq = seq[:510]   # RNA-FM max tokens = 512 incl. special tokens
            inputs = tokenizer(seq, return_tensors="pt").to(DEVICE)
            out    = rnafm_model(**inputs)
            # mean-pool over residue tokens (exclude [CLS] and [EOS])
            emb = out.last_hidden_state[0, 1:-1].mean(0)
            X_rnafm[idx] = emb.cpu().numpy()
            if (idx + 1) % 20 == 0:
                log.info(f"    RNA-FM: {idx+1}/{n} done")

    non_zero = (X_rnafm.sum(1) != 0).sum()
    log.info(f"  RNA-FM done — {non_zero}/{n} non-zero embeddings")

except Exception as e:
    log.warning(f"  RNA-FM failed: {e} — using zeros")

# ── 2. ViennaRNA secondary structure features ─────────────────────────────────
log.info("\nComputing ViennaRNA secondary structure features ...")
SS_DIM  = 8
X_ss    = np.zeros((n, SS_DIM), dtype=np.float32)

for idx, pdb_id in enumerate(ids):
    seq = df_seq.loc[pdb_id, "rna_sequence"] if pdb_id in df_seq.index else ""
    seq = str(seq).strip().upper().replace("T", "U")
    seq = "".join(c for c in seq if c in "AUGC")
    if len(seq) < 5:
        continue
    try:
        # MFE structure
        struct, mfe = RNA.fold(seq)
        n_bp        = struct.count("(")
        n_unp       = struct.count(".")
        n_total     = len(struct)
        frac_bp     = n_bp / n_total
        frac_unp    = n_unp / n_total

        # Partition function for base-pair probabilities
        RNA.pf_fold(seq)
        bppm = RNA.get_pr(seq, struct)  # mean bp prob (approx)

        # Ensemble diversity via McCaskill model
        fc   = RNA.fold_compound(seq)
        fc.pf()
        ed   = fc.mean_bp_distance()

        # GC content
        gc   = (seq.count("G") + seq.count("C")) / len(seq)

        X_ss[idx] = [mfe, n_bp, n_unp, frac_bp, frac_unp,
                     float(bppm) if bppm else 0.0, ed, gc]
    except Exception:
        try:
            # fallback: MFE only
            struct, mfe = RNA.fold(seq)
            n_bp  = struct.count("(")
            frac_bp = n_bp / len(struct) if struct else 0.0
            frac_unp = struct.count(".") / len(struct) if struct else 0.0
            gc = (seq.count("G") + seq.count("C")) / len(seq)
            X_ss[idx] = [mfe, n_bp, len(struct)-n_bp,
                         frac_bp, frac_unp, 0.0, 0.0, gc]
        except Exception:
            pass

log.info(f"  Secondary structure features shape: {X_ss.shape}")

# ── 3. k-mer composition ───────────────────────────────────────────────────────
log.info("\nComputing k-mer features (1,2,3-mer) ...")
NTS = ["A", "U", "G", "C"]
kmers_1 = NTS
kmers_2 = ["".join(p) for p in product(NTS, repeat=2)]
kmers_3 = ["".join(p) for p in product(NTS, repeat=3)]
ALL_KMERS = kmers_1 + kmers_2 + kmers_3   # 84 total
KMER_DIM  = len(ALL_KMERS)

X_kmer = np.zeros((n, KMER_DIM), dtype=np.float32)
for idx, pdb_id in enumerate(ids):
    seq = df_seq.loc[pdb_id, "rna_sequence"] if pdb_id in df_seq.index else ""
    seq = str(seq).strip().upper().replace("T", "U")
    seq = "".join(c for c in seq if c in "AUGC")
    if len(seq) < 1:
        continue
    vec = []
    for k, kmer_list in [(1, kmers_1), (2, kmers_2), (3, kmers_3)]:
        total = max(len(seq) - k + 1, 1)
        cnt   = Counter(seq[i:i+k] for i in range(len(seq)-k+1))
        vec.extend([cnt.get(km, 0) / total for km in kmer_list])
    X_kmer[idx] = vec

log.info(f"  k-mer features shape: {X_kmer.shape}")

# ── combine all features ───────────────────────────────────────────────────────
X_all = np.concatenate([X_sgt, X_rnafm, X_ss, X_kmer], axis=1)
log.info(f"\nFull feature matrix: {X_all.shape}")
log.info(f"  SGT (Step 5):  {X_sgt.shape[1]}")
log.info(f"  RNA-FM:         {X_rnafm.shape[1]}")
log.info(f"  Secondary str.: {X_ss.shape[1]}")
log.info(f"  k-mer (84):     {X_kmer.shape[1]}")

npz_path = OUT_DIR / "step07_full_features.npz"
np.savez_compressed(npz_path, X=X_all, y=y, ids=np.array(ids),
                    subtypes=subtypes)
log.info(f"  Saved → {npz_path}")

# ── subtype-specific nested CV ────────────────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("SUBTYPE-SPECIFIC RIDGE — SGT + RNA-FM + SS + k-mer")
log.info("=" * 70)

ALPHA_GRID = {"est__alpha": [1, 10, 100, 1000, 10000, 100000]}

def make_pipe():
    return Pipeline([
        ("vt",  VarianceThreshold(threshold=1e-4)),
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=SEED)),
        ("est", Ridge()),
    ])

def run_cv(X_sub, y_sub, label):
    n_sub = len(y_sub)
    if n_sub < 5:
        log.info(f"  [{label}] n={n_sub} — too small, skip")
        return None, None

    cv_inner = KFold(n_splits=5, shuffle=True, random_state=SEED)
    if n_sub < 15:
        outer = LeaveOneOut()
        cv_name = "LOO"
    else:
        outer = KFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_name = "5-fold"

    oof = np.zeros(n_sub, dtype=np.float32)
    for tr_idx, te_idx in outer.split(X_sub):
        n_inner = min(5, len(tr_idx))
        gs = GridSearchCV(make_pipe(), ALPHA_GRID,
                          cv=n_inner, scoring="r2", n_jobs=10, refit=True)
        gs.fit(X_sub[tr_idx], y_sub[tr_idx])
        oof[te_idx] = gs.predict(X_sub[te_idx])

    r,   _ = stats.pearsonr(y_sub, oof)
    rho, _ = stats.spearmanr(y_sub, oof)
    rmse   = float(np.sqrt(mean_squared_error(y_sub, oof)))
    mae    = float(mean_absolute_error(y_sub, oof))
    r2     = float(r2_score(y_sub, oof))
    log.info(f"  [{label}] n={n_sub:3d} | {cv_name} | r={r:.4f} | ρ={rho:.4f} | "
             f"RMSE={rmse:.4f} | R²={r2:.4f}")
    return oof, {"subtype": label, "n": n_sub, "Pearson_r": round(r,4),
                 "Spearman_rho": round(rho,4), "RMSE": round(rmse,4),
                 "MAE": round(mae,4), "R2": round(r2,4)}

subtype_counts = Counter(subtypes)
all_oof   = np.zeros(n, dtype=np.float32)
all_valid = np.zeros(n, dtype=bool)
results   = []
t0 = time.time()

for st in sorted(subtype_counts.keys()):
    mask  = subtypes == st
    idx   = np.where(mask)[0]
    oof_sub, met = run_cv(X_all[idx], y[idx], st)
    if oof_sub is not None and met is not None:
        all_oof[idx]   = oof_sub
        all_valid[idx] = True
        results.append(met)

# ── overall metrics ───────────────────────────────────────────────────────────
y_v   = y[all_valid]
oof_v = all_oof[all_valid]
r_all,   _ = stats.pearsonr(y_v, oof_v)
rho_all, _ = stats.spearmanr(y_v, oof_v)
rmse_all   = float(np.sqrt(mean_squared_error(y_v, oof_v)))
mae_all    = float(mean_absolute_error(y_v, oof_v))
r2_all     = float(r2_score(y_v, oof_v))

log.info(f"\n{'='*70}")
log.info("COMBINED OOF (all subtypes)")
log.info(f"{'='*70}")
log.info(f"  n evaluated    : {all_valid.sum()}/{n}")
log.info(f"  Pearson r      : {r_all:.4f}")
log.info(f"  Spearman ρ     : {rho_all:.4f}")
log.info(f"  RMSE           : {rmse_all:.4f}")
log.info(f"  MAE            : {mae_all:.4f}")
log.info(f"  R²             : {r2_all:.4f}")
log.info(f"  Time           : {time.time()-t0:.0f}s")

for step, sr in [("Step 4 global Ridge", 0.5029),
                  ("Step 5 expanded Ridge", 0.5165),
                  ("Step 6 subtype Ridge", 0.4691)]:
    log.info(f"  {step}: r = {sr:.4f}  Δ = {r_all-sr:+.4f}")

BENCHMARKS = [("AffiGrapher",0.498),("RLaffinity",0.559),
               ("RLASIF",0.666),("DeepRSMA",0.784),("RSAPred",0.830)]
log.info("\n  Benchmark comparison:")
for bname, br in BENCHMARKS:
    sym = "✓ BEAT" if r_all > br else "✗ below"
    log.info(f"    {sym}  {bname}: {br:.3f}")

# ── save + figure ─────────────────────────────────────────────────────────────
df_res = pd.DataFrame(results)
df_res.to_csv(RES_DIR / "step07_results.csv", index=False)

COLORS = {"riboswitch":"#4C72B0","aptamer":"#55A868",
          "ribosomal_asite":"#C44E52","viral_tar":"#DD8452","other":"#8172B2"}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor("white")
fig.suptitle(f"SGT-RNA  |  Step 7: + RNA-FM + SS + k-mer  (r = {r_all:.4f})",
             fontsize=15, fontweight="bold")

ax = axes[0]
for st in sorted(subtype_counts.keys()):
    mask = (subtypes == st) & all_valid
    if mask.sum() == 0: continue
    ax.scatter(all_oof[mask], y[mask], label=f"{st} (n={mask.sum()})",
               color=COLORS.get(st,"#888"), s=40, alpha=0.8,
               edgecolors="white", linewidths=0.3)
slope, inter, *_ = stats.linregress(oof_v, y_v)
xr = np.linspace(oof_v.min()-.3, oof_v.max()+.3, 200)
ax.plot(xr, slope*xr+inter, "k-", lw=2, label=f"r={r_all:.4f}")
ax.plot([y.min()-.5,y.max()+.5],[y.min()-.5,y.max()+.5],":",color="gray",lw=1.2)
ax.set_xlabel("Predicted pKd"); ax.set_ylabel("Observed pKd")
ax.set_title("A  |  Pred vs Obs", fontweight="bold", loc="left")
ax.legend(fontsize=8.5); ax.grid(alpha=0.3, ls="--")

ax = axes[1]
if results:
    df_p = pd.DataFrame(results).sort_values("Pearson_r", ascending=False)
    cols = [COLORS.get(r,"#888") for r in df_p["subtype"]]
    bars = ax.bar(df_p["subtype"], df_p["Pearson_r"], color=cols,
                  alpha=0.85, edgecolor="white")
    for bar, rv in zip(bars, df_p["Pearson_r"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{rv:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    for label, val, ls, col in [
        ("Step 5 global", 0.5165, "--", "black"),
        ("Step 6 subtype", 0.4691, "-.", "gray"),
        ("Step 7 combined", r_all, "-", "red"),
    ]:
        ax.axhline(val, ls=ls, color=col, lw=1.5, label=f"{label} ({val:.4f})")
    ax.set_ylabel("Pearson r (OOF)"); ax.set_ylim(0, 1.05)
    ax.set_title("B  |  Per-Subtype Performance", fontweight="bold", loc="left")
    ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=0.3, ls="--")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

plt.tight_layout(rect=[0,0,1,0.95])
fig_path = FIG_DIR / "step07_rnafm_results.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
log.info(f"\n  Figure saved → {fig_path}")
log.info("\n" + "="*70)
log.info("STEP 7 COMPLETE")
log.info("="*70)
