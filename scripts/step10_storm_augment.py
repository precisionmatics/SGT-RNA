"""
RNA-PDFL · Step 10: STORM Database Augmentation

Adds 900 STORM entries (confidence_score > 0.7) with experimental pKd
to expand the training set from NL2020's 143 complexes.

For STORM entries we compute the same non-PDFL features as NL2020:
  - Morgan ECFP4 2048-bit + MACCS 167-bit     (from SMILES)
  - Ligand physico: MW, n_rings, HBD, HBA, RotB, TPSA  (from SMILES)
  - Nucleotide composition: 10 features         (from RNA sequence)
  - k-mer 1/2/3: 84 features                   (from RNA sequence)
  - ViennaRNA SS: MFE, base pairs, fractions, diversity (from RNA seq)
  - RNA-FM embeddings: 640-dim                  (from RNA sequence)

Total non-PDFL feature vector: 2,963 features per complex
(Same slice that exists in NL2020's step09 NPZ: columns 36000–38963)

Combined dataset: NL2020 (143) + STORM (900) = 1,043 complexes
Train per-subtype Ridge on combined data using non-PDFL features.
Also train stacked ensemble: combined-model OOF + NL2020-only PDFL model.
"""

import logging, warnings, time
from pathlib import Path
from datetime import datetime
from collections import Counter
from itertools import product

import numpy as np
import pandas as pd
import torch
from multimolecule import RnaFmModel, RnaTokenizer
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from scipy import stats

import RNA  # ViennaRNA

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
PDFL_ROOT  = Path("/home/stalin/Desktop/PDFL-RNA/RNA_PDFL")
STORM_BASE = Path("/home/stalin/Desktop/RNA_Database/Dataset/Boltz2_Results")
CSV_DIR    = Path("/home/stalin/Desktop/RNA_Database/Dataset/CSV_Files")

NPZ_S9     = PDFL_ROOT / "data" / "features" / "step09_full_features.npz"
OUT_NPZ    = PDFL_ROOT / "data" / "features" / "step10_storm_features.npz"
RES_DIR    = PDFL_ROOT / "results"
FIG_DIR    = PDFL_ROOT / "results" / "figures"
LOG_DIR    = PDFL_ROOT / "results" / "logs"
for d in [RES_DIR, FIG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── logging ───────────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"step10_{ts}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("RNA-PDFL · Step 10: STORM Database Augmentation")
log.info("=" * 70)

SEED   = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"  Device: {DEVICE}")
np.random.seed(SEED)

# ── subtype map: STORM RNA type → our subtype labels ──────────────────────────
STORM_SUBTYPE = {
    "aptamers":   "aptamer",
    "riboswitch": "riboswitch",
    "viral_rna":  "viral_tar",
    "mirna":      "other_misc",
    "ribosomal":  "ribosomal_asite",
    "repeats":    "other_misc",
}
CONF_THRESH = 0.7
RNA_MAX_LEN = 200

# ── build STORM master table ──────────────────────────────────────────────────
log.info("\nBuilding STORM master table ...")
all_pkd = pd.read_csv(CSV_DIR / "all_pkd_extracted.csv")
frames  = []

for rna_type, subtype in STORM_SUBTYPE.items():
    pkd = (all_pkd[all_pkd["rna_type"] == rna_type]
           [["Entry_ID", "Molecule_ID", "SMILES", "Target_RNA_sequence", "pKd"]]
           .dropna(subset=["pKd"]))

    ymap = pd.read_csv(STORM_BASE / rna_type / "results" / "yaml_map.csv")
    tsv  = pd.read_csv(STORM_BASE / rna_type / "results" /
                       f"{rna_type}_boltz2_all.tsv", sep="\t")

    merged = (pkd
              .merge(ymap[["yaml_id", "Molecule_ID"]], on="Molecule_ID", how="inner")
              .merge(tsv[["yaml_id", "confidence_score", "RNA_sequence"]],
                     on="yaml_id", how="inner"))

    cpx_dir = STORM_BASE / rna_type / "results" / "complexes"
    existing = set(p.name for p in cpx_dir.glob("*.pdb"))
    merged["pdb_path"] = merged["yaml_id"].apply(
        lambda x: str(cpx_dir / f"{x}.pdb"))
    merged = merged[merged["yaml_id"].apply(lambda x: f"{x}.pdb" in existing)]
    merged = merged.drop_duplicates("Entry_ID")

    rna_len = merged["RNA_sequence"].apply(lambda s: len(str(s)))
    merged  = merged[(merged["confidence_score"] >= CONF_THRESH) &
                     (rna_len <= RNA_MAX_LEN)]
    merged["subtype"]  = subtype
    merged["rna_type"] = rna_type
    frames.append(merged)
    log.info(f"  {rna_type:<15}: {len(merged):4d} entries")

storm_df = pd.concat(frames, ignore_index=True)
log.info(f"  Total STORM entries: {len(storm_df)}")

# ── load NL2020 step09 features and strip PDFL ────────────────────────────────
log.info("\nLoading NL2020 step09 features ...")
d9      = np.load(NPZ_S9, allow_pickle=True)
X9_full = d9["X"].astype(np.float32)          # (143, 38963)
y_nl    = d9["y"].astype(np.float32)
ids_nl  = [str(i) for i in d9["ids"]]
sub_nl  = list(d9["subtypes"])

# Feature layout of step09 NPZ (38963 total):
# [0:36000]   PDFL (36000)
# [36000:38048] Morgan ECFP4 (2048)
# [38048:38058] nuc composition (10)
# [38058:38064] ligand physico (6)
# [38064:38704] RNA-FM (640)
# [38704:38712] ViennaRNA SS (8)
# [38712:38796] k-mer 1/2/3 (84)
# [38796:38963] MACCS (167)
NON_PDFL_START = 36000
X9_noPDFL = X9_full[:, NON_PDFL_START:]   # (143, 2963)
N_SEQ_FEAT = X9_noPDFL.shape[1]
log.info(f"  NL2020 non-PDFL features: {X9_noPDFL.shape}")

# ── feature extraction functions ──────────────────────────────────────────────
log.info("\nLoading RNA-FM model ...")
tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnafm")
rnafm     = RnaFmModel.from_pretrained("multimolecule/rnafm").to(DEVICE)
rnafm.eval()

PURINES = {"A", "G"}
BASES   = ["A", "U", "G", "C"]
KMERS   = (["".join(b) for b in product(BASES, repeat=1)] +
           ["".join(b) for b in product(BASES, repeat=2)] +
           ["".join(b) for b in product(BASES, repeat=3)])  # 4+16+64=84

def get_morgan(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(2048, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return np.array(fp, dtype=np.float32)

def get_maccs(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(167, dtype=np.float32)
    return np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)

def get_physico(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(6, dtype=np.float32)
    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.RingCount(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.TPSA(mol),
    ], dtype=np.float32)

def get_nuc_comp(seq: str) -> np.ndarray:
    seq = seq.upper().replace("T", "U")
    n = len(seq)
    if n == 0:
        return np.zeros(10, dtype=np.float32)
    cnts = {b: seq.count(b) for b in BASES}
    gc   = (cnts["G"] + cnts["C"]) / n
    pur  = (cnts["A"] + cnts["G"]) / n
    return np.array([cnts["A"], cnts["U"], cnts["G"], cnts["C"],
                     cnts["A"]/n, cnts["U"]/n, cnts["G"]/n, cnts["C"]/n,
                     gc, pur], dtype=np.float32)

def get_kmer(seq: str) -> np.ndarray:
    seq = seq.upper().replace("T", "U")
    n = len(seq)
    feat = np.zeros(len(KMERS), dtype=np.float32)
    for i, km in enumerate(KMERS):
        k = len(km)
        cnt = sum(seq[j:j+k] == km for j in range(n - k + 1))
        feat[i] = cnt / max(1, n - k + 1)
    return feat

def get_rnafm(seq: str) -> np.ndarray:
    seq = seq.upper().replace("T", "U")
    seq = seq[:510]
    try:
        inp = tokenizer(seq, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = rnafm(**inp)
        emb = out.last_hidden_state[0, 1:-1].mean(0).cpu().numpy()
        return emb.astype(np.float32)
    except Exception:
        return np.zeros(640, dtype=np.float32)

def get_vienna(seq: str) -> np.ndarray:
    seq = seq.upper().replace("T", "U")
    try:
        struct, mfe = RNA.fold(seq)
        n  = len(seq)
        bp = struct.count("(")
        fc = RNA.fold_compound(seq)
        _, ens_e = fc.pf()
        ed = fc.ensemble_defect(struct)
        stem_chars = struct.count("(") + struct.count(")")
        return np.array([
            mfe, bp, bp / max(1, n),
            struct.count(".") / max(1, n),
            stem_chars / max(1, n),
            ens_e, ed, n
        ], dtype=np.float32)
    except Exception:
        return np.zeros(8, dtype=np.float32)

def extract_features(row) -> np.ndarray:
    """Morgan(2048) + nuc(10) + physico(6) + RNA-FM(640) + SS(8) + kmer(84) + MACCS(167) = 2963"""
    smiles = str(row["SMILES"])
    seq    = str(row["RNA_sequence"]).upper().replace("T", "U")
    return np.concatenate([
        get_morgan(smiles),
        get_nuc_comp(seq),
        get_physico(smiles),
        get_rnafm(seq),
        get_vienna(seq),
        get_kmer(seq),
        get_maccs(smiles),
    ]).astype(np.float32)

# ── compute STORM features ────────────────────────────────────────────────────
log.info(f"\nComputing features for {len(storm_df)} STORM entries ...")
log.info("  (Morgan + NucComp + Physico + RNA-FM + ViennaRNA + k-mer + MACCS)")

storm_feats = []
for i, (_, row) in enumerate(storm_df.iterrows()):
    f = extract_features(row)
    storm_feats.append(f)
    if (i + 1) % 50 == 0:
        log.info(f"  [{i+1}/{len(storm_df)}] done ...")

storm_feats = np.array(storm_feats, dtype=np.float32)
storm_feats = np.nan_to_num(storm_feats, nan=0.0, posinf=0.0, neginf=0.0)
log.info(f"  STORM features shape: {storm_feats.shape}")

# ── save ──────────────────────────────────────────────────────────────────────
np.savez_compressed(
    OUT_NPZ,
    X=storm_feats,
    y=storm_df["pKd"].values.astype(np.float32),
    ids=storm_df["yaml_id"].values,
    subtypes=storm_df["subtype"].values,
)
log.info(f"  STORM features saved → {OUT_NPZ}")

# ── build combined dataset ────────────────────────────────────────────────────
log.info("\nBuilding combined NL2020 + STORM dataset ...")

X_combined = np.vstack([X9_noPDFL, storm_feats])       # (1043, 2963)
y_combined = np.concatenate([y_nl, storm_df["pKd"].values.astype(np.float32)])
ids_combined  = ids_nl + list(storm_df["yaml_id"].values)
sub_combined  = np.array(sub_nl + list(storm_df["subtype"].values))
source_combined = np.array(["nl2020"]*len(y_nl) + ["storm"]*len(storm_df))

X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)
log.info(f"  Combined: {X_combined.shape}, NL2020={len(y_nl)}, STORM={len(storm_df)}")

dist = Counter(sub_combined)
log.info("  Subtype distribution:")
for k, v in sorted(dist.items(), key=lambda x: -x[1]):
    nl_n  = sum(1 for s, src in zip(sub_combined, source_combined) if s==k and src=="nl2020")
    stm_n = sum(1 for s, src in zip(sub_combined, source_combined) if s==k and src=="storm")
    log.info(f"    {k:<20}: {v:4d} (NL2020={nl_n}, STORM={stm_n})")

# ── pipeline ──────────────────────────────────────────────────────────────────
ALPHA_GRID = {"reg__alpha": [1, 10, 100, 1000, 10_000, 100_000]}

def make_pipe():
    return Pipeline([
        ("vt",  VarianceThreshold(threshold=1e-4)),
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=SEED)),
        ("reg", Ridge()),
    ])

def run_cv(Xs, ys, ns, n_jobs=10):
    cv = LeaveOneOut() if ns < 15 else KFold(5, shuffle=True, random_state=SEED)
    oof = np.full(ns, np.nan)
    for tr, te in cv.split(Xs):
        ni = min(5, len(tr))
        gs = GridSearchCV(make_pipe(), ALPHA_GRID, cv=ni, scoring="r2",
                          n_jobs=n_jobs, refit=True)
        gs.fit(Xs[tr], ys[tr])
        preds = gs.best_estimator_.predict(Xs[te])
        oof[te] = np.clip(preds, ys[tr].min()-3, ys[tr].max()+3)
    return oof

def metrics(yt, yp):
    r   = float(np.corrcoef(yt, yp)[0, 1])
    rho = float(stats.spearmanr(yt, yp).statistic)
    rms = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2  = float(r2_score(yt, yp))
    return r, rho, rms, mae, r2

# ── per-subtype CV on combined dataset ────────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("PER-SUBTYPE CV — Combined NL2020 + STORM (seq features only)")
log.info("=" * 70)

all_oof_comb = np.full(len(y_combined), np.nan)
results_comb = []
t0 = time.time()

for sub in sorted(set(sub_combined)):
    mask = sub_combined == sub
    Xs, ys = X_combined[mask], y_combined[mask]
    ns = int(mask.sum())
    cv_name = "LOO" if ns < 15 else "5-fold"

    if ns < 8:
        log.info(f"  [{sub}] n={ns} — too small, skip")
        continue

    log.info(f"  [{sub}] n={ns:4d} | {cv_name}")
    oof = run_cv(Xs, ys, ns, n_jobs=10)
    r, rho, rms, mae, r2 = metrics(ys, oof)
    all_oof_comb[mask] = oof
    log.info(f"    r={r:.4f} | ρ={rho:.4f} | RMSE={rms:.4f}")
    results_comb.append({"subtype": sub, "n": ns, "Pearson_r": round(r,4),
                          "Spearman_rho": round(rho,4), "RMSE": round(rms,4),
                          "MAE": round(mae,4), "R2": round(r2,4)})

valid = ~np.isnan(all_oof_comb)
r_comb = float(np.corrcoef(y_combined[valid], all_oof_comb[valid])[0, 1])
log.info(f"\n  Combined OOF r = {r_comb:.4f} | n_eval={valid.sum()}/{len(y_combined)}")

# ── NL2020-only evaluation on combined model predictions ─────────────────────
log.info("\n" + "=" * 70)
log.info("NL2020 SUBSET — how does combined model score on NL2020 only?")
log.info("=" * 70)

nl_mask = source_combined == "nl2020"
nl_valid = nl_mask & valid
r_nl = float(np.corrcoef(y_combined[nl_valid], all_oof_comb[nl_valid])[0, 1])
log.info(f"  NL2020 r (combined model) = {r_nl:.4f}")

# ── per-subtype on NL2020-only subset ────────────────────────────────────────
log.info("\n  Per-subtype on NL2020:")
for sub in sorted(set(sub_nl)):
    mask = (sub_combined == sub) & nl_mask & valid
    if mask.sum() < 4:
        continue
    r_sub = float(np.corrcoef(y_combined[mask], all_oof_comb[mask])[0, 1])
    log.info(f"    {sub:<20}: n={mask.sum():3d}  r={r_sub:.4f}")

# ── benchmark comparison ──────────────────────────────────────────────────────
BENCHMARKS = {"AffiGrapher":0.498, "RLaffinity":0.559,
              "RLASIF":0.666, "DeepRSMA":0.784, "RSAPred":0.830}
log.info("\n  Benchmark vs NL2020 r:")
for bm, bv in BENCHMARKS.items():
    sym = "✓" if r_nl >= bv else "✗"
    log.info(f"    {sym} {'above' if r_nl >= bv else 'below'}  {bm}: {bv}")

log.info(f"\n  Step 9 (NL2020 only, PDFL): 0.5350")
log.info(f"  Step 10 (combined, seq only): NL2020 r = {r_nl:.4f}")
log.info(f"  Time: {int(time.time()-t0)}s")

# ── save results ──────────────────────────────────────────────────────────────
pd.DataFrame(results_comb).to_csv(RES_DIR / "step10_results.csv", index=False)
log.info(f"\n  Results → {RES_DIR / 'step10_results.csv'}")

# ── figure ────────────────────────────────────────────────────────────────────
COLORS = {"riboswitch":"#4C72B0","aptamer":"#55A868","ribosomal_asite":"#C44E52",
          "viral_tar":"#DD8452","g_quadruplex":"#E377C2","duplex_groove":"#7F7F7F",
          "other_misc":"#8172B2"}

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(f"Step 10 — STORM Augmentation   NL2020 r = {r_nl:.4f}",
             fontsize=14, fontweight="bold")

ax = axes[0]
for sub in sorted(set(sub_combined)):
    mask = (sub_combined == sub) & valid
    if mask.sum() == 0:
        continue
    nl = (source_combined[mask] == "nl2020")
    ax.scatter(y_combined[mask][nl], all_oof_comb[mask][nl],
               label=f"{sub} (NL)", color=COLORS.get(sub,"gray"),
               alpha=0.8, s=50, edgecolors="none")
    ax.scatter(y_combined[mask][~nl], all_oof_comb[mask][~nl],
               color=COLORS.get(sub,"gray"), alpha=0.3, s=15,
               marker="^", edgecolors="none")
lims = [y_combined.min()-0.5, y_combined.max()+0.5]
ax.plot(lims, lims, "k--", lw=1)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Experimental pKd"); ax.set_ylabel("Predicted pKd")
ax.set_title("Parity — circles=NL2020, triangles=STORM")
ax.legend(fontsize=6, ncol=2)

ax2 = axes[1]
df_bar = pd.DataFrame(results_comb).sort_values("Pearson_r", ascending=True)
ax2.barh(df_bar["subtype"], df_bar["Pearson_r"],
         color=[COLORS.get(s,"gray") for s in df_bar["subtype"]])
ax2.axvline(0, color="k", lw=0.8)
ax2.axvline(r_nl, color="purple", lw=1.5, ls="--", label=f"NL2020 r={r_nl:.3f}")
ax2.axvline(0.5350, color="green", lw=1.2, ls=":", label="Step9 r=0.535")
for bm, bv in BENCHMARKS.items():
    ax2.axvline(bv, color="gray", lw=0.8, ls=":")
    ax2.text(bv+0.005, 0.05, bm, fontsize=6, rotation=90, va="bottom",
             transform=ax2.get_xaxis_transform())
ax2.set_xlabel("Pearson r")
ax2.set_title("Per-subtype r (combined NL2020+STORM)")
ax2.legend(fontsize=7)

plt.tight_layout()
plt.savefig(FIG_DIR / "step10_results.png", dpi=150, bbox_inches="tight")
plt.close()
log.info(f"  Figure → {FIG_DIR / 'step10_results.png'}")

log.info("\n" + "=" * 70)
log.info("STEP 10 COMPLETE")
log.info("=" * 70)
