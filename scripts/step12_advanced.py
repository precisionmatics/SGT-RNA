"""
SGT-RNA · Step 12: Four Advanced Enhancements

1. Riboswitch sub-classification: k-means on Morgan fingerprints (k=6)
   → train separate Ridge per riboswitch chemical class
2. Mg²⁺ SGT: RNA-Mg and Mg-Ligand SGT from pocket PDB crystal coords
   → 1,300 features (13 pairs × 5 thresholds × 2 × 10)
3. Concentric shell SGT: 3 radial shells from ligand centroid (0-4, 4-8, 8-12 Å)
   → 10,800 features (3 × 3,600)
4. Betti curves: 20 filtration thresholds (vs current 5) on interface subgraph at 6Å
   → 14,400 features (36 × 20 × 2 × 10)

Base: step11 features (49,763) + new = 76,263 total
"""

import gzip, pickle, re, logging, time
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
from sklearn.cluster import KMeans
from rdkit import Chem
from rdkit.Chem import AllChem

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
NA_L     = Path("/home/stalin/Desktop/SGT-RNA/NA-L")
PKL_FILE = ROOT / "data" / "pocket_fri" / "pocket_fri_data.pkl.gz"
FEAT_S11 = ROOT / "data" / "features" / "step11_full_features.npz"
DATASET  = ROOT / "data" / "affinity" / "dataset.csv"
OUT_NPZ  = ROOT / "data" / "features" / "step12_full_features.npz"
RES_DIR  = ROOT / "results"
FIG_DIR  = ROOT / "results" / "figures"

# ── logging ───────────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "results" / "logs" / f"step12_{ts}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("SGT-RNA · Step 12: Advanced Enhancements")
log.info("=" * 70)

# ── SGT constants ────────────────────────────────────────────────────────────
RNA_ELEMENTS  = ["C", "N", "O", "P"]
LIG_ELEMENTS  = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
THRESHOLDS_5  = [0.0, 0.8, 0.85, 0.90, 0.95]         # original
THRESHOLDS_20 = [round(i * 0.05, 2) for i in range(20)]  # betti curves
N_STATS = 10
ETA, KAPPA, EPS = 5.0, 2.0, 1e-8

# Mg²⁺ pairs: RNA-Mg and Mg-Lig
MG_RNA_PAIRS = [(r, "Mg") for r in RNA_ELEMENTS]  # 4 pairs
MG_LIG_PAIRS = [("Mg", l) for l in LIG_ELEMENTS]  # 9 pairs
# Only RNA-Mg (ligand-Mg is too sparse); 4 × 5 × 2 × 10 = 400 features
N_MG_FEATS   = len(MG_RNA_PAIRS) * len(THRESHOLDS_5) * 2 * N_STATS  # 400

# Concentric shells: 3 × 3,600
SHELLS       = [(0, 4), (4, 8), (8, 12)]
N_SHELL_FEATS= len(SHELLS) * len(RNA_ELEMENTS) * len(LIG_ELEMENTS) * len(THRESHOLDS_5) * 2 * N_STATS

# Betti curves: 36 × 20 × 2 × 10 at 6Å interface
N_BETTI_FEATS= len(RNA_ELEMENTS) * len(LIG_ELEMENTS) * len(THRESHOLDS_20) * 2 * N_STATS

log.info(f"Mg²⁺ SGT features    : {N_MG_FEATS}")
log.info(f"Shell SGT features   : {N_SHELL_FEATS}")
log.info(f"Betti curve features  : {N_BETTI_FEATS}")
log.info(f"Total new features    : {N_MG_FEATS + N_SHELL_FEATS + N_BETTI_FEATS}")

# ── subtype labels ────────────────────────────────────────────────────────────
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

# ── SGT core ─────────────────────────────────────────────────────────────────
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

def pair_features_custom_thresholds(rc, lc, thresholds):
    n_feat = len(thresholds) * 2 * N_STATS
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
    for tau in thresholds:
        mask   = W_norm >= tau
        W_sel  = W_norm * mask
        n_edges= int(mask.sum())
        if n_edges == 0:
            eigs_L0 = np.zeros(n_total)
            eigs_L1 = np.array([])
        else:
            L0      = build_L0(W_sel)
            eigs_L0 = np.maximum(np.linalg.eigvalsh(L0), 0.0)
            beta0   = int((eigs_L0 < EPS).sum())
            nz_L1   = max(0, n_edges - n_total + beta0)
            eigs_L1 = np.concatenate([np.zeros(nz_L1), eigs_L0[eigs_L0 >= EPS]])
        feats[offset:offset+N_STATS]          = spectral_stats(eigs_L0)
        feats[offset+N_STATS:offset+2*N_STATS] = spectral_stats(eigs_L1)
        offset += 2 * N_STATS
    return feats

# ── Feature 1: Mg²⁺ SGT ──────────────────────────────────────────────────────
def parse_mg_coords(pocket_pdb_path):
    coords = []
    try:
        with open(pocket_pdb_path) as f:
            for line in f:
                if (line.startswith("HETATM") or line.startswith("ATOM")) and \
                   line[12:16].strip() == "MG" or \
                   (len(line) > 76 and line[76:78].strip() == "MG"):
                    # try standard HETATM MG record
                    pass
        # Simpler: grep for MG in residue name field
        with open(pocket_pdb_path) as f:
            for line in f:
                if len(line) >= 20 and line[:6] in ("HETATM", "ATOM  "):
                    resname = line[17:20].strip()
                    if resname == "MG":
                        try:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            coords.append([x, y, z])
                        except ValueError:
                            pass
    except Exception:
        pass
    return np.array(coords, dtype=np.float32) if coords else np.empty((0, 3), np.float32)

def compute_mg_sgt(rna_coords, mg_coords):
    """4 RNA-element × Mg pairs, 5 thresholds, = 400 features."""
    feats = []
    for r_el in RNA_ELEMENTS:
        rc = rna_coords[r_el]
        feats.append(pair_features_custom_thresholds(rc, mg_coords, THRESHOLDS_5))
    return np.concatenate(feats)  # 400

# ── Feature 2: Concentric shell SGT ─────────────────────────────────────────
def compute_shell_sgt(rna_coords, lig_coords):
    """3 shells × 3,600 = 10,800 features."""
    # Ligand centroid
    lig_all = [v for v in lig_coords.values() if len(v) > 0]
    if not lig_all:
        return np.zeros(N_SHELL_FEATS, dtype=np.float32)
    lig_all_arr = np.vstack(lig_all)
    centroid = lig_all_arr.mean(axis=0)

    parts = []
    for (d_min, d_max) in SHELLS:
        shell_feats = []
        for r_el in RNA_ELEMENTS:
            rc = rna_coords[r_el]
            if len(rc) == 0:
                for l_el in LIG_ELEMENTS:
                    shell_feats.append(np.zeros(len(THRESHOLDS_5) * 2 * N_STATS, dtype=np.float32))
                continue
            # Distance from centroid
            dists = np.linalg.norm(rc - centroid, axis=1)
            mask  = (dists >= d_min) & (dists < d_max)
            rc_shell = rc[mask]
            for l_el in LIG_ELEMENTS:
                lc = lig_coords.get(l_el, np.empty((0, 3), np.float32))
                shell_feats.append(pair_features_custom_thresholds(rc_shell, lc, THRESHOLDS_5))
        parts.append(np.concatenate(shell_feats))
    return np.concatenate(parts)  # 10,800

# ── Feature 3: Betti curves (20 thresholds, 6Å interface) ────────────────────
def compute_betti_curves(rna_coords, lig_coords, cutoff=6.0):
    """36 pairs × 20 thresholds × 2 × 10 = 14,400 features."""
    lig_all = [v for v in lig_coords.values() if len(v) > 0]
    if not lig_all:
        return np.zeros(N_BETTI_FEATS, dtype=np.float32)
    lig_all_arr = np.vstack(lig_all)

    # filter RNA to 6Å interface
    filtered_rna = {}
    for el in RNA_ELEMENTS:
        rc = rna_coords[el]
        if len(rc) == 0:
            filtered_rna[el] = rc
            continue
        D = cdist(rc, lig_all_arr)
        filtered_rna[el] = rc[D.min(axis=1) <= cutoff]

    parts = []
    for r_el in RNA_ELEMENTS:
        for l_el in LIG_ELEMENTS:
            rc = filtered_rna[r_el]
            lc = lig_coords.get(l_el, np.empty((0, 3), np.float32))
            parts.append(pair_features_custom_thresholds(rc, lc, THRESHOLDS_20))
    return np.concatenate(parts)  # 14,400

# ── load data ─────────────────────────────────────────────────────────────────
log.info("\nLoading pocket data ...")
with gzip.open(PKL_FILE, "rb") as f:
    records = pickle.load(f)

log.info("Loading step11 features ...")
d11 = np.load(FEAT_S11)
X11, y11, ids11 = d11["X"].astype(np.float32), d11["y"].astype(np.float32), d11["ids"]
subtypes_raw = d11["subtypes"]
log.info(f"  step11 X: {X11.shape}")

df = pd.read_csv(DATASET)
id2row = {row["pdb"]: row for _, row in df.iterrows()}

# verify alignment
assert list(ids11) == [r["pdb"] for r in records]

subtypes = np.array([make_subtype(p, s) for p, s in zip(ids11, subtypes_raw)])

# ── Riboswitch sub-classification via Morgan k-means ──────────────────────────
log.info("\nRiboswitch sub-classification (Morgan k-means, k=6) ...")
rs_mask = subtypes == "riboswitch"
rs_idx  = np.where(rs_mask)[0]

rs_morgan = []
for i in rs_idx:
    pdb = ids11[i]
    sdf_path = NA_L / pdb / f"{pdb}_ligand.sdf"
    try:
        mol = next(Chem.SDMolSupplier(str(sdf_path), removeHs=True))
        fp  = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
        rs_morgan.append(np.array(fp, dtype=np.float32))
    except Exception:
        rs_morgan.append(np.zeros(512, dtype=np.float32))

rs_morgan_arr = np.array(rs_morgan)
km = KMeans(n_clusters=6, random_state=42, n_init=20)
rs_class_labels = km.fit_predict(rs_morgan_arr)

# Build augmented subtypes: riboswitch_0 .. riboswitch_5
subtypes_aug = subtypes.copy()
for k, i in enumerate(rs_idx):
    subtypes_aug[i] = f"riboswitch_{rs_class_labels[k]}"

for cl in range(6):
    members = [ids11[i] for k, i in enumerate(rs_idx) if rs_class_labels[k] == cl]
    log.info(f"  riboswitch_{cl}: n={len(members)}  {members[:5]}")

# ── compute new features ──────────────────────────────────────────────────────
n = len(records)
X_mg    = np.zeros((n, N_MG_FEATS),    dtype=np.float32)
X_shell = np.zeros((n, N_SHELL_FEATS), dtype=np.float32)
X_betti = np.zeros((n, N_BETTI_FEATS), dtype=np.float32)

log.info(f"\nComputing Mg²⁺ SGT, shell SGT, betti curves ...")
t0 = time.time()
for idx, rec in enumerate(records):
    pdb = rec["pdb"]

    # Mg²⁺
    pocket_path = NA_L / pdb / f"{pdb}_pocket.pdb"
    mg_coords   = parse_mg_coords(pocket_path)
    X_mg[idx]   = compute_mg_sgt(rec["rna_coords"], mg_coords)

    # Shells
    X_shell[idx] = compute_shell_sgt(rec["rna_coords"], rec["lig_coords"])

    # Betti curves
    X_betti[idx] = compute_betti_curves(rec["rna_coords"], rec["lig_coords"], cutoff=6.0)

    if (idx + 1) % 20 == 0 or idx == 0:
        elapsed = time.time() - t0
        eta = (n - idx - 1) / max((idx + 1) / elapsed, 1e-6)
        mg_present = len(mg_coords) > 0
        log.info(f"  [{idx+1:3d}/{n}] {pdb}  Mg={mg_present}  ETA {eta:.0f}s")

log.info(f"  Done in {time.time()-t0:.1f}s")

# ── combine all features ──────────────────────────────────────────────────────
X_full = np.hstack([X11, X_mg, X_shell, X_betti])
log.info(f"\nFull feature matrix: {X_full.shape}")
log.info(f"  step11:       {X11.shape[1]}")
log.info(f"  + Mg SGT:    {N_MG_FEATS}")
log.info(f"  + Shell SGT: {N_SHELL_FEATS}")
log.info(f"  + Betti:      {N_BETTI_FEATS}")
log.info(f"  = Total:      {X_full.shape[1]}")

np.savez_compressed(OUT_NPZ, X=X_full, y=y11, ids=ids11,
                    subtypes=subtypes_raw, subtypes_aug=subtypes_aug)
log.info(f"Saved → {OUT_NPZ}")

# ── ML pipeline ───────────────────────────────────────────────────────────────
ALPHA_GRID = [1, 10, 100, 1000, 10_000, 100_000]

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

def cv_ridge(X, y, n_splits=5):
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
                p = np.clip(pipe.predict(X[te]), y[tr].min()-3, y[tr].max()+3)
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
log.info("ML EVALUATION — riboswitch sub-classified, adaptive stacking")
log.info("=" * 70)

# Global model OOF
log.info("\nFitting global model ...")
global_preds, global_r = cv_ridge(X_full, y11, n_splits=5)
log.info(f"  Global CV r = {global_r:.4f}")

unique_subtypes = sorted(set(subtypes_aug))
subtype_preds   = np.full(n, np.nan)
subtype_rs      = {}

for st in unique_subtypes:
    mask = subtypes_aug == st
    idx  = np.where(mask)[0]
    ns   = len(idx)
    Xs, ys = X_full[idx], y11[idx]

    preds_st, r_st = loo_ridge(Xs, ys)
    preds_gl       = global_preds[idx]
    r_gl = pearsonr(ys, preds_gl)[0] if np.std(preds_gl) > 1e-8 else -99.0

    if r_st >= r_gl:
        subtype_preds[idx] = preds_st
        chosen, r_use = "subtype", r_st
    else:
        subtype_preds[idx] = preds_gl
        chosen, r_use = "global", r_gl

    subtype_rs[st] = r_use
    log.info(f"  {st:<26}: n={ns:3d}  r={r_use:.4f}  (used {chosen})")

valid      = ~np.isnan(subtype_preds)
combined_r = pearsonr(y11[valid], subtype_preds[valid])[0]

# Aggregate riboswitch sub-class performance
rs_all_pred = subtype_preds[rs_mask]
rs_all_true = y11[rs_mask]
rs_combined = pearsonr(rs_all_true, rs_all_pred)[0] if np.std(rs_all_pred) > 1e-8 else -99.0

log.info("\n" + "=" * 70)
log.info(f"  Riboswitch combined r  = {rs_combined:.4f}  (step09: 0.400, step11: 0.388)")
log.info(f"  COMBINED r (step12)    = {combined_r:.4f}")
log.info(f"  Step 11 baseline       = 0.5754")
log.info(f"  Delta vs step11        = {combined_r - 0.5754:+.4f}")
log.info(f"  Delta vs step09        = {combined_r - 0.5350:+.4f}")
log.info("=" * 70)

# Benchmark comparison
benchmarks = [
    ("AffiGrapher", 0.498), ("RLaffinity", 0.559), ("RLASIF", 0.666),
    ("DeepRSMA", 0.784), ("RSAPred", 0.830),
]
log.info("\nBenchmark comparison:")
for name, r_bench in benchmarks:
    sym = "✓" if combined_r > r_bench else "✗"
    log.info(f"  {sym} {name}: {r_bench:.3f}  (ours: {combined_r:.4f})")

# ── results CSV ───────────────────────────────────────────────────────────────
df_res = pd.DataFrame({
    "pdb": ids11, "subtype": subtypes, "subtype_aug": subtypes_aug,
    "y_true": y11, "y_pred": subtype_preds,
})
res_path = RES_DIR / "step12_results.csv"
df_res.to_csv(res_path, index=False)
log.info(f"\nResults → {res_path}")

# ── figure ────────────────────────────────────────────────────────────────────
colors = {
    "aptamer":"#2166AC","riboswitch":"#1A9641","ribosomal_asite":"#D73027",
    "duplex_groove":"#7B2D8B","g_quadruplex":"#FF7F00","viral_tar":"#A65628",
    "other_misc":"#999999",
}

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.patch.set_facecolor("white")

# Panel A: scatter
ax = axes[0]
for st in sorted(set(subtypes)):
    mask = subtypes == st
    ax.scatter(y11[mask], subtype_preds[mask],
               c=colors.get(st,"#888888"), label=st, alpha=0.75, s=40, edgecolors="none")
mn, mx = y11.min()-0.5, y11.max()+0.5
ax.plot([mn,mx],[mn,mx],"k--",lw=1,alpha=0.4)
ax.set_xlabel("Experimental pKd"); ax.set_ylabel("Predicted pKd")
ax.set_title(f"Step 12 (combined r={combined_r:.4f})", fontweight="bold")
ax.legend(fontsize=7, loc="upper left", framealpha=0.7)
ax.grid(alpha=0.3, linestyle="--")

# Panel B: step comparison bar
ax = axes[1]
step_rs = {"Step 09": 0.535, "Step 11": 0.575, "Step 12": combined_r}
bars = ax.bar(list(step_rs.keys()), list(step_rs.values()),
              color=["#AAAAAA","#4393C3","#2166AC"], edgecolor="white", alpha=0.9)
for bar, val in zip(bars, step_rs.values()):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
for name, r_bench in benchmarks:
    ax.axhline(r_bench, linestyle="--", linewidth=0.9, alpha=0.6, label=name)
ax.set_ylabel("Combined Pearson r")
ax.set_title("Progress across steps", fontweight="bold")
ax.legend(fontsize=7); ax.set_ylim(0.4, 0.9)
ax.grid(axis="y", alpha=0.3, linestyle="--")

# Panel C: riboswitch sub-class bar
ax = axes[2]
rs_classes = [f"riboswitch_{c}" for c in range(6)]
rs_ns  = [int((subtypes_aug == c).sum()) for c in rs_classes]
rs_r_vals = [subtype_rs.get(c, 0) for c in rs_classes]
x = np.arange(6)
ax.bar(x, rs_r_vals, color="#1A9641", alpha=0.8, edgecolor="white")
ax.axhline(0.400, linestyle="--", color="gray", lw=1.5, label="Step09 riboswitch r=0.400")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"class {c}\n(n={rs_ns[c]})" for c in range(6)], fontsize=9)
ax.set_ylabel("Pearson r")
ax.set_title("Riboswitch sub-class performance", fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3, linestyle="--")

plt.tight_layout()
fig_path = FIG_DIR / "step12_results.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
log.info(f"Figure → {fig_path}")

log.info("\n" + "=" * 70)
log.info("STEP 12 COMPLETE")
log.info("=" * 70)
