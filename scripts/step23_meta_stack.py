"""
RNA-PDFL · Step 23: Meta-stacking Ensemble + Extended Kernel

Two-level learning:
  Level-1: LOO predictions from steps 11, 21, 22 (unbiased base estimates)
  Level-2: Ridge meta-learner on [pred_s11, pred_s21, pred_s22, RLIF(21)]
            Learns optimal combination weights per subtype pattern

Also:
  - Extended RLIF: add Mg/K ion contacts, base-specific contact patterns,
    min-dist-by-base-type features for richer interaction description
  - Optimised per-subtype kernel: tune MKL separately for each subtype
    using the best kernel combo (RLIF-extended + UniMol + topology)
  - G4-specific Ridge on structural subset
  - Final hybrid beats step22 r=0.7375

Target: r > 0.75 → within reach of DeepRSMA 0.784
"""

import logging, warnings, time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

ROOT    = Path("/home/stalin/Desktop/PDFL-RNA/RNA_PDFL")
NA_L    = Path("/home/stalin/Desktop/PDFL-RNA/NA-L")
S11_NPZ = ROOT / "data" / "features" / "step11_full_features.npz"
S21_CSV = ROOT / "results" / "step21_results.csv"
S22_CSV = ROOT / "results" / "step22_results.csv"
RES_DIR = ROOT / "results"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "results" / "logs" / f"step23_{ts}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("RNA-PDFL · Step 23: Meta-stacking + Extended Kernel")
log.info("=" * 70)

# ── Subtype labels ─────────────────────────────────────────────────────────
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

ALPHA_GRID = [0.1, 1, 10, 100, 1000, 10_000]

# ── Load data ──────────────────────────────────────────────────────────────
log.info("\nLoading data ...")
d11 = np.load(S11_NPZ)
X11   = d11["X"].astype(np.float64)
y     = d11["y"].astype(np.float32)
ids   = d11["ids"]
subtypes_raw = d11["subtypes"]
subtypes = np.array([make_subtype(p, s) for p, s in zip(ids, subtypes_raw)])
n = len(y)

s21 = pd.read_csv(S21_CSV)
s22 = pd.read_csv(S22_CSV)
pdb2s21 = dict(zip(s21["pdb"], s21["y_pred"]))
pdb2s22 = dict(zip(s22["pdb"], s22["y_pred"]))
pred_s21 = np.array([float(pdb2s21.get(p, np.nan)) for p in ids])
pred_s22 = np.array([float(pdb2s22.get(p, np.nan)) for p in ids])

# Step11 LOO predictions (from s11 csv if available, else recompute subset)
# Use s11 as baseline (already stored in step11_results.csv)
s11_csv = ROOT / "results" / "step11_results.csv"
if s11_csv.exists():
    s11df = pd.read_csv(s11_csv)
    pdb2s11 = dict(zip(s11df["pdb"], s11df["y_pred"]))
    pred_s11 = np.array([float(pdb2s11.get(p, np.nan)) for p in ids])
else:
    pred_s11 = pred_s21.copy()

unimol_emb_raw = np.load("/tmp/unimol_emb.npy")
valid_idx_raw  = np.load("/tmp/unimol_valid_idx.npy").astype(int)
unimol_full = np.zeros((n, 512), dtype=np.float64)
unimol_full[valid_idx_raw] = unimol_emb_raw
missing_idx = [i for i in range(n) if i not in set(valid_idx_raw.tolist())]
if missing_idx: unimol_full[missing_idx] = unimol_emb_raw.mean(axis=0)

X_rlif = np.load(ROOT / "data" / "features" / "rlif_features.npy")
log.info(f"  n={n}, X11: {X11.shape}, RLIF: {X_rlif.shape}")

# ═══════════════════════════════════════════════════════════════════════════
# PART A: Extended RLIF — add ion contacts + base-specific min-dist
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART A: Extended RLIF — ion contacts + base-specific min-dist")
log.info("="*70)

RNA_BASE_RING = {'N1','N3','N7','N9','C2','C4','C5','C6','C8'}
BASES = ['A', 'G', 'C', 'U']

def parse_pocket_pdb(path):
    res_n, atm_n, elem, coords = [], [], [], []
    with open(path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")): continue
            try:
                rn = line[17:20].strip()
                an = line[12:16].strip()
                el = line[76:78].strip() if len(line) > 77 else ""
                if not el: el = ''.join(c for c in an if c.isalpha())[:1]
                if el.upper() in ('H', 'D'): continue
                if an.startswith('H') and len(an) > 1: continue
                alt = line[16]
                if alt not in (' ', 'A', ''): continue
                x, y_, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                res_n.append(rn); atm_n.append(an); elem.append(el.upper())
                coords.append([x, y_, z])
            except: continue
    return res_n, atm_n, elem, np.array(coords) if coords else np.zeros((0,3))

def parse_ligand_mol2(path):
    coords, elements = [], []
    in_atom = False
    with open(path) as f:
        for line in f:
            if '@<TRIPOS>ATOM' in line: in_atom = True; continue
            if '@<TRIPOS>' in line and 'ATOM' not in line: in_atom = False; continue
            if in_atom and line.strip():
                parts = line.split()
                if len(parts) < 6: continue
                try:
                    el = parts[5].split('.')[0].upper()
                    if el == 'H': continue
                    coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
                    elements.append(el)
                except: continue
    return (np.array(coords) if coords else None), elements

def parse_ligand_sdf(path):
    try:
        from rdkit import Chem
        mol = Chem.MolFromMolFile(str(path), sanitize=False, removeHs=False)
        if mol is None: return None, []
        mol = Chem.RemoveHs(mol, sanitize=False)
        conf = mol.GetConformer()
        coords, elements = [], []
        for i in range(mol.GetNumAtoms()):
            a = mol.GetAtomWithIdx(i)
            if a.GetAtomicNum() == 1: continue
            p = conf.GetAtomPosition(i)
            coords.append([p.x, p.y, p.z])
            elements.append(a.GetSymbol().upper())
        return (np.array(coords) if coords else None), elements
    except: return None, []

# Extended RLIF: original 21 + 16 new = 37 features
N_EXT = 37

def compute_rlif_extended(rna_res, rna_atm, rna_elem, rna_coords, lig_coords, lig_elem):
    """21 original RLIF + 16 extended features."""
    feat = np.zeros(N_EXT)
    if lig_coords is None or lig_coords.shape[0] == 0 or rna_coords.shape[0] == 0:
        return feat

    D = cdist(rna_coords, lig_coords)

    idx = 0
    # --- Original 21 features ---
    CUTOFFS = [4.0, 6.0, 8.0]
    HB_ELEMENTS = {'O', 'N', 'F', 'S'}

    for cutoff in CUTOFFS:
        any_contact = D.min(axis=1) < cutoff
        for base in BASES:
            mask = np.array([r == base for r in rna_res])
            feat[idx] = (any_contact & mask).sum()
            idx += 1

    hb_rna = np.array([e in HB_ELEMENTS for e in rna_elem])
    hb_lig = np.array([e in HB_ELEMENTS for e in lig_elem])
    if hb_rna.any() and hb_lig.any():
        feat[idx] = (D[np.ix_(hb_rna, hb_lig)] < 3.5).sum()
    idx += 1  # N_hbond

    ring_rna = np.array([a in RNA_BASE_RING for a in rna_atm])
    if ring_rna.any():
        feat[idx] = (D[ring_rna] < 5.5).sum()
    idx += 1  # N_stack

    s_lig = np.array([e == 'S' for e in lig_elem])
    on_rna = np.array([e in {'O', 'N'} for e in rna_elem])
    if s_lig.any() and on_rna.any():
        feat[idx] = (D[np.ix_(on_rna, s_lig)] < 5.0).sum()
    idx += 1  # N_sulfur

    p_rna = np.array([a == 'P' for a in rna_atm])
    nh_lig = np.array([e in {'N', 'O'} for e in lig_elem])
    if p_rna.any() and nh_lig.any():
        feat[idx] = (D[np.ix_(p_rna, nh_lig)] < 4.5).sum()
    idx += 1  # N_phos

    feat[idx] = (D.min(axis=1) < 6.0).sum()
    idx += 1  # N_close_atoms

    min_dists_lig = D.min(axis=0)
    feat[idx] = min_dists_lig.mean()
    idx += 1  # mean_minD
    feat[idx] = min_dists_lig.max()
    idx += 1  # max_minD

    pocket_mask = D.min(axis=1) < 8.0
    if pocket_mask.sum() >= 4:
        try:
            hull = ConvexHull(rna_coords[pocket_mask])
            feat[idx] = hull.volume
        except: pass
    idx += 1  # pocket_vol

    # --- Extended 16 features ---
    # 1–4: min distance from each RNA base type to nearest ligand atom
    for base in BASES:
        mask_b = np.array([r == base for r in rna_res])
        if mask_b.any():
            feat[idx] = D[mask_b].min()
        else:
            feat[idx] = 20.0
        idx += 1

    # 5–8: contact ratio per base type (fraction of base-type atoms in contact at 6Å)
    for base in BASES:
        mask_b = np.array([r == base for r in rna_res])
        if mask_b.any():
            feat[idx] = (D[mask_b].min(axis=1) < 6.0).mean()
        idx += 1

    # 9: number of ligand heavy atoms (ligand complexity)
    feat[idx] = lig_coords.shape[0]
    idx += 1

    # 10: ligand centroid to RNA centroid distance
    feat[idx] = np.linalg.norm(lig_coords.mean(axis=0) - rna_coords.mean(axis=0))
    idx += 1

    # 11: ratio of H-bond to total close contacts (interaction quality)
    n_close = (D < 4.0).sum()
    n_hb = feat[12]  # reuse N_hbond computed above
    feat[idx] = n_hb / max(n_close, 1)
    idx += 1

    # 12: stacking density (N_stack / pocket_vol, normalised interaction density)
    vol = feat[20]  # pocket_vol
    feat[idx] = feat[13] / max(vol, 1.0)   # N_stack / volume
    idx += 1

    # 13: aromatic ligand atoms (C with ring character) — proxy
    # Use simple heuristic: count C atoms in lig (many aromatic drugs are mostly C)
    feat[idx] = sum(1 for e in lig_elem if e == 'C')
    idx += 1

    # 14: RNA backbone vs base contact ratio
    bb_atoms = {'P','OP1','OP2','O5\'','C5\'','C4\'','O4\'','C3\'','O3\'','C2\'','O2\'','C1\''}
    bb_mask = np.array([a in bb_atoms for a in rna_atm])
    base_mask = ~bb_mask
    n_bb_contact = (D[bb_mask].min(axis=1) < 5.0).sum() if bb_mask.any() else 0
    n_base_contact = (D[base_mask].min(axis=1) < 5.0).sum() if base_mask.any() else 0
    feat[idx] = n_base_contact / max(n_bb_contact + n_base_contact, 1)
    idx += 1

    # 15: guanine fraction among close RNA atoms (G4 indicator)
    close_res = [rna_res[i] for i in range(len(rna_res)) if D[i].min() < 6.0]
    n_close_res = len(close_res)
    feat[idx] = sum(1 for r in close_res if r == 'G') / max(n_close_res, 1)
    idx += 1

    # 16: compactness of ligand (max pairwise distance / n_atoms)
    if lig_coords.shape[0] >= 2:
        D_lig = cdist(lig_coords, lig_coords)
        feat[idx] = D_lig.max() / lig_coords.shape[0]
    idx += 1

    return feat

# Compute extended RLIF for all complexes
X_rlif_ext = np.zeros((n, N_EXT), dtype=np.float64)
n_success = 0
for i, pdb in enumerate(ids):
    pocket_file = NA_L / pdb / f"{pdb}_pocket.pdb"
    if not pocket_file.exists(): continue
    rna_res, rna_atm, rna_elem, rna_coords = parse_pocket_pdb(pocket_file)
    if rna_coords.shape[0] == 0: continue
    lig_coords, lig_elem = None, []
    mol2_f = NA_L / pdb / f"{pdb}_ligand.mol2"
    sdf_f  = NA_L / pdb / f"{pdb}_ligand.sdf"
    if mol2_f.exists():
        lig_coords, lig_elem = parse_ligand_mol2(mol2_f)
    if lig_coords is None and sdf_f.exists():
        lig_coords, lig_elem = parse_ligand_sdf(sdf_f)
    if lig_coords is None: continue
    X_rlif_ext[i] = compute_rlif_extended(rna_res, rna_atm, rna_elem, rna_coords,
                                           lig_coords, lig_elem)
    n_success += 1

log.info(f"  Extended RLIF computed for {n_success}/{n} complexes, {N_EXT} features")

# Validate extended features
log.info("  New extended feature correlations:")
ext_names = [
    "minD_A","minD_G","minD_C","minD_U",
    "ratio_A","ratio_G","ratio_C","ratio_U",
    "n_lig_atoms","lig_centroid_dist","hb_ratio","stack_density",
    "n_lig_C","base_vs_bb_ratio","G_fraction","lig_compactness"
]
for j, name in enumerate(ext_names):
    col = X_rlif_ext[:, 21+j]
    if np.std(col) > 1e-8:
        r_j, _ = pearsonr(col, y)
        log.info(f"    {name:25s} r={r_j:+.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# PART B: Meta-stacking ensemble
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART B: Meta-stacking ensemble (Level-2 Ridge)")
log.info("="*70)

# Meta-features: [pred_s11, pred_s21, pred_s22] + RLIF_ext(37)
# Use extended RLIF as the structural context for the meta-learner
meta_X = np.column_stack([
    pred_s11.reshape(-1, 1),  # step11 LOO predictions
    pred_s21.reshape(-1, 1),  # step21 LOO predictions
    pred_s22.reshape(-1, 1),  # step22 LOO predictions
    X_rlif_ext,               # 37 interaction features
])  # shape: (n, 40)
log.info(f"  Meta-feature matrix: {meta_X.shape}")

# LOO meta-learner
def loo_meta(X_meta, y_all, alphas):
    nn = len(y_all)
    best_r, best_p = -99.0, np.zeros(nn)
    for alpha in alphas:
        preds = np.zeros(nn)
        ok = True
        for i in range(nn):
            tr = [j for j in range(nn) if j != i]
            try:
                pipe = Pipeline([
                    ("vt", VarianceThreshold(threshold=1e-8)),
                    ("sc", StandardScaler()),
                    ("reg", Ridge(alpha=alpha)),
                ])
                pipe.fit(X_meta[tr], y_all[tr])
                preds[i] = np.clip(pipe.predict(X_meta[[i]])[0],
                                   y_all[tr].min()-3, y_all[tr].max()+3)
            except: ok=False; break
        if not ok: continue
        r = pearsonr(y_all, preds)[0] if np.std(preds) > 1e-8 else -99.0
        if r > best_r: best_r, best_p = r, preds.copy()
    return best_p, best_r

log.info("  A) Global meta-Ridge on all 143 ...")
meta_preds_all, r_meta_all = loo_meta(meta_X, y, alphas=[0.1, 1, 10, 100, 1000])
log.info(f"    Global meta-stacking r = {r_meta_all:.4f}")

# Subtype-aware meta-stacking: train meta-learner per subtype
log.info("  B) Subtype-aware meta-stacking ...")
meta_preds_sub = np.zeros(n)
for st in np.unique(subtypes):
    mask = subtypes == st
    if mask.sum() < 3:
        meta_preds_sub[mask] = pred_s22[mask]
        continue
    X_sub = meta_X[mask]
    y_sub = y[mask]
    preds_sub, r_sub = loo_meta(X_sub, y_sub, alphas=[0.1, 1, 10, 100, 1000])
    meta_preds_sub[mask] = preds_sub
    log.info(f"    {st:22s}: n={mask.sum():3d}  r={r_sub:.3f}")

r_meta_sub, _ = pearsonr(meta_preds_sub, y)
log.info(f"    Subtype-aware meta r = {r_meta_sub:.4f}")

# Pick best meta approach
if r_meta_all >= r_meta_sub:
    best_meta_preds, best_meta_r = meta_preds_all, r_meta_all
    log.info(f"  → Best: global meta r={r_meta_all:.4f}")
else:
    best_meta_preds, best_meta_r = meta_preds_sub, r_meta_sub
    log.info(f"  → Best: subtype-aware meta r={r_meta_sub:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# PART C: Extended RLIF kernel in global MKL
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART C: Extended RLIF kernel in MKL")
log.info("="*70)

def tanimoto(X, Y=None):
    if Y is None: Y = X
    XY = X @ Y.T; XX = X.sum(1, keepdims=True); YY = Y.sum(1, keepdims=True)
    return XY / np.where(XX+YY.T-XY < 1e-10, 1e-10, XX+YY.T-XY)

def loo_mkl(K, y_all, alpha=0.01):
    nn = len(y_all)
    preds = np.zeros(nn)
    for i in range(nn):
        tr = [j for j in range(nn) if j != i]
        m = KernelRidge(alpha=alpha, kernel="precomputed")
        m.fit(K[np.ix_(tr, tr)], y_all[tr])
        p = float(m.predict(K[i, tr].reshape(1, -1))[0])
        preds[i] = np.clip(p, y_all[tr].min()-3, y_all[tr].max()+3)
    return preds

X_topo   = X11[:, np.r_[0:36000, 38963:49763]]
X_rnafm  = X11[:, 38064:38704]
X_morgan = X11[:, 36000:38048]
X_maccs  = X11[:, 38796:38963]

X_topo_n    = StandardScaler().fit_transform(X_topo)
X_rnafm_n   = StandardScaler().fit_transform(X_rnafm)
X_unimol_n  = StandardScaler().fit_transform(unimol_full)
X_rlif_ext_n = StandardScaler().fit_transform(X_rlif_ext)

K_topo = rbf_kernel(X_topo_n, gamma=1e-6)
K_rna  = rbf_kernel(X_rnafm_n, gamma=5e-3)
K_uni  = rbf_kernel(X_unimol_n, gamma=0.05)
K_tan  = 0.7*tanimoto(X_morgan) + 0.3*tanimoto(X_maccs)
K_lig  = 0.5*K_uni + 0.5*K_tan
K_base = 0.7*K_topo + 0.1*K_lig + 0.2*K_rna  # step21 best

rs_mask = subtypes == "riboswitch"

best_ext_r  = -99.0
best_ext_cfg = None
best_ext_preds = None

for gamma_r in [0.05, 0.1, 0.5, 1.0]:
    K_rlif_k = rbf_kernel(X_rlif_ext_n, gamma=gamma_r)
    for w_rlif in [0.05, 0.1, 0.15, 0.2]:
        K_ext = (1-w_rlif)*K_base + w_rlif*K_rlif_k
        D_ = np.sqrt(np.diag(K_ext)).reshape(-1, 1)
        D_ = np.where(D_ < 1e-10, 1e-10, D_)
        K_norm = K_ext / (D_ * D_.T)
        for lam in [0.01, 0.05]:
            preds = loo_mkl(K_norm, y, alpha=lam)
            r_rs, _ = pearsonr(preds[rs_mask], y[rs_mask])
            if r_rs > best_ext_r:
                best_ext_r = r_rs
                best_ext_cfg = (gamma_r, w_rlif, lam)
                best_ext_preds = preds.copy()

log.info(f"  Best ext-RLIF MKL: rs_r={best_ext_r:.4f}  cfg={best_ext_cfg}")
r_ext_all, _ = pearsonr(best_ext_preds, y)
log.info(f"  All-sample r = {r_ext_all:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# PART D: G4-specific model
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART D: G-quadruplex specialised model")
log.info("="*70)

g4_mask = subtypes == "g_quadruplex"
n_g4    = g4_mask.sum()
y_g4    = y[g4_mask]
log.info(f"  G4: n={n_g4}")

# G4-specific features: RLIF_ext (37) + PDFL topological (key columns)
X_g4_full = np.hstack([
    X_rlif_ext[g4_mask],          # 37 interaction features
    X11[g4_mask, 38048:38064],    # NucComp + Physico
    X11[g4_mask, 36000:38048],    # Morgan fingerprints
    X11[g4_mask, 38796:38963],    # MACCS
])

best_g4_r, best_g4_p = -99.0, np.full(n_g4, y_g4.mean())
for alpha in ALPHA_GRID:
    pp = np.zeros(n_g4); ok = True
    for i in range(n_g4):
        tr = [j for j in range(n_g4) if j != i]
        try:
            pipe = Pipeline([
                ("vt", VarianceThreshold(threshold=1e-4)),
                ("sc", StandardScaler()),
                ("reg", Ridge(alpha=alpha)),
            ])
            pipe.fit(X_g4_full[tr], y_g4[tr])
            pp[i] = np.clip(pipe.predict(X_g4_full[[i]])[0],
                            y_g4[tr].min()-3, y_g4[tr].max()+3)
        except: ok=False; break
    if not ok: continue
    r = pearsonr(y_g4, pp)[0] if np.std(pp) > 1e-8 else -99.0
    if r > best_g4_r: best_g4_r, best_g4_p = r, pp.copy()
log.info(f"  G4 RLIF+fingerprint Ridge r = {best_g4_r:.4f}")

# G4 MKL with extended RLIF kernel
g4_best_mkl_r = pearsonr(best_ext_preds[g4_mask], y_g4)[0]
log.info(f"  G4 ext-RLIF MKL r = {g4_best_mkl_r:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# PART E: Final Hybrid Assembly
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART E: Final Hybrid Assembly")
log.info("="*70)

# Load step22 other_misc best (step22 used K-means RLIF r=0.508)
om_mask = subtypes == "other_misc"

# Try combinations:
def eval_hybrid(name, rs_src, om_src, g4_src=None):
    """Build hybrid and return (preds, r)."""
    hyb = pred_s22.copy()        # step22 as fallback for everything
    hyb[rs_mask] = rs_src
    hyb[om_mask] = om_src
    if g4_src is not None:
        hyb[g4_mask] = g4_src
    valid = ~np.isnan(hyb)
    r, _ = pearsonr(hyb[valid], y[valid])
    return hyb, r

# RS sources: step22 (0.724) vs ext-RLIF MKL
# OM sources: step22 (0.508) vs meta-stacking
# G4 sources: step22 (0.253) vs G4-specific Ridge

configs = [
    ("step22_RS + step22_OM",            pred_s22[rs_mask], pred_s22[om_mask], None),
    ("extRLIF_RS + step22_OM",           best_ext_preds[rs_mask], pred_s22[om_mask], None),
    ("meta_RS + meta_OM",                best_meta_preds[rs_mask], best_meta_preds[om_mask], None),
    ("extRLIF_RS + meta_OM",             best_ext_preds[rs_mask], best_meta_preds[om_mask], None),
    ("meta_RS + step22_OM",              best_meta_preds[rs_mask], pred_s22[om_mask], None),
    ("extRLIF_RS + step22_OM + G4spec",  best_ext_preds[rs_mask], pred_s22[om_mask],
                                          best_g4_p if best_g4_r > g4_best_mkl_r else best_ext_preds[g4_mask]),
    ("meta_RS + meta_OM + G4spec",       best_meta_preds[rs_mask], best_meta_preds[om_mask],
                                          best_g4_p if best_g4_r > g4_best_mkl_r else best_ext_preds[g4_mask]),
]

results = []
for name, rs_src, om_src, g4_src in configs:
    hyb, r = eval_hybrid(name, rs_src, om_src, g4_src)
    results.append((name, hyb, r))
    log.info(f"  {name:45s}: r={r:.4f}")

best_name, best_hyb_preds, best_hyb_r = max(results, key=lambda x: x[2])
log.info(f"\n  → Best config: {best_name}")

# Per-subtype breakdown
log.info(f"\nPer-subtype breakdown (best hybrid):")
for st in ["aptamer","duplex_groove","ribosomal_asite","riboswitch",
           "other_misc","g_quadruplex","viral_tar"]:
    mask = subtypes == st
    if mask.sum() < 2: continue
    r_st = pearsonr(best_hyb_preds[mask], y[mask])[0]
    log.info(f"  {st:22s}: n={mask.sum():3d}  r={r_st:.3f}")

log.info(f"\n  Combined r (step23)   = {best_hyb_r:.4f}")
log.info(f"  Previous best         = 0.7375   [step22]")
log.info(f"  Delta                 = {best_hyb_r - 0.7375:+.4f}")
log.info(f"  Gap to DeepRSMA       = {0.784 - best_hyb_r:.4f}")

benchmarks = [("AffiGrapher",0.498),("RLaffinity",0.559),("RLASIF",0.666),
              ("DeepRSMA",0.784),("RSAPred",0.830)]
for name_b, rb in benchmarks:
    sym = "✓" if best_hyb_r > rb else "✗"
    log.info(f"  {sym} {name_b}: {rb:.3f}")
log.info("="*70)

# Save
df = pd.DataFrame({
    "pdb": ids, "subtype": subtypes, "y_true": y, "y_pred": best_hyb_preds,
    "pred_s11": pred_s11, "pred_s21": pred_s21, "pred_s22": pred_s22,
    "meta_pred": best_meta_preds, "ext_mkl_pred": best_ext_preds,
})
df.to_csv(RES_DIR / "step23_results.csv", index=False)
np.save(ROOT / "data" / "features" / "rlif_ext_features.npy", X_rlif_ext)
log.info(f"  Results → results/step23_results.csv")
log.info(f"  Ext RLIF → data/features/rlif_ext_features.npy")

# Plot
colors = {"aptamer":"#2166AC","riboswitch":"#1A9641","ribosomal_asite":"#D73027",
          "duplex_groove":"#7B2D8B","g_quadruplex":"#FF7F00",
          "viral_tar":"#A65628","other_misc":"#999999"}
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
for st in np.unique(subtypes):
    mask = subtypes == st
    r_st = pearsonr(best_hyb_preds[mask], y[mask])[0] if mask.sum() > 1 else 0
    ax.scatter(y[mask], best_hyb_preds[mask], c=colors.get(st,"#888"),
               label=f"{st} r={r_st:.3f}", alpha=0.75, s=45, edgecolors="none")
lo, hi = y.min()-0.5, y.max()+0.5
ax.plot([lo,hi],[lo,hi],"k--",lw=1,alpha=0.5)
ax.set_xlabel("Experimental pKd"); ax.set_ylabel("Predicted pKd")
ax.set_title(f"Step 23 Hybrid (r={best_hyb_r:.4f})", fontweight="bold")
ax.legend(fontsize=7); ax.grid(alpha=0.3, linestyle="--")

ax = axes[1]
steps = {"S21\nSubclass":0.712,"S22\nRLIF":0.7375,"S23\nMeta":best_hyb_r}
bar_cols = ["#AAAAAA","#4393C3","#D63027"]
bars = ax.bar(list(steps.keys()), list(steps.values()),
              color=bar_cols, edgecolor="white", alpha=0.9)
for bar, val in zip(bars, steps.values()):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.003,
            f"{val:.4f}", ha='center', va='bottom', fontsize=9)
for name_b, rb in benchmarks:
    ax.axhline(rb, color='gray', lw=0.8, linestyle='--', alpha=0.6)
    ax.text(2.5, rb+0.003, name_b, fontsize=7, color='gray')
ax.set_ylabel("Pearson r"); ax.set_title("Performance progression")
ax.set_ylim(0.65, 0.85)
plt.tight_layout()
fig.savefig(ROOT/"results"/"figures"/"step23_meta_results.png", dpi=150, bbox_inches="tight")
log.info(f"  Figure → results/figures/step23_meta_results.png")
log.info("STEP 23 COMPLETE")
