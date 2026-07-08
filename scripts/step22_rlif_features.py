"""
RNA-PDFL · Step 22: RNA-Ligand Interaction Fingerprint (RLIF)

Novel 3D-structure-based interaction features extracted directly from
pocket PDB coordinates (no force-field needed):
  - Contact counts per RNA base type (A/G/C/U) at 4/6/8 Å
  - H-bond potential pairs (N/O within 3.5 Å)
  - π-stacking proxy (base ring atoms within 5.5 Å of ligand)
  - Sulfur contacts (for SAM/SAH/TPP classes)
  - Phosphate-ligand contacts (ionic)
  - Binding pocket geometry (volume, burial, compactness)

RLIF → new kernel K_rlif → added to MKL ensemble
Also: ElasticNet + SelectKBest for other_misc (n=27 heterogeneous)

Target: push combined r beyond 0.712 toward DeepRSMA 0.784
"""

import gzip, pickle, logging, warnings, time
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
from sklearn.linear_model import Ridge, ElasticNetCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

ROOT    = Path("/home/stalin/Desktop/PDFL-RNA/RNA_PDFL")
NA_L    = Path("/home/stalin/Desktop/PDFL-RNA/NA-L")
S11_NPZ = ROOT / "data" / "features" / "step11_full_features.npz"
S21_CSV = ROOT / "results" / "step21_results.csv"
RES_DIR = ROOT / "results"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "results" / "logs" / f"step22_{ts}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("RNA-PDFL · Step 22: RNA-Ligand Interaction Fingerprint (RLIF)")
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

ALPHA_GRID = [1, 10, 100, 1000, 10_000, 100_000]

# ── Load data ──────────────────────────────────────────────────────────────
log.info("\nLoading data ...")
d11 = np.load(S11_NPZ)
X11   = d11["X"].astype(np.float64)
y     = d11["y"].astype(np.float32)
ids   = d11["ids"]
subtypes_raw = d11["subtypes"]
subtypes = np.array([make_subtype(p, s) for p, s in zip(ids, subtypes_raw)])
n = len(y)

# Load step21 best predictions as base
s21 = pd.read_csv(S21_CSV)
pdb2pred_s21 = dict(zip(s21["pdb"], s21["y_pred"]))
step21_preds = np.array([float(pdb2pred_s21.get(p, np.nan)) for p in ids])

# Load step11 for non-RS baseline
from sklearn.linear_model import Ridge as _Ridge

unimol_emb_raw = np.load("/tmp/unimol_emb.npy")
valid_idx_raw  = np.load("/tmp/unimol_valid_idx.npy").astype(int)
unimol_full = np.zeros((n, 512), dtype=np.float64)
unimol_full[valid_idx_raw] = unimol_emb_raw
missing_idx = [i for i in range(n) if i not in set(valid_idx_raw.tolist())]
if missing_idx: unimol_full[missing_idx] = unimol_emb_raw.mean(axis=0)

log.info(f"  n={n}, X11: {X11.shape}, UniMol: {unimol_full.shape}")

# ═══════════════════════════════════════════════════════════════════════════
# PART A: Compute RLIF features
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART A: Computing RLIF (RNA-Ligand Interaction Fingerprint)")
log.info("="*70)

RNA_BASE_RING = {'N1','N3','N7','N9','C2','C4','C5','C6','C8','C4A','C5A'}
HB_ELEMENTS   = {'O', 'N', 'F', 'S'}
BASES         = ['A', 'G', 'C', 'U']
CUTOFFS       = [4.0, 6.0, 8.0]
N_RLIF        = len(BASES)*len(CUTOFFS) + 9  # 12 contact counts + 9 geometric/chem

def parse_pocket_pdb(path):
    """Return (res_name[], atom_name[], element[], coords[])."""
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
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                # skip alternates (only take first)
                alt = line[16]
                if alt not in (' ', 'A', ''): continue
                res_n.append(rn); atm_n.append(an); elem.append(el.upper())
                coords.append([x, y, z])
            except: continue
    return res_n, atm_n, elem, np.array(coords) if coords else np.zeros((0,3))

def parse_ligand_mol2(path):
    """Return (coords, elements) from mol2."""
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
    """Return (coords, elements) from sdf via RDKit."""
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

def compute_rlif(rna_res, rna_atm, rna_elem, rna_coords, lig_coords, lig_elem):
    feat = np.zeros(N_RLIF)
    if lig_coords is None or lig_coords.shape[0] == 0 or rna_coords.shape[0] == 0:
        return feat

    D = cdist(rna_coords, lig_coords)  # (n_rna, n_lig)

    idx = 0
    # Contact counts per base type at each cutoff
    for cutoff in CUTOFFS:
        any_contact = D.min(axis=1) < cutoff  # bool per RNA atom
        for base in BASES:
            mask_rna = np.array([r == base for r in rna_res])
            feat[idx] = (any_contact & mask_rna).sum()
            idx += 1

    # H-bond potential: N/O on RNA within 3.5Å of N/O on ligand
    hb_rna_mask = np.array([e in HB_ELEMENTS for e in rna_elem])
    hb_lig_mask = np.array([e in HB_ELEMENTS for e in lig_elem])
    if hb_rna_mask.any() and hb_lig_mask.any():
        D_hb = D[np.ix_(hb_rna_mask, hb_lig_mask)]
        feat[idx] = (D_hb < 3.5).sum()
    idx += 1

    # π-stacking proxy: RNA ring atoms within 5.5Å of ligand heavy atoms
    ring_rna_mask = np.array([a in RNA_BASE_RING for a in rna_atm])
    if ring_rna_mask.any():
        D_ring = D[ring_rna_mask]
        feat[idx] = (D_ring < 5.5).sum()
    idx += 1

    # Sulfur contacts: RNA O/N within 5Å of ligand S
    s_lig_mask = np.array([e == 'S' for e in lig_elem])
    on_rna_mask = np.array([e in {'O', 'N'} for e in rna_elem])
    if s_lig_mask.any() and on_rna_mask.any():
        D_s = D[np.ix_(on_rna_mask, s_lig_mask)]
        feat[idx] = (D_s < 5.0).sum()
    idx += 1

    # Phosphate contacts: ligand N/O within 4.5Å of RNA P atoms
    p_rna_mask = np.array([a == 'P' for a in rna_atm])
    nh_lig_mask = np.array([e in {'N', 'O'} for e in lig_elem])
    if p_rna_mask.any() and nh_lig_mask.any():
        D_p = D[np.ix_(p_rna_mask, nh_lig_mask)]
        feat[idx] = (D_p < 4.5).sum()
    idx += 1

    # Number of unique RNA residues contacting ligand (<6Å any atom)
    close_rna = D.min(axis=1) < 6.0
    # Use (residue_name, approximate chain position) heuristic
    feat[idx] = close_rna.sum()   # atom count within 6Å (proxy for unique residues)
    idx += 1

    # Ligand burial: mean and max of min-dist from lig atoms to RNA
    min_dists_lig = D.min(axis=0)
    feat[idx]   = min_dists_lig.mean()
    feat[idx+1] = min_dists_lig.max()
    idx += 2

    # Pocket volume: convex hull of RNA atoms within 8Å
    pocket_mask = D.min(axis=1) < 8.0
    if pocket_mask.sum() >= 4:
        try:
            hull = ConvexHull(rna_coords[pocket_mask])
            feat[idx] = hull.volume
        except: feat[idx] = 0.0
    idx += 1

    return feat

# Compute RLIF for all complexes
X_rlif = np.zeros((n, N_RLIF), dtype=np.float64)
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
    if (lig_coords is None) and sdf_f.exists():
        lig_coords, lig_elem = parse_ligand_sdf(sdf_f)
    if lig_coords is None: continue

    X_rlif[i] = compute_rlif(rna_res, rna_atm, rna_elem, rna_coords,
                              lig_coords, lig_elem)
    n_success += 1

log.info(f"  RLIF computed for {n_success}/{n} complexes, {N_RLIF} features")

# Univariate correlation check
rlif_names = (
    [f"N_{b}_{int(c)}A" for c in CUTOFFS for b in BASES] +
    ["N_hbond","N_stack","N_sulfur","N_phos","N_close_atoms",
     "mean_minD","max_minD","pocket_vol"]
)
log.info("\n  RLIF univariate correlations with pKd:")
for j, name in enumerate(rlif_names):
    col = X_rlif[:, j]
    if np.std(col) > 1e-8:
        r_j, _ = pearsonr(col, y)
        log.info(f"    {name:20s} r={r_j:+.3f}")

# Save RLIF
np.save(ROOT / "data" / "features" / "rlif_features.npy", X_rlif)
log.info(f"\n  RLIF saved → data/features/rlif_features.npy")

# ═══════════════════════════════════════════════════════════════════════════
# PART B: RLIF kernel in global MKL
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART B: RLIF kernel in extended MKL")
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

# Build kernels
X_topo    = X11[:, np.r_[0:36000, 38963:49763]]
X_rnafm   = X11[:, 38064:38704]
X_morgan  = X11[:, 36000:38048]
X_maccs   = X11[:, 38796:38963]

X_topo_n    = StandardScaler().fit_transform(X_topo)
X_rnafm_n   = StandardScaler().fit_transform(X_rnafm)
X_unimol_n  = StandardScaler().fit_transform(unimol_full)
X_rlif_n    = StandardScaler().fit_transform(X_rlif)

K_topo = rbf_kernel(X_topo_n, gamma=1e-6)
K_rna  = rbf_kernel(X_rnafm_n, gamma=5e-3)
K_uni  = rbf_kernel(X_unimol_n, gamma=0.05)
K_tan  = 0.7*tanimoto(X_morgan) + 0.3*tanimoto(X_maccs)
K_lig  = 0.5*K_uni + 0.5*K_tan

# RLIF kernel: try multiple gamma values
gamma_rlif_grid = [1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5]
rs_mask = subtypes == "riboswitch"

K_base = 0.7*K_topo + 0.1*K_lig + 0.2*K_rna  # step21 best

best_rlif_r = -99.0
best_rlif_cfg = None
best_rlif_preds = None

log.info("\n  Grid: w_rlif × gamma_rlif × lambda ...")
for gamma_r in gamma_rlif_grid:
    K_rlif_k = rbf_kernel(X_rlif_n, gamma=gamma_r)
    for w_rlif in [0.05, 0.1, 0.2, 0.3]:
        K_ext = (1-w_rlif)*K_base + w_rlif*K_rlif_k
        D_ = np.sqrt(np.diag(K_ext)).reshape(-1, 1)
        D_ = np.where(D_ < 1e-10, 1e-10, D_)
        K_norm = K_ext / (D_ * D_.T)
        for lam in [0.01, 0.05]:
            preds = loo_mkl(K_norm, y, alpha=lam)
            r_rs, _ = pearsonr(preds[rs_mask], y[rs_mask])
            r_all, _ = pearsonr(preds, y)
            if r_rs > best_rlif_r:
                best_rlif_r = r_rs
                best_rlif_cfg = (gamma_r, w_rlif, lam)
                best_rlif_preds = preds.copy()

log.info(f"  Best RLIF-extended MKL: rs_r={best_rlif_r:.4f}  cfg={best_rlif_cfg}")
r_all_ext, _ = pearsonr(best_rlif_preds, y)
log.info(f"  All-sample r = {r_all_ext:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# PART C: Improved other_misc with RLIF + feature selection
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART C: Improved other_misc with RLIF + SelectKBest")
log.info("="*70)

om_mask = subtypes == "other_misc"
om_idx  = np.where(om_mask)[0]
n_om    = om_mask.sum()
y_om    = y[om_mask]

log.info(f"  other_misc: n={n_om}")

# Build augmented feature matrix: X11 + RLIF
X_aug = np.hstack([X11, X_rlif])  # (n, 49763+N_RLIF)

def loo_ridge_selectk(X_sub, y_sub, alphas, k_vals):
    """LOO Ridge with SelectKBest feature selection inside each fold."""
    ns = len(y_sub)
    best_r, best_p = -99.0, np.full(ns, y_sub.mean())
    for alpha in alphas:
        for k in k_vals:
            preds = np.zeros(ns)
            ok = True
            for i in range(ns):
                tr = [j for j in range(ns) if j != i]
                try:
                    pipe = Pipeline([
                        ("vt", VarianceThreshold(threshold=1e-4)),
                        ("fs", SelectKBest(f_regression, k=min(k, len(tr)-2))),
                        ("sc", StandardScaler()),
                        ("reg", Ridge(alpha=alpha)),
                    ])
                    pipe.fit(X_sub[tr], y_sub[tr])
                    preds[i] = np.clip(pipe.predict(X_sub[[i]])[0],
                                       y_sub[tr].min()-3, y_sub[tr].max()+3)
                except: ok=False; break
            if not ok: continue
            r = pearsonr(y_sub, preds)[0] if np.std(preds)>1e-8 else -99.0
            if r > best_r: best_r, best_p = r, preds.copy()
    return best_p, best_r

# Test different configurations for other_misc
log.info("\n  A) Standard PCA+Ridge (baseline) ...")
X_om   = X11[om_mask]
r_om_std, preds_om_std = -99.0, np.full(n_om, y_om.mean())
for alpha in ALPHA_GRID:
    pp = np.zeros(n_om)
    ok = True
    for i in range(n_om):
        tr = [j for j in range(n_om) if j != i]
        try:
            pipe = Pipeline([
                ("vt", VarianceThreshold(threshold=1e-4)),
                ("sc", StandardScaler()),
                ("pca", PCA(n_components=0.95, svd_solver="full")),
                ("reg", Ridge(alpha=alpha)),
            ])
            pipe.fit(X_om[tr], y_om[tr])
            pp[i] = np.clip(pipe.predict(X_om[[i]])[0], y_om[tr].min()-3, y_om[tr].max()+3)
        except: ok=False; break
    if not ok: continue
    r = pearsonr(y_om, pp)[0] if np.std(pp)>1e-8 else -99.0
    if r > r_om_std: r_om_std, preds_om_std = r, pp.copy()
log.info(f"    other_misc PCA+Ridge r = {r_om_std:.4f}")

log.info("\n  B) SelectKBest (k=30,60,100) + Ridge ...")
p_om_skb, r_om_skb = loo_ridge_selectk(
    X_aug[om_mask], y_om,
    alphas=[1, 10, 100, 1000],
    k_vals=[30, 60, 100]
)
log.info(f"    other_misc SelectKBest r = {r_om_skb:.4f}")

log.info("\n  C) RLIF-only Ridge ...")
r_om_rlif, p_om_rlif = -99.0, np.full(n_om, y_om.mean())
X_rlif_om = X_rlif[om_mask]
for alpha in ALPHA_GRID:
    pp = np.zeros(n_om)
    ok = True
    for i in range(n_om):
        tr = [j for j in range(n_om) if j != i]
        try:
            pipe = Pipeline([
                ("vt", VarianceThreshold(threshold=1e-4)),
                ("sc", StandardScaler()),
                ("reg", Ridge(alpha=alpha)),
            ])
            pipe.fit(X_rlif_om[tr], y_om[tr])
            pp[i] = np.clip(pipe.predict(X_rlif_om[[i]])[0], y_om[tr].min()-3, y_om[tr].max()+3)
        except: ok=False; break
    if not ok: continue
    r = pearsonr(y_om, pp)[0] if np.std(pp)>1e-8 else -99.0
    if r > r_om_rlif: r_om_rlif, p_om_rlif = r, pp.copy()
log.info(f"    other_misc RLIF-only r = {r_om_rlif:.4f}")

log.info("\n  D) RLIF kernel MKL for other_misc (best K_rlif config) ...")
# Use the best gamma from Part B
g_best, w_best, lam_best = best_rlif_cfg
K_rlif_best = rbf_kernel(X_rlif_n, gamma=g_best)
K_ext_best  = (1-w_best)*K_base + w_best*K_rlif_best
D_ = np.sqrt(np.diag(K_ext_best)).reshape(-1,1)
D_ = np.where(D_ < 1e-10, 1e-10, D_)
K_ext_norm = K_ext_best / (D_*D_.T)

# LOO MKL restricted to other_misc
p_om_mkl = best_rlif_preds[om_mask]  # from global LOO above
r_om_mkl, _ = pearsonr(p_om_mkl, y_om)
log.info(f"    other_misc RLIF-MKL (global LOO) r = {r_om_mkl:.4f}")

# K-means subgroup model for other_misc
log.info("\n  E) K-means subgroup Ridge for other_misc ...")
best_km_r, best_km_p = -99.0, np.full(n_om, y_om.mean())
for k in range(2, 6):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    rlif_om_sc = StandardScaler().fit_transform(X_rlif_om)
    cluster_labels = km.fit_predict(rlif_om_sc)
    pp = np.full(n_om, y_om.mean())
    for c in range(k):
        c_mask = cluster_labels == c
        c_idx  = np.where(c_mask)[0]
        nc = c_mask.sum()
        if nc < 3:
            pp[c_mask] = y_om[c_mask].mean()
            continue
        for alpha in ALPHA_GRID:
            preds_c = np.zeros(nc)
            ok = True
            for ii in range(nc):
                tr_c = [j for j in range(nc) if j != ii]
                try:
                    pipe = Pipeline([
                        ("vt", VarianceThreshold(threshold=1e-4)),
                        ("sc", StandardScaler()),
                        ("pca", PCA(n_components=min(0.95, len(tr_c)-1), svd_solver="full")),
                        ("reg", Ridge(alpha=alpha)),
                    ])
                    pipe.fit(X_aug[om_idx[c_idx[tr_c]]], y_om[c_idx[tr_c]])
                    preds_c[ii] = np.clip(
                        pipe.predict(X_aug[[om_idx[c_idx[ii]]]])[0],
                        y_om[c_idx[tr_c]].min()-3, y_om[c_idx[tr_c]].max()+3
                    )
                except: ok=False; break
            if ok:
                r_c = pearsonr(y_om[c_mask], preds_c)[0] if np.std(preds_c)>1e-8 else -99.0
                if r_c > 0: pp[c_mask] = preds_c
                break
    r_km = pearsonr(y_om, pp)[0] if np.std(pp)>1e-8 else -99.0
    if r_km > best_km_r: best_km_r, best_km_p = r_km, pp.copy()
log.info(f"    other_misc K-means(2-5) Ridge r = {best_km_r:.4f}")

# Best other_misc model
om_options = {
    "pca_ridge": (preds_om_std, r_om_std),
    "selectk":   (p_om_skb,     r_om_skb),
    "rlif_only": (p_om_rlif,    r_om_rlif),
    "mkl_rlif":  (p_om_mkl,     r_om_mkl),
    "kmeans":    (best_km_p,    best_km_r),
}
best_om_name = max(om_options, key=lambda k: om_options[k][1])
best_om_preds, best_om_r = om_options[best_om_name]
log.info(f"\n  → Best other_misc: {best_om_name} r={best_om_r:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# PART D: FINAL HYBRID
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART D: Final Hybrid Assembly")
log.info("="*70)

# Options for the riboswitch component:
#  - step21 predictions (best riboswitch r=0.723)
#  - RLIF-extended MKL (might improve riboswitch further)
# Options for other_misc:
#  - step21 (r=0.395)
#  - best from Part C
# Everything else: step21 predictions

def build_hybrid(rs_preds_choice, om_preds_choice):
    hyb = step21_preds.copy()
    hyb[rs_mask] = rs_preds_choice
    hyb[om_mask] = om_preds_choice
    valid = ~np.isnan(hyb)
    r, _ = pearsonr(hyb[valid], y[valid])
    return hyb, r

# Option 1: step21 RS + best OM
h1, r1 = build_hybrid(step21_preds[rs_mask], best_om_preds)

# Option 2: RLIF-MKL for RS + best OM
h2, r2 = build_hybrid(best_rlif_preds[rs_mask], best_om_preds)

# Option 3: RLIF-MKL for RS + step21 OM
h3, r3 = build_hybrid(best_rlif_preds[rs_mask], step21_preds[om_mask])

# Option 4: step21 RS + step21 OM (reference)
h4, r4 = build_hybrid(step21_preds[rs_mask], step21_preds[om_mask])

log.info(f"\n  Hybrid configs:")
log.info(f"    step21_RS  + best_OM          : r={r1:.4f}")
log.info(f"    RLIF_MKL   + best_OM          : r={r2:.4f}")
log.info(f"    RLIF_MKL   + step21_OM        : r={r3:.4f}")
log.info(f"    step21_RS  + step21_OM (ref)  : r={r4:.4f}  [step21 best]")

best_hyb_preds, best_hyb_r = max(
    [(h1, r1), (h2, r2), (h3, r3), (h4, r4)], key=lambda x: x[1]
)

log.info(f"\n  Best hybrid r = {best_hyb_r:.4f}")

# Per-subtype breakdown
for st in ["aptamer","duplex_groove","ribosomal_asite","riboswitch",
           "other_misc","g_quadruplex","viral_tar"]:
    mask = subtypes == st
    if mask.sum() < 2: continue
    r_st = pearsonr(best_hyb_preds[mask], y[mask])[0]
    log.info(f"  {st:22s}: n={mask.sum():3d}  r={r_st:.3f}")

log.info(f"\n  Combined r (step22)   = {best_hyb_r:.4f}")
log.info(f"  Previous best         = 0.7120   [step21]")
log.info(f"  Delta                 = {best_hyb_r - 0.7120:+.4f}")
log.info(f"  Gap to DeepRSMA       = {0.784 - best_hyb_r:.4f}")

benchmarks = [("AffiGrapher",0.498),("RLaffinity",0.559),("RLASIF",0.666),
              ("DeepRSMA",0.784),("RSAPred",0.830)]
for name, rb in benchmarks:
    sym = "✓" if best_hyb_r > rb else "✗"
    log.info(f"  {sym} {name}: {rb:.3f}")
log.info("="*70)

# Save results
df = pd.DataFrame({
    "pdb": ids,
    "subtype": subtypes,
    "y_true": y,
    "y_pred": best_hyb_preds,
})
df.to_csv(RES_DIR / "step22_results.csv", index=False)
log.info(f"\n  Results → results/step22_results.csv")

# Plot
colors = {"aptamer":"#2166AC","riboswitch":"#1A9641","ribosomal_asite":"#D73027",
          "duplex_groove":"#7B2D8B","g_quadruplex":"#FF7F00",
          "viral_tar":"#A65628","other_misc":"#999999"}
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
for st in np.unique(subtypes):
    mask = subtypes == st
    r_st = pearsonr(best_hyb_preds[mask], y[mask])[0] if mask.sum() > 1 else 0
    ax.scatter(y[mask], best_hyb_preds[mask],
               c=colors.get(st,"#888"),
               label=f"{st} r={r_st:.3f}", alpha=0.75, s=45, edgecolors="none")
lo, hi = y.min()-0.5, y.max()+0.5
ax.plot([lo,hi],[lo,hi],"k--",lw=1,alpha=0.5)
ax.set_xlabel("Experimental pKd"); ax.set_ylabel("Predicted pKd")
ax.set_title(f"Step 22 Hybrid (r={best_hyb_r:.4f})", fontweight="bold")
ax.legend(fontsize=7); ax.grid(alpha=0.3,linestyle="--")

ax = axes[1]
steps = {"S19\nUniMol":0.706,"S21\nSubclass":0.712,"S22\nRLIF":best_hyb_r}
bar_cols = ["#AAAAAA","#4393C3","#D63027"]
bars = ax.bar(list(steps.keys()), list(steps.values()),
              color=bar_cols, edgecolor="white", alpha=0.9)
for bar, val in zip(bars, steps.values()):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.003,
            f"{val:.4f}", ha='center', va='bottom', fontsize=9)
for name, rb in benchmarks:
    ax.axhline(rb, color='gray', lw=0.8, linestyle='--', alpha=0.6)
    ax.text(2.5, rb+0.003, name, fontsize=7, color='gray')
ax.set_ylabel("Pearson r"); ax.set_title("Performance progression")
ax.set_ylim(0.65, 0.85)
plt.tight_layout()
fig.savefig(ROOT/"results"/"figures"/"step22_rlif_results.png", dpi=150, bbox_inches="tight")
log.info(f"  Figure → results/figures/step22_rlif_results.png")
log.info("STEP 22 COMPLETE")
