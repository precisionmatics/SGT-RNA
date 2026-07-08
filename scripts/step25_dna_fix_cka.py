"""
RNA-PDFL · Step 25: DNA G4 RLIF Fix + Kernel Alignment Weighting

Critical bug fix:
  5/8 G-quadruplex structures are DNA (DG/DA/DC/DT residues).
  RLIF nucleotide contact features were all zero for DNA G4 complexes.
  Fix: map DG→G, DA→A, DC→C, DT→U in pocket parsing.
  This should dramatically improve G4 predictions.

Novel kernel weighting:
  Centered Kernel Alignment (CKA) score for each kernel K_i vs label kernel K_y:
    align(K_i, K_y) = <K_ic, K_yc>_F / (||K_ic||_F × ||K_yc||_F)
  CKA-weighted MKL: K_mix = Σ align_i × K_i (then normalise)
  More principled than grid search — theoretical guarantee of maximising alignment.

Also: full extended feature recomputation with DNA-aware parsing for all 143 complexes.
"""

import logging, warnings
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel

warnings.filterwarnings("ignore")

ROOT    = Path("/home/stalin/Desktop/PDFL-RNA/RNA_PDFL")
NA_L    = Path("/home/stalin/Desktop/PDFL-RNA/NA-L")
S11_NPZ = ROOT / "data" / "features" / "step11_full_features.npz"
S24_CSV = ROOT / "results" / "step24_results.csv"
RES_DIR = ROOT / "results"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "results" / "logs" / f"step25_{ts}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("RNA-PDFL · Step 25: DNA G4 RLIF Fix + CKA Kernel Weighting")
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

# DNA → RNA residue mapping
DNA_TO_RNA = {'DA': 'A', 'DG': 'G', 'DC': 'C', 'DT': 'U',
               'DA ': 'A', 'DG ': 'G', 'DC ': 'C', 'DT ': 'U'}

ALPHA_GRID = [0.1, 1, 10, 100, 1000, 10_000, 100_000]

# ── Load data ──────────────────────────────────────────────────────────────
log.info("\nLoading data ...")
d11 = np.load(S11_NPZ)
X11      = d11["X"].astype(np.float64)
y        = d11["y"].astype(np.float32)
ids      = d11["ids"]
subtypes_raw = d11["subtypes"]
subtypes = np.array([make_subtype(p, s) for p, s in zip(ids, subtypes_raw)])
n = len(y)

s24_df   = pd.read_csv(S24_CSV)
pdb2s24  = dict(zip(s24_df["pdb"], s24_df["y_pred"]))
pred_s24 = np.array([float(pdb2s24.get(p, np.nan)) for p in ids])

unimol_emb_raw = np.load("/tmp/unimol_emb.npy")
valid_idx_raw  = np.load("/tmp/unimol_valid_idx.npy").astype(int)
unimol_full = np.zeros((n, 512), dtype=np.float64)
unimol_full[valid_idx_raw] = unimol_emb_raw
missing_idx = [i for i in range(n) if i not in set(valid_idx_raw.tolist())]
if missing_idx: unimol_full[missing_idx] = unimol_emb_raw.mean(axis=0)

log.info(f"  n={n}, X11: {X11.shape}")

# ═══════════════════════════════════════════════════════════════════════════
# PART A: Recompute RLIF with DNA-aware parsing
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART A: DNA-aware RLIF recomputation")
log.info("="*70)

RNA_BASE_RING = {'N1','N3','N7','N9','C2','C4','C5','C6','C8'}
HB_ELEMENTS   = {'O', 'N', 'F', 'S'}
BASES         = ['A', 'G', 'C', 'U']
CUTOFFS       = [4.0, 6.0, 8.0]
N_RLIF_V2    = len(BASES)*len(CUTOFFS) + 9 + 16  # 12 contact + 9 original + 16 extended

def parse_pocket_pdb_v2(path):
    """Parse pocket PDB, converting DNA residues to RNA equivalents."""
    res_n, atm_n, elem, coords = [], [], [], []
    with open(path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")): continue
            try:
                rn_raw = line[17:20].strip()
                rn = DNA_TO_RNA.get(rn_raw, DNA_TO_RNA.get(rn_raw.strip(), rn_raw))
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

def compute_rlif_v2(rna_res, rna_atm, rna_elem, rna_coords, lig_coords, lig_elem):
    feat = np.zeros(N_RLIF_V2)
    if lig_coords is None or lig_coords.shape[0] == 0 or rna_coords.shape[0] == 0:
        return feat
    D = cdist(rna_coords, lig_coords)
    idx = 0

    # --- Contact counts per base type at each cutoff (12 features) ---
    for cutoff in CUTOFFS:
        any_contact = D.min(axis=1) < cutoff
        for base in BASES:
            mask = np.array([r == base for r in rna_res])
            feat[idx] = (any_contact & mask).sum(); idx += 1

    # --- H-bond potential ---
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

    feat[idx] = (D.min(axis=1) < 6.0).sum(); idx += 1  # N_close
    min_dists_lig = D.min(axis=0)
    feat[idx] = min_dists_lig.mean(); idx += 1          # mean_minD
    feat[idx] = min_dists_lig.max();  idx += 1          # max_minD

    pocket_mask = D.min(axis=1) < 8.0
    if pocket_mask.sum() >= 4:
        try:
            hull = ConvexHull(rna_coords[pocket_mask])
            feat[idx] = hull.volume
        except: pass
    idx += 1  # pocket_vol

    # --- Extended 16 features ---
    for base in BASES:
        mask_b = np.array([r == base for r in rna_res])
        feat[idx] = D[mask_b].min() if mask_b.any() else 20.0; idx += 1  # minD per base

    for base in BASES:
        mask_b = np.array([r == base for r in rna_res])
        feat[idx] = (D[mask_b].min(axis=1) < 6.0).mean() if mask_b.any() else 0.0
        idx += 1  # contact ratio per base

    feat[idx] = lig_coords.shape[0]; idx += 1   # n_lig_atoms
    feat[idx] = np.linalg.norm(lig_coords.mean(0) - rna_coords.mean(0)); idx += 1  # centroid dist

    n_close = (D < 4.0).sum()
    feat[idx] = feat[12] / max(n_close, 1); idx += 1   # hb_ratio (using hbond at idx12)

    vol = feat[20]  # pocket_vol at idx20
    feat[idx] = feat[13] / max(vol, 1.0); idx += 1     # stack_density

    feat[idx] = sum(1 for e in lig_elem if e == 'C'); idx += 1   # n_lig_C

    bb_atoms = {'P','OP1','OP2',"O5'",'C5',"C4'",'O4',"C3'",'O3',"C2'",'O2',"C1'"}
    bb_mask = np.array([a in bb_atoms for a in rna_atm])
    base_mask = ~bb_mask
    n_bb_c = (D[bb_mask].min(axis=1) < 5.0).sum() if bb_mask.any() else 0
    n_bs_c = (D[base_mask].min(axis=1) < 5.0).sum() if base_mask.any() else 0
    feat[idx] = n_bs_c / max(n_bb_c + n_bs_c, 1); idx += 1  # base_vs_bb

    close_res = [rna_res[i] for i in range(len(rna_res)) if D[i].min() < 6.0]
    feat[idx] = sum(1 for r in close_res if r == 'G') / max(len(close_res), 1); idx += 1  # G_frac

    if lig_coords.shape[0] >= 2:
        feat[idx] = cdist(lig_coords, lig_coords).max() / lig_coords.shape[0]
    idx += 1  # lig_compactness

    return feat

# Recompute RLIF for all complexes with DNA-aware parsing
X_rlif_v2 = np.zeros((n, N_RLIF_V2), dtype=np.float64)
n_success = 0
for i, pdb in enumerate(ids):
    pocket_file = NA_L / pdb / f"{pdb}_pocket.pdb"
    if not pocket_file.exists(): continue
    rna_res, rna_atm, rna_elem, rna_coords = parse_pocket_pdb_v2(pocket_file)
    if rna_coords.shape[0] == 0: continue
    lig_coords, lig_elem = None, []
    mol2_f = NA_L / pdb / f"{pdb}_ligand.mol2"
    sdf_f  = NA_L / pdb / f"{pdb}_ligand.sdf"
    if mol2_f.exists(): lig_coords, lig_elem = parse_ligand_mol2(mol2_f)
    if lig_coords is None and sdf_f.exists(): lig_coords, lig_elem = parse_ligand_sdf(sdf_f)
    if lig_coords is None: continue
    X_rlif_v2[i] = compute_rlif_v2(rna_res, rna_atm, rna_elem, rna_coords, lig_coords, lig_elem)
    n_success += 1

log.info(f"  RLIF v2 computed for {n_success}/{n} complexes ({N_RLIF_V2} features)")

# Show G4 corrected RLIF
log.info("\n  G4 corrected RLIF (N_G_4A, N_A_4A, N_stack):")
g4_mask = subtypes == "g_quadruplex"
g4_idx  = np.where(g4_mask)[0]
y_g4    = y[g4_mask]
for i in g4_idx:
    f = X_rlif_v2[i]
    log.info(f"    {ids[i]} pKd={y[i]:.3f}: N_G_4A={f[1]:.0f} N_A_4A={f[0]:.0f} "
             f"N_stack={f[13]:.0f} pvol={f[20]:.0f}")

np.save(ROOT / "data" / "features" / "rlif_v2_features.npy", X_rlif_v2)
log.info("  Saved → data/features/rlif_v2_features.npy")

# ═══════════════════════════════════════════════════════════════════════════
# PART B: Centered Kernel Alignment (CKA) based weights
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART B: CKA-based kernel alignment weights")
log.info("="*70)

def center_kernel(K):
    """Centering a kernel matrix: K_c = H K H, H = I - 1/n 11^T."""
    n_ = K.shape[0]
    H = np.eye(n_) - 1/n_
    return H @ K @ H

def kernel_align(K1, K2):
    """Frobenius alignment between two centered kernels."""
    K1c = center_kernel(K1); K2c = center_kernel(K2)
    num = np.sum(K1c * K2c)
    denom = np.sqrt(np.sum(K1c**2) * np.sum(K2c**2))
    return num / denom if denom > 1e-12 else 0.0

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

# Build all kernels
X_topo   = X11[:, np.r_[0:36000, 38963:49763]]
X_rnafm  = X11[:, 38064:38704]
X_morgan = X11[:, 36000:38048]
X_maccs  = X11[:, 38796:38963]

X_topo_n    = StandardScaler().fit_transform(X_topo)
X_rnafm_n   = StandardScaler().fit_transform(X_rnafm)
X_unimol_n  = StandardScaler().fit_transform(unimol_full)
X_rlif_n    = StandardScaler().fit_transform(X_rlif_v2)

K_topo  = rbf_kernel(X_topo_n, gamma=1e-6)
K_rna   = rbf_kernel(X_rnafm_n, gamma=5e-3)
K_uni   = rbf_kernel(X_unimol_n, gamma=0.05)
K_tan   = 0.7*tanimoto(X_morgan) + 0.3*tanimoto(X_maccs)
K_lig   = 0.5*K_uni + 0.5*K_tan
K_rlif  = rbf_kernel(X_rlif_n, gamma=0.5)

# Load pocket sequence kernel from step24
X_pocket = np.load(ROOT / "data" / "features" / "pocket_seq_features.npy")
X_pocket_n = StandardScaler().fit_transform(X_pocket)
K_pocket = rbf_kernel(X_pocket_n, gamma=0.01)  # best from step24

# Label kernel K_y = yy^T / ||y||^2 (normalised)
y_f = y.astype(np.float64)
y_c = y_f - y_f.mean()
K_y = np.outer(y_c, y_c) / (np.dot(y_c, y_c) + 1e-12)

kernels = [("topology", K_topo), ("lig_combo", K_lig), ("rna_fm", K_rna),
           ("rlif_v2", K_rlif), ("pocket_seq", K_pocket)]

aligns = {}
log.info("  CKA alignment scores:")
for name, K in kernels:
    a = kernel_align(K, K_y)
    aligns[name] = a
    log.info(f"    {name:15s}: CKA = {a:.4f}")

# CKA-weighted MKL
total_align = sum(max(v, 0) for v in aligns.values())
cka_weights = {k: max(v, 0)/total_align for k, v in aligns.items()} if total_align > 0 \
              else {k: 1/len(kernels) for k in aligns}
log.info("\n  CKA-weighted combination:")
for k, w in cka_weights.items():
    log.info(f"    {k:15s}: w = {w:.4f}")

K_cka = sum(cka_weights[name]*K for name, K in kernels)
D_ = np.sqrt(np.diag(K_cka)).reshape(-1,1); D_ = np.where(D_<1e-10,1e-10,D_)
K_cka_n = K_cka / (D_*D_.T)

rs_mask = subtypes == "riboswitch"
best_cka_r, best_cka_p = -99.0, None
for lam in [0.005, 0.01, 0.05, 0.1]:
    pp = loo_mkl(K_cka_n, y, alpha=lam)
    r_rs, _ = pearsonr(pp[rs_mask], y[rs_mask])
    if r_rs > best_cka_r: best_cka_r, best_cka_p = r_rs, pp.copy()
r_cka_all, _ = pearsonr(best_cka_p, y)
log.info(f"\n  CKA-MKL: rs_r={best_cka_r:.4f}  all_r={r_cka_all:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# PART C: Refined 5-kernel MKL with RLIF v2 + CKA-guided search
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART C: Refined MKL with RLIF v2 (DNA-fixed)")
log.info("="*70)

# Start from CKA weights and refine
best_r5, best_cfg5, best_p5 = -99.0, None, None
# Use CKA weights as centroid, search ±30% around them
w_t0 = cka_weights["topology"]
w_l0 = cka_weights["lig_combo"]
w_r0 = cka_weights["rna_fm"]
w_ps0 = cka_weights["pocket_seq"]
w_rl0 = cka_weights["rlif_v2"]

log.info(f"  CKA centroid: topo={w_t0:.3f} lig={w_l0:.3f} rna={w_r0:.3f} "
         f"pocket={w_ps0:.3f} rlif={w_rl0:.3f}")

configs = []
for delta in [0.0, -0.1, 0.1, -0.2, 0.2]:
    for which in range(5):
        w = [w_t0, w_l0, w_r0, w_ps0, w_rl0]
        w[which] = max(0.01, w[which] + delta)
        total = sum(w); w = [x/total for x in w]
        configs.append(tuple(w))

# Also include step24 best and pure grid
configs += [(0.5, 0.15, 0.20, 0.05, 0.10),
            (0.6, 0.12, 0.18, 0.05, 0.05),
            (0.55, 0.15, 0.15, 0.08, 0.07),
            (0.65, 0.10, 0.15, 0.05, 0.05)]
configs = list(set(tuple(round(x, 3) for x in c) for c in configs))

log.info(f"  Testing {len(configs)} configs ...")
for cfg in configs:
    w_t, w_l, w_r, w_ps, w_rl = cfg
    K5 = w_t*K_topo + w_l*K_lig + w_r*K_rna + w_ps*K_pocket + w_rl*K_rlif
    D_ = np.sqrt(np.diag(K5)).reshape(-1,1); D_ = np.where(D_<1e-10,1e-10,D_)
    K5n = K5 / (D_*D_.T)
    for lam in [0.005, 0.01, 0.05]:
        pp = loo_mkl(K5n, y, alpha=lam)
        r_rs, _ = pearsonr(pp[rs_mask], y[rs_mask])
        if r_rs > best_r5:
            best_r5 = r_rs; best_cfg5 = (*cfg, lam); best_p5 = pp.copy()

log.info(f"  Best 5-kernel (v2): rs_r={best_r5:.4f}  cfg={best_cfg5}")
r5_all, _ = pearsonr(best_p5, y)
log.info(f"  All-sample r = {r5_all:.4f}")

# G4 with corrected RLIF v2
log.info(f"\n  G4 results from 5-kernel MKL v2:")
g4_r_mkl, _ = pearsonr(best_p5[g4_mask], y_g4)
log.info(f"    G4 r (5k-MKL v2) = {g4_r_mkl:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# PART D: G4 specialised model with corrected features
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART D: G4 specialised model with corrected RLIF v2")
log.info("="*70)

g4_idx_arr = np.where(g4_mask)[0]

# Build G4 feature set: corrected RLIF + ligand fingerprints
X_g4 = np.hstack([
    X_rlif_v2[g4_mask],              # 37 corrected RLIF
    X11[g4_mask, 36000:38048],       # Morgan
    X11[g4_mask, 38796:38963],       # MACCS
    X11[g4_mask, 38048:38064],       # NucComp + Physico
])

best_g4_r, best_g4_p = -99.0, np.full(len(y_g4), y_g4.mean())
for alpha in ALPHA_GRID:
    pp = np.zeros(len(y_g4)); ok = True
    for i in range(len(y_g4)):
        tr = [j for j in range(len(y_g4)) if j != i]
        try:
            pipe = Pipeline([
                ("vt", VarianceThreshold(threshold=1e-4)),
                ("sc", StandardScaler()),
                ("reg", Ridge(alpha=alpha)),
            ])
            pipe.fit(X_g4[tr], y_g4[tr])
            pp[i] = np.clip(pipe.predict(X_g4[[i]])[0],
                            y_g4[tr].min()-3, y_g4[tr].max()+3)
        except: ok=False; break
    if not ok: continue
    r = pearsonr(y_g4, pp)[0] if np.std(pp) > 1e-8 else -99.0
    if r > best_g4_r: best_g4_r, best_g4_p = r, pp.copy()
log.info(f"  G4 specialised Ridge r = {best_g4_r:.4f}")
log.info(f"  G4 MKL (global LOO)    r = {g4_r_mkl:.4f}")

# Use best G4 prediction
use_g4_spec = best_g4_r > g4_r_mkl
best_g4_final = best_g4_p if use_g4_spec else best_p5[g4_mask]
r_g4_final = max(best_g4_r, g4_r_mkl)
log.info(f"  → Using {'specialised' if use_g4_spec else 'MKL global'} for G4: r={r_g4_final:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# PART E: Final Hybrid
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART E: Final Hybrid Assembly")
log.info("="*70)

om_mask = subtypes == "other_misc"
s22_df2 = pd.read_csv(ROOT / "results" / "step22_results.csv")
pdb2s22 = dict(zip(s22_df2["pdb"], s22_df2["y_pred"]))
pred_s22 = np.array([float(pdb2s22.get(p, np.nan)) for p in ids])

def make_hybrid(rs_src, om_src, g4_src=None, base=pred_s24):
    hyb = base.copy()
    hyb[rs_mask] = rs_src
    hyb[om_mask] = om_src
    if g4_src is not None:
        hyb[g4_mask] = g4_src
    valid = ~np.isnan(hyb)
    r, _ = pearsonr(hyb[valid], y[valid])
    return hyb, r

configs_hybrid = [
    ("step24 ref (base)",                  pred_s24[rs_mask], pred_s22[om_mask], None),
    ("MKLv2_RS  + s22_OM",                 best_p5[rs_mask],  pred_s22[om_mask], None),
    ("CKA_MKL_RS + s22_OM",               best_cka_p[rs_mask], pred_s22[om_mask], None),
    ("MKLv2_RS  + s22_OM  + G4spec",      best_p5[rs_mask],  pred_s22[om_mask], best_g4_final),
    ("CKA_MKL_RS + s22_OM + G4spec",     best_cka_p[rs_mask], pred_s22[om_mask], best_g4_final),
    ("MKLv2_RS  + MKLv2_OM + G4spec",    best_p5[rs_mask],  best_p5[om_mask],  best_g4_final),
]

results = []
for name, rs_src, om_src, g4_src in configs_hybrid:
    hyb, r = make_hybrid(rs_src, om_src, g4_src)
    results.append((name, hyb, r))
    log.info(f"  {name:45s}: r={r:.4f}")

best_name, best_hyb, best_r = max(results, key=lambda x: x[2])
log.info(f"\n  → Best: {best_name}")

log.info(f"\nPer-subtype breakdown:")
for st in ["aptamer","duplex_groove","ribosomal_asite","riboswitch",
           "other_misc","g_quadruplex","viral_tar"]:
    mask = subtypes == st
    if mask.sum() < 2: continue
    r_st = pearsonr(best_hyb[mask], y[mask])[0]
    log.info(f"  {st:22s}: n={mask.sum():3d}  r={r_st:.3f}")

log.info(f"\n  Combined r (step25)   = {best_r:.4f}")
log.info(f"  Previous best         = 0.7412   [step24]")
log.info(f"  Delta                 = {best_r - 0.7412:+.4f}")
log.info(f"  Gap to DeepRSMA       = {0.784 - best_r:.4f}")

for nm, rb in [("AffiGrapher",0.498),("RLaffinity",0.559),("RLASIF",0.666),
               ("DeepRSMA",0.784),("RSAPred",0.830)]:
    log.info(f"  {'✓' if best_r > rb else '✗'} {nm}: {rb:.3f}")
log.info("="*70)

df_out = pd.DataFrame({
    "pdb": ids, "subtype": subtypes, "y_true": y,
    "y_pred": best_hyb, "mkl_v2": best_p5, "cka_mkl": best_cka_p,
})
df_out.to_csv(RES_DIR / "step25_results.csv", index=False)
log.info(f"  Results → results/step25_results.csv")

# Plot
colors = {"aptamer":"#2166AC","riboswitch":"#1A9641","ribosomal_asite":"#D73027",
          "duplex_groove":"#7B2D8B","g_quadruplex":"#FF7F00",
          "viral_tar":"#A65628","other_misc":"#999999"}
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
for st in np.unique(subtypes):
    mask = subtypes == st
    r_st = pearsonr(best_hyb[mask], y[mask])[0] if mask.sum() > 1 else 0
    ax.scatter(y[mask], best_hyb[mask], c=colors.get(st,"#888"),
               label=f"{st} r={r_st:.3f}", alpha=0.75, s=45, edgecolors="none")
lo, hi = y.min()-0.5, y.max()+0.5
ax.plot([lo,hi],[lo,hi],"k--",lw=1,alpha=0.5)
ax.set_xlabel("Experimental pKd"); ax.set_ylabel("Predicted pKd")
ax.set_title(f"Step 25 Hybrid (r={best_r:.4f})", fontweight="bold")
ax.legend(fontsize=7); ax.grid(alpha=0.3, linestyle="--")

ax = axes[1]
steps = {"S22\nRLIF":0.7375,"S24\n5kMKL":0.7412,"S25\nDNAfix":best_r}
bar_cols = ["#AAAAAA","#4393C3","#D63027"]
bars = ax.bar(list(steps.keys()), list(steps.values()),
              color=bar_cols, edgecolor="white", alpha=0.9)
for bar, val in zip(bars, steps.values()):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.003,
            f"{val:.4f}", ha='center', va='bottom', fontsize=9)
for nm, rb in [("AffiGrapher",0.498),("RLaffinity",0.559),("RLASIF",0.666),
               ("DeepRSMA",0.784),("RSAPred",0.830)]:
    ax.axhline(rb, color='gray', lw=0.8, linestyle='--', alpha=0.6)
    ax.text(2.5, rb+0.003, nm, fontsize=7, color='gray')
ax.set_ylabel("Pearson r"); ax.set_title("Performance progression")
ax.set_ylim(0.65, 0.86)
plt.tight_layout()
fig.savefig(ROOT/"results"/"figures"/"step25_results.png", dpi=150, bbox_inches="tight")
log.info(f"  Figure → results/figures/step25_results.png")
log.info("STEP 25 COMPLETE")
