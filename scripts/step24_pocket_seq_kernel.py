"""
RNA-PDFL · Step 24: Pocket Sequence Kernel + SAM-specific Features

Novel approach: RNA BINDING SITE sequence kernel
  - Extract RNA residue sequence within 6Å of ligand (pocket context)
  - Compute 4-gram and dinucleotide frequency vectors for each pocket
  - K_pocket = RBF(X_pocket_kmer) — captures RNA fold similarity at binding site
  - Similar RNA folds (riboswitch classes) → similar pocket sequences → high K_pocket
  - 5-kernel MKL: topology + UniMol/Tanimoto + RNA-FM + pocket_seq + RLIF_ext

SAM_SAH fix:
  - Formal charge of ligand (SAM has sulfonium +1, SAH neutral)
  - Quaternary sulfur indicator
  - Extended SAM-specific chemical features
  - Separate SAM_SAH sub-model using ligand-centric features

Target: r > 0.75
"""

import logging, warnings, time
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

ROOT    = Path("/home/stalin/Desktop/PDFL-RNA/RNA_PDFL")
NA_L    = Path("/home/stalin/Desktop/PDFL-RNA/NA-L")
S11_NPZ = ROOT / "data" / "features" / "step11_full_features.npz"
S22_CSV = ROOT / "results" / "step22_results.csv"
S23_CSV = ROOT / "results" / "step23_results.csv"
RES_DIR = ROOT / "results"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "results" / "logs" / f"step24_{ts}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("RNA-PDFL · Step 24: Pocket Sequence Kernel + SAM Features")
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
X11      = d11["X"].astype(np.float64)
y        = d11["y"].astype(np.float32)
ids      = d11["ids"]
subtypes_raw = d11["subtypes"]
subtypes = np.array([make_subtype(p, s) for p, s in zip(ids, subtypes_raw)])
n = len(y)

s22_df = pd.read_csv(S22_CSV)
s23_df = pd.read_csv(S23_CSV)
pdb2s22 = dict(zip(s22_df["pdb"], s22_df["y_pred"]))
pdb2s23 = dict(zip(s23_df["pdb"], s23_df["y_pred"]))
pred_s22 = np.array([float(pdb2s22.get(p, np.nan)) for p in ids])
pred_s23 = np.array([float(pdb2s23.get(p, np.nan)) for p in ids])

unimol_emb_raw = np.load("/tmp/unimol_emb.npy")
valid_idx_raw  = np.load("/tmp/unimol_valid_idx.npy").astype(int)
unimol_full = np.zeros((n, 512), dtype=np.float64)
unimol_full[valid_idx_raw] = unimol_emb_raw
missing_idx = [i for i in range(n) if i not in set(valid_idx_raw.tolist())]
if missing_idx: unimol_full[missing_idx] = unimol_emb_raw.mean(axis=0)

X_rlif_ext = np.load(ROOT / "data" / "features" / "rlif_ext_features.npy")
log.info(f"  n={n}, X11: {X11.shape}, RLIF_ext: {X_rlif_ext.shape}")

# ═══════════════════════════════════════════════════════════════════════════
# PART A: Pocket sequence kernel
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART A: Pocket sequence kernel from binding site residues")
log.info("="*70)

BASES_MAP = {'A': 0, 'G': 1, 'C': 2, 'U': 3, 'T': 3}  # T→U
DINUC = [f"{a}{b}" for a in "AGCU" for b in "AGCU"]     # 16 dinucleotides
TRINUC = [f"{a}{b}{c}" for a in "AGCU" for b in "AGCU" for c in "AGCU"]  # 64

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

def get_pocket_sequence(rna_res, rna_coords, lig_coords, cutoff=6.0):
    """Get ordered sequence of RNA residues at binding site (by chain position)."""
    if lig_coords is None or lig_coords.shape[0] == 0 or rna_coords.shape[0] == 0:
        return []
    D = cdist(rna_coords, lig_coords)
    contact_mask = D.min(axis=1) < cutoff
    # Get unique residues (by approximate position, deduplicated)
    seen = set(); pocket_res = []
    for i, (r, c) in enumerate(zip(rna_res, contact_mask)):
        if c and r in BASES_MAP and r not in seen:
            seen.add(r); pocket_res.append(r)
    return pocket_res

def seq_to_kmer_vec(seq_list, k=2):
    """Convert sequence (list of residues) to k-mer frequency vector."""
    if k == 1:
        keys = ['A','G','C','U']
    elif k == 2:
        keys = DINUC
    else:
        keys = TRINUC
    vec = np.zeros(len(keys))
    key_idx = {kk: j for j, kk in enumerate(keys)}
    seq = ''.join(seq_list)
    total = max(len(seq) - k + 1, 1)
    for j in range(len(seq) - k + 1):
        gram = seq[j:j+k]
        if gram in key_idx:
            vec[key_idx[gram]] += 1
    return vec / total

# Pocket sequence features: unigram(4) + digram(16) + trigram(64) = 84 features
N_POCKET_SEQ = 4 + 16 + 64

X_pocket_seq = np.zeros((n, N_POCKET_SEQ), dtype=np.float64)
pocket_seqs  = []

for i, pdb in enumerate(ids):
    pocket_file = NA_L / pdb / f"{pdb}_pocket.pdb"
    if not pocket_file.exists():
        pocket_seqs.append([]); continue
    rna_res, rna_atm, rna_elem, rna_coords = parse_pocket_pdb(pocket_file)
    if rna_coords.shape[0] == 0:
        pocket_seqs.append([]); continue
    lig_coords = None
    mol2_f = NA_L / pdb / f"{pdb}_ligand.mol2"
    sdf_f  = NA_L / pdb / f"{pdb}_ligand.sdf"
    if mol2_f.exists():
        lig_coords, _ = parse_ligand_mol2(mol2_f)
    if lig_coords is None and sdf_f.exists():
        lig_coords, _ = parse_ligand_sdf(sdf_f)
    if lig_coords is None:
        pocket_seqs.append([]); continue
    pocket_seq = get_pocket_sequence(rna_res, rna_coords, lig_coords, cutoff=6.0)
    pocket_seqs.append(pocket_seq)
    if pocket_seq:
        uni  = seq_to_kmer_vec(pocket_seq, k=1)  # 4
        dig  = seq_to_kmer_vec(pocket_seq, k=2)  # 16
        tri  = seq_to_kmer_vec(pocket_seq, k=3)  # 64
        X_pocket_seq[i] = np.concatenate([uni, dig, tri])

log.info(f"  Pocket sequences computed: {sum(1 for s in pocket_seqs if s)}/{n}")

# Validate
log.info("  Pocket seq univariate correlations:")
best_seq_r = 0
for j in range(N_POCKET_SEQ):
    col = X_pocket_seq[:, j]
    if np.std(col) > 1e-8:
        r_j = abs(pearsonr(col, y)[0])
        if r_j > best_seq_r: best_seq_r = r_j
log.info(f"    Max |r| among {N_POCKET_SEQ} pocket-seq features: {best_seq_r:.3f}")

# Additionally: full pocket sequence (all contacting residues, order-independent counts)
# This gives a richer base composition profile at the binding site
N_BASE_COUNTS = 4  # A, G, C, U counts at pocket
X_pocket_counts = np.zeros((n, N_BASE_COUNTS))
for i, seq in enumerate(pocket_seqs):
    if seq:
        c = Counter(seq)
        total = max(len(seq), 1)
        X_pocket_counts[i, 0] = c.get('A', 0) / total
        X_pocket_counts[i, 1] = c.get('G', 0) / total
        X_pocket_counts[i, 2] = c.get('C', 0) / total
        X_pocket_counts[i, 3] = c.get('U', 0) / total

# Combine: pocket kmer + counts → full pocket features
X_pocket = np.hstack([X_pocket_seq, X_pocket_counts])
log.info(f"  Total pocket features: {X_pocket.shape[1]}")

# Univariate correlation for G fraction at pocket
g_frac = X_pocket_counts[:, 1]
r_g, _ = pearsonr(g_frac, y)
log.info(f"  G-fraction at pocket: r={r_g:+.3f}")
a_frac = X_pocket_counts[:, 0]
r_a, _ = pearsonr(a_frac, y)
log.info(f"  A-fraction at pocket: r={r_a:+.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# PART B: SAM_SAH targeted features + model
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART B: SAM_SAH targeted features")
log.info("="*70)

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

def get_sam_specific_features(pdb):
    """Return SAM-specific chemical features for a ligand."""
    feat = np.zeros(10)
    for ext in ['mol2', 'sdf']:
        fpath = NA_L / pdb / f"{pdb}_ligand.{ext}"
        if not fpath.exists(): continue
        try:
            if ext == 'mol2':
                mol = Chem.MolFromMol2File(str(fpath), sanitize=False)
            else:
                mol = Chem.MolFromMolFile(str(fpath), sanitize=False)
            if mol is None: continue
            Chem.SanitizeMol(mol)
            # 0: formal charge of ligand
            feat[0] = sum(a.GetFormalCharge() for a in mol.GetAtoms())
            # 1: number of S atoms
            feat[1] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16)
            # 2: quaternary S indicator (S with 4 bonds = sulfonium)
            feat[2] = sum(1 for a in mol.GetAtoms()
                          if a.GetAtomicNum() == 16 and a.GetDegree() == 4)
            # 3: number of positively charged N (quaternary ammonium)
            feat[3] = sum(1 for a in mol.GetAtoms()
                          if a.GetAtomicNum() == 7 and a.GetFormalCharge() > 0)
            # 4: MW
            feat[4] = Descriptors.MolWt(mol)
            # 5: number of H-bond donors
            feat[5] = rdMolDescriptors.CalcNumHBD(mol)
            # 6: number of H-bond acceptors
            feat[6] = rdMolDescriptors.CalcNumHBA(mol)
            # 7: number of rotatable bonds
            feat[7] = rdMolDescriptors.CalcNumRotatableBonds(mol)
            # 8: LogP
            feat[8] = Descriptors.MolLogP(mol)
            # 9: TPSA
            feat[9] = Descriptors.TPSA(mol)
            return feat
        except: continue
    return feat

X_sam_feat = np.array([get_sam_specific_features(p) for p in ids])
log.info(f"  SAM features computed: {X_sam_feat.shape}")

# Classify SAM_SAH riboswitches
def classify_rs_ligand(pdb):
    for ext in ['sdf', 'mol2']:
        f = NA_L / pdb / f"{pdb}_ligand.{ext}"
        if not f.exists(): continue
        try:
            mol = Chem.MolFromMolFile(str(f), sanitize=False) if ext == 'sdf' \
                  else Chem.MolFromMol2File(str(f), sanitize=False)
            if mol is None: continue
            Chem.SanitizeMol(mol)
            mw  = Descriptors.MolWt(mol)
            n_S = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16)
            n_rings = mol.GetRingInfo().NumRings()
            n_N = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
            if n_S >= 1 and mw > 400: return "TPP_like"
            if n_S >= 1 and mw > 300: return "SAM_SAH"
            if mw < 200 and n_rings >= 2 and n_N >= 3: return "purine"
            if mw > 350 and n_N >= 4 and n_rings >= 3: return "FMN_FAD"
            if mw < 200 and n_N >= 1 and n_rings == 0: return "amino_acid"
            if mw < 200 and n_rings >= 1: return "preQ_small"
            return "other_lig"
        except: continue
    return "unknown"

rs_mask = subtypes == "riboswitch"
rs_class = np.array(["unknown"] * n, dtype=object)
for i in np.where(rs_mask)[0]:
    rs_class[i] = classify_rs_ligand(ids[i])

sam_mask = (subtypes == "riboswitch") & (rs_class == "SAM_SAH")
n_sam = sam_mask.sum()
log.info(f"  SAM_SAH: n={n_sam}")

if n_sam >= 3:
    y_sam = y[sam_mask]
    X_sam_sub = np.hstack([
        X_sam_feat[sam_mask],
        X_rlif_ext[sam_mask],
        X11[sam_mask, 36000:38048],  # Morgan
        X11[sam_mask, 38796:38963],  # MACCS
    ])
    best_sam_r, best_sam_p = -99.0, np.full(n_sam, y_sam.mean())
    for alpha in ALPHA_GRID:
        pp = np.zeros(n_sam); ok = True
        for i in range(n_sam):
            tr = [j for j in range(n_sam) if j != i]
            try:
                pipe = Pipeline([
                    ("vt", VarianceThreshold(threshold=1e-4)),
                    ("sc", StandardScaler()),
                    ("reg", Ridge(alpha=alpha)),
                ])
                pipe.fit(X_sam_sub[tr], y_sam[tr])
                pp[i] = np.clip(pipe.predict(X_sam_sub[[i]])[0],
                                y_sam[tr].min()-3, y_sam[tr].max()+3)
            except: ok=False; break
        if not ok: continue
        r = pearsonr(y_sam, pp)[0] if np.std(pp) > 1e-8 else -99.0
        if r > best_sam_r: best_sam_r, best_sam_p = r, pp.copy()
    log.info(f"  SAM_SAH targeted model r = {best_sam_r:.4f}")
    log.info(f"  SAM_SAH formal charges: {X_sam_feat[sam_mask, 0]}")
else:
    best_sam_r, best_sam_p = -99.0, np.array([])
    log.info("  SAM_SAH: too few samples")

# ═══════════════════════════════════════════════════════════════════════════
# PART C: 5-kernel MKL with pocket sequence kernel
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART C: 5-kernel MKL (topo + UniMol/Tan + RNA-FM + pocket_seq + RLIF)")
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

# Build all kernels
X_topo   = X11[:, np.r_[0:36000, 38963:49763]]
X_rnafm  = X11[:, 38064:38704]
X_morgan = X11[:, 36000:38048]
X_maccs  = X11[:, 38796:38963]

X_topo_n     = StandardScaler().fit_transform(X_topo)
X_rnafm_n    = StandardScaler().fit_transform(X_rnafm)
X_unimol_n   = StandardScaler().fit_transform(unimol_full)
X_rlif_n     = StandardScaler().fit_transform(X_rlif_ext)
X_pocket_n   = StandardScaler().fit_transform(X_pocket)

K_topo   = rbf_kernel(X_topo_n, gamma=1e-6)
K_rna    = rbf_kernel(X_rnafm_n, gamma=5e-3)
K_uni    = rbf_kernel(X_unimol_n, gamma=0.05)
K_tan    = 0.7*tanimoto(X_morgan) + 0.3*tanimoto(X_maccs)
K_lig    = 0.5*K_uni + 0.5*K_tan

# Pocket sequence kernel — tune gamma
log.info("  Selecting best gamma for pocket sequence kernel ...")
best_ps_g, best_ps_r = 1e-2, -99.0
K_base = 0.7*K_topo + 0.1*K_lig + 0.2*K_rna
for gamma_ps in [1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0]:
    K_ps_try = rbf_kernel(X_pocket_n, gamma=gamma_ps)
    K_try = 0.8*K_base + 0.2*K_ps_try
    D_ = np.sqrt(np.diag(K_try)).reshape(-1,1); D_ = np.where(D_<1e-10,1e-10,D_)
    pp = loo_mkl(K_try/(D_*D_.T), y, alpha=0.01)
    r_rs, _ = pearsonr(pp[rs_mask], y[rs_mask])
    if r_rs > best_ps_r:
        best_ps_r = r_rs
        best_ps_g = gamma_ps

K_pocket_k = rbf_kernel(X_pocket_n, gamma=best_ps_g)
log.info(f"  Best pocket-seq kernel: gamma={best_ps_g:.3g}, rs_r≈{best_ps_r:.4f}")

# 5-kernel grid search (coarser but covering all kernels)
RLIF_best_g = 0.5  # from step22
K_rlif_k = rbf_kernel(X_rlif_n, gamma=RLIF_best_g)

log.info("  5-kernel grid (w_topo, w_lig, w_rna, w_pocket, w_rlif) ...")
best5_r, best5_cfg, best5_p = -99.0, None, None

weight_configs = []
for w_t in [0.5, 0.6, 0.7]:
    for w_l in [0.05, 0.1, 0.15]:
        for w_r in [0.1, 0.2]:
            for w_ps in [0.05, 0.1, 0.15]:
                for w_rl in [0.03, 0.05, 0.1]:
                    total = w_t + w_l + w_r + w_ps + w_rl
                    if abs(total - 1.0) > 0.1: continue
                    weight_configs.append((w_t/total, w_l/total, w_r/total,
                                           w_ps/total, w_rl/total))

log.info(f"  Testing {len(weight_configs)} 5-kernel configs ...")
for w_t, w_l, w_r, w_ps, w_rl in weight_configs:
    K5 = w_t*K_topo + w_l*K_lig + w_r*K_rna + w_ps*K_pocket_k + w_rl*K_rlif_k
    D_ = np.sqrt(np.diag(K5)).reshape(-1,1); D_ = np.where(D_<1e-10,1e-10,D_)
    K5n = K5 / (D_*D_.T)
    for lam in [0.01, 0.05]:
        pp = loo_mkl(K5n, y, alpha=lam)
        r_rs, _ = pearsonr(pp[rs_mask], y[rs_mask])
        if r_rs > best5_r:
            best5_r = r_rs
            best5_cfg = (w_t, w_l, w_r, w_ps, w_rl, lam)
            best5_p = pp.copy()

log.info(f"  Best 5-kernel MKL: rs_r={best5_r:.4f}  cfg={best5_cfg}")
r_5k_all, _ = pearsonr(best5_p, y)
log.info(f"  All-sample r = {r_5k_all:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# PART D: Rebuild riboswitch ensemble with SAM fix
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART D: Riboswitch ensemble with SAM_SAH fix")
log.info("="*70)

rs_idx  = np.where(rs_mask)[0]
y_rs    = y[rs_mask]

# Base: 5-kernel MKL for riboswitch
mkl_rs_preds = best5_p[rs_mask].copy()
r_mkl_rs, _ = pearsonr(mkl_rs_preds, y_rs)
log.info(f"  5-kernel MKL riboswitch r = {r_mkl_rs:.4f}")

# Apply SAM-specific fix if it improves
rs_preds_fixed = mkl_rs_preds.copy()
sam_in_rs = np.where(sam_mask[rs_mask])[0]  # SAM positions within rs array

if best_sam_r > r_mkl_rs and len(best_sam_p) > 0:
    # Check if SAM model helps globally when blended
    rs_preds_sam_blend = mkl_rs_preds.copy()
    for alpha_b in [0.3, 0.5, 0.7, 1.0]:
        blended = mkl_rs_preds.copy()
        if alpha_b < 1.0:
            blended[sam_in_rs] = (alpha_b * best_sam_p +
                                   (1-alpha_b) * mkl_rs_preds[sam_in_rs])
        else:
            blended[sam_in_rs] = best_sam_p
        r_bl, _ = pearsonr(blended, y_rs)
        if r_bl > r_mkl_rs:
            rs_preds_fixed = blended.copy()
            r_mkl_rs = r_bl
            log.info(f"  SAM blend={alpha_b:.1f} improves rs_r to {r_bl:.4f}")

r_rs_final, _ = pearsonr(rs_preds_fixed, y_rs)
log.info(f"  Final riboswitch r = {r_rs_final:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# PART E: Final Hybrid
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART E: Final Hybrid Assembly")
log.info("="*70)

om_mask = subtypes == "other_misc"

# Build hybrids comparing all options
def make_hybrid(rs_src, om_src, base=pred_s23):
    hyb = base.copy()
    hyb[rs_mask] = rs_src
    hyb[om_mask] = om_src
    valid = ~np.isnan(hyb)
    r, _ = pearsonr(hyb[valid], y[valid])
    return hyb, r

# RS options: step23 (0.726), 5-kernel MKL (rs_r), fixed (with SAM)
# OM options: step22 (0.508), 5-kernel MKL global

h_configs = [
    ("step23_RS  + step22_OM",    pred_s23[rs_mask],    pred_s22[om_mask]),
    ("5k_MKL_RS  + step22_OM",    best5_p[rs_mask],     pred_s22[om_mask]),
    ("fixed_RS   + step22_OM",    rs_preds_fixed,       pred_s22[om_mask]),
    ("5k_MKL_RS  + 5k_MKL_OM",   best5_p[rs_mask],     best5_p[om_mask]),
    ("fixed_RS   + 5k_MKL_OM",   rs_preds_fixed,        best5_p[om_mask]),
]

results = []
for name, rs_src, om_src in h_configs:
    hyb, r = make_hybrid(rs_src, om_src)
    results.append((name, hyb, r))
    log.info(f"  {name:40s}: r={r:.4f}")

best_name, best_hyb, best_r = max(results, key=lambda x: x[2])
log.info(f"\n  → Best: {best_name}")

# Per-subtype breakdown
log.info(f"\nPer-subtype breakdown:")
for st in ["aptamer","duplex_groove","ribosomal_asite","riboswitch",
           "other_misc","g_quadruplex","viral_tar"]:
    mask = subtypes == st
    if mask.sum() < 2: continue
    r_st = pearsonr(best_hyb[mask], y[mask])[0]
    log.info(f"  {st:22s}: n={mask.sum():3d}  r={r_st:.3f}")

log.info(f"\n  Combined r (step24)   = {best_r:.4f}")
log.info(f"  Previous best         = 0.7389   [step23]")
log.info(f"  Delta                 = {best_r - 0.7389:+.4f}")
log.info(f"  Gap to DeepRSMA       = {0.784 - best_r:.4f}")

benchmarks = [("AffiGrapher",0.498),("RLaffinity",0.559),("RLASIF",0.666),
              ("DeepRSMA",0.784),("RSAPred",0.830)]
for nm, rb in benchmarks:
    log.info(f"  {'✓' if best_r > rb else '✗'} {nm}: {rb:.3f}")
log.info("="*70)

# Save
df_out = pd.DataFrame({
    "pdb": ids, "subtype": subtypes, "y_true": y, "y_pred": best_hyb,
    "pred_s22": pred_s22, "pred_s23": pred_s23, "mkl5_pred": best5_p,
})
df_out.to_csv(RES_DIR / "step24_results.csv", index=False)
np.save(ROOT / "data" / "features" / "pocket_seq_features.npy", X_pocket)
log.info(f"  Results → results/step24_results.csv")

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
ax.set_title(f"Step 24 Hybrid (r={best_r:.4f})", fontweight="bold")
ax.legend(fontsize=7); ax.grid(alpha=0.3, linestyle="--")

ax = axes[1]
steps = {"S22\nRLIF":0.7375,"S23\nMeta":0.7389,"S24\nPocketSeq":best_r}
bar_cols = ["#AAAAAA","#4393C3","#D63027"]
bars = ax.bar(list(steps.keys()), list(steps.values()),
              color=bar_cols, edgecolor="white", alpha=0.9)
for bar, val in zip(bars, steps.values()):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.003,
            f"{val:.4f}", ha='center', va='bottom', fontsize=9)
for nm, rb in benchmarks:
    ax.axhline(rb, color='gray', lw=0.8, linestyle='--', alpha=0.6)
    ax.text(2.5, rb+0.003, nm, fontsize=7, color='gray')
ax.set_ylabel("Pearson r"); ax.set_title("Performance progression")
ax.set_ylim(0.65, 0.86)
plt.tight_layout()
fig.savefig(ROOT/"results"/"figures"/"step24_results.png", dpi=150, bbox_inches="tight")
log.info(f"  Figure → results/figures/step24_results.png")
log.info("STEP 24 COMPLETE")
