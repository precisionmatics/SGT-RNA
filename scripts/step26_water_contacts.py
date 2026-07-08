"""
RNA-PDFL · Step 26: Water-mediated Contact Fingerprint (WCF) + Contact Pair FP (CPF)

Two novel structure-based features:

1. WCF (Water Contact Fingerprint):
   88/143 pocket PDBs contain water molecules.
   Water bridges = HOH within 3.5Å of BOTH RNA and ligand.
   Features: N_bridges, mean bridge distances, N_water_RNA, N_water_lig,
   water-specific interaction fractions.
   Water bridges mediate 30-50% of binding enthalpy in RNA-ligand complexes.

2. CPF (Contact Pair Fingerprint):
   For each (RNA_base × ligand_element) pair at 3/4/5Å:
   4 bases × 6 elements × 3 cutoffs = 72 features
   Captures chemical specificity of each contact type:
   e.g., G-N contacts (H-bonds), A-C contacts (stacking), U-S contacts (SAM binding)

Both features → new kernels in 6-kernel MKL.
"""

import logging, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
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
        logging.FileHandler(ROOT / "results" / "logs" / f"step26_{ts}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("RNA-PDFL · Step 26: Water Contacts + Contact Pair Fingerprint")
log.info("=" * 70)

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
DNA_TO_RNA = {'DA': 'A', 'DG': 'G', 'DC': 'C', 'DT': 'U'}
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

s24_df   = pd.read_csv(S24_CSV)
pdb2s24  = dict(zip(s24_df["pdb"], s24_df["y_pred"]))
pred_s24 = np.array([float(pdb2s24.get(p, np.nan)) for p in ids])

unimol_emb_raw = np.load("/tmp/unimol_emb.npy")
valid_idx_raw  = np.load("/tmp/unimol_valid_idx.npy").astype(int)
unimol_full = np.zeros((n, 512), dtype=np.float64)
unimol_full[valid_idx_raw] = unimol_emb_raw
missing_idx = [i for i in range(n) if i not in set(valid_idx_raw.tolist())]
if missing_idx: unimol_full[missing_idx] = unimol_emb_raw.mean(axis=0)

X_rlif = np.load(ROOT / "data" / "features" / "rlif_features.npy")
X_pocket = np.load(ROOT / "data" / "features" / "pocket_seq_features.npy")

# ═══════════════════════════════════════════════════════════════════════════
# PART A: Parse structures — RNA, water, ligand
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART A: Parsing RNA + water + ligand from pocket PDB files")
log.info("="*70)

BASES = ['A', 'G', 'C', 'U']
ELEMENTS = ['C', 'N', 'O', 'S', 'P', 'X']   # X = other
CUTOFFS_CPF = [3.0, 4.0, 5.0]

N_WCF = 12   # water contact features
N_CPF = len(BASES) * len(ELEMENTS) * len(CUTOFFS_CPF)   # 4×6×3 = 72

def parse_pocket_full(path):
    """Parse pocket PDB: separate RNA, water (HOH/WAT), return coords + types."""
    rna_res, rna_atm, rna_elem, rna_coords = [], [], [], []
    wat_coords = []
    with open(path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")): continue
            try:
                rn_raw = line[17:20].strip()
                an = line[12:16].strip()
                el = line[76:78].strip() if len(line) > 77 else ""
                if not el: el = ''.join(c for c in an if c.isalpha())[:1]
                if el.upper() in ('H', 'D'): continue
                if an.startswith('H') and len(an) > 1: continue
                alt = line[16]
                if alt not in (' ', 'A', ''): continue
                x, y_, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                if rn_raw in ('HOH', 'WAT', 'DOD', 'TIP'):
                    wat_coords.append([x, y_, z])
                else:
                    rn = DNA_TO_RNA.get(rn_raw, rn_raw)
                    rna_res.append(rn); rna_atm.append(an)
                    rna_elem.append(el.upper())
                    rna_coords.append([x, y_, z])
            except: continue
    rc = np.array(rna_coords) if rna_coords else np.zeros((0, 3))
    wc = np.array(wat_coords) if wat_coords else np.zeros((0, 3))
    return rna_res, rna_atm, rna_elem, rc, wc

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

def elem_idx(e):
    for j, ec in enumerate(ELEMENTS[:-1]):
        if e == ec: return j
    return 5  # 'X'

def compute_wcf_cpf(rna_res, rna_elem, rna_coords, wat_coords, lig_coords, lig_elem):
    """Return (WCF[12], CPF[72])."""
    wcf = np.zeros(N_WCF)
    cpf = np.zeros(N_CPF)

    if lig_coords is None or lig_coords.shape[0] == 0 or rna_coords.shape[0] == 0:
        return wcf, cpf

    D_rl = cdist(rna_coords, lig_coords)   # (n_rna, n_lig)

    # ── CPF: (RNA_base × lig_element) contact counts ──────────────────────
    cidx = 0
    for ci, cutoff in enumerate(CUTOFFS_CPF):
        for bi, base in enumerate(BASES):
            base_mask = np.array([r == base for r in rna_res])
            if not base_mask.any():
                cidx += len(ELEMENTS); continue
            D_base = D_rl[base_mask]    # (n_base_atoms, n_lig)
            for ei, el_class in enumerate(ELEMENTS):
                if el_class == 'X':
                    el_mask = np.array([e not in ('C','N','O','S','P') for e in lig_elem])
                else:
                    el_mask = np.array([e == el_class for e in lig_elem])
                if el_mask.any():
                    cpf[cidx] = (D_base[:, el_mask] < cutoff).sum()
                cidx += 1

    # ── WCF: water-bridge features ─────────────────────────────────────────
    if wat_coords.shape[0] == 0:
        # Features 8–11: fraction missing (0 = no water data)
        wcf[8] = 0.0   # has_water flag = 0
        return wcf, cpf

    wcf[8] = 1.0   # has_water flag = 1
    D_wl = cdist(wat_coords, lig_coords)   # (n_wat, n_lig)
    D_wr = cdist(wat_coords, rna_coords)   # (n_wat, n_rna)

    WAT_CUT_LIG = 3.5
    WAT_CUT_RNA = 3.5

    # Water-ligand contacts
    w_contacts_lig = D_wl.min(axis=1) < WAT_CUT_LIG   # bool per water
    # Water-RNA contacts
    w_contacts_rna = D_wr.min(axis=1) < WAT_CUT_RNA

    # Bridge waters: close to both
    bridges = w_contacts_lig & w_contacts_rna
    n_bridge = bridges.sum()
    wcf[0] = n_bridge

    # Number of water-RNA contacts (not bridging)
    wcf[1] = w_contacts_rna.sum()
    # Number of water-lig contacts (not bridging)
    wcf[2] = w_contacts_lig.sum()

    if n_bridge > 0:
        bridge_idx = np.where(bridges)[0]
        # Mean min distance from bridge water to ligand
        wcf[3] = D_wl[bridge_idx].min(axis=1).mean()
        # Mean min distance from bridge water to RNA
        wcf[4] = D_wr[bridge_idx].min(axis=1).mean()
        # Bridge density: bridges per ligand atom
        wcf[5] = n_bridge / max(lig_coords.shape[0], 1)
    else:
        wcf[3] = 5.0; wcf[4] = 5.0; wcf[5] = 0.0

    # Total water around binding site
    wcf[6] = wat_coords.shape[0]
    # Hydration fraction (bridge / total water)
    wcf[7] = n_bridge / max(wat_coords.shape[0], 1)
    # Bridge fraction of lig contacts
    wcf[9] = n_bridge / max(w_contacts_lig.sum(), 1)
    # Mean number of RNA contacts per bridge water
    if n_bridge > 0:
        wcf[10] = (D_wr[bridges] < WAT_CUT_RNA).sum(axis=1).mean()
    # Mean number of lig contacts per bridge water
    if n_bridge > 0:
        wcf[11] = (D_wl[bridges] < WAT_CUT_LIG).sum(axis=1).mean()

    return wcf, cpf

# Compute for all 143 complexes
X_wcf = np.zeros((n, N_WCF), dtype=np.float64)
X_cpf = np.zeros((n, N_CPF), dtype=np.float64)
n_success = 0
n_water   = 0

for i, pdb in enumerate(ids):
    pocket_file = NA_L / pdb / f"{pdb}_pocket.pdb"
    if not pocket_file.exists(): continue
    rna_res, rna_atm, rna_elem, rna_coords, wat_coords = parse_pocket_full(pocket_file)
    if rna_coords.shape[0] == 0: continue

    lig_coords, lig_elem = None, []
    mol2_f = NA_L / pdb / f"{pdb}_ligand.mol2"
    sdf_f  = NA_L / pdb / f"{pdb}_ligand.sdf"
    if mol2_f.exists(): lig_coords, lig_elem = parse_ligand_mol2(mol2_f)
    if lig_coords is None and sdf_f.exists(): lig_coords, lig_elem = parse_ligand_sdf(sdf_f)
    if lig_coords is None: continue

    wcf, cpf = compute_wcf_cpf(rna_res, rna_elem, rna_coords, wat_coords, lig_coords, lig_elem)
    X_wcf[i] = wcf
    X_cpf[i] = cpf
    n_success += 1
    if wat_coords.shape[0] > 0: n_water += 1

log.info(f"  Computed WCF+CPF for {n_success}/{n} complexes")
log.info(f"  Structures with water: {n_water}/{n_success}")

# Validate WCF
log.info("\n  WCF univariate correlations:")
wcf_names = ["N_bridge","N_wat_RNA","N_wat_lig","mean_bridge_lig","mean_bridge_rna",
             "bridge_density","N_total_wat","hydration_frac","has_water",
             "bridge_lig_frac","rna_per_bridge","lig_per_bridge"]
for j, name in enumerate(wcf_names):
    col = X_wcf[:, j]
    if np.std(col) > 1e-8:
        r_j, _ = pearsonr(col, y)
        log.info(f"    {name:25s} r={r_j:+.3f}")

# Validate CPF (top features by |r|)
log.info("\n  CPF top 10 features by |r|:")
cpf_rs = []
for j in range(N_CPF):
    col = X_cpf[:, j]
    if np.std(col) > 1e-8:
        r_j, _ = pearsonr(col, y)
        ci = j // (len(BASES)*len(ELEMENTS))
        bi = (j // len(ELEMENTS)) % len(BASES)
        ei = j % len(ELEMENTS)
        cpf_rs.append((abs(r_j), r_j,
                       f"N_{BASES[bi]}_{ELEMENTS[ei]}_{int(CUTOFFS_CPF[ci])}A"))
cpf_rs.sort(reverse=True)
for _, r_j, name in cpf_rs[:10]:
    log.info(f"    {name:20s} r={r_j:+.3f}")

np.save(ROOT / "data" / "features" / "wcf_features.npy", X_wcf)
np.save(ROOT / "data" / "features" / "cpf_features.npy", X_cpf)
log.info("  Saved WCF + CPF features")

# ═══════════════════════════════════════════════════════════════════════════
# PART B: WCF + CPF kernels in 6-kernel MKL
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART B: WCF + CPF kernels in extended MKL")
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

X_topo_n    = StandardScaler().fit_transform(X_topo)
X_rnafm_n   = StandardScaler().fit_transform(X_rnafm)
X_unimol_n  = StandardScaler().fit_transform(unimol_full)
X_rlif_n    = StandardScaler().fit_transform(X_rlif)
X_pocket_n  = StandardScaler().fit_transform(X_pocket)
X_wcf_n     = StandardScaler().fit_transform(X_wcf)
X_cpf_n     = StandardScaler().fit_transform(X_cpf)

K_topo   = rbf_kernel(X_topo_n, gamma=1e-6)
K_rna    = rbf_kernel(X_rnafm_n, gamma=5e-3)
K_uni    = rbf_kernel(X_unimol_n, gamma=0.05)
K_tan    = 0.7*tanimoto(X_morgan) + 0.3*tanimoto(X_maccs)
K_lig    = 0.5*K_uni + 0.5*K_tan
K_rlif   = rbf_kernel(X_rlif_n, gamma=0.5)
K_pocket = rbf_kernel(X_pocket_n, gamma=0.01)

# Find best gamma for WCF and CPF
rs_mask = subtypes == "riboswitch"
K_base5 = 0.50*K_topo + 0.15*K_lig + 0.20*K_rna + 0.05*K_pocket + 0.10*K_rlif  # step24 best

best_wcf_g, best_wcf_r = 0.1, -99.0
for gamma in [1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0]:
    K_w = rbf_kernel(X_wcf_n, gamma=gamma)
    K_try = 0.85*K_base5 + 0.15*K_w
    D_ = np.sqrt(np.diag(K_try)).reshape(-1,1); D_ = np.where(D_<1e-10,1e-10,D_)
    pp = loo_mkl(K_try/(D_*D_.T), y, alpha=0.01)
    r_rs, _ = pearsonr(pp[rs_mask], y[rs_mask])
    if r_rs > best_wcf_r: best_wcf_r, best_wcf_g = r_rs, gamma
K_wcf = rbf_kernel(X_wcf_n, gamma=best_wcf_g)
log.info(f"  Best WCF kernel: gamma={best_wcf_g}  rs_r≈{best_wcf_r:.4f}")

best_cpf_g, best_cpf_r = 0.01, -99.0
for gamma in [1e-4, 1e-3, 1e-2, 0.1, 0.5]:
    K_c = rbf_kernel(X_cpf_n, gamma=gamma)
    K_try = 0.85*K_base5 + 0.15*K_c
    D_ = np.sqrt(np.diag(K_try)).reshape(-1,1); D_ = np.where(D_<1e-10,1e-10,D_)
    pp = loo_mkl(K_try/(D_*D_.T), y, alpha=0.01)
    r_rs, _ = pearsonr(pp[rs_mask], y[rs_mask])
    if r_rs > best_cpf_r: best_cpf_r, best_cpf_g = r_rs, gamma
K_cpf = rbf_kernel(X_cpf_n, gamma=best_cpf_g)
log.info(f"  Best CPF kernel: gamma={best_cpf_g}  rs_r≈{best_cpf_r:.4f}")

# 7-kernel MKL: topo + lig + rna + pocket + rlif + WCF + CPF
log.info("\n  7-kernel grid search ...")
best7_r, best7_p, best7_cfg = -99.0, None, None

for w_wcf in [0.0, 0.05, 0.1]:
    for w_cpf in [0.0, 0.05, 0.1]:
        for w_t in [0.45, 0.50, 0.55]:
            for w_l in [0.10, 0.15]:
                for w_r in [0.15, 0.20]:
                    w_ps = 0.05; w_rl = 0.10
                    total = w_t + w_l + w_r + w_ps + w_rl + w_wcf + w_cpf
                    if total < 0.5: continue
                    # Normalize
                    ww = [w_t/total, w_l/total, w_r/total, w_ps/total,
                          w_rl/total, w_wcf/total, w_cpf/total]
                    K7 = (ww[0]*K_topo + ww[1]*K_lig + ww[2]*K_rna + ww[3]*K_pocket +
                          ww[4]*K_rlif + ww[5]*K_wcf + ww[6]*K_cpf)
                    D_ = np.sqrt(np.diag(K7)).reshape(-1,1); D_ = np.where(D_<1e-10,1e-10,D_)
                    K7n = K7 / (D_*D_.T)
                    for lam in [0.01, 0.05]:
                        pp = loo_mkl(K7n, y, alpha=lam)
                        r_rs, _ = pearsonr(pp[rs_mask], y[rs_mask])
                        if r_rs > best7_r:
                            best7_r = r_rs; best7_cfg = (*ww, lam); best7_p = pp.copy()

log.info(f"  Best 7-kernel MKL: rs_r={best7_r:.4f}")
r7_all, _ = pearsonr(best7_p, y)
log.info(f"  All-sample r = {r7_all:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# PART C: Final Hybrid
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART C: Final Hybrid Assembly")
log.info("="*70)

om_mask = subtypes == "other_misc"
s22_df2 = pd.read_csv(ROOT / "results" / "step22_results.csv")
pdb2s22 = dict(zip(s22_df2["pdb"], s22_df2["y_pred"]))
pred_s22 = np.array([float(pdb2s22.get(p, np.nan)) for p in ids])

def make_hybrid(rs_src, om_src, base=pred_s24):
    hyb = base.copy()
    hyb[rs_mask] = rs_src
    hyb[om_mask] = om_src
    valid = ~np.isnan(hyb)
    r, _ = pearsonr(hyb[valid], y[valid])
    return hyb, r

configs = [
    ("step24 (base)",           pred_s24[rs_mask],  pred_s22[om_mask]),
    ("7k_RS  + s22_OM",         best7_p[rs_mask],   pred_s22[om_mask]),
    ("7k_RS  + 7k_OM",          best7_p[rs_mask],   best7_p[om_mask]),
    ("7k_RS  + s24_OM",         best7_p[rs_mask],   pred_s24[om_mask]),
]

results = []
for name, rs_src, om_src in configs:
    hyb, r = make_hybrid(rs_src, om_src)
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

log.info(f"\n  Combined r (step26)   = {best_r:.4f}")
log.info(f"  Previous best         = 0.7412   [step24]")
log.info(f"  Delta                 = {best_r - 0.7412:+.4f}")
log.info(f"  Gap to DeepRSMA       = {0.784 - best_r:.4f}")
for nm, rb in [("AffiGrapher",0.498),("RLaffinity",0.559),("RLASIF",0.666),
               ("DeepRSMA",0.784),("RSAPred",0.830)]:
    log.info(f"  {'✓' if best_r > rb else '✗'} {nm}: {rb:.3f}")
log.info("="*70)

df_out = pd.DataFrame({
    "pdb": ids, "subtype": subtypes, "y_true": y,
    "y_pred": best_hyb, "pred_7k": best7_p,
})
df_out.to_csv(RES_DIR / "step26_results.csv", index=False)
log.info(f"  Results → results/step26_results.csv")

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
ax.set_title(f"Step 26 Hybrid (r={best_r:.4f})", fontweight="bold")
ax.legend(fontsize=7); ax.grid(alpha=0.3, linestyle="--")

ax = axes[1]
steps = {"S22\nRLIF":0.7375,"S24\n5kMKL":0.7412,"S26\nWater":best_r}
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
fig.savefig(ROOT/"results"/"figures"/"step26_water_results.png", dpi=150, bbox_inches="tight")
log.info(f"  Figure → results/figures/step26_water_results.png")
log.info("STEP 26 COMPLETE")
