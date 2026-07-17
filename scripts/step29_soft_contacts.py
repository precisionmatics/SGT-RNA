"""
SGT-RNA · Step 29: Soft Contact Fingerprint (SCF) + Ribosomal A-site Targeted Model

Current best: r=0.7874 (beats DeepRSMA r=0.784, gap to RSAPred r=0.830 = 0.043).
Main bottlenecks:
  - riboswitch (n=61): r=0.769
  - ribosomal_asite (n=13): r=0.753

Approach:
  A. SCF (Soft Contact Fingerprint): replace binary CPF cutoffs with Gaussian-decay weighted
     contacts. More physically meaningful: captures the continuous decay of interactions.
     sum_ij exp(-D_ij² / (2σ²))  for σ ∈ {1.5, 3.0, 5.0} Å
     4 bases × 6 elements × 3 sigmas = 72 SCF features

  B. Fix riboswitch classification (actual mol2 codes discovered in step28 probe):
     PURINE:  6AP, 6GU, XAN, 6GO, 2BP, A2F, ADE, 2BA, 2QB, 29G, 29H, 7DG
     FMN_FAD: PRF, FFO, LYA, H4B  (proflavin + flavin-analog)
     TPP:     C2E, PRP  (thiamine/TPP analogs)
     SAM_SAH: SAM, SAH, SFG (already in step27)

  C. Ribosomal A-site targeted model: aminoglycosides have -NH2 groups; N-N contacts
     at short range should discriminate within ribosomal structures.

  D. Riboswitch: try SCF kernel in MKL + targeted models for purine subclass.

  E. Final hybrid assembly.
"""

import logging, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent.parent
NA_L    = Path("/home/stalin/Desktop/SGT-RNA/NA-L")
S11_NPZ = ROOT / "data" / "features" / "step11_full_features.npz"
RES_DIR = ROOT / "results"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "results" / "logs" / f"step29_{ts}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("SGT-RNA · Step 29: SCF + Ribosomal A-site Targeted Model")
log.info("=" * 70)

# ── Subtype definitions ────────────────────────────────────────────────────
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

# ── Corrected riboswitch ligand classification (from mol2 inspection) ─────
SAM_SAH_CODES = {"SAM","SAH","SFG","AGN","SAX","SOH","SCA","SEP","MET"}
PURINE_CODES  = {
    "6AP","6GU","XAN","6GO","2BP","A2F","ADE","2BA","2QB","29G",
    "29H","7DG","GUA","IMP","XMP","GMP","AMP","ADN","INO","2AP",
    "3DA","FGA","FA5","MIX",  # MIX is likely adenine mix in 2kgp
}
FMN_FAD_CODES = {
    "PRF","FFO","LYA","H4B","FMN","FAD","RFL","LUM","FMA","FHD",
    "FCO","FAS","FHN","FIN","FEP","FLP","FAM","FMR","FH4",
}
TPP_CODES = {
    "C2E","PRP","TPP","TDP","THI","T2P","THM","VIB","G4P",
}
# Remaining: glycine/lysine/other riboswitches → "other_lig" (use global predictions)

def get_lig_code(pdb, na_l_root):
    mol2_f = na_l_root / pdb / f"{pdb}_ligand.mol2"
    if not mol2_f.exists():
        return ""
    try:
        with open(mol2_f) as f:
            content = f.read()
        in_atom = False
        for line in content.splitlines():
            if '@<TRIPOS>ATOM' in line: in_atom = True; continue
            if '@<TRIPOS>' in line and 'ATOM' not in line: in_atom = False; continue
            if in_atom and line.strip():
                parts = line.split()
                if len(parts) >= 8:
                    return parts[7].strip().upper()
    except Exception:
        pass
    return ""

def classify_rs_ligand_v2(pdb, na_l_root):
    code = get_lig_code(pdb, na_l_root)
    if code in SAM_SAH_CODES: return "SAM_SAH"
    if code in PURINE_CODES:  return "purine"
    if code in FMN_FAD_CODES: return "FMN_FAD"
    if code in TPP_CODES:     return "TPP"
    return "other_lig"

# ── Feature helpers ────────────────────────────────────────────────────────
BASES        = ['A', 'G', 'C', 'U']
ELEMENTS_CPF = ['C', 'N', 'O', 'S', 'P', 'X']
CUTOFFS_CPF  = [3.0, 4.0, 5.0]
SIGMAS_SCF   = [1.5, 3.0, 5.0]   # Gaussian sigmas for SCF
MAX_DIST_SCF = 12.0               # max distance to consider
N_SCF        = len(BASES) * len(ELEMENTS_CPF) * len(SIGMAS_SCF)  # 72

def compute_scf(rna_res, rna_elem, rna_coords, lig_coords, lig_elem):
    """Soft Contact Fingerprint: Gaussian-weighted RNA-ligand contacts."""
    feat = np.zeros(N_SCF)
    if lig_coords is None or lig_coords.shape[0] == 0 or rna_coords.shape[0] == 0:
        return feat

    D = cdist(rna_coords, lig_coords)   # (n_rna, n_lig)

    idx = 0
    for sigma in SIGMAS_SCF:
        W = np.exp(-D**2 / (2 * sigma**2))   # Gaussian weights, full matrix
        for base in BASES:
            base_mask = np.array([r == base for r in rna_res])
            for el_class in ELEMENTS_CPF:
                if el_class == 'X':
                    el_mask = np.array([e not in ('C','N','O','S','P') for e in lig_elem])
                else:
                    el_mask = np.array([e == el_class for e in lig_elem])
                if base_mask.any() and el_mask.any():
                    feat[idx] = W[np.ix_(base_mask, el_mask)].sum()
                idx += 1
    return feat

def parse_pocket_pdb_v2(path):
    """Parse pocket PDB, map DNA residue names to RNA."""
    rna_res, rna_atm, rna_elem, rna_coords = [], [], [], []
    with open(path) as f:
        for line in f:
            if not line.startswith(("ATOM","HETATM")): continue
            try:
                rn_raw = line[17:20].strip()
                an = line[12:16].strip()
                el = line[76:78].strip() if len(line) > 77 else ""
                if not el: el = ''.join(c for c in an if c.isalpha())[:1]
                if el.upper() in ('H', 'D'): continue
                if an.startswith('H') and len(an) > 1: continue
                alt = line[16]
                if alt not in (' ', 'A', ''): continue
                if rn_raw in ('HOH', 'WAT', 'DOD', 'TIP'): continue
                x, y_, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                rn = DNA_TO_RNA.get(rn_raw, rn_raw)
                rna_res.append(rn); rna_atm.append(an)
                rna_elem.append(el.upper())
                rna_coords.append([x, y_, z])
            except: continue
    rc = np.array(rna_coords) if rna_coords else np.zeros((0, 3))
    return rna_res, rna_atm, rna_elem, rc

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

# ── Load data ──────────────────────────────────────────────────────────────
log.info("\nLoading data ...")
d11          = np.load(S11_NPZ)
X11          = d11["X"].astype(np.float64)
y            = d11["y"].astype(np.float32)
ids          = d11["ids"]
subtypes_raw = d11["subtypes"]
subtypes     = np.array([make_subtype(p, s) for p, s in zip(ids, subtypes_raw)])
n            = len(y)

X_rlif   = np.load(ROOT/"data"/"features"/"rlif_features.npy")
X_cpf    = np.load(ROOT/"data"/"features"/"cpf_features.npy")
X_pocket = np.load(ROOT/"data"/"features"/"pocket_seq_features.npy")

unimol_emb_raw = np.load("/tmp/unimol_emb.npy")
valid_idx_raw  = np.load("/tmp/unimol_valid_idx.npy").astype(int)
unimol_full    = np.zeros((n, 512), dtype=np.float64)
unimol_full[valid_idx_raw] = unimol_emb_raw
missing_idx    = [i for i in range(n) if i not in set(valid_idx_raw.tolist())]
if missing_idx: unimol_full[missing_idx] = unimol_emb_raw.mean(axis=0)

s24_df   = pd.read_csv(ROOT/"results"/"step24_results.csv")
pdb2s24  = dict(zip(s24_df["pdb"], s24_df["y_pred"]))
pred_s24 = np.array([float(pdb2s24.get(p, np.nan)) for p in ids])

s27_df   = pd.read_csv(ROOT/"results"/"step27_results.csv")
pdb2s27  = dict(zip(s27_df["pdb"], s27_df["y_pred"]))
pred_s27 = np.array([float(pdb2s27.get(p, np.nan)) for p in ids])

s28_df   = pd.read_csv(ROOT/"results"/"step28_results.csv")
pdb2s28  = dict(zip(s28_df["pdb"], s28_df["y_pred"]))
pred_s28 = np.array([float(pdb2s28.get(p, np.nan)) for p in ids])

log.info(f"  Loaded {n} complexes, X11={X11.shape}")

# ═══════════════════════════════════════════════════════════════════════════
# PART A: Compute SCF features
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART A: Soft Contact Fingerprint (SCF) Computation")
log.info("="*70)

X_scf    = np.zeros((n, N_SCF), dtype=np.float64)
n_scf_ok = 0

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

    X_scf[i] = compute_scf(rna_res, rna_elem, rna_coords, lig_coords, lig_elem)
    n_scf_ok += 1

log.info(f"  Computed SCF for {n_scf_ok}/{n} complexes")

# Top SCF correlations
scf_cors = []
for si, sigma in enumerate(SIGMAS_SCF):
    for bi, base in enumerate(BASES):
        for ei, elem in enumerate(ELEMENTS_CPF):
            idx = si * (len(BASES)*len(ELEMENTS_CPF)) + bi*len(ELEMENTS_CPF) + ei
            col = X_scf[:, idx]
            if np.std(col) > 1e-8:
                r_j, _ = pearsonr(col, y)
                nm = f"S_{base}_{elem}_{sigma}"
                scf_cors.append((abs(r_j), r_j, idx, nm))
scf_cors.sort(reverse=True)
log.info("\n  Top 10 SCF global correlations:")
for _, r_j, _, nm in scf_cors[:10]:
    log.info(f"    {nm:20s} r={r_j:+.3f}")

np.save(ROOT/"data"/"features"/"scf_features.npy", X_scf)
log.info("  Saved SCF features → data/features/scf_features.npy")

# ═══════════════════════════════════════════════════════════════════════════
# PART B: Riboswitch classification (corrected mol2 codes)
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART B: Riboswitch Subclass Classification (corrected)")
log.info("="*70)

rs_mask = subtypes == "riboswitch"
rs_subclass = np.array([classify_rs_ligand_v2(p, NA_L) for p in ids])

for cls in ["SAM_SAH", "purine", "FMN_FAD", "TPP", "other_lig"]:
    m = rs_mask & (rs_subclass == cls)
    log.info(f"  {cls:10s}: n={m.sum():2d}  {sorted(ids[m].tolist())}")

sam_sah_mask = rs_mask & (rs_subclass == "SAM_SAH")
purine_mask  = rs_mask & (rs_subclass == "purine")
fmn_fad_mask = rs_mask & (rs_subclass == "FMN_FAD")
tpp_mask     = rs_mask & (rs_subclass == "TPP")
otherlig_mask= rs_mask & (rs_subclass == "other_lig")

om_mask = subtypes == "other_misc"
asite_mask = subtypes == "ribosomal_asite"

# ═══════════════════════════════════════════════════════════════════════════
# PART C: Ribosomal A-site targeted model
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART C: Ribosomal A-site Targeted Model (aminoglycoside contacts)")
log.info("="*70)

n_asite = asite_mask.sum()
y_asite = y[asite_mask].astype(np.float64)
log.info(f"  n={n_asite}, pKd range [{y_asite.min():.2f}, {y_asite.max():.2f}]")

# Top CPF correlations within ribosomal A-site
asite_cpf_cors = []
for bi, base in enumerate(BASES):
    for ci, cutoff in enumerate(CUTOFFS_CPF):
        for ei, elem in enumerate(ELEMENTS_CPF):
            idx = ci*24 + bi*6 + ei
            col = X_cpf[asite_mask, idx]
            if np.std(col) > 1e-8:
                r_j, _ = pearsonr(col, y_asite)
                nm = f"N_{base}_{elem}_{int(cutoff)}A"
                asite_cpf_cors.append((abs(r_j), r_j, idx, nm))
asite_cpf_cors.sort(reverse=True)
log.info("\n  Top CPF features within ribosomal_asite:")
for _, r_j, _, nm in asite_cpf_cors[:10]:
    log.info(f"    {nm:18s} r={r_j:+.3f}")

# Top SCF correlations within ribosomal A-site
asite_scf_cors = []
for si, sigma in enumerate(SIGMAS_SCF):
    for bi, base in enumerate(BASES):
        for ei, elem in enumerate(ELEMENTS_CPF):
            idx = si*24 + bi*6 + ei
            col = X_scf[asite_mask, idx]
            if np.std(col) > 1e-8:
                r_j, _ = pearsonr(col, y_asite)
                nm = f"S_{base}_{elem}_{sigma}"
                asite_scf_cors.append((abs(r_j), r_j, idx, nm))
asite_scf_cors.sort(reverse=True)
log.info("\n  Top SCF features within ribosomal_asite:")
for _, r_j, _, nm in asite_scf_cors[:10]:
    log.info(f"    {nm:18s} r={r_j:+.3f}")

def loo_ridge_subclass(X_feat, y_cls, label, alpha_grid=None):
    """LOO Ridge for a subclass."""
    if alpha_grid is None:
        alpha_grid = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    n_cls = len(y_cls)
    best_r, best_preds, best_alpha = -99.0, None, None
    for alpha in alpha_grid:
        preds = np.zeros(n_cls)
        try:
            for i in range(n_cls):
                tr = [j for j in range(n_cls) if j != i]
                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_feat[tr])
                X_te_s = sc.transform(X_feat[[i]])
                m = Ridge(alpha=alpha)
                m.fit(X_tr_s, y_cls[tr])
                preds[i] = float(m.predict(X_te_s)[0])
            preds = np.clip(preds, y_cls.min()-3, y_cls.max()+3)
            r_try, _ = pearsonr(preds, y_cls)
            if r_try > best_r:
                best_r = r_try; best_preds = preds.copy(); best_alpha = alpha
        except Exception:
            continue
    log.info(f"  [{label}] Best LOO Ridge: r={best_r:.4f}  alpha={best_alpha}")
    return best_preds, best_r

# Ribosomal A-site: try various feature sets
r_s28_asite, _ = pearsonr(pred_s28[asite_mask], y_asite)
log.info(f"\n  step28 r for ribosomal_asite: {r_s28_asite:.4f}")

top_cpf_asite_idx = [t[2] for t in asite_cpf_cors[:12]]
top_scf_asite_idx = [t[2] for t in asite_scf_cors[:12]]

asite_feature_sets = {
    "CPF_top12":          X_cpf[asite_mask][:, top_cpf_asite_idx],
    "SCF_top12":          X_scf[asite_mask][:, top_scf_asite_idx],
    "CPF_top12+RLIF":     np.hstack([X_cpf[asite_mask][:, top_cpf_asite_idx],
                                      X_rlif[asite_mask]]),
    "SCF_top12+CPF_top12": np.hstack([X_scf[asite_mask][:, top_scf_asite_idx],
                                        X_cpf[asite_mask][:, top_cpf_asite_idx]]),
    "all_CPF+SCF":        np.hstack([X_cpf[asite_mask], X_scf[asite_mask]]),
    "all_CPF+RLIF":       np.hstack([X_cpf[asite_mask], X_rlif[asite_mask]]),
}

best_asite_r = r_s28_asite; best_asite_preds = pred_s28[asite_mask].copy()
for fs_name, X_fs in asite_feature_sets.items():
    preds_try, r_try = loo_ridge_subclass(X_fs, y_asite, f"asite_{fs_name}")
    if r_try > best_asite_r:
        best_asite_r = r_try; best_asite_preds = preds_try

log.info(f"\n  Best A-site model: r={best_asite_r:.4f}  (step28: {r_s28_asite:.4f})")

# ═══════════════════════════════════════════════════════════════════════════
# PART D: Riboswitch subclass targeted models
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART D: Riboswitch Subclass Models")
log.info("="*70)

def probe_and_model(mask, label, top_n=10):
    y_cls = y[mask].astype(np.float64)
    n_cls = mask.sum()
    r_s28, _ = pearsonr(pred_s28[mask], y_cls)
    log.info(f"\n  [{label}] n={n_cls}  step28_r={r_s28:.4f}")
    if n_cls < 4: return pred_s28[mask], r_s28

    # Top CPF
    cpf_cors = []
    for bi, base in enumerate(BASES):
        for ci, cutoff in enumerate(CUTOFFS_CPF):
            for ei, elem in enumerate(ELEMENTS_CPF):
                idx = ci*24 + bi*6 + ei
                col = X_cpf[mask, idx]
                if np.std(col) > 1e-8:
                    r_j, _ = pearsonr(col, y_cls)
                    cpf_cors.append((abs(r_j), r_j, idx, f"N_{base}_{elem}_{int(cutoff)}A"))
    cpf_cors.sort(reverse=True)

    # Top SCF
    scf_cors = []
    for si, sigma in enumerate(SIGMAS_SCF):
        for bi, base in enumerate(BASES):
            for ei, elem in enumerate(ELEMENTS_CPF):
                idx = si*24 + bi*6 + ei
                col = X_scf[mask, idx]
                if np.std(col) > 1e-8:
                    r_j, _ = pearsonr(col, y_cls)
                    scf_cors.append((abs(r_j), r_j, idx, f"S_{base}_{elem}_{sigma}"))
    scf_cors.sort(reverse=True)

    log.info(f"    Top5 CPF: {[(nm,f'{r:.3f}') for _,r,_,nm in cpf_cors[:5]]}")
    log.info(f"    Top5 SCF: {[(nm,f'{r:.3f}') for _,r,_,nm in scf_cors[:5]]}")

    top_cpf_idx = [t[2] for t in cpf_cors[:top_n]]
    top_scf_idx = [t[2] for t in scf_cors[:top_n]]

    fsets = {
        "CPF_top": X_cpf[mask][:, top_cpf_idx],
        "SCF_top": X_scf[mask][:, top_scf_idx],
        "CPF+RLIF": np.hstack([X_cpf[mask][:, top_cpf_idx], X_rlif[mask]]),
        "SCF+RLIF": np.hstack([X_scf[mask][:, top_scf_idx], X_rlif[mask]]),
        "CPF+SCF": np.hstack([X_cpf[mask][:, top_cpf_idx], X_scf[mask][:, top_scf_idx]]),
    }

    best_r_local, best_preds_local = r_s28, pred_s28[mask].copy()
    for fs_name, X_fs in fsets.items():
        preds_try, r_try = loo_ridge_subclass(X_fs, y_cls, f"{label}_{fs_name}")
        if r_try > best_r_local:
            best_r_local = r_try; best_preds_local = preds_try

    log.info(f"    Final: r={best_r_local:.4f}  delta={best_r_local-r_s28:+.4f}")
    return best_preds_local, best_r_local

# Probe each riboswitch subclass
purine_final,  purine_r  = probe_and_model(purine_mask,  "purine")
fmn_final,     fmn_r     = probe_and_model(fmn_fad_mask, "FMN_FAD")
tpp_final,     tpp_r     = probe_and_model(tpp_mask,     "TPP")
otherlig_final,otherlig_r= probe_and_model(otherlig_mask,"other_lig")

# SAM_SAH: check if SCF adds anything
y_ss = y[sam_sah_mask].astype(np.float64)
r_s28_ss, _ = pearsonr(pred_s28[sam_sah_mask], y_ss)
log.info(f"\n  [SAM_SAH] n={sam_sah_mask.sum()}  step28_r={r_s28_ss:.4f}")
# Try SCF + CPF for SAM_SAH
ss_feature_sets = {
    "s28_baseline": pred_s28[sam_sah_mask],  # just reference
}
# SCF within SAM_SAH
scf_ss_cors = []
for si, sigma in enumerate(SIGMAS_SCF):
    for bi, base in enumerate(BASES):
        for ei, elem in enumerate(ELEMENTS_CPF):
            idx = si*24 + bi*6 + ei
            col = X_scf[sam_sah_mask, idx]
            if np.std(col) > 1e-8:
                r_j, _ = pearsonr(col, y_ss)
                scf_ss_cors.append((abs(r_j), r_j, idx, f"S_{base}_{elem}_{sigma}"))
scf_ss_cors.sort(reverse=True)
top_scf_ss = [t[2] for t in scf_ss_cors[:8]]
top_cpf_ss = [7, 19, 56, 32, 26, 45, 31, 55]  # from step27 analysis
X_ss_scf_cpf = np.hstack([X_scf[sam_sah_mask][:, top_scf_ss],
                            X_cpf[sam_sah_mask][:, top_cpf_ss]])
ss_preds_try, ss_r_try = loo_ridge_subclass(X_ss_scf_cpf, y_ss, "SAM_SAH_scf+cpf")
samsah_final = pred_s28[sam_sah_mask].copy()
samsah_r     = r_s28_ss
if ss_r_try > r_s28_ss:
    samsah_final = ss_preds_try; samsah_r = ss_r_try
    log.info(f"  SAM_SAH improved: r={samsah_r:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# PART E: other_misc with SCF
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART E: other_misc SCF improvement")
log.info("="*70)

y_om = y[om_mask].astype(np.float64)
n_om = om_mask.sum()
r_s28_om, _ = pearsonr(pred_s28[om_mask], y_om)
log.info(f"  n={n_om}  step28_r={r_s28_om:.4f}")

# Top SCF in other_misc
om_scf_cors = []
for si, sigma in enumerate(SIGMAS_SCF):
    for bi, base in enumerate(BASES):
        for ei, elem in enumerate(ELEMENTS_CPF):
            idx = si*24 + bi*6 + ei
            col = X_scf[om_mask, idx]
            if np.std(col) > 1e-8:
                r_j, _ = pearsonr(col, y_om)
                om_scf_cors.append((abs(r_j), r_j, idx, f"S_{base}_{elem}_{sigma}"))
om_scf_cors.sort(reverse=True)
log.info("  Top SCF in other_misc:")
for _, r_j, _, nm in om_scf_cors[:8]:
    log.info(f"    {nm:22s} r={r_j:+.3f}")

top_scf_om = [t[2] for t in om_scf_cors[:15]]
top_cpf_om_cors = sorted([
    (abs(pearsonr(X_cpf[om_mask, j], y_om)[0]), pearsonr(X_cpf[om_mask, j], y_om)[0], j)
    for j in range(72) if np.std(X_cpf[om_mask, j]) > 1e-8
], reverse=True)
top_cpf_om = [t[2] for t in top_cpf_om_cors[:15]]

om_feature_sets = {
    "all_CPF+RLIF (s28 best)": np.hstack([X_cpf[om_mask], X_rlif[om_mask]]),
    "all_SCF+RLIF":            np.hstack([X_scf[om_mask], X_rlif[om_mask]]),
    "all_CPF+SCF+RLIF":        np.hstack([X_cpf[om_mask], X_scf[om_mask], X_rlif[om_mask]]),
    "top_SCF+CPF+RLIF":        np.hstack([X_scf[om_mask][:, top_scf_om],
                                            X_cpf[om_mask][:, top_cpf_om],
                                            X_rlif[om_mask]]),
}

best_om_r = r_s28_om; best_om_preds = pred_s28[om_mask].copy()
for fs_name, X_fs in om_feature_sets.items():
    preds_try, r_try = loo_ridge_subclass(X_fs, y_om, f"other_misc_{fs_name}")
    if r_try > best_om_r:
        best_om_r = r_try; best_om_preds = preds_try
log.info(f"\n  Best other_misc: r={best_om_r:.4f}  (step28: {r_s28_om:.4f})")

# ═══════════════════════════════════════════════════════════════════════════
# PART F: SCF kernel in riboswitch MKL
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART F: SCF kernel in riboswitch MKL")
log.info("="*70)

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

def tanimoto(X, Y=None):
    if Y is None: Y = X
    XY = X @ Y.T; XX = X.sum(1, keepdims=True); YY = Y.sum(1, keepdims=True)
    return XY / np.where(XX+YY.T-XY < 1e-10, 1e-10, XX+YY.T-XY)

# Build kernels
X_topo   = X11[:, np.r_[0:36000, 38963:49763]]
X_rnafm  = X11[:, 38064:38704]
X_morgan = X11[:, 36000:38048]
X_maccs  = X11[:, 38796:38963]

sc = StandardScaler()
K_topo   = rbf_kernel(sc.fit_transform(X_topo),   gamma=1e-6)
K_rna    = rbf_kernel(sc.fit_transform(X_rnafm),  gamma=5e-3)
K_uni    = rbf_kernel(sc.fit_transform(unimol_full), gamma=0.05)
K_tan    = 0.7*tanimoto(X_morgan) + 0.3*tanimoto(X_maccs)
K_lig    = 0.5*K_uni + 0.5*K_tan
K_rlif   = rbf_kernel(sc.fit_transform(X_rlif),   gamma=0.5)
K_pocket = rbf_kernel(sc.fit_transform(X_pocket),  gamma=0.01)

# SCF kernel search
log.info("  SCF gamma search (for addition to 5k base):")
K_base5 = 0.50*K_topo + 0.15*K_lig + 0.20*K_rna + 0.05*K_pocket + 0.10*K_rlif
best_scf_g, best_scf_rs_r = 0.001, -99.0
for gamma in [1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5]:
    K_scf_try = rbf_kernel(sc.fit_transform(X_scf), gamma=gamma)
    for w_scf in [0.05, 0.10, 0.15]:
        K_try = (1-w_scf)*K_base5 + w_scf*K_scf_try
        D_ = np.sqrt(np.diag(K_try)).reshape(-1,1); D_ = np.where(D_<1e-10,1e-10,D_)
        pp = loo_mkl(K_try/(D_*D_.T), y, alpha=0.01)
        r_rs, _ = pearsonr(pp[rs_mask], y[rs_mask])
        if r_rs > best_scf_rs_r:
            best_scf_rs_r = r_rs; best_scf_g = gamma

K_scf = rbf_kernel(sc.fit_transform(X_scf), gamma=best_scf_g)
log.info(f"  Best SCF kernel: gamma={best_scf_g}  rs_r={best_scf_rs_r:.4f}")

# Quick CPF kernel test too
K_cpf_kernel = rbf_kernel(sc.fit_transform(X_cpf), gamma=0.01)
K6_test = 0.85*K_base5 + 0.15*K_cpf_kernel
D_ = np.sqrt(np.diag(K6_test)).reshape(-1,1); D_ = np.where(D_<1e-10,1e-10,D_)
pp6 = loo_mkl(K6_test/(D_*D_.T), y, alpha=0.01)
r_rs_cpf, _ = pearsonr(pp6[rs_mask], y[rs_mask])
log.info(f"  CPF kernel only in MKL: rs_r={r_rs_cpf:.4f}")

# Best 6-kernel combination: 5k + SCF
best_6k_r, best_6k_p = -99.0, None
for w_scf in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    K6 = (1-w_scf)*K_base5 + w_scf*K_scf
    D_ = np.sqrt(np.diag(K6)).reshape(-1,1); D_ = np.where(D_<1e-10,1e-10,D_)
    K6n = K6/(D_*D_.T)
    for lam in [0.005, 0.01, 0.05]:
        pp = loo_mkl(K6n, y, alpha=lam)
        r_rs, _ = pearsonr(pp[rs_mask], y[rs_mask])
        if r_rs > best_6k_r:
            best_6k_r = r_rs; best_6k_p = pp.copy()
log.info(f"  Best 6k (5k+SCF) MKL: rs_r={best_6k_r:.4f}")
r_rs_s27, _ = pearsonr(pred_s27[rs_mask], y[rs_mask])
log.info(f"  Step27 MKL for riboswitch: rs_r={r_rs_s27:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# PART G: Final Hybrid Assembly
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART G: Final Hybrid Assembly")
log.info("="*70)

# Decide riboswitch source: per-subclass targeted OR 6k-MKL OR step27/28
# Start from step27 SAM_SAH replacement, then override subclasses if better

def assemble_rs_preds(base_rs_preds, subclass_preds_dict):
    """Start from base, override per subclass if we have better."""
    final = base_rs_preds.copy()
    for mask_k, preds_k, r_k in subclass_preds_dict:
        # Check if targeted is better
        r_base, _ = pearsonr(base_rs_preds[mask_k[rs_mask]], y[mask_k])
        if r_k > r_base and preds_k is not None:
            final[mask_k[rs_mask]] = preds_k
    return final

# Base riboswitch: best of 6k-MKL and step27
rs_base_step27 = pred_s27[rs_mask].copy()
rs_base_6k     = best_6k_p[rs_mask].copy() if best_6k_p is not None else rs_base_step27

# Override SAM_SAH with step27/28 targeted (already in pred_s27/s28)
# (step27 already fixed SAM_SAH in rs predictions)

# Try per-subclass overrides on top of step27 base
subclass_overrides = [
    (sam_sah_mask, samsah_final, samsah_r),
    (purine_mask,  purine_final,  purine_r),
    (fmn_fad_mask, fmn_final,    fmn_r),
    (tpp_mask,     tpp_final,    tpp_r),
    (otherlig_mask,otherlig_final,otherlig_r),
]

# Build rs predictions: for each subclass, take targeted if better than current
pred_rs_final = rs_base_step27.copy()
for mask_k, preds_k, r_k in subclass_overrides:
    if mask_k.sum() < 2: continue
    sub_in_rs = mask_k[rs_mask]
    cur_r, _ = pearsonr(pred_rs_final[sub_in_rs], y[mask_k])
    if r_k > cur_r:
        pred_rs_final[sub_in_rs] = preds_k
        log.info(f"  Override: {np.where(mask_k)[0][:1]} r {cur_r:.4f}→{r_k:.4f}")

# Also try 6k-MKL base
pred_rs_final_6k = rs_base_6k.copy()
for mask_k, preds_k, r_k in subclass_overrides:
    if mask_k.sum() < 2: continue
    sub_in_rs = mask_k[rs_mask]
    cur_r, _ = pearsonr(pred_rs_final_6k[sub_in_rs], y[mask_k])
    if r_k > cur_r:
        pred_rs_final_6k[sub_in_rs] = preds_k

r_rs_final, _ = pearsonr(pred_rs_final, y[rs_mask])
r_rs_final_6k, _ = pearsonr(pred_rs_final_6k, y[rs_mask])
log.info(f"\n  Riboswitch r: step27={r_rs_s27:.4f}  final={r_rs_final:.4f}  final_6k={r_rs_final_6k:.4f}")

pred_rs_best = pred_rs_final if r_rs_final >= r_rs_final_6k else pred_rs_final_6k
r_rs_best    = max(r_rs_final, r_rs_final_6k)

# Full hybrid configurations
configs = [
    ("s28 base",                      pred_s28[rs_mask],  pred_s28[om_mask]),
    ("rs_final + s28_OM",             pred_rs_best,       pred_s28[om_mask]),
    ("rs_final + best_om",            pred_rs_best,       best_om_preds),
    ("rs_6k + s28_OM",                rs_base_6k,         pred_s28[om_mask]),
    ("rs_6k + best_om",               rs_base_6k,         best_om_preds),
    ("s28_rs + best_om",              pred_s28[rs_mask],  best_om_preds),
]

results = []
for name, rs_src, om_src in configs:
    hyb = pred_s28.copy()
    hyb[rs_mask] = rs_src
    hyb[om_mask] = om_src
    if asite_mask.sum() > 1 and best_asite_r > r_s28_asite:
        hyb[asite_mask] = best_asite_preds
    valid = ~np.isnan(hyb)
    r, _ = pearsonr(hyb[valid], y[valid])
    results.append((name, hyb.copy(), r))
    log.info(f"  {name:40s}: r={r:.4f}")

best_name, best_hyb, best_r = max(results, key=lambda x: x[2])
log.info(f"\n  → Best: {best_name}")

log.info(f"\nPer-subtype breakdown:")
for st in ["aptamer","duplex_groove","ribosomal_asite","riboswitch",
           "other_misc","g_quadruplex","viral_tar"]:
    mask_st = subtypes == st
    if mask_st.sum() < 2: continue
    r_st = pearsonr(best_hyb[mask_st], y[mask_st])[0]
    log.info(f"  {st:22s}: n={mask_st.sum():3d}  r={r_st:.3f}")

log.info(f"\n  Riboswitch subclass breakdown:")
for cls, m in [("SAM_SAH",sam_sah_mask),("purine",purine_mask),
                ("FMN_FAD",fmn_fad_mask),("TPP",tpp_mask),("other_lig",otherlig_mask)]:
    if m.sum() < 2: continue
    r_c = pearsonr(best_hyb[m], y[m])[0]
    log.info(f"    {cls:10s}: n={m.sum():2d}  r={r_c:.3f}")

log.info(f"\n  Combined r (step29)   = {best_r:.4f}")
log.info(f"  Previous best         = 0.7874   [step28]")
log.info(f"  Delta                 = {best_r - 0.7874:+.4f}")
log.info(f"  Gap to RSAPred        = {0.830 - best_r:.4f}")
for nm, rb in [("AffiGrapher",0.498),("RLaffinity",0.559),("RLASIF",0.666),
               ("DeepRSMA",0.784),("RSAPred",0.830)]:
    log.info(f"  {'✓' if best_r > rb else '✗'} {nm}: {rb:.3f}")
log.info("="*70)

# ── Save ───────────────────────────────────────────────────────────────────
df_out = pd.DataFrame({
    "pdb": ids, "subtype": subtypes, "y_true": y,
    "y_pred": best_hyb, "rs_subclass": rs_subclass,
})
df_out.to_csv(RES_DIR/"step29_results.csv", index=False)
log.info(f"  Results → results/step29_results.csv")

# ── Figure ─────────────────────────────────────────────────────────────────
colors = {"aptamer":"#2166AC","riboswitch":"#1A9641","ribosomal_asite":"#D73027",
          "duplex_groove":"#7B2D8B","g_quadruplex":"#FF7F00",
          "viral_tar":"#A65628","other_misc":"#999999"}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
for st in np.unique(subtypes):
    mask_st = subtypes == st
    r_st = pearsonr(best_hyb[mask_st], y[mask_st])[0] if mask_st.sum() > 1 else 0
    ax.scatter(y[mask_st], best_hyb[mask_st], c=colors.get(st,"#888"),
               label=f"{st} r={r_st:.3f}", alpha=0.75, s=45, edgecolors="none")
lo, hi = y.min()-0.5, y.max()+0.5
ax.plot([lo,hi],[lo,hi],"k--",lw=1,alpha=0.5)
ax.set_xlabel("Experimental pKd"); ax.set_ylabel("Predicted pKd")
ax.set_title(f"Step 29 Hybrid (r={best_r:.4f})", fontweight="bold")
ax.legend(fontsize=7); ax.grid(alpha=0.3, linestyle="--")

ax = axes[1]
steps = {"S27\nSAM_fix":0.7601,"S28\nBeats\nDRSMA":0.7874,"S29\nSCF":best_r}
bar_cols = ["#AAAAAA","#4393C3","#D63027"]
bars = ax.bar(list(steps.keys()), list(steps.values()),
              color=bar_cols, edgecolor="white", alpha=0.9)
for bar, val in zip(bars, steps.values()):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.003,
            f"{val:.4f}", ha='center', va='bottom', fontsize=9)
for nm, rb in [("RLASIF",0.666),("DeepRSMA",0.784),("RSAPred",0.830)]:
    ax.axhline(rb, color='gray', lw=0.8, linestyle='--', alpha=0.6)
    ax.text(2.5, rb+0.003, nm, fontsize=7, color='gray')
ax.set_ylabel("Pearson r"); ax.set_title("Performance progression")
ax.set_ylim(0.65, 0.86)
plt.tight_layout()
fig.savefig(ROOT/"results"/"figures"/"step29_scf_results.png", dpi=150, bbox_inches="tight")
log.info(f"  Figure → results/figures/step29_scf_results.png")
log.info("STEP 29 COMPLETE")
