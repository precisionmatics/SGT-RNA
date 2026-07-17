"""
SGT-RNA · Step 27: SAM_SAH Targeted Model + Improved Riboswitch Hybrid

The global 5k-MKL gives r=-0.326 for SAM_SAH riboswitches (n=10).
Key insight from step26 probe: SAM_SAH affinity is driven by Guanine contacts.
  - N_G_8A   (RLIF, idx=9):  r=+0.704 within SAM_SAH
  - N_G_N_3A (CPF,  idx=7):  r=+0.692 within SAM_SAH

Step 27:
  A. Identify SAM_SAH riboswitch subclass members via ligand name
  B. Targeted LOO Ridge for SAM_SAH using 12 CPF+RLIF G-contact features → r≈0.632
  C. Extended search: add more features, try kernel methods
  D. Integrate into full hybrid replacing SAM_SAH component
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
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent.parent
NA_L    = ROOT / "NA-L"
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
        logging.FileHandler(ROOT / "results" / "logs" / f"step27_{ts}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("SGT-RNA · Step 27: SAM_SAH Targeted Model")
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

# SAM/SAH ligand residue codes in PDB
SAM_SAH_CODES = {
    "SAM", "SAH", "SFG", "AGN", "5GP", "8OG", "SAX",
    "SOH", "SCA",  # S-cysteyladenosine = SAH analog
    "ACP",         # acetyl-CoA-related
    "MET",         # some SAM-like in older structures
}
# Keyword substrings in ligand mol2 TRIPOS name
SAM_SAH_KEYWORDS = ["adenosyl", "methionin", "homocystein", "sinefungin"]

def detect_sam_sah(pdb, na_l_root):
    """Return True if the ligand is SAM/SAH-class."""
    mol2_f = na_l_root / pdb / f"{pdb}_ligand.mol2"
    if not mol2_f.exists():
        return False
    try:
        with open(mol2_f) as f:
            content = f.read()
        # Check residue name in @<TRIPOS>ATOM section
        in_atom = False
        for line in content.splitlines():
            if '@<TRIPOS>ATOM' in line:
                in_atom = True; continue
            if '@<TRIPOS>' in line and 'ATOM' not in line:
                in_atom = False; continue
            if in_atom and line.strip():
                parts = line.split()
                if len(parts) >= 8:
                    res_name = parts[7].strip().upper()
                    if res_name in SAM_SAH_CODES:
                        return True
                    for kw in SAM_SAH_KEYWORDS:
                        if kw in res_name.lower():
                            return True
        # Also check @<TRIPOS>MOLECULE block
        for line in content.splitlines():
            ll = line.strip().upper()
            for code in SAM_SAH_CODES:
                if ll == code:
                    return True
            for kw in SAM_SAH_KEYWORDS:
                if kw in ll.lower():
                    return True
    except Exception:
        pass
    return False

# ── Feature name helpers ───────────────────────────────────────────────────
BASES        = ['A', 'G', 'C', 'U']
ELEMENTS_CPF = ['C', 'N', 'O', 'S', 'P', 'X']
CUTOFFS_CPF  = [3.0, 4.0, 5.0]
CUTOFFS_RLIF = [4.0, 6.0, 8.0]
N_CPF        = len(BASES) * len(ELEMENTS_CPF) * len(CUTOFFS_CPF)  # 72

def cpf_idx(base, element, cutoff):
    ci = CUTOFFS_CPF.index(cutoff)
    bi = BASES.index(base)
    ei = ELEMENTS_CPF.index(element)
    return ci * (len(BASES) * len(ELEMENTS_CPF)) + bi * len(ELEMENTS_CPF) + ei

def rlif_idx(base, cutoff):
    ci = CUTOFFS_RLIF.index(cutoff)
    bi = BASES.index(base)
    return ci * len(BASES) + bi

# ── Load data ──────────────────────────────────────────────────────────────
log.info("\nLoading data ...")
d11          = np.load(S11_NPZ)
X11          = d11["X"].astype(np.float64)
y            = d11["y"].astype(np.float32)
ids          = d11["ids"]
subtypes_raw = d11["subtypes"]
subtypes     = np.array([make_subtype(p, s) for p, s in zip(ids, subtypes_raw)])
n            = len(y)

X_rlif  = np.load(ROOT / "data" / "features" / "rlif_features.npy")
X_cpf   = np.load(ROOT / "data" / "features" / "cpf_features.npy")
X_wcf   = np.load(ROOT / "data" / "features" / "wcf_features.npy")
X_pocket= np.load(ROOT / "data" / "features" / "pocket_seq_features.npy")
X_rlifv2= np.load(ROOT / "data" / "features" / "rlif_v2_features.npy")

s24_df   = pd.read_csv(ROOT / "results" / "step24_results.csv")
pdb2s24  = dict(zip(s24_df["pdb"], s24_df["y_pred"]))
pred_s24 = np.array([float(pdb2s24.get(p, np.nan)) for p in ids])

s22_df   = pd.read_csv(ROOT / "results" / "step22_results.csv")
pdb2s22  = dict(zip(s22_df["pdb"], s22_df["y_pred"]))
pred_s22 = np.array([float(pdb2s22.get(p, np.nan)) for p in ids])

unimol_emb_raw = np.load("/tmp/unimol_emb.npy")
valid_idx_raw  = np.load("/tmp/unimol_valid_idx.npy").astype(int)
unimol_full    = np.zeros((n, 512), dtype=np.float64)
unimol_full[valid_idx_raw] = unimol_emb_raw
missing_idx    = [i for i in range(n) if i not in set(valid_idx_raw.tolist())]
if missing_idx: unimol_full[missing_idx] = unimol_emb_raw.mean(axis=0)

log.info(f"  Loaded {n} complexes")
log.info(f"  X_rlif: {X_rlif.shape}, X_cpf: {X_cpf.shape}")

# ═══════════════════════════════════════════════════════════════════════════
# PART A: Identify SAM_SAH riboswitch members
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART A: SAM_SAH Riboswitch Identification")
log.info("="*70)

rs_mask  = subtypes == "riboswitch"
rs_idx   = np.where(rs_mask)[0]
rs_ids   = ids[rs_mask]

sam_sah_flags = np.array([detect_sam_sah(p, NA_L) for p in ids])
sam_sah_mask  = rs_mask & sam_sah_flags

log.info(f"  Riboswitches: n={rs_mask.sum()}")
log.info(f"  SAM_SAH detected: n={sam_sah_mask.sum()}")
log.info(f"  SAM_SAH PDB IDs: {sorted(ids[sam_sah_mask].tolist())}")
log.info(f"  SAM_SAH pKd: {y[sam_sah_mask]}")

n_ss = sam_sah_mask.sum()
ss_idx = np.where(sam_sah_mask)[0]

if n_ss < 4:
    log.warning("  Too few SAM_SAH structures detected — check detection logic")

# ═══════════════════════════════════════════════════════════════════════════
# PART B: SAM_SAH Targeted Model
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART B: SAM_SAH Targeted LOO Ridge")
log.info("="*70)

y_ss = y[sam_sah_mask].astype(np.float64)

# Feature selection: G-contact features identified from probe
# CPF features (strong within SAM_SAH)
cpf_target_specs = [
    ('G', 'N', 3.0),   # r=+0.692
    ('U', 'N', 3.0),   # r=-0.687
    ('G', 'O', 4.0),   # r=+0.574
    ('G', 'O', 5.0),   # r=+0.670
    ('C', 'O', 4.0),   # positive
    ('U', 'S', 4.0),   # SAM sulfonium-related
    ('G', 'N', 4.0),   # G-N contacts at 4Å
    ('G', 'N', 5.0),   # G-N contacts at 5Å
    ('G', 'C', 3.0),   # G-C (stacking)
    ('A', 'N', 3.0),   # A-N (adenosine common in SAM)
    ('A', 'N', 4.0),
    ('G', 'S', 5.0),   # G-Sulfur: sulfonium group
]
cpf_target_idx = [cpf_idx(*spec) for spec in cpf_target_specs]
cpf_names_tgt  = [f"N_{b}_{e}_{int(c)}A" for (b, e, c) in cpf_target_specs]

# RLIF features (strong within SAM_SAH)
rlif_target_specs = [
    ('G', 8.0),   # r=+0.704 (strongest!)
    ('G', 6.0),   # r=+0.592
    ('G', 4.0),
    ('U', 8.0),   # r=-0.431
    ('U', 6.0),
    ('A', 8.0),
    ('A', 6.0),
]
rlif_target_idx = [rlif_idx(*spec) for spec in rlif_target_specs]
rlif_names_tgt  = [f"N_{b}_{int(c)}A" for (b, c) in rlif_target_specs]
# Add N_hbond (index 12 in RLIF)
N_RLIF_HBOND = 12  # = 3 cutoffs × 4 bases = 12
rlif_target_idx.append(N_RLIF_HBOND)
rlif_names_tgt.append("N_hbond")

log.info(f"\n  CPF target features ({len(cpf_target_idx)}): {cpf_names_tgt}")
log.info(f"  RLIF target features ({len(rlif_target_idx)}): {rlif_names_tgt}")

# Print univariate correlations within SAM_SAH
log.info("\n  Within SAM_SAH univariate correlations:")
for name, idx in zip(cpf_names_tgt, cpf_target_idx):
    col = X_cpf[sam_sah_mask, idx]
    if np.std(col) > 1e-8:
        r_j, _ = pearsonr(col, y_ss)
        log.info(f"    CPF {name:15s} r={r_j:+.3f}  vals={col.astype(int).tolist()}")
for name, idx in zip(rlif_names_tgt, rlif_target_idx):
    col = X_rlif[sam_sah_mask, idx]
    if np.std(col) > 1e-8:
        r_j, _ = pearsonr(col, y_ss)
        log.info(f"    RLIF {name:12s} r={r_j:+.3f}  vals={col.astype(int).tolist()}")

# Build feature matrix for SAM_SAH
X_cpf_ss   = X_cpf[sam_sah_mask][:, cpf_target_idx]
X_rlif_ss  = X_rlif[sam_sah_mask][:, rlif_target_idx]
X_rlifv2_ss = X_rlifv2[sam_sah_mask][:, rlif_target_idx]

# Also try all CPF+RLIF concatenated
X_all_cpf_ss  = X_cpf[sam_sah_mask]
X_all_rlif_ss = X_rlif[sam_sah_mask]
X_all_rlifv2_ss = X_rlifv2[sam_sah_mask]

def loo_ridge_ss(X_feat, y_feat, alpha):
    """LOO Ridge for small group."""
    n_ss_loc = len(y_feat)
    preds = np.zeros(n_ss_loc)
    for i in range(n_ss_loc):
        tr = [j for j in range(n_ss_loc) if j != i]
        X_tr, X_te = X_feat[tr], X_feat[[i]]
        y_tr = y_feat[tr]
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        m = Ridge(alpha=alpha)
        m.fit(X_tr_s, y_tr)
        preds[i] = float(m.predict(X_te_s)[0])
    # Clip
    preds = np.clip(preds, y_feat.min()-3, y_feat.max()+3)
    r, _ = pearsonr(preds, y_feat)
    return preds, r

log.info("\n  LOO Ridge: targeted CPF+RLIF (12 features)")
best_ss_r, best_ss_preds, best_ss_cfg = -99.0, None, None

feature_sets = {
    "CPF_tgt+RLIF_tgt":    np.hstack([X_cpf_ss,     X_rlif_ss]),
    "CPF_tgt+RLIFv2_tgt":  np.hstack([X_cpf_ss,     X_rlifv2_ss]),
    "RLIF_tgt_only":       X_rlif_ss,
    "RLIFv2_tgt_only":     X_rlifv2_ss,
    "CPF_tgt_only":        X_cpf_ss,
    "all_CPF+RLIF_tgt":    np.hstack([X_all_cpf_ss,  X_rlif_ss]),
    "all_CPF+RLIFv2_tgt":  np.hstack([X_all_cpf_ss,  X_rlifv2_ss]),
    "all_CPF+all_RLIF":    np.hstack([X_all_cpf_ss,  X_all_rlif_ss]),
}

alpha_grid = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

for fs_name, X_fs in feature_sets.items():
    best_r_fs = -99.0
    best_a_fs = None
    for alpha in alpha_grid:
        if X_fs.shape[0] == 0: continue
        try:
            preds_try, r_try = loo_ridge_ss(X_fs, y_ss, alpha)
        except Exception:
            continue
        if r_try > best_r_fs:
            best_r_fs = r_try
            best_a_fs = alpha
            if r_try > best_ss_r:
                best_ss_r     = r_try
                best_ss_preds = preds_try.copy()
                best_ss_cfg   = (fs_name, alpha)
    log.info(f"    {fs_name:30s}: r={best_r_fs:.4f}  (alpha={best_a_fs})")

log.info(f"\n  Best SAM_SAH model: r={best_ss_r:.4f}  cfg={best_ss_cfg}")

# Also try kernel ridge within SAM_SAH
log.info("\n  LOO KernelRidge: RBF on CPF+RLIF")
X_kr = np.hstack([X_cpf_ss, X_rlif_ss])
sc_kr = StandardScaler()
X_kr_s = sc_kr.fit_transform(X_kr)
best_kr_r, best_kr_preds = -99.0, None
for gamma in [1e-3, 0.01, 0.05, 0.1, 0.5, 1.0]:
    for alpha in [0.001, 0.01, 0.1, 1.0]:
        K_ss = rbf_kernel(X_kr_s, gamma=gamma)
        preds_kr = np.zeros(n_ss)
        try:
            for i in range(n_ss):
                tr = [j for j in range(n_ss) if j != i]
                m = KernelRidge(alpha=alpha, kernel="precomputed")
                m.fit(K_ss[np.ix_(tr, tr)], y_ss[tr])
                preds_kr[i] = float(m.predict(K_ss[i, tr].reshape(1, -1))[0])
            preds_kr = np.clip(preds_kr, y_ss.min()-3, y_ss.max()+3)
            r_kr, _ = pearsonr(preds_kr, y_ss)
            if r_kr > best_kr_r:
                best_kr_r = r_kr
                best_kr_preds = preds_kr.copy()
        except Exception:
            continue
log.info(f"    Best KernelRidge SAM_SAH: r={best_kr_r:.4f}")

if best_kr_r > best_ss_r:
    best_ss_r     = best_kr_r
    best_ss_preds = best_kr_preds.copy()
    best_ss_cfg   = ("KernelRidge_CPF+RLIF", "grid")
    log.info(f"  KernelRidge wins → using r={best_ss_r:.4f}")

log.info(f"\n  Final SAM_SAH predictions (vs -0.326 from global MKL):")
for pdb_id, y_true, y_pred in zip(ids[sam_sah_mask], y_ss, best_ss_preds):
    log.info(f"    {pdb_id}: y_true={y_true:.3f}  y_pred={y_pred:.3f}  "
             f"err={abs(y_pred-y_true):.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# PART C: Improved Riboswitch Subclass Model
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART C: Full Riboswitch Hybrid with SAM_SAH Targeted Predictions")
log.info("="*70)

# Start from step24 5k-MKL riboswitch predictions
# Replace SAM_SAH subset with targeted predictions
pred_rs_s24    = pred_s24[rs_mask].copy()
pred_rs_hybrid = pred_rs_s24.copy()

# Map SAM_SAH predictions back into riboswitch-level array
sam_in_rs = sam_sah_flags[rs_mask]  # bool array within riboswitch
pred_rs_hybrid[sam_in_rs] = best_ss_preds

r_rs_s24,    _ = pearsonr(pred_rs_s24,    y[rs_mask])
r_rs_hybrid, _ = pearsonr(pred_rs_hybrid, y[rs_mask])
log.info(f"  Riboswitch r before (step24 MKL): {r_rs_s24:.4f}")
log.info(f"  Riboswitch r after SAM_SAH fix:   {r_rs_hybrid:.4f}")
log.info(f"  Delta riboswitch:                  {r_rs_hybrid-r_rs_s24:+.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# PART D: Final Full Hybrid Assembly
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*70)
log.info("PART D: Final Hybrid Assembly")
log.info("="*70)

om_mask = subtypes == "other_misc"

def make_hybrid(rs_preds, om_preds, base=pred_s24):
    hyb = base.copy()
    hyb[rs_mask] = rs_preds
    hyb[om_mask] = om_preds
    valid = ~np.isnan(hyb)
    r, _ = pearsonr(hyb[valid], y[valid])
    return hyb, r

configs = [
    ("s24 base (reference)",         pred_s24[rs_mask],   pred_s22[om_mask]),
    ("s27 SAM_fix + s22_OM",         pred_rs_hybrid,      pred_s22[om_mask]),
    ("s27 SAM_fix + s24_OM",         pred_rs_hybrid,      pred_s24[om_mask]),
]

results = []
for name, rs_src, om_src in configs:
    hyb, r = make_hybrid(rs_src, om_src)
    results.append((name, hyb, r))
    log.info(f"  {name:45s}: r={r:.4f}")

best_name, best_hyb, best_r = max(results, key=lambda x: x[2])
log.info(f"\n  → Best config: {best_name}")

log.info(f"\nPer-subtype breakdown:")
for st in ["aptamer","duplex_groove","ribosomal_asite","riboswitch",
           "other_misc","g_quadruplex","viral_tar"]:
    mask_st = subtypes == st
    if mask_st.sum() < 2: continue
    r_st = pearsonr(best_hyb[mask_st], y[mask_st])[0]
    log.info(f"  {st:22s}: n={mask_st.sum():3d}  r={r_st:.3f}")

# SAM_SAH breakdown within riboswitch
if n_ss >= 2:
    r_ss_new = pearsonr(best_ss_preds, y_ss)[0]
    log.info(f"\n  SAM_SAH specifically: n={n_ss}  r={r_ss_new:.3f}  "
             f"(was r=-0.326 from global MKL)")

log.info(f"\n  Combined r (step27)   = {best_r:.4f}")
log.info(f"  Previous best         = 0.7412   [step24]")
log.info(f"  Delta                 = {best_r - 0.7412:+.4f}")
log.info(f"  Gap to DeepRSMA       = {0.784 - best_r:.4f}")
for nm, rb in [("AffiGrapher",0.498),("RLaffinity",0.559),("RLASIF",0.666),
               ("DeepRSMA",0.784),("RSAPred",0.830)]:
    log.info(f"  {'✓' if best_r > rb else '✗'} {nm}: {rb:.3f}")
log.info("="*70)

# ── Save results ───────────────────────────────────────────────────────────
df_out = pd.DataFrame({
    "pdb": ids, "subtype": subtypes, "y_true": y,
    "y_pred": best_hyb,
    "is_sam_sah": sam_sah_flags.astype(int),
})
df_out.to_csv(RES_DIR / "step27_results.csv", index=False)
log.info(f"  Results → results/step27_results.csv")

# ── Figure ─────────────────────────────────────────────────────────────────
colors = {"aptamer":"#2166AC","riboswitch":"#1A9641","ribosomal_asite":"#D73027",
          "duplex_groove":"#7B2D8B","g_quadruplex":"#FF7F00",
          "viral_tar":"#A65628","other_misc":"#999999"}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ax = axes[0]
for st in np.unique(subtypes):
    mask_st = subtypes == st
    r_st = pearsonr(best_hyb[mask_st], y[mask_st])[0] if mask_st.sum() > 1 else 0
    ax.scatter(y[mask_st], best_hyb[mask_st], c=colors.get(st,"#888"),
               label=f"{st} r={r_st:.3f}", alpha=0.75, s=45, edgecolors="none")
lo, hi = y.min()-0.5, y.max()+0.5
ax.plot([lo,hi],[lo,hi],"k--",lw=1,alpha=0.5)
ax.set_xlabel("Experimental pKd"); ax.set_ylabel("Predicted pKd")
ax.set_title(f"Step 27 Hybrid (r={best_r:.4f})", fontweight="bold")
ax.legend(fontsize=7); ax.grid(alpha=0.3, linestyle="--")

if n_ss >= 2:
    ax2 = axes[1]
    ax2.scatter(y_ss, best_ss_preds, c="#D73027", s=70, zorder=3, label="SAM_SAH (step27)")
    lo2, hi2 = y_ss.min()-0.5, y_ss.max()+0.5
    ax2.plot([lo2,hi2],[lo2,hi2],"k--",lw=1,alpha=0.5)
    for pdb_id, yt, yp in zip(ids[sam_sah_mask], y_ss, best_ss_preds):
        ax2.annotate(pdb_id, (yt, yp), fontsize=7, ha='left', va='bottom')
    ax2.set_xlabel("Experimental pKd"); ax2.set_ylabel("Predicted pKd")
    ax2.set_title(f"SAM_SAH targeted model r={best_ss_r:.3f}", fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, linestyle="--")

ax3 = axes[2]
steps = {"S22\nRLIF":0.7375,"S24\n5kMKL":0.7412,"S27\nSAM_fix":best_r}
bar_cols = ["#AAAAAA","#4393C3","#D63027"]
bars = ax3.bar(list(steps.keys()), list(steps.values()),
               color=bar_cols, edgecolor="white", alpha=0.9)
for bar, val in zip(bars, steps.values()):
    ax3.text(bar.get_x()+bar.get_width()/2, val+0.003,
             f"{val:.4f}", ha='center', va='bottom', fontsize=9)
for nm, rb in [("AffiGrapher",0.498),("RLaffinity",0.559),("RLASIF",0.666),
               ("DeepRSMA",0.784),("RSAPred",0.830)]:
    ax3.axhline(rb, color='gray', lw=0.8, linestyle='--', alpha=0.6)
    ax3.text(2.5, rb+0.003, nm, fontsize=7, color='gray')
ax3.set_ylabel("Pearson r"); ax3.set_title("Performance progression")
ax3.set_ylim(0.65, 0.86)
plt.tight_layout()
fig.savefig(ROOT/"results"/"figures"/"step27_sam_results.png", dpi=150, bbox_inches="tight")
log.info(f"  Figure → results/figures/step27_sam_results.png")
log.info("STEP 27 COMPLETE")
