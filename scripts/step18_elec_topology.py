"""
RNA-PDFL · Step 18: Electrostatic Persistent Topology

For each complex:
  1. pdb2pqr (AMBER FF) → RNA atom partial charges
  2. RDKit Gasteiger     → ligand atom partial charges
  3. Interface atoms (RNA ≤8Å from ligand centroid + all ligand atoms)
  4. Coulomb potential φ_i = Σ_{j≠i} q_j / r_ij at each atom
  5. Gudhi alpha complex filtered by φ → persistent homology
  6. Betti curves (H0,H1,H2) over 50 thresholds → 150 features
  7. Persistence diagram stats (H0,H1,H2) × 5 → 15 features
  8. Scalar charge features → 10 features
  Total new: 175 features

Combine with step11 features (49,763) → 49,938 total
Evaluate hybrid model.
"""

import subprocess, gzip, pickle, logging, time, os, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel

warnings.filterwarnings("ignore")

ROOT    = Path("/home/stalin/Desktop/PDFL-RNA/RNA_PDFL")
NA_L    = Path("/home/stalin/Desktop/PDFL-RNA/NA-L")
PKL_FILE= ROOT / "data" / "pocket_fri" / "pocket_fri_data.pkl.gz"
S11_NPZ = ROOT / "data" / "features" / "step11_full_features.npz"
S11_CSV = ROOT / "results"  / "step11_results.csv"
OUT_NPZ = ROOT / "data" / "features" / "step18_full_features.npz"
RES_DIR = ROOT / "results"
FIG_DIR = ROOT / "results" / "figures"
TMPDIR  = Path("/tmp/elec_rna")
TMPDIR.mkdir(parents=True, exist_ok=True)
N_WORKERS = 20

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "results" / "logs" / f"step18_{ts}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("RNA-PDFL · Step 18: Electrostatic Persistent Topology")
log.info("=" * 70)

# ── subtype labels ─────────────────────────────────────────────────────────────
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
N_BETTI_THRESH = 50   # thresholds for Betti curves
N_ELEC_FEAT    = 175  # 150 betti + 15 pers-stats + 10 scalar

# ─────────────────────────────────────────────────────────────────────────────
# Worker (must be module-level for multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────
def _elec_worker(args):
    """Compute electrostatic topology features for one complex."""
    pdb, lig_centroid, lig_coords_flat, lig_charges_flat = args

    import subprocess
    import numpy as np
    from pathlib import Path
    from scipy.spatial.distance import cdist
    import gudhi
    from rdkit import Chem
    from rdkit.Chem import AllChem

    NA_L   = Path("/home/stalin/Desktop/PDFL-RNA/NA-L")
    TMPDIR = Path("/tmp/elec_rna")
    nan_feat = np.full(175, np.nan, dtype=np.float32)

    pocket_pdb = NA_L / pdb / f"{pdb}_pocket.pdb"
    pqr_file   = TMPDIR / f"{pdb}_pocket.pqr"

    # ── Step 1: pdb2pqr → RNA charges ────────────────────────────────────────
    if not pqr_file.exists():
        try:
            r = subprocess.run(
                ["pdb2pqr", "--ff", "AMBER", "--nodebump", "--noopt",
                 str(pocket_pdb), str(pqr_file)],
                capture_output=True, text=True, timeout=30
            )
            if not pqr_file.exists() or pqr_file.stat().st_size == 0:
                return pdb, nan_feat
        except Exception:
            return pdb, nan_feat

    # ── Step 2: parse PQR → RNA (coords, charges) ────────────────────────────
    rna_coords_list, rna_charges_list = [], []
    try:
        with open(pqr_file) as f:
            for line in f:
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                parts = line.split()
                if len(parts) < 10:
                    continue
                try:
                    x, y, z = float(parts[5]), float(parts[6]), float(parts[7])
                    q = float(parts[8])
                    rna_coords_list.append([x, y, z])
                    rna_charges_list.append(q)
                except (ValueError, IndexError):
                    continue
    except Exception:
        return pdb, nan_feat

    if len(rna_coords_list) < 3:
        return pdb, nan_feat

    rna_coords  = np.array(rna_coords_list, dtype=np.float64)
    rna_charges = np.array(rna_charges_list, dtype=np.float64)

    # ── Step 3: ligand coords + charges (passed in) ───────────────────────────
    lig_coords  = np.array(lig_coords_flat,  dtype=np.float64).reshape(-1, 3)
    lig_charges = np.array(lig_charges_flat, dtype=np.float64)

    if len(lig_coords) < 1:
        return pdb, nan_feat

    # ── Step 4: interface filter — RNA atoms ≤8Å from any ligand atom ────────
    D = cdist(rna_coords, lig_coords)
    iface_mask = D.min(axis=1) <= 8.0
    if iface_mask.sum() < 3:
        return pdb, nan_feat

    iface_rna_coords  = rna_coords[iface_mask]
    iface_rna_charges = rna_charges[iface_mask]

    # Combined point cloud: interface RNA + ligand
    all_coords  = np.vstack([iface_rna_coords, lig_coords])
    all_charges = np.concatenate([iface_rna_charges, lig_charges])
    N = len(all_coords)

    # ── Step 5: Coulomb potential φ_i = Σ_{j≠i} q_j / r_ij ─────────────────
    D_all = cdist(all_coords, all_coords)
    np.fill_diagonal(D_all, np.inf)
    phi = (all_charges[np.newaxis, :] / np.maximum(D_all, 0.5)).sum(axis=1)
    # Clip to avoid extreme values
    phi = np.clip(phi, -50.0, 50.0)

    # ── Step 6: Persistent homology on alpha complex filtered by φ ───────────
    try:
        ac = gudhi.AlphaComplex(points=all_coords)
        st = ac.create_simplex_tree(max_alpha_square=100.0)

        # Override filtration: each simplex gets max(φ) of its vertices
        new_filt = []
        for s, _ in st.get_filtration():
            fval = max(phi[v] for v in s)
            new_filt.append((s, fval))
        for s, fval in new_filt:
            st.assign_filtration(s, fval)
        st.make_filtration_non_decreasing()
        st.compute_persistence()

        # Betti curves over 50 thresholds from phi.min to phi.max
        thresh = np.linspace(phi.min(), phi.max(), N_BETTI_THRESH)
        betti_curves = np.zeros((3, N_BETTI_THRESH), dtype=np.float32)
        for d in range(min(3, st.dimension() + 1)):
            diag = st.persistence_intervals_in_dimension(d)
            if len(diag) == 0:
                continue
            births = diag[:, 0]
            deaths = np.where(np.isinf(diag[:, 1]), phi.max() + 1, diag[:, 1])
            for ti, t in enumerate(thresh):
                betti_curves[d, ti] = np.sum((births <= t) & (deaths > t))

        # Persistence diagram statistics per dimension
        pers_stats = []
        for d in range(3):
            diag = st.persistence_intervals_in_dimension(d)
            if len(diag) == 0:
                pers_stats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                continue
            pers = diag[:, 1] - diag[:, 0]
            pers = pers[~np.isinf(pers)]
            if len(pers) == 0:
                pers_stats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                continue
            pers_stats.extend([
                len(pers),          # n_bars
                float(pers.mean()), # mean persistence
                float(pers.max()),  # max persistence (most persistent feature)
                float(pers.std()),  # std
                float(np.sum(pers)) # total persistence
            ])

    except Exception:
        return pdb, nan_feat

    # ── Step 7: Scalar charge features (10) ───────────────────────────────────
    E_rna_lig = (
        (iface_rna_charges[:, None] * lig_charges[None, :]) /
        np.maximum(cdist(iface_rna_coords, lig_coords), 0.5)
    ).sum()  # Coulomb energy (kcal/mol units irrelevant — just correlation)

    # multi-scale charge FRI: Σ q_i*q_j * exp(-η*r²) for 5 η values
    D_iface = cdist(iface_rna_coords, lig_coords)
    charge_outer = iface_rna_charges[:, None] * lig_charges[None, :]
    eta_vals = [0.1, 0.5, 1.0, 2.0, 5.0]
    charge_fri = [
        float((charge_outer * np.exp(-eta * D_iface**2)).sum())
        for eta in eta_vals
    ]

    scalar_feats = np.array([
        E_rna_lig,
        float(iface_rna_charges.sum()),   # net charge of interface RNA
        float(iface_rna_charges.std()),   # std of interface RNA charges
        float(lig_charges.sum()),          # net charge of ligand
        float(lig_charges.std()),          # std of ligand charges
    ] + charge_fri, dtype=np.float32)     # 5 scalar + 5 charge-FRI = 10

    # ── Assemble all features: 150 + 15 + 10 = 175 ───────────────────────────
    feat = np.concatenate([
        betti_curves.ravel(),   # 3 × 50 = 150
        np.array(pers_stats, dtype=np.float32),  # 3 × 5  = 15
        scalar_feats,                             #          10
    ])
    return pdb, feat.astype(np.float32)


# ── Need subprocess in worker — add import ────────────────────────────────────
import subprocess  # already imported at top, but worker needs it in its scope


# ── Load data ──────────────────────────────────────────────────────────────────
log.info("\nLoading pocket data ...")
with gzip.open(PKL_FILE, "rb") as f:
    records = pickle.load(f)
rec_map = {r["pdb"]: r for r in records}

log.info("Loading step11 features ...")
d11 = np.load(S11_NPZ)
X11 = d11["X"].astype(np.float64)
y   = d11["y"].astype(np.float32)
ids = d11["ids"]
subtypes_raw = d11["subtypes"]
subtypes = np.array([make_subtype(p, s) for p, s in zip(ids, subtypes_raw)])
n = len(y)
log.info(f"  X11: {X11.shape}")

# ── Build ligand charge args from RDKit ────────────────────────────────────────
log.info("\nComputing ligand Gasteiger charges ...")
from rdkit import Chem
from rdkit.Chem import AllChem

def get_lig_charge_args(pdb, rec_data):
    """Return (lig_centroid, lig_coords_flat, lig_charges_flat) or None."""
    lig_sdf = NA_L / pdb / f"{pdb}_ligand.sdf"
    lig_mol2 = NA_L / pdb / f"{pdb}_ligand.mol2"

    mol = None
    if lig_sdf.exists():
        mol = Chem.MolFromMolFile(str(lig_sdf), removeHs=False, sanitize=False)
    if mol is None and lig_mol2.exists():
        mol = Chem.MolFromMol2File(str(lig_mol2), removeHs=False, sanitize=False)
    if mol is None:
        return None

    try:
        Chem.SanitizeMol(mol)
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        return None

    conf = mol.GetConformer()
    coords, charges = [], []
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            continue
        pos = conf.GetAtomPosition(i)
        q   = atom.GetDoubleProp("_GasteigerCharge")
        if np.isfinite(q):
            coords.append([pos.x, pos.y, pos.z])
            charges.append(q)

    if len(coords) < 1:
        return None

    coords  = np.array(coords)
    charges = np.array(charges)
    centroid = coords.mean(axis=0)
    return centroid.tolist(), coords.ravel().tolist(), charges.tolist()

job_args = []
for pdb in ids:
    rec_data = rec_map.get(pdb)
    if rec_data is None:
        job_args.append(None)
        continue
    res = get_lig_charge_args(pdb, rec_data)
    if res is None:
        job_args.append(None)
    else:
        job_args.append((pdb,) + res)

valid_jobs = [(i, a) for i, a in enumerate(job_args) if a is not None]
log.info(f"  Valid ligand charge jobs: {len(valid_jobs)}/{n}")

# ── Run parallel electrostatic feature computation ─────────────────────────────
log.info(f"\nComputing electrostatic topology features [{N_WORKERS} workers] ...")
elec_feats = np.zeros((n, N_ELEC_FEAT), dtype=np.float32)
pdb_to_idx = {pdb: i for i, pdb in enumerate(ids)}

t0 = time.time()
done = 0
with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
    futures = {pool.submit(_elec_worker, a): a[0] for _, a in valid_jobs}
    for fut in as_completed(futures):
        pdb, feat = fut.result()
        idx = pdb_to_idx[pdb]
        elec_feats[idx] = feat
        done += 1
        elapsed = time.time() - t0
        eta = (len(valid_jobs) - done) / max(done / elapsed, 1e-6)
        if done % 20 == 0 or done == 1 or done == len(valid_jobs):
            n_ok = (~np.isnan(feat)).sum()
            log.info(f"  [{done:3d}/{len(valid_jobs)}] {pdb}  feats_ok={n_ok}/175  ETA {eta:.0f}s")

log.info(f"\nDone in {time.time()-t0:.1f}s")
n_valid = np.sum(~np.all(elec_feats == 0, axis=1))
n_nan   = np.sum(np.any(np.isnan(elec_feats), axis=1))
log.info(f"  Valid rows: {n_valid}/{n},  NaN rows: {n_nan}")

# Impute NaN columns with column median
for col in range(N_ELEC_FEAT):
    col_vals = elec_feats[:, col]
    nan_mask = np.isnan(col_vals)
    if nan_mask.all():
        elec_feats[:, col] = 0.0
    elif nan_mask.any():
        elec_feats[nan_mask, col] = np.nanmedian(col_vals)

# ── Combine with step11 features ──────────────────────────────────────────────
X18 = np.hstack([X11, elec_feats.astype(np.float64)])
log.info(f"\nX18 shape: {X18.shape}  (step11 + elec topology)")

np.savez_compressed(OUT_NPZ,
    X=X18.astype(np.float32), y=y, ids=ids, subtypes=subtypes_raw)
log.info(f"Saved: {OUT_NPZ}")

# Quick correlation of each feature block with y
r_betti  = pearsonr(elec_feats[:, :150].mean(axis=1), y)[0]
r_scalar = pearsonr(elec_feats[:, 165], y)[0]  # E_coulomb
log.info(f"\nElec feature sanity: Betti-mean r={r_betti:.3f}, E_coulomb r={r_scalar:.3f}")

# ── Hybrid model evaluation ────────────────────────────────────────────────────
def make_pipe(alpha):
    return Pipeline([
        ("vt",  VarianceThreshold(threshold=1e-4)),
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
        ("reg", Ridge(alpha=alpha)),
    ])

def loo_ridge_best(X_sub, y_sub):
    ns = len(y_sub)
    if ns < 3:
        return np.full(ns, y_sub.mean()), -99.0
    best_r, best_p = -99.0, np.full(ns, y_sub.mean())
    for alpha in ALPHA_GRID:
        preds = np.zeros(ns)
        ok = True
        for i in range(ns):
            tr = [j for j in range(ns) if j != i]
            try:
                pipe = make_pipe(alpha)
                pipe.fit(X_sub[tr], y_sub[tr])
                preds[i] = np.clip(
                    pipe.predict(X_sub[[i]])[0],
                    y_sub[tr].min() - 3, y_sub[tr].max() + 3
                )
            except Exception:
                ok = False; break
        if not ok:
            continue
        r = pearsonr(y_sub, preds)[0] if np.std(preds) > 1e-8 else -99.0
        if r > best_r:
            best_r, best_p = r, preds.copy()
    return best_p, best_r

def tanimoto_kernel(X, Y=None):
    if Y is None: Y = X
    XY  = X @ Y.T
    XX  = X.sum(1, keepdims=True)
    YY  = Y.sum(1, keepdims=True)
    denom = np.where(XX + YY.T - XY < 1e-10, 1e-10, XX + YY.T - XY)
    return XY / denom

log.info("\n" + "=" * 70)
log.info("HYBRID MODEL EVALUATION (step18 features)")
log.info("=" * 70)

# Per-subtype Ridge on X18
log.info("\n--- Per-subtype Ridge LOO (X18) ---")
s18_ridge_preds = np.full(n, np.nan)
s18_ridge_rs    = {}
for st in np.unique(subtypes):
    mask = subtypes == st
    ns_idx = np.where(mask)[0]
    if len(ns_idx) < 3:
        s18_ridge_preds[mask] = y[mask].mean()
        s18_ridge_rs[st] = np.nan
        continue
    preds, r = loo_ridge_best(X18[mask], y[mask])
    s18_ridge_preds[mask] = preds
    s18_ridge_rs[st] = r
    log.info(f"  {st:22s}: n={mask.sum():3d}  r={r:.3f}")

# Global MKL for riboswitch (step11 slice — unchanged kernel)
log.info("\n--- Global MKL LOO (riboswitch, step11 kernel) ---")
X_topo = X11[:, np.r_[0:36000, 38963:49763]]
X_lig  = X11[:, 36000:38048]
X_maccs= X11[:, 38796:38963]
X_rna  = X11[:, 38064:38704]

X_topo_n = StandardScaler().fit_transform(X_topo)
X_rna_n  = StandardScaler().fit_transform(X_rna)

K_topo = rbf_kernel(X_topo_n, gamma=1e-6)
K_lig  = 0.7 * tanimoto_kernel(X_lig) + 0.3 * tanimoto_kernel(X_maccs)
K_rna  = rbf_kernel(X_rna_n, gamma=5e-3)
K_full = 0.7 * K_topo + 0.1 * K_lig + 0.2 * K_rna

mkl_preds = np.zeros(n)
for i in range(n):
    tr = [j for j in range(n) if j != i]
    m = KernelRidge(alpha=0.01, kernel="precomputed")
    m.fit(K_full[np.ix_(tr, tr)], y[tr])
    p = float(m.predict(K_full[i, tr].reshape(1, -1))[0])
    mkl_preds[i] = np.clip(p, y[tr].min() - 3, y[tr].max() + 3)

rs_mask = subtypes == "riboswitch"
r_mkl, _ = pearsonr(mkl_preds[rs_mask], y[rs_mask])
log.info(f"  Riboswitch MKL r = {r_mkl:.3f}")

# Hybrid
log.info("\n--- HYBRID (step18 Ridge + MKL riboswitch) ---")
hybrid_preds = s18_ridge_preds.copy()
hybrid_preds[rs_mask] = mkl_preds[rs_mask]

r_hybrid, _ = pearsonr(hybrid_preds, y)
log.info(f"  Combined r = {r_hybrid:.4f}")

log.info("\nPer-subtype (Hybrid step18+MKL):")
for st in ["aptamer","duplex_groove","ribosomal_asite","riboswitch","other_misc","g_quadruplex","viral_tar"]:
    mask = subtypes == st
    if mask.sum() < 2: continue
    r, _ = pearsonr(hybrid_preds[mask], y[mask])
    log.info(f"  {st:22s}: n={mask.sum():3d}  r={r:.3f}")

log.info(f"\n{'='*60}")
log.info(f"FINAL RESULTS")
log.info(f"  step18 Hybrid r = {r_hybrid:.4f}")
log.info(f"  Previous best   = 0.6954")
log.info(f"  Delta           = {r_hybrid - 0.6954:+.4f}")
log.info(f"  Gap to DeepRSMA = {0.784 - r_hybrid:.4f}")
benchmarks = [("AffiGrapher",0.498),("RLaffinity",0.559),("RLASIF",0.666),
              ("DeepRSMA",0.784),("RSAPred",0.830)]
for name, rb in benchmarks:
    sym = "✓" if r_hybrid > rb else "✗"
    log.info(f"  {sym} {name}: {rb:.3f}")
log.info(f"{'='*60}")

# Save results
df = pd.DataFrame({"pdb":ids,"subtype":subtypes,"y_true":y,"y_pred":hybrid_preds})
df.to_csv(RES_DIR / "step18_results.csv", index=False)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = {"aptamer":"#2166AC","riboswitch":"#1A9641","ribosomal_asite":"#D73027",
          "duplex_groove":"#7B2D8B","g_quadruplex":"#FF7F00",
          "viral_tar":"#A65628","other_misc":"#999999"}

ax = axes[0]
for st in np.unique(subtypes):
    mask = subtypes == st
    r_st, _ = pearsonr(hybrid_preds[mask], y[mask]) if mask.sum() > 1 else (0, 0)
    ax.scatter(y[mask], hybrid_preds[mask], c=colors.get(st,"#888"),
               label=f"{st} r={r_st:.3f}", alpha=0.75, s=45, edgecolors="none")
lo, hi = y.min()-0.5, y.max()+0.5
ax.plot([lo,hi],[lo,hi],"k--",lw=1,alpha=0.5)
ax.set_xlabel("Experimental pKd"); ax.set_ylabel("Predicted pKd")
ax.set_title(f"Step 18: Elec Topology (r={r_hybrid:.4f})", fontweight="bold")
ax.legend(fontsize=7); ax.grid(alpha=0.3, linestyle="--")

ax = axes[1]
steps = {"S11+MKL":0.695,"S15":0.695,"S16":0.695,"S18":r_hybrid}
cols  = ["#AAAAAA","#888888","#4393C3","#D63027"]
bars = ax.bar(list(steps.keys()), list(steps.values()), color=cols, edgecolor="white", alpha=0.9)
for bar, val in zip(bars, steps.values()):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.005, f"{val:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
for name, rb in benchmarks:
    ax.axhline(rb, linestyle="--", lw=0.9, alpha=0.6, label=f"{name} {rb:.3f}")
ax.set_ylim(0.5, 0.85); ax.set_ylabel("Combined Pearson r")
ax.set_title("Progress", fontweight="bold")
ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3, linestyle="--")

plt.tight_layout()
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / "step18_results.png", dpi=150, bbox_inches="tight")
plt.close()

log.info(f"\nFigure → {FIG_DIR/'step18_results.png'}")
log.info("=" * 70)
log.info("STEP 18 COMPLETE")
log.info("=" * 70)
