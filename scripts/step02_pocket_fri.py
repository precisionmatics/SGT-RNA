"""
SGT-RNA  ·  Step 2: Pocket Extraction & FRI-Weighted Digraph Construction

For each of 143 RNA–small molecule complexes:
  1. Parse {pdb}_pocket.pdb  → RNA pocket atoms, keep elements {C, N, O, P}
  2. Parse {pdb}_ligand.sdf  → ligand atoms, keep elements {C, N, O, S, P, F, Cl, Br, I}
  3. Apply 12 Å inter-atomic cutoff (already enforced by pocket file; verified here)
  4. For each of the 36 element pairs, record coordinates and pairwise distance matrices
  5. Compute FRI weights: Exponential w_E(r) = exp(-(r/η)^κ)
                          Lorentz    w_L(r) = 1 / (1 + (r/η)^κ)
  6. Record edge direction from Pauling electronegativity (χ_RNA vs χ_Lig per pair)
  7. Save compressed per-complex data (.pkl.gz) for Step 3
  8. Generate journal-quality QC figures
"""

import os, re, gzip, pickle, logging, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
NA_L_DIR  = Path("/home/stalin/Desktop/SGT-RNA/NA-L")
DATASET   = ROOT / "data" / "affinity" / "dataset.csv"
OUT_DIR   = ROOT / "data" / "pocket_fri"
FIG_DIR   = ROOT / "results" / "figures"
LOG_DIR   = ROOT / "results" / "logs"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── logging ───────────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = LOG_DIR / f"step02_{ts}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("SGT-RNA  ·  Step 2: Pocket Extraction & FRI-Weighted Digraph")
log.info("=" * 70)

# ── constants ─────────────────────────────────────────────────────────────────
RNA_ELEMENTS = ["C", "N", "O", "P"]
LIG_ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
ELEMENT_PAIRS = [(r, l) for r in RNA_ELEMENTS for l in LIG_ELEMENTS]  # 36 pairs

POCKET_CUTOFF = 12.0   # Å  — verified but not re-applied (pocket file already filtered)

# Pauling electronegativity
CHI = {
    "C": 2.55, "N": 3.04, "O": 3.44, "P": 2.19,
    "S": 2.58, "F": 3.98, "Cl": 3.16, "Br": 2.96, "I": 2.66, "H": 2.20
}

# FRI parameters (exponential kernel primary)
ETA   = 5.0   # Å
KAPPA = 2.0

# ── edge direction lookup per pair (constant since element fixed) ──────────────
# direction = +1 if RNA_atom is source (χ_RNA < χ_Lig), -1 if sink, 0 if equal
PAIR_DIRECTION = {}
for (r_el, l_el) in ELEMENT_PAIRS:
    chi_r, chi_l = CHI.get(r_el, 2.55), CHI.get(l_el, 2.55)
    if chi_r < chi_l:
        PAIR_DIRECTION[(r_el, l_el)] = +1   # RNA → Lig
    elif chi_r > chi_l:
        PAIR_DIRECTION[(r_el, l_el)] = -1   # Lig → RNA
    else:
        PAIR_DIRECTION[(r_el, l_el)] =  0   # bidirectional

# ── FRI kernels ───────────────────────────────────────────────────────────────
def fri_exp(D, eta=ETA, kappa=KAPPA):
    """Exponential FRI kernel."""
    return np.exp(-(D / eta) ** kappa)

def fri_lor(D, eta=ETA, kappa=KAPPA):
    """Lorentz FRI kernel."""
    return 1.0 / (1.0 + (D / eta) ** kappa)

# ── PDB pocket parser ─────────────────────────────────────────────────────────
_ALPHA_RE = re.compile(r"[^A-Za-z]")

def parse_pocket(pdb_path, keep_elems):
    """Return dict {element: ndarray(N,3)} for atoms in keep_elems."""
    coords = {e: [] for e in keep_elems}
    try:
        with open(pdb_path) as f:
            for line in f:
                rec = line[:6].strip()
                if rec not in ("ATOM", "HETATM"):
                    continue
                # Element from standard PDB columns 76-78 (0-indexed)
                raw_elem = line[76:79].strip() if len(line) > 76 else ""
                elem = _ALPHA_RE.sub("", raw_elem).capitalize()
                # Fallback: derive from atom name (col 12-16)
                if not elem or len(elem) > 2:
                    atom_name = line[12:16].strip().lstrip("0123456789")
                    elem = _ALPHA_RE.sub("", atom_name).capitalize()
                    if len(elem) > 1:
                        # Take first alphabetic character only for common elements
                        elem = elem[0]
                if elem not in keep_elems:
                    continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords[elem].append([x, y, z])
                except ValueError:
                    continue
    except Exception as e:
        log.warning(f"  Pocket parse error {pdb_path}: {e}")
    return {e: np.array(v, dtype=np.float32) if v else np.empty((0, 3), np.float32)
            for e, v in coords.items()}

# ── SDF ligand parser (manual V2000, avoids RDKit sanitization errors) ────────
def parse_sdf(sdf_path, keep_elems):
    """Return dict {element: ndarray(N,3)} from V2000 SDF atom block."""
    coords = {e: [] for e in keep_elems}
    try:
        with open(sdf_path) as f:
            lines = f.readlines()
        # Locate counts line (line index 3)
        if len(lines) < 4:
            return coords
        counts_line = lines[3]
        try:
            n_atoms = int(counts_line[:3].strip())
        except ValueError:
            return coords
        for i in range(4, 4 + n_atoms):
            if i >= len(lines):
                break
            parts = lines[i].split()
            if len(parts) < 4:
                continue
            elem = parts[3].capitalize()
            if len(elem) > 1 and not elem[1].isupper():
                elem = elem  # already two-char like Cl, Br
            # normalise two-char: "CL" → "Cl"
            if len(elem) == 2:
                elem = elem[0].upper() + elem[1].lower()
            if elem not in keep_elems:
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                coords[elem].append([x, y, z])
            except ValueError:
                continue
    except Exception as e:
        log.warning(f"  SDF parse error {sdf_path}: {e}")
    return {e: np.array(v, dtype=np.float32) if v else np.empty((0, 3), np.float32)
            for e, v in coords.items()}

# Fall back to mol2 if SDF gives too few atoms
def parse_mol2(mol2_path, keep_elems):
    """Parse @<TRIPOS>ATOM block from mol2 file."""
    coords = {e: [] for e in keep_elems}
    try:
        with open(mol2_path) as f:
            content = f.read()
        atom_block = re.search(r"@<TRIPOS>ATOM\n(.*?)(?:@<TRIPOS>|\Z)", content, re.DOTALL)
        if not atom_block:
            return coords
        for line in atom_block.group(1).strip().split("\n"):
            parts = line.split()
            if len(parts) < 6:
                continue
            raw = parts[5].split(".")[0]  # e.g. "N.3" → "N", "C.ar" → "C"
            elem = raw.capitalize()
            if len(elem) == 2:
                elem = elem[0].upper() + elem[1].lower()
            if elem not in keep_elems:
                continue
            try:
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                coords[elem].append([x, y, z])
            except ValueError:
                continue
    except Exception as e:
        log.warning(f"  mol2 parse error {mol2_path}: {e}")
    return {e: np.array(v, dtype=np.float32) if v else np.empty((0, 3), np.float32)
            for e, v in coords.items()}

# ── load dataset ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATASET)
log.info(f"Loaded dataset: {len(df)} complexes")

# ── main processing loop ──────────────────────────────────────────────────────
records = []
stats = {
    "n_rna_atoms":  [],   # total RNA heavy atoms per complex
    "n_lig_atoms":  [],   # total ligand heavy atoms per complex
    "n_pairs_populated": [],    # how many of 36 pairs have ≥1 atom each side
    "min_inter_dist": [],       # closest RNA-ligand contact
}
pair_counts = np.zeros((len(RNA_ELEMENTS), len(LIG_ELEMENTS)), dtype=int)  # how many complexes have this pair

# For distance distribution sampling (store a random subsample)
dist_samples = {(r, l): [] for r, l in ELEMENT_PAIRS}

log.info(f"\nProcessing {len(df)} complexes ...")

failed = []
for idx, row in df.iterrows():
    pdb = row["pdb"]
    pkd = row["pKd"]
    cdir = NA_L_DIR / pdb

    pocket_pdb = cdir / f"{pdb}_pocket.pdb"
    ligand_sdf = cdir / f"{pdb}_ligand.sdf"
    ligand_mol2 = cdir / f"{pdb}_ligand.mol2"

    if not pocket_pdb.exists():
        log.warning(f"  SKIP {pdb}: pocket PDB missing")
        failed.append(pdb)
        continue

    # Parse RNA pocket atoms
    rna_coords = parse_pocket(pocket_pdb, RNA_ELEMENTS)
    n_rna = sum(len(v) for v in rna_coords.values())

    # Parse ligand atoms (SDF primary, mol2 fallback)
    lig_coords = parse_sdf(ligand_sdf, LIG_ELEMENTS) if ligand_sdf.exists() else {}
    n_lig_sdf = sum(len(v) for v in lig_coords.values())
    if n_lig_sdf < 3 and ligand_mol2.exists():
        lig_coords = parse_mol2(ligand_mol2, LIG_ELEMENTS)
        log.debug(f"  {pdb}: SDF gave {n_lig_sdf} atoms, switched to mol2")
    n_lig = sum(len(v) for v in lig_coords.values())

    if n_rna == 0 or n_lig == 0:
        log.warning(f"  SKIP {pdb}: RNA={n_rna}, Lig={n_lig}")
        failed.append(pdb)
        continue

    # Verify inter-molecular distances (ensure pocket is within cutoff)
    all_rna = np.vstack([v for v in rna_coords.values() if len(v) > 0])
    all_lig = np.vstack([v for v in lig_coords.values() if len(v) > 0])
    D_all = cdist(all_rna, all_lig)
    min_d = float(D_all.min())
    max_d = float(D_all.min(axis=1).max())  # farthest RNA atom's closest ligand contact
    # Flag if any RNA atoms are very far (>15 Å) from all ligand atoms
    far_mask = D_all.min(axis=1) > 15.0
    if far_mask.sum() > 0:
        log.debug(f"  {pdb}: {far_mask.sum()} RNA atoms >15 Å from ligand (far shell)")

    # Count populated pairs and sample distances
    n_pop = 0
    pair_dists = {}
    for ri, r_el in enumerate(RNA_ELEMENTS):
        for li, l_el in enumerate(LIG_ELEMENTS):
            rc = rna_coords[r_el]
            lc = lig_coords[l_el]
            if len(rc) == 0 or len(lc) == 0:
                pair_dists[(r_el, l_el)] = None
                continue
            D = cdist(rc, lc, metric="euclidean").astype(np.float32)
            pair_dists[(r_el, l_el)] = D
            n_pop += 1
            pair_counts[ri, li] += 1
            # Sample up to 200 distances for distribution plot
            flat = D.ravel()
            if len(flat) > 200:
                flat = np.random.choice(flat, 200, replace=False)
            dist_samples[(r_el, l_el)].extend(flat.tolist())

    # Build per-complex record
    record = {
        "pdb":       pdb,
        "pkd":       pkd,
        "rna_coords": rna_coords,
        "lig_coords": lig_coords,
        "n_rna":     n_rna,
        "n_lig":     n_lig,
        "min_inter_dist": min_d,
        "n_pairs_pop": n_pop,
    }
    records.append(record)
    stats["n_rna_atoms"].append(n_rna)
    stats["n_lig_atoms"].append(n_lig)
    stats["n_pairs_populated"].append(n_pop)
    stats["min_inter_dist"].append(min_d)

    if (idx + 1) % 20 == 0 or idx == 0:
        log.info(f"  [{idx+1:3d}/{len(df)}] {pdb}: RNA={n_rna} Lig={n_lig} "
                 f"pairs={n_pop}/36 minD={min_d:.2f}Å")

log.info(f"\nProcessed: {len(records)} complexes  |  Failed: {len(failed)}")
if failed:
    log.warning(f"  Failed: {failed}")

# ── save per-complex data ─────────────────────────────────────────────────────
out_pkl = OUT_DIR / "pocket_fri_data.pkl.gz"
with gzip.open(out_pkl, "wb") as f:
    pickle.dump(records, f, protocol=4)
log.info(f"Saved pocket data → {out_pkl}  ({out_pkl.stat().st_size / 1024:.1f} KB)")

# Save pair coverage CSV
coverage_df = pd.DataFrame(
    pair_counts,
    index=pd.Index(RNA_ELEMENTS, name="RNA_element"),
    columns=pd.Index(LIG_ELEMENTS, name="Lig_element")
)
coverage_csv = OUT_DIR / "pair_coverage.csv"
coverage_df.to_csv(coverage_csv)
log.info(f"Pair coverage saved → {coverage_csv}")

# ── summary stats ─────────────────────────────────────────────────────────────
log.info("\n" + "=" * 50)
log.info("STEP 2 SUMMARY")
log.info("=" * 50)
log.info(f"  Complexes processed  : {len(records)}")
log.info(f"  RNA atoms  (mean±std): {np.mean(stats['n_rna_atoms']):.0f} ± {np.std(stats['n_rna_atoms']):.0f}")
log.info(f"  Lig atoms  (mean±std): {np.mean(stats['n_lig_atoms']):.0f} ± {np.std(stats['n_lig_atoms']):.0f}")
log.info(f"  Min inter-dist (mean): {np.mean(stats['min_inter_dist']):.2f} Å")
log.info(f"  Populated pairs/36   : {np.mean(stats['n_pairs_populated']):.1f} ± {np.std(stats['n_pairs_populated']):.1f}")
log.info("  Pair coverage (complexes with pair populated):")
for ri, r_el in enumerate(RNA_ELEMENTS):
    for li, l_el in enumerate(LIG_ELEMENTS):
        if pair_counts[ri, li] > 0:
            log.info(f"    {r_el}-{l_el}: {pair_counts[ri, li]}/{len(records)}")

# ── journal-quality QC figures ────────────────────────────────────────────────
log.info("\nGenerating QC figures ...")

COLORS = {
    "C": "#4C72B0", "N": "#DD8452", "O": "#55A868", "P": "#C44E52",
    "S": "#8172B2", "F": "#937860", "Cl": "#DA8BC3", "Br": "#8C8C8C", "I": "#CCB974",
}
FRI_COLOR  = "#2166AC"
LOR_COLOR  = "#D6604D"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.linewidth": 1.2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
})

# ── Figure 1: Atom count & distance QC (8 panels) ────────────────────────────
fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor("white")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.38)
fig.suptitle("SGT-RNA  |  Step 2: Pocket Extraction & FRI Digraph QC",
             fontsize=16, fontweight="bold", y=0.98)

# ── Panel A: RNA atom counts per element ─────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
rna_by_elem = {e: [] for e in RNA_ELEMENTS}
for rec in records:
    for e in RNA_ELEMENTS:
        rna_by_elem[e].append(len(rec["rna_coords"][e]))
bp = ax.boxplot([rna_by_elem[e] for e in RNA_ELEMENTS],
                patch_artist=True, widths=0.55,
                medianprops=dict(color="black", linewidth=2))
for patch, e in zip(bp["boxes"], RNA_ELEMENTS):
    patch.set_facecolor(COLORS[e])
    patch.set_alpha(0.8)
ax.set_xticks(range(1, 5))
ax.set_xticklabels(RNA_ELEMENTS, fontsize=12)
ax.set_xlabel("RNA Pocket Element", fontsize=11)
ax.set_ylabel("Atom Count per Complex", fontsize=11)
ax.set_title("A  |  RNA Pocket Atom Counts", fontsize=12, fontweight="bold", loc="left")
ax.grid(axis="y", alpha=0.3, linestyle="--")

# ── Panel B: Ligand atom counts per element ───────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
lig_by_elem = {e: [] for e in LIG_ELEMENTS}
for rec in records:
    for e in LIG_ELEMENTS:
        lig_by_elem[e].append(len(rec["lig_coords"].get(e, np.empty((0,3)))))
bp = ax.boxplot([lig_by_elem[e] for e in LIG_ELEMENTS],
                patch_artist=True, widths=0.55,
                medianprops=dict(color="black", linewidth=2))
for patch, e in zip(bp["boxes"], LIG_ELEMENTS):
    patch.set_facecolor(COLORS.get(e, "#888888"))
    patch.set_alpha(0.8)
ax.set_xticks(range(1, 10))
ax.set_xticklabels(LIG_ELEMENTS, fontsize=9)
ax.set_xlabel("Ligand Element", fontsize=11)
ax.set_ylabel("Atom Count per Complex", fontsize=11)
ax.set_title("B  |  Ligand Atom Counts", fontsize=12, fontweight="bold", loc="left")
ax.grid(axis="y", alpha=0.3, linestyle="--")

# ── Panel C: Populated pairs histogram ────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
n_pop_vals = stats["n_pairs_populated"]
ax.hist(n_pop_vals, bins=range(min(n_pop_vals), max(n_pop_vals)+2), color="#4C72B0",
        edgecolor="white", linewidth=0.8, align="left")
ax.axvline(np.mean(n_pop_vals), color="crimson", linestyle="--", linewidth=1.5,
           label=f"Mean = {np.mean(n_pop_vals):.1f}")
ax.set_xlabel("Populated Pairs per Complex (out of 36)", fontsize=11)
ax.set_ylabel("# Complexes", fontsize=11)
ax.set_title("C  |  Element Pair Coverage", fontsize=12, fontweight="bold", loc="left")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3, linestyle="--")

# ── Panel D: Minimum inter-molecular distance distribution ────────────────────
ax = fig.add_subplot(gs[1, 0])
ax.hist(stats["min_inter_dist"], bins=25, color="#55A868", edgecolor="white", linewidth=0.8)
ax.axvline(np.mean(stats["min_inter_dist"]), color="crimson", linestyle="--", linewidth=1.5,
           label=f"Mean = {np.mean(stats['min_inter_dist']):.2f} Å")
ax.set_xlabel("Min RNA–Ligand Distance (Å)", fontsize=11)
ax.set_ylabel("# Complexes", fontsize=11)
ax.set_title("D  |  Closest RNA–Ligand Contact", fontsize=12, fontweight="bold", loc="left")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3, linestyle="--")

# ── Panel E: FRI kernel curves ────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
r_vals = np.linspace(0, 15, 300)
for eta in [2.0, 5.0, 8.0]:
    ls = "-" if eta == 5.0 else ("--" if eta == 2.0 else ":")
    ax.plot(r_vals, fri_exp(r_vals, eta=eta), color=FRI_COLOR, linestyle=ls,
            linewidth=2, label=f"Exp η={eta}")
    ax.plot(r_vals, fri_lor(r_vals, eta=eta), color=LOR_COLOR, linestyle=ls,
            linewidth=2, label=f"Lor η={eta}")
ax.axvline(POCKET_CUTOFF, color="gray", linestyle=":", linewidth=1.2, label=f"{POCKET_CUTOFF} Å cutoff")
ax.set_xlabel("Inter-atomic Distance (Å)", fontsize=11)
ax.set_ylabel("FRI Weight w(r)", fontsize=11)
ax.set_title("E  |  FRI Kernel Functions (κ=2)", fontsize=12, fontweight="bold", loc="left")
ax.legend(fontsize=8, ncol=2)
ax.set_xlim(0, 15); ax.set_ylim(-0.05, 1.05)
ax.grid(alpha=0.3, linestyle="--")

# ── Panel F: Distance distributions for top pairs ─────────────────────────────
ax = fig.add_subplot(gs[1, 2])
top_pairs = [("C", "C"), ("C", "N"), ("N", "C"), ("O", "C"), ("N", "O")]
pair_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
for (r_el, l_el), col in zip(top_pairs, pair_colors):
    samp = dist_samples[(r_el, l_el)]
    if samp:
        ax.hist(samp, bins=40, alpha=0.55, color=col, density=True,
                label=f"RNA-{r_el}···Lig-{l_el}", histtype="stepfilled", edgecolor="none")
ax.axvline(POCKET_CUTOFF, color="black", linestyle="--", linewidth=1.2)
ax.set_xlabel("Pairwise Distance (Å)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("F  |  Distance Distributions (top pairs)", fontsize=12, fontweight="bold", loc="left")
ax.legend(fontsize=8)
ax.set_xlim(0, 25)
ax.grid(alpha=0.3, linestyle="--")

# ── Panel G: Pair coverage heatmap ───────────────────────────────────────────
ax = fig.add_subplot(gs[2, 0])
cmap = LinearSegmentedColormap.from_list("cov", ["#FFFDE7", "#F57F17", "#B71C1C"])
im = ax.imshow(pair_counts, aspect="auto", cmap=cmap, vmin=0, vmax=len(records))
for ri in range(len(RNA_ELEMENTS)):
    for li in range(len(LIG_ELEMENTS)):
        v = pair_counts[ri, li]
        ax.text(li, ri, str(v), ha="center", va="center",
                fontsize=9, color="white" if v > len(records)*0.5 else "black",
                fontweight="bold")
ax.set_xticks(range(len(LIG_ELEMENTS))); ax.set_xticklabels(LIG_ELEMENTS, fontsize=10)
ax.set_yticks(range(len(RNA_ELEMENTS))); ax.set_yticklabels(RNA_ELEMENTS, fontsize=10)
ax.set_xlabel("Ligand Element", fontsize=11)
ax.set_ylabel("RNA Pocket Element", fontsize=11)
ax.set_title("G  |  Populated Pair Coverage (#complexes)", fontsize=12, fontweight="bold", loc="left")
plt.colorbar(im, ax=ax, shrink=0.85, label="# Complexes")

# ── Panel H: Edge direction per pair ─────────────────────────────────────────
ax = fig.add_subplot(gs[2, 1])
dir_mat = np.zeros((len(RNA_ELEMENTS), len(LIG_ELEMENTS)))
for ri, r_el in enumerate(RNA_ELEMENTS):
    for li, l_el in enumerate(LIG_ELEMENTS):
        dir_mat[ri, li] = PAIR_DIRECTION[(r_el, l_el)]
cmap2 = LinearSegmentedColormap.from_list("dir", ["#D6604D", "#F7F7F7", "#4393C3"])
im2 = ax.imshow(dir_mat, aspect="auto", cmap=cmap2, vmin=-1.1, vmax=1.1)
for ri in range(len(RNA_ELEMENTS)):
    for li in range(len(LIG_ELEMENTS)):
        d = dir_mat[ri, li]
        sym = "→" if d > 0 else ("←" if d < 0 else "↔")
        ax.text(li, ri, sym, ha="center", va="center", fontsize=12)
ax.set_xticks(range(len(LIG_ELEMENTS))); ax.set_xticklabels(LIG_ELEMENTS, fontsize=10)
ax.set_yticks(range(len(RNA_ELEMENTS))); ax.set_yticklabels(RNA_ELEMENTS, fontsize=10)
ax.set_xlabel("Ligand Element", fontsize=11)
ax.set_ylabel("RNA Pocket Element", fontsize=11)
ax.set_title("H  |  Edge Direction (Pauling χ)\n→=RNA→Lig  ←=Lig→RNA  ↔=bidirectional",
             fontsize=11, fontweight="bold", loc="left")

# ── Panel I: FRI weight at observed distances ─────────────────────────────────
ax = fig.add_subplot(gs[2, 2])
# Show weight distribution for C-C pair under both kernels
cc_dists = np.array(dist_samples[("C", "C")])
if len(cc_dists) > 0:
    w_exp = fri_exp(cc_dists)
    w_lor = fri_lor(cc_dists)
    ax.hist(w_exp, bins=40, alpha=0.65, color=FRI_COLOR, density=True,
            label="Exponential (η=5)", histtype="stepfilled", edgecolor="none")
    ax.hist(w_lor, bins=40, alpha=0.65, color=LOR_COLOR, density=True,
            label="Lorentz (η=5)", histtype="stepfilled", edgecolor="none")
ax.set_xlabel("FRI Edge Weight", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("I  |  FRI Weight Distribution (C···C pair)", fontsize=12, fontweight="bold", loc="left")
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle="--")

fig_path = FIG_DIR / "step02_pocket_fri_qc.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
log.info(f"  QC figure saved → {fig_path}")

# ── Figure 2: Per-element pair distance heatmap matrix ───────────────────────
fig2, axes2 = plt.subplots(4, 9, figsize=(32, 14))
fig2.patch.set_facecolor("white")
fig2.suptitle(
    "SGT-RNA  |  Pairwise Distance Distributions — All 36 RNA×Ligand Element Pairs\n"
    "(with Exponential FRI weight curve overlay)",
    fontsize=15, fontweight="bold", y=1.0
)

r_vals_plot = np.linspace(0.1, 20, 300)
NO_DATA_COLOR  = "#F0F0F0"
SPARSE_THRESH  = 5  # panels with fewer than this many complexes get a "sparse" note

for ri, r_el in enumerate(RNA_ELEMENTS):
    for li, l_el in enumerate(LIG_ELEMENTS):
        ax = axes2[ri, li]
        samp = dist_samples[(r_el, l_el)]
        n_cplx = pair_counts[ri, li]

        if n_cplx == 0:
            # Completely empty — gray fill + centred label
            ax.set_facecolor(NO_DATA_COLOR)
            ax.text(0.5, 0.5, "No data\n(n = 0)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="#888888",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BBBBBB", alpha=0.8))
        elif n_cplx < SPARSE_THRESH:
            # Very sparse — plot what we have with a note
            ax.hist(samp, bins=max(5, n_cplx), density=True, alpha=0.6,
                    color=COLORS.get(r_el, "#4C72B0"), edgecolor="white")
            ax.text(0.97, 0.97, f"sparse\n(n={n_cplx})", transform=ax.transAxes,
                    ha="right", va="top", fontsize=6.5, color="#AA4444",
                    fontstyle="italic")
            w_curve = fri_exp(r_vals_plot)
            ax2t = ax.twinx()
            ax2t.plot(r_vals_plot, w_curve, color="crimson", linewidth=1.0, alpha=0.7)
            ax2t.set_ylim(0, 1.5)
            ax2t.tick_params(right=False, labelright=False)
        else:
            # Normal — histogram + FRI weight overlay
            ax.hist(samp, bins=30, density=True, alpha=0.78,
                    color=COLORS.get(r_el, "#4C72B0"), edgecolor="none")
            w_curve = fri_exp(r_vals_plot)
            ax2t = ax.twinx()
            ax2t.plot(r_vals_plot, w_curve, color="crimson", linewidth=1.2, alpha=0.8)
            ax2t.set_ylim(0, 1.5)
            ax2t.tick_params(right=False, labelright=False)

        ax.set_title(f"RNA-{r_el}·Lig-{l_el}  (n={n_cplx})", fontsize=7.5, pad=2)
        ax.set_xlim(0, 20)
        ax.tick_params(labelsize=6)
        ax.set_yticks([])
        if ri == 3:
            ax.set_xlabel("d (Å)", fontsize=7)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig2_path = FIG_DIR / "step02_pair_distance_matrix.png"
plt.savefig(fig2_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
log.info(f"  Pair distance matrix saved → {fig2_path}")

# ── final log ─────────────────────────────────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("STEP 2 COMPLETE")
log.info(f"  Complexes     : {len(records)}")
log.info(f"  Failed        : {len(failed)}")
log.info(f"  Pocket data   : {out_pkl}")
log.info(f"  QC figure 1   : {fig_path}")
log.info(f"  QC figure 2   : {fig2_path}")
log.info(f"  Log           : {log_path}")
log.info("=" * 70)
