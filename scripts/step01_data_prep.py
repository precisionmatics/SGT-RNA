"""
RNA-PDFL Project — Step 1: Data Preparation & Exploratory Data Analysis
========================================================================
Authors : Stalin (PI) + Claude
Goal    : Build a clean, labeled dataset for RNA–small molecule binding
          affinity prediction using PDBbind NL2020 structures + affinities.

Inputs
------
  INDEX_general_NL.2020  — PDBbind affinity index (149 entries)
  NA-L/                  — Processed structural data (143 complexes)
                           Each folder: {pdb}_nucleic_acid.pdb
                                        {pdb}_pocket.pdb
                                        {pdb}_ligand.sdf
                                        {pdb}_ligand.mol2

Outputs (all under RNA_PDFL/)
------------------------------
  data/affinity/dataset.csv          — clean labeled dataset
  results/logs/step01_YYYYMMDD.log
  results/figures/step01_eda.png     — journal-quality EDA panel (300 dpi)
  results/figures/step01_elements.png — element composition heatmap (300 dpi)
"""

import os, re, math, logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# ─── 0. Paths ────────────────────────────────────────────────────────────────
ROOT      = Path("/home/stalin/Desktop/PDFL-RNA/RNA_PDFL")
NA_L_DIR  = Path("/home/stalin/Desktop/PDFL-RNA/NA-L")          # extracted tar
INDEX_FILE = Path("/run/media/stalin/PortableSSD/ML_Projects/"
                  "CAML_RNA/data/raw/NL/index/INDEX_general_NL.2020")

DATA_DIR   = ROOT / "data"
AFF_DIR    = DATA_DIR / "affinity"
FIG_DIR    = ROOT / "results" / "figures"
LOG_DIR    = ROOT / "results" / "logs"

for d in [AFF_DIR, FIG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── 1. Logging ──────────────────────────────────────────────────────────────
ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"step01_{ts}.log"
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)s | %(message)s",
    handlers= [logging.FileHandler(log_file), logging.StreamHandler()]
)
log = logging.getLogger()
log.info("=" * 70)
log.info("RNA-PDFL  ·  Step 1: Data Preparation & EDA")
log.info("=" * 70)

# ─── 2. Affinity parser ──────────────────────────────────────────────────────
UNIT = {"pm": 1e-12, "nm": 1e-9, "um": 1e-6, "mm": 1e-3, "m": 1.0}

def parse_affinity(raw: str):
    """Parse raw affinity string → (value_M, measure_type) or (None, None)."""
    raw = raw.strip()
    m = re.match(r"(Kd|Ki|IC50|Ka|EC50)[=<>~]+(.+)", raw, re.IGNORECASE)
    if not m:
        return None, None
    mtype, val_str = m.group(1).upper(), m.group(2).strip()
    val_str = re.split(r"[,\s\(]", val_str)[0]
    val_str = re.sub(r"[<>~]", "", val_str)
    nm = re.match(r"([\d.]+)\s*([a-zA-Z]+)", val_str)
    if not nm:
        return None, None
    num  = float(nm.group(1))
    unit = nm.group(2).lower()
    if unit not in UNIT:
        return None, None
    val_M = num * UNIT[unit]
    if mtype == "KA":
        val_M, mtype = 1.0 / val_M, "Kd"
    return val_M, mtype

def to_pkd(val_M):
    if val_M is None or val_M <= 0:
        return None
    return round(-math.log10(val_M), 4)

# ─── 3. Parse INDEX file ─────────────────────────────────────────────────────
log.info(f"Parsing index: {INDEX_FILE}")
index_records = []
with open(INDEX_FILE) as fh:
    for line in fh:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        pdb      = parts[0].lower()
        res      = parts[1]
        year     = parts[2]
        aff_raw  = parts[3]
        val_M, mtype = parse_affinity(aff_raw)
        pkd = to_pkd(val_M)
        index_records.append({
            "pdb"          : pdb,
            "resolution"   : res,
            "year"         : int(year) if year.isdigit() else None,
            "affinity_raw" : aff_raw,
            "affinity_M"   : val_M,
            "affinity_type": mtype,
            "pKd"          : pkd,
        })

df_index = pd.DataFrame(index_records)
log.info(f"  Total entries in index : {len(df_index)}")
log.info(f"  Parseable affinities   : {df_index['pKd'].notna().sum()}")

# ─── 4. Match with NA-L structural data ──────────────────────────────────────
log.info(f"\nScanning NA-L structural data: {NA_L_DIR}")

struct_records = []
for pdb_dir in sorted(NA_L_DIR.iterdir()):
    if not pdb_dir.is_dir():
        continue
    pdb_id = pdb_dir.name.lower()
    pocket_pdb  = pdb_dir / f"{pdb_id}_pocket.pdb"
    rna_pdb     = pdb_dir / f"{pdb_id}_nucleic_acid.pdb"
    lig_sdf     = pdb_dir / f"{pdb_id}_ligand.sdf"
    lig_mol2    = pdb_dir / f"{pdb_id}_ligand.mol2"

    if not (pocket_pdb.exists() and rna_pdb.exists() and lig_sdf.exists()):
        log.warning(f"  SKIP {pdb_id}: missing files")
        continue

    struct_records.append({
        "pdb"       : pdb_id,
        "pocket_pdb": str(pocket_pdb),
        "rna_pdb"   : str(rna_pdb),
        "lig_sdf"   : str(lig_sdf),
        "lig_mol2"  : str(lig_mol2),
    })

df_struct = pd.DataFrame(struct_records)
log.info(f"  Structural entries found: {len(df_struct)}")

# ─── 5. Merge on PDB ID ──────────────────────────────────────────────────────
df = pd.merge(df_index, df_struct, on="pdb", how="inner")
log.info(f"\nAfter inner join (index ∩ NA-L): {len(df)} complexes")

# Drop entries with unparseable affinity
before = len(df)
df = df[df["pKd"].notna()].reset_index(drop=True)
log.info(f"Dropped {before - len(df)} entries with unparseable affinity → {len(df)} remain")

# ─── 6. Parse structural features from PDB + SDF ────────────────────────────
log.info("\nParsing structural features from PDB / SDF files ...")

RNA_ELEMENTS  = {"C", "N", "O", "P", "S"}
PARS = PDBParser(QUIET=True)

rows = []
for _, r in df.iterrows():
    # --- RNA pocket atom types -----------------------------------------------
    rna_elem_counts = {e: 0 for e in ["C","N","O","P","S","other"]}
    n_rna_atoms = 0
    try:
        struct = PARS.get_structure(r["pdb"], r["pocket_pdb"])
        for atom in struct.get_atoms():
            elem = atom.element.strip().upper() if atom.element else "?"
            n_rna_atoms += 1
            if elem in RNA_ELEMENTS:
                rna_elem_counts[elem] += 1
            else:
                rna_elem_counts["other"] += 1
    except Exception as e:
        log.warning(f"  {r['pdb']} pocket parse error: {e}")

    # --- Ligand (SDF) features -----------------------------------------------
    lig_elem_counts = {e: 0 for e in ["C","N","O","S","P","F","Cl","Br","I","other"]}
    n_lig_atoms = 0
    mw = n_rings = n_hbd = n_hba = n_rot = tpsa = 0.0
    try:
        suppl = Chem.SDMolSupplier(r["lig_sdf"], removeHs=True)
        mol   = next((m for m in suppl if m is not None), None)
        if mol:
            for atom in mol.GetAtoms():
                sym = atom.GetSymbol()
                n_lig_atoms += 1
                if sym in lig_elem_counts:
                    lig_elem_counts[sym] += 1
                else:
                    lig_elem_counts["other"] += 1
            mw      = Descriptors.MolWt(mol)
            n_rings = rdMolDescriptors.CalcNumRings(mol)
            n_hbd   = rdMolDescriptors.CalcNumHBD(mol)
            n_hba   = rdMolDescriptors.CalcNumHBA(mol)
            n_rot   = rdMolDescriptors.CalcNumRotatableBonds(mol)
            tpsa    = Descriptors.TPSA(mol)
    except Exception as e:
        log.warning(f"  {r['pdb']} ligand parse error: {e}")

    row = {**r.to_dict(),
           "n_rna_atoms"  : n_rna_atoms,
           "n_lig_atoms"  : n_lig_atoms,
           "rna_C"        : rna_elem_counts["C"],
           "rna_N"        : rna_elem_counts["N"],
           "rna_O"        : rna_elem_counts["O"],
           "rna_P"        : rna_elem_counts["P"],
           "rna_S"        : rna_elem_counts["S"],
           "lig_C"        : lig_elem_counts["C"],
           "lig_N"        : lig_elem_counts["N"],
           "lig_O"        : lig_elem_counts["O"],
           "lig_S"        : lig_elem_counts["S"],
           "lig_F"        : lig_elem_counts["F"],
           "lig_Cl"       : lig_elem_counts["Cl"],
           "lig_Br"       : lig_elem_counts["Br"],
           "lig_I"        : lig_elem_counts["I"],
           "lig_P"        : lig_elem_counts["P"],
           "mol_weight"   : mw,
           "n_rings"      : n_rings,
           "n_hbd"        : n_hbd,
           "n_hba"        : n_hba,
           "n_rot_bonds"  : n_rot,
           "tpsa"         : tpsa,
           }
    rows.append(row)

df_final = pd.DataFrame(rows)
log.info(f"  Structural parsing complete: {len(df_final)} complexes")

# ─── 7. Save dataset CSV ────────────────────────────────────────────────────
csv_out = AFF_DIR / "dataset.csv"
df_final.to_csv(csv_out, index=False)
log.info(f"\nDataset saved → {csv_out}")

# ─── 8. Summary statistics ──────────────────────────────────────────────────
log.info("\n" + "=" * 50)
log.info("DATASET SUMMARY")
log.info("=" * 50)
log.info(f"  Total complexes         : {len(df_final)}")
log.info(f"  pKd range               : {df_final['pKd'].min():.3f} – {df_final['pKd'].max():.3f}")
log.info(f"  pKd mean ± std          : {df_final['pKd'].mean():.3f} ± {df_final['pKd'].std():.3f}")
log.info(f"  pKd median              : {df_final['pKd'].median():.3f}")
log.info(f"  Affinity types:\n{df_final['affinity_type'].value_counts().to_string()}")
log.info(f"  RNA pocket atoms (mean) : {df_final['n_rna_atoms'].mean():.1f}")
log.info(f"  Ligand atoms (mean)     : {df_final['n_lig_atoms'].mean():.1f}")
log.info(f"  Mol weight (mean)       : {df_final['mol_weight'].mean():.1f} Da")
log.info("=" * 50)

# ─── 9. Journal-quality EDA Figure ──────────────────────────────────────────
log.info("\nGenerating journal-quality EDA figure ...")

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.size"        : 11,
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 12,
    "axes.titleweight" : "bold",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "xtick.direction"  : "out",
    "ytick.direction"  : "out",
    "figure.dpi"       : 300,
    "savefig.dpi"      : 300,
    "pdf.fonttype"     : 42,
})

PALETTE = {
    "blue"   : "#2166AC",
    "red"    : "#D6604D",
    "green"  : "#4DAC26",
    "orange" : "#F4A582",
    "purple" : "#762A83",
    "gray"   : "#878787",
}
TYPE_COLORS = {"KD": PALETTE["blue"], "KI": PALETTE["red"],
               "IC50": PALETTE["green"], "EC50": PALETTE["orange"]}

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig,
                        hspace=0.42, wspace=0.38,
                        left=0.07, right=0.97, top=0.93, bottom=0.07)

# ── Panel A: pKd distribution (KDE + rug) ───────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
pkd = df_final["pKd"].dropna()
ax.hist(pkd, bins=18, color=PALETTE["blue"], edgecolor="white",
        linewidth=0.6, alpha=0.85, zorder=2)
ax.axvline(pkd.mean(),   color=PALETTE["red"],    lw=1.8, ls="--",
           label=f"Mean = {pkd.mean():.2f}")
ax.axvline(pkd.median(), color=PALETTE["orange"], lw=1.8, ls=":",
           label=f"Median = {pkd.median():.2f}")
ax.set_xlabel("pKd  (−log₁₀[M])")
ax.set_ylabel("Count")
ax.set_title("A  |  Binding Affinity (pKd) Distribution")
ax.legend(fontsize=9, frameon=False)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# ── Panel B: Affinity measurement type ──────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
atype = df_final["affinity_type"].value_counts()
colors_bar = [TYPE_COLORS.get(k, PALETTE["gray"]) for k in atype.index]
bars = ax.bar(atype.index, atype.values, color=colors_bar,
              edgecolor="white", linewidth=0.6)
for bar, val in zip(bars, atype.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xlabel("Measurement Type")
ax.set_ylabel("Count")
ax.set_title("B  |  Affinity Measurement Types")
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# ── Panel C: pKd by affinity type (violin) ──────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
groups  = [df_final[df_final["affinity_type"] == t]["pKd"].dropna().values
           for t in atype.index]
vp = ax.violinplot(groups, positions=range(len(groups)),
                   showmedians=True, showextrema=True)
for i, (body, col) in enumerate(zip(vp["bodies"],
                                    [TYPE_COLORS.get(k, PALETTE["gray"])
                                     for k in atype.index])):
    body.set_facecolor(col); body.set_alpha(0.7)
vp["cmedians"].set_color("white"); vp["cmedians"].set_linewidth(2)
vp["cmins"].set_color(PALETTE["gray"]); vp["cmaxes"].set_color(PALETTE["gray"])
ax.set_xticks(range(len(groups))); ax.set_xticklabels(atype.index)
ax.set_ylabel("pKd")
ax.set_title("C  |  pKd by Affinity Type")

# ── Panel D: Deposition year ─────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
years = df_final["year"].dropna().astype(int)
yr_range = range(int(years.min()), int(years.max()) + 2)
ax.hist(years, bins=yr_range, color=PALETTE["purple"],
        edgecolor="white", linewidth=0.6, alpha=0.85)
ax.set_xlabel("Deposition Year")
ax.set_ylabel("Count")
ax.set_title("D  |  Deposition Year")
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# ── Panel E: RNA pocket atom count distribution ──────────────────────────────
ax = fig.add_subplot(gs[1, 1])
ax.hist(df_final["n_rna_atoms"], bins=20, color=PALETTE["green"],
        edgecolor="white", linewidth=0.6, alpha=0.85)
ax.axvline(df_final["n_rna_atoms"].mean(), color=PALETTE["red"],
           lw=1.8, ls="--",
           label=f"Mean = {df_final['n_rna_atoms'].mean():.0f}")
ax.set_xlabel("RNA Pocket Atom Count")
ax.set_ylabel("Count")
ax.set_title("E  |  RNA Pocket Size")
ax.legend(fontsize=9, frameon=False)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# ── Panel F: Ligand atom count vs pKd (scatter) ─────────────────────────────
ax = fig.add_subplot(gs[1, 2])
sc = ax.scatter(df_final["n_lig_atoms"], df_final["pKd"],
                c=df_final["pKd"], cmap="RdYlBu_r",
                s=55, alpha=0.75, edgecolors="white", linewidths=0.4,
                vmin=pkd.min(), vmax=pkd.max())
plt.colorbar(sc, ax=ax, label="pKd", pad=0.02, fraction=0.046)
ax.set_xlabel("Ligand Heavy Atom Count")
ax.set_ylabel("pKd")
ax.set_title("F  |  Ligand Size vs Affinity")

# ── Panel G: Ligand molecular weight distribution ───────────────────────────
ax = fig.add_subplot(gs[2, 0])
mw = df_final["mol_weight"][df_final["mol_weight"] > 0]
ax.hist(mw, bins=20, color=PALETTE["orange"],
        edgecolor="white", linewidth=0.6, alpha=0.9)
ax.axvline(500, color=PALETTE["red"], lw=1.5, ls="--", label="Lipinski MW=500")
ax.set_xlabel("Molecular Weight (Da)")
ax.set_ylabel("Count")
ax.set_title("G  |  Ligand Molecular Weight")
ax.legend(fontsize=9, frameon=False)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# ── Panel H: Ligand drug-likeness properties (radar not suitable for 1 axis)
#    → show HBD / HBA / rings / rot bonds as grouped bar
ax = fig.add_subplot(gs[2, 1])
prop_means = [df_final["n_hbd"].mean(), df_final["n_hba"].mean(),
              df_final["n_rings"].mean(), df_final["n_rot_bonds"].mean()]
prop_stds  = [df_final["n_hbd"].std(),  df_final["n_hba"].std(),
              df_final["n_rings"].std(), df_final["n_rot_bonds"].std()]
labels = ["HB Donors", "HB Acceptors", "Rings", "Rot. Bonds"]
cols   = [PALETTE["blue"], PALETTE["green"], PALETTE["purple"], PALETTE["orange"]]
xpos   = np.arange(len(labels))
bars   = ax.bar(xpos, prop_means, yerr=prop_stds,
                color=cols, edgecolor="white", linewidth=0.6,
                capsize=4, error_kw={"elinewidth": 1.2})
ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Mean Count")
ax.set_title("H  |  Ligand Drug-likeness Properties")
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# ── Panel I: pKd cumulative distribution ─────────────────────────────────────
ax = fig.add_subplot(gs[2, 2])
sorted_pkd = np.sort(pkd)
cdf = np.arange(1, len(sorted_pkd) + 1) / len(sorted_pkd)
ax.plot(sorted_pkd, cdf, color=PALETTE["blue"], lw=2.2)
ax.fill_between(sorted_pkd, cdf, alpha=0.12, color=PALETTE["blue"])
ax.axvline(pkd.mean(), color=PALETTE["red"], lw=1.5, ls="--",
           label=f"Mean = {pkd.mean():.2f}")
ax.set_xlabel("pKd")
ax.set_ylabel("Cumulative Fraction")
ax.set_title("I  |  Cumulative pKd Distribution")
ax.legend(fontsize=9, frameon=False)
ax.set_ylim(0, 1.05)

# ── Main title ────────────────────────────────────────────────────────────────
fig.suptitle(
    f"RNA–Small Molecule Binding Affinity Dataset (PDBbind NL2020)  "
    f"|  n = {len(df_final)} complexes",
    fontsize=15, fontweight="bold", y=0.975
)

fig_path = FIG_DIR / "step01_eda.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
log.info(f"  EDA figure saved → {fig_path}")

# ─── 10. Element Composition Heatmap ────────────────────────────────────────
log.info("Generating element composition heatmap ...")

rna_elems = ["C", "N", "O", "P", "S"]
lig_elems = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]

# Build 36-pair count matrix (mean atom counts per pair)
mat = np.zeros((len(rna_elems), len(lig_elems)))
for i, re_ in enumerate(rna_elems):
    for j, le in enumerate(lig_elems):
        r_col = f"rna_{re_}"
        l_col = f"lig_{le}"
        if r_col in df_final.columns and l_col in df_final.columns:
            # number of complexes where BOTH element types are present
            both = ((df_final[r_col] > 0) & (df_final[l_col] > 0)).sum()
            mat[i, j] = both

fig2, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
plt.colorbar(im, ax=ax, label="# Complexes with both element types present",
             fraction=0.03, pad=0.03)
ax.set_xticks(range(len(lig_elems))); ax.set_xticklabels(lig_elems, fontsize=12)
ax.set_yticks(range(len(rna_elems))); ax.set_yticklabels(rna_elems, fontsize=12)
ax.set_xlabel("Ligand Element", fontsize=13, fontweight="bold")
ax.set_ylabel("RNA Pocket Element", fontsize=13, fontweight="bold")
ax.set_title(
    "RNA × Ligand Element Co-occurrence Across Dataset\n"
    "(Defines the 36 Interaction Pairs for RNA-PDFL Feature Extraction)",
    fontsize=13, fontweight="bold", pad=12
)
# Annotate cells
for i in range(len(rna_elems)):
    for j in range(len(lig_elems)):
        val = int(mat[i, j])
        color = "white" if mat[i, j] > mat.max() * 0.6 else "black"
        ax.text(j, i, str(val), ha="center", va="center",
                fontsize=11, fontweight="bold", color=color)

# Draw grid lines
for x in np.arange(-0.5, len(lig_elems), 1):
    ax.axvline(x, color="white", lw=0.8)
for y in np.arange(-0.5, len(rna_elems), 1):
    ax.axhline(y, color="white", lw=0.8)

plt.tight_layout()
fig2_path = FIG_DIR / "step01_element_pairs.png"
fig2.savefig(fig2_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig2)
log.info(f"  Element heatmap saved → {fig2_path}")

# ─── 11. Final log summary ───────────────────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("STEP 1 COMPLETE")
log.info(f"  Dataset CSV   : {csv_out}")
log.info(f"  EDA figure    : {fig_path}")
log.info(f"  Element map   : {fig2_path}")
log.info(f"  Log file      : {log_file}")
log.info(f"  Complexes     : {len(df_final)}")
log.info(f"  pKd range     : {df_final['pKd'].min():.3f} – {df_final['pKd'].max():.3f}")
log.info(f"  Mean pKd      : {df_final['pKd'].mean():.3f} ± {df_final['pKd'].std():.3f}")
log.info("=" * 70)
