# SGT-RNA: Spectral Graph Topology for RNA-Ligand Binding Affinity Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

SGT-RNA is a topology-guided machine learning framework for predicting RNA-small molecule binding affinity. It characterizes RNA-ligand complexes through the eigenvalue spectra of element-type-specific bipartite graph Laplacians weighted by an exponential Flexibility-Rigidity Index (FRI) kernel, combined with complementary contact fingerprints and ligand descriptors. SGT-RNA achieves state-of-the-art performance on the PDBbind NL2020 benchmark (143 complexes, LOO-CV Pearson r = 0.830).

The framework integrates:
- SGT topological features (3,600-dim eigenvalue spectra from vertex and edge Laplacians)
- Contact Pair Fingerprint (CPF, 72-dim)
- Soft Contact Fingerprint (SCF, 72-dim, Gaussian-decay)
- RNA-Ligand Interaction Fingerprint (RLIF, 21-dim)
- Ligand features (Morgan, MACCS, UniMol, RNA-FM)
- Subtype-specific residual correction

## Method

### Spectral Graph Topology (SGT) Features

For each RNA-ligand complex, element-type-specific bipartite graphs are constructed between RNA heavy atoms and ligand heavy atoms. Edge weights are assigned using an exponential FRI kernel:

```
W_ij = exp(-(d_ij / eta)^kappa)
```

where eta = 5.0 Å and kappa = 2.0. At each of five distance filtration thresholds (tau = 4, 5, 6, 7, 8 Å), the vertex Laplacian L0 and edge Laplacian L1 are computed. Ten spectral statistics are extracted from each (minimum, maximum, mean, median, variance, standard deviation, sum, sum of squares, spectral rank, and Betti number), yielding a 3,600-dimensional feature vector per complex.

### Contact Fingerprints

Three complementary structural fingerprints describe the RNA-ligand interface:

- **CPF**: Hard-cutoff contact counts between each RNA nucleotide base type (A, G, C, U) and each ligand element type at three distance thresholds (72-dim)
- **SCF**: Gaussian-decay soft contact counts over the same base-element pairs (72-dim)
- **RLIF**: Per-residue contact counts between RNA backbone/base and the ligand as a whole, augmented by base-pair co-contact features (21-dim)

### Global Ensemble Model

All feature representations are combined via multi-kernel learning (MKL) with ridge regression. All model parameters (kernel weights, regularization) are tuned strictly within leave-one-out cross-validation to prevent information leakage.

### Subtype-Specific Residual Correction

A rule-based RNA subtype classifier partitions the 143 complexes into seven primary categories (riboswitch, aptamer, ribozyme, ribosome, tRNA, G-quadruplex, other). Subtype-specific residual correction models are applied for riboswitch subclasses where the global model shows systematic bias.

## Benchmark Results (PDBbind NL2020, LOO-CV)

| Method | Pearson r | Reference |
|--------|-----------|-----------|
| AffiGrapher | 0.498 | Zhao et al. |
| RLaffinity | 0.559 | Wang et al. |
| RLASIF | 0.666 | Li et al. |
| DeepRSMA | 0.784 | Su et al. |
| RSAPred | 0.830 | Zhang et al. |
| **SGT-RNA (this work)** | **0.830** | This work |

## Requirements

See `requirements.txt`. Key dependencies:
- Python >= 3.8
- numpy, pandas, scipy, scikit-learn, joblib
- rdkit (for ligand feature computation)
- matplotlib (for visualization)

## Installation

```bash
git clone https://github.com/precisionmatics/SGT-RNA.git
cd SGT-RNA
pip install -r requirements.txt
```

## Dataset

The PDBbind NL2020 benchmark is available from [http://www.pdbbind.org.cn/](http://www.pdbbind.org.cn/).
Place the 143 complex folders under `NA-L/` with the structure:
```
NA-L/
  {pdbid}/
    {pdbid}_pocket.pdb
    {pdbid}_nucleic_acid.pdb
    {pdbid}_ligand.sdf
    {pdbid}_ligand.mol2
```

## Reproducing Results

Run the pipeline scripts in order:

```bash
# Steps 1-12: SGT feature extraction and contact fingerprint computation
python scripts/step01_build_graphs.py
python scripts/step02_compute_laplacians.py
python scripts/step03_extract_sgt_features.py
# ... (see individual script headers for details)

# Global ensemble model (steps 13-30)
python scripts/step30_final_blend.py

# Subtype-specific residual corrections (steps 35-36)
python scripts/step35_residual_correction.py
python scripts/step36_prespecified_correction.py
```

Results are saved to `results/step36_results.csv`.

**Note:** Pre-computed feature arrays and LOO prediction CSVs for all 36 steps are provided in `data/features/` and `results/` respectively, so the full pipeline does not need to be re-run to inspect intermediate outputs.

## Repository Structure

```
SGT-RNA/
  scripts/         Core pipeline scripts (steps 1-36)
  data/
    features/      Pre-computed feature arrays (.npy, .npz)
  results/         LOO prediction CSVs for each pipeline step
  requirements.txt
  README.md
```

