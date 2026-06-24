# PDFL-RNA: Persistent Directed Flag Laplacian for RNA-Ligand Binding Affinity Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

PDFL-RNA is a topology-guided machine learning framework for predicting RNA-small molecule binding affinity. It adapts the Persistent Directed Flag Laplacian (PDFL) method to RNA-ligand systems and achieves state-of-the-art performance on the PDBbind NL2020 benchmark (143 complexes, LOO-CV Pearson r = 0.830).

The framework integrates:
- PDFL topological features (36,000-dim eigenvalue spectra)
- Contact Pair Fingerprint (CPF, 72-dim)
- Soft Contact Fingerprint (SCF, 72-dim, Gaussian-decay)
- RNA-Ligand Interaction Fingerprint (RLIF, 21-dim)
- Ligand features (Morgan, MACCS, UniMol, RNA-FM)
- Subtype-specific residual correction

## Benchmark Results (PDBbind NL2020, LOO-CV)

| Method | Pearson r | Reference |
|--------|-----------|-----------|
| AffiGrapher | 0.498 | Zhao et al. |
| RLaffinity | 0.559 | Wang et al. |
| RLASIF | 0.666 | Li et al. |
| DeepRSMA | 0.784 | Su et al. |
| RSAPred | 0.830 | Zhang et al. |
| **PDFL-RNA (this work)** | **0.830** | This work |

## Requirements

See `requirements.txt`. Key dependencies:
- Python >= 3.8
- numpy, pandas, scipy, scikit-learn, joblib
- rdkit (for ligand feature computation)
- matplotlib (for visualization)

## Installation

```bash
git clone https://github.com/[username]/PDFL-RNA.git
cd PDFL-RNA
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
# Global ensemble model (steps 1-30)
python scripts/step30_final_blend.py

# Residual corrections (steps 35-36)
python scripts/step35_residual_correction.py
python scripts/step36_prespecified_correction.py
```

Results are saved to `results/step36_results.csv`.

## Repository Structure

```
PDFL-RNA/
  scripts/         Core pipeline scripts
  data/
    features/      Pre-computed feature arrays (.npy, .npz)
  results/         LOO prediction CSVs for each step
  manuscript/      Figures and manuscript files
  requirements.txt
  README.md
```

## Citation

If you use PDFL-RNA in your research, please cite:

```bibtex
@article{pdfl_rna_2026,
  title   = {Persistent Directed Flag Laplacian with Subtype-Specific Residual Correction for RNA-Ligand Binding Affinity Prediction},
  author  = {[Authors]},
  journal = {Journal of Chemical Information and Modeling},
  year    = {2026},
  doi     = {[DOI]}
}
```

## Contact

For questions, please open an issue or contact [email].
