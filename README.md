# Zero-Field Quantum Hall Effect in Graphene/TMD Heterostructures

A computational investigation of anomalous spin-split band gap behavior near the Dirac point in graphene-transition metal dichalcogenide (TMD) heterostructures, exploring the physical origins of experimentally observed gap widening at low energies.

## Overview

This repository implements a continuum Hamiltonian model for graphene/TMD heterostructures to investigate the zero-field quantum Hall effect. The code systematically explores whether the experimentally observed anomalous gap widening near the Dirac point can be explained by:

1. Intrinsic spin-orbit coupling mechanisms (Kane-Mele, Rashba, valley-Zeeman)
2. Sublattice asymmetry and pseudospin-inversion asymmetry
3. Extrinsic disorder effects from charge puddles

## Physical Model

The system is described by a 4×4 continuum Hamiltonian in the basis |A↑⟩, |A↓⟩, |B↑⟩, |B↓⟩:

```
H = ℏvF (τkx σx + ky σy) ⊗ Is          (kinetic term)
  + λVZ τ σz ⊗ sz                       (valley-Zeeman coupling)
  + λR (τ σx ⊗ sy - σy ⊗ sx)            (Rashba spin-orbit coupling)
  + λKM τ σz ⊗ sz                       (Kane-Mele intrinsic SOC)
  + Δ σz ⊗ Is                           (sublattice potential)
  + λPIA k-dependent terms              (pseudospin-inversion asymmetry)
```

All energy parameters are in meV, wavevectors in Å⁻¹, and Fermi velocity in meV·Å.

## Repository Structure

```
.
├── hamiltonian.py       # Continuum Hamiltonian implementation
├── optimizer.py         # Parameter optimization (differential evolution, Monte Carlo)
├── charge_puddles.py    # Disorder modeling via charge puddle landscape
├── main.py              # Complete analysis pipeline
└── outputs/             # Generated figures and data
```

## Methodology

### Step 1: Baseline Model
Establishes the reference band structure using experimentally constrained parameters from the literature. Computes the inner-band gap (spin splitting) as a function of energy.

### Step 2: Unreasonable Scenario
Demonstrates that achieving ~20 meV gap widening requires unphysically large Kane-Mele coupling (λKM ~ 20 meV), orders of magnitude beyond theoretical predictions for graphene/TMD systems.

### Step 3: Optimization Search
Employs two complementary global optimization strategies to search for physically reasonable parameter combinations (all couplings < 5 meV) that could reproduce the anomalous behavior:

- **Differential Evolution**: Deterministic evolutionary algorithm with adaptive mutation
- **Monte Carlo with Simulated Annealing**: Stochastic exploration with temperature scheduling

The objective function maximizes the gap slope dΔ/dE near the Dirac point.

### Step 4: Charge Puddle Disorder
Models nanoscale electron-hole puddles arising from substrate impurities. Computes disorder-averaged band structure to assess whether spatial potential fluctuations can create an apparent gap widening artifact in transport measurements.

## Installation

### Requirements
- Python 3.8+
- NumPy
- SciPy
- Matplotlib

### Setup
```bash
git clone https://github.com/soham164/zerofield-.git
cd zerofield-
pip install numpy scipy matplotlib
```

## Usage

Run the complete analysis pipeline:

```bash
python main.py
```

This executes all four analysis steps and generates publication-quality figures in the `outputs/` directory:

- `step1_baseline_dispersion.png` - Reference band structure
- `step2_unreasonable_scenario.png` - Parameter sweep showing λKM effects
- `step3_optimizer_results.png` - Optimization convergence and results
- `step4_charge_puddles.png` - Disorder-averaged band structure
- `step4_puddle_landscape.png` - Spatial potential visualization
- `optimizer_log.csv` - Detailed optimization trajectory

### Individual Module Usage

```python
from hamiltonian import Params, dispersion
import numpy as np

# Define parameters
params = Params(lambdaVZ=2.45, lambdaR=0.56, lambdaKM=0.12)

# Compute band structure
k_array = np.linspace(-0.07, 0.07, 200)
bands = dispersion(k_array, params)
```

## Key Findings

The analysis reveals:

1. **Intrinsic mechanisms insufficient**: Within physically reasonable parameter bounds (< 5 meV), intrinsic spin-orbit couplings cannot reproduce the observed ~20 meV gap widening.

2. **Charge puddle effects**: Disorder-averaged calculations show that charge puddles can create apparent gap modifications near the Dirac point, but the magnitude depends critically on disorder strength and spatial correlation length.

3. **Parameter sensitivity**: The gap behavior exhibits strong sensitivity to the interplay between Kane-Mele coupling, sublattice asymmetry, and pseudospin-inversion asymmetry terms.

## Physical Parameters

Default values based on experimental constraints:

| Parameter | Symbol | Value | Unit | Physical Origin |
|-----------|--------|-------|------|-----------------|
| Fermi velocity | vF | 810 | meV·Å | Graphene band structure |
| Valley-Zeeman | λVZ | 2.45 | meV | Proximity-induced exchange |
| Rashba SOC | λR | 0.56 | meV | Broken inversion symmetry |
| Kane-Mele SOC | λKM | 0.12 | meV | Intrinsic graphene SOC |
| Sublattice potential | Δ | 0.15 | meV | Substrate asymmetry |
| PIA coupling | λPIA | ±0.20 | meV | Sublattice-dependent hopping |

## Computational Details

- **k-space resolution**: 200 points over [-0.07, 0.07] Å⁻¹
- **Optimization iterations**: 80 generations (DE), 2000 steps (MC)
- **Disorder averaging**: 30 samples per k-point
- **Charge puddle parameters**: W = 4 meV, correlation length ~ 10 nm

## Output Interpretation

The gap slope score quantifies anomalous behavior:
- **Positive score**: Gap widens toward Dirac point (anomalous)
- **Negative score**: Gap shrinks toward Dirac point (conventional)
- **Zero score**: Energy-independent gap

## Citation

If you use this code in your research, please cite:

```bibtex
@software{zerofield2024,
  author = {[Author Name]},
  title = {Zero-Field Quantum Hall Effect in Graphene/TMD Heterostructures},
  year = {2024},
  url = {https://github.com/soham164/zerofield-}
}
```

## License

This project is available under the MIT License.

## Contact

For questions or collaboration inquiries, please open an issue on GitHub.

## Acknowledgments

This work builds upon experimental observations of zero-field quantum Hall effects in graphene/WSe₂ heterostructures and theoretical frameworks for spin-orbit coupling in two-dimensional materials.
