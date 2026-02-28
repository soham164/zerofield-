"""
charge_puddles.py
=================
Step 4: Model charge puddle disorder near the Dirac point.

Charge puddles are nanoscale regions of electron/hole doping caused by
substrate impurities. They dominate transport below ~30 meV and can
create an apparent gap widening — potentially an artefact.

We model them as a random smooth potential landscape overlaid on the
clean Hamiltonian, then compute an effective disordered band structure
via spatial averaging (disorder-averaged Green's function approach).
"""

import numpy as np
from typing import Tuple
from hamiltonian import Params, build_hamiltonian, dispersion


# ── Puddle landscape generation ───────────────────────────────────────────────

def generate_puddle_landscape(
    n_sites: int = 200,
    box_size: float = 100.0,   # nm
    n_puddles: int = 30,
    puddle_strength: float = 5.0,   # meV, disorder amplitude
    puddle_radius: float = 10.0,    # nm, spatial extent of each puddle
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a 2D charge puddle potential on a real-space grid.

    Each puddle is a Gaussian-shaped potential bump at a random position.
    The sum gives a smooth correlated disorder landscape.

    Returns
    -------
    x, y : 1D coordinate arrays (nm)
    V    : 2D potential array (meV)
    """
    rng = np.random.default_rng(seed)
    x   = np.linspace(0, box_size, n_sites)
    y   = np.linspace(0, box_size, n_sites)
    X, Y = np.meshgrid(x, y)
    V    = np.zeros_like(X)

    for _ in range(n_puddles):
        cx = rng.uniform(0, box_size)
        cy = rng.uniform(0, box_size)
        amp = rng.choice([-1, 1]) * rng.uniform(0.5, 1.0) * puddle_strength
        V += amp * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * puddle_radius**2))

    return x, y, V


def puddle_broadened_dispersion(
    k_array: np.ndarray,
    params: Params,
    puddle_strength: float = 3.0,
    n_samples: int = 40,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute disorder-averaged band structure by sampling random local
    potentials from the puddle distribution and averaging eigenvalues.

    For each k-point, we add a random scalar potential drawn from a
    Gaussian distribution N(0, puddle_strength²) — representing the
    local doping from a nearby charge puddle — and average the resulting
    eigenvalue shifts over many samples.

    Returns
    -------
    bands_clean    : (len(k_array), 4) clean band energies
    bands_disorder : (len(k_array), 4) disorder-averaged band energies
    """
    rng = np.random.default_rng(seed)

    bands_clean    = np.array([
        np.sort(np.linalg.eigvalsh(build_hamiltonian(kx, 0, params)))
        for kx in k_array
    ])

    # Disorder-averaged: for each k, sample n_samples random potentials
    bands_all = np.zeros((n_samples, len(k_array), 4))
    for s in range(n_samples):
        for i, kx in enumerate(k_array):
            H = build_hamiltonian(kx, 0, params)
            # Local scalar potential (charge puddle): shifts all eigenvalues
            # k-dependent weight: stronger near Dirac point (low k)
            k_weight = np.exp(-kx**2 / 0.001)   # peaks at k=0
            V_local  = rng.normal(0, puddle_strength * k_weight)
            H_disordered = H + V_local * np.eye(4)
            bands_all[s, i] = np.sort(np.linalg.eigvalsh(H_disordered))

    bands_disorder = bands_all.mean(axis=0)
    return bands_clean, bands_disorder


def compute_effective_gap(
    bands: np.ndarray,
    k_array: np.ndarray,
    energy_bins: int = 30,
    e_max: float = 80.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract gap-vs-|k| curve, expressed in energy units by using the
    average of the two inner band energies as the 'energy' axis label.

    Returns
    -------
    energies : |E_mid| values at each k-point (meV)  — sorted ascending
    gaps     : corresponding inner-band gap values (meV)
    """
    e_mid = np.abs(0.5 * (bands[:, 1] + bands[:, 2]))
    gap   = bands[:, 2] - bands[:, 1]

    # Sort by |E_mid| so the curve reads left (Dirac) → right (away)
    order    = np.argsort(e_mid)
    energies = e_mid[order]
    gaps_out = gap[order]

    return energies, gaps_out
