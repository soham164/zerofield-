"""
hamiltonian.py
==============
Continuum Hamiltonian for graphene / TMD heterostructure (Eq. 2 from paper).

Basis: |A↑⟩, |A↓⟩, |B↑⟩, |B↓⟩

H = ℏv_F (τ k_x σ_x + k_y σ_y) ⊗ I_s
  + λ_VZ  τ σ_z ⊗ s_z          (valley-Zeeman)
  + λ_R   (τ σ_x ⊗ s_y - σ_y ⊗ s_x)   (Rashba)
  + λ_KM  τ σ_z ⊗ s_z          (Kane-Mele intrinsic SOC)
  + Δ     σ_z ⊗ I_s             (sublattice potential)
  + λ_PIA terms  (pseudospin-inversion asymmetry, k-dependent)

All energies in meV, k in Å⁻¹, v_F in meV·Å
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple

# ── Pauli matrices ─────────────────────────────────────────────────────────────
σ0 = np.eye(2, dtype=complex)
σx = np.array([[0, 1], [1, 0]], dtype=complex)
σy = np.array([[0, -1j], [1j, 0]], dtype=complex)
σz = np.array([[1, 0], [0, -1]], dtype=complex)


@dataclass
class Params:
    """All Hamiltonian parameters in meV (v_F in meV·Å)."""
    vF:          float = 810.0   # Fermi velocity
    lambdaVZ:    float = 2.45    # valley-Zeeman
    lambdaR:     float = 0.56    # Rashba
    lambdaKM:    float = 0.12    # Kane-Mele
    Delta:       float = 0.15    # sublattice potential
    lambdaPIA_A: float = 0.20    # PIA on sublattice A
    lambdaPIA_B: float = -0.20   # PIA on sublattice B

    def to_array(self) -> np.ndarray:
        return np.array([self.lambdaKM, self.Delta,
                         self.lambdaPIA_A, self.lambdaPIA_B])

    @classmethod
    def unreasonable(cls) -> "Params":
        """The ~20 meV scenario from the paper (Step 2)."""
        p = cls()
        p.lambdaKM = 20.0
        p.Delta    = 20.0
        return p


def build_hamiltonian(kx: float, ky: float,
                      params: Params, valley: int = 1) -> np.ndarray:
    """
    Build the full 4×4 Hamiltonian matrix at a given (kx, ky) point.

    Parameters
    ----------
    kx, ky  : wave-vector components (Å⁻¹)
    params  : Params dataclass
    valley  : +1 for K, -1 for K'

    Returns
    -------
    H : (4, 4) complex numpy array in meV
    """
    p   = params
    tau = valley
    k   = np.hypot(kx, ky)

    # ── Kinetic term: ℏv_F (τ kx σx + ky σy) ⊗ I_s ─────────────────────────
    H_kin = p.vF * (tau * kx * np.kron(σx, σ0)
                  +       ky * np.kron(σy, σ0))

    # ── Valley-Zeeman: λ_VZ τ σz ⊗ sz ───────────────────────────────────────
    H_VZ  = p.lambdaVZ * tau * np.kron(σz, σz)

    # ── Rashba: λ_R (τ σx ⊗ sy - σy ⊗ sx) ──────────────────────────────────
    H_R   = p.lambdaR * (tau * np.kron(σx, σy) - np.kron(σy, σx))

    # ── Kane-Mele: λ_KM τ σz ⊗ sz ───────────────────────────────────────────
    H_KM  = p.lambdaKM * tau * np.kron(σz, σz)

    # ── Sublattice potential: Δ σz ⊗ I_s ─────────────────────────────────────
    H_sub = p.Delta * np.kron(σz, σ0)

    # ── PIA (k-dependent, sublattice-asymmetric) ─────────────────────────────
    # λ_PIA^A couples A-sublattice off-diagonal with k, λ_PIA^B for B
    # Implemented as k-linear correction to the off-diagonal hopping blocks
    k_factor = k / 0.05   # normalise by typical k scale
    PA = p.lambdaPIA_A * k_factor
    PB = p.lambdaPIA_B * k_factor

    H_PIA = np.zeros((4, 4), dtype=complex)
    # PIA adds to off-diagonal sublattice blocks (rows/cols 0-1 ↔ 2-3)
    H_PIA[0, 2] += PA * tau
    H_PIA[2, 0] += PA * tau
    H_PIA[1, 3] += PB * tau
    H_PIA[3, 1] += PB * tau

    H = H_kin + H_VZ + H_R + H_KM + H_sub + H_PIA

    # Enforce Hermiticity (numerical safety)
    return 0.5 * (H + H.conj().T)


def eigenvalues(kx: float, ky: float,
                params: Params, valley: int = 1) -> np.ndarray:
    """Return the 4 sorted real eigenvalues at (kx, ky)."""
    H = build_hamiltonian(kx, ky, params, valley)
    evals = np.linalg.eigvalsh(H)   # eigvalsh guarantees real output for Hermitian H
    return np.sort(evals)


def dispersion(k_array: np.ndarray, params: Params,
               valley: int = 1, ky: float = 0.0) -> np.ndarray:
    """
    Compute band energies for an array of kx values.

    Returns
    -------
    bands : (len(k_array), 4) array of energies in meV
    """
    return np.array([eigenvalues(kx, ky, params, valley) for kx in k_array])


def inner_band_gap(k_array: np.ndarray, params: Params) -> np.ndarray:
    """
    Gap between the two inner bands (bands[1] and bands[2]) for each k.
    This is the experimentally observed spin-split gap.
    """
    bands = dispersion(k_array, params)
    return bands[:, 2] - bands[:, 1]   # band 3 – band 2 (0-indexed)
