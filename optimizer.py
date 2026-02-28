"""
optimizer.py
============
Step 3: Search for physically reasonable parameters (all < 5 meV) that
reproduce the anomalous gap widening near the Dirac point.

Two strategies:
  1. scipy.optimize.differential_evolution  — deterministic global search
  2. Monte Carlo random walk                — explore the landscape broadly

Objective: maximize dGap/dE at low energy (gap should INCREASE toward E=0).
"""

import numpy as np
from scipy.optimize import differential_evolution, OptimizeResult
from dataclasses import asdict
from typing import Callable, List, Tuple, Dict
import time

from hamiltonian import Params, dispersion, inner_band_gap


# ── Helper: measure gap slope near Dirac point ────────────────────────────────

def gap_slope_score(params: Params, n_k: int = 80) -> float:
    """
    Score = mean gap near k=0  -  mean gap at large |k|.
    Positive → gap widens toward Dirac point (anomalous, what we want to reproduce).
    Negative → gap shrinks toward Dirac point (normal behavior).

    We use |k| directly as the proxy (closer to Dirac = smaller |k|).
    """
    k_arr  = np.linspace(-0.07, 0.07, n_k)
    bands  = dispersion(k_arr, params)
    gap    = bands[:, 2] - bands[:, 1]   # inner-band gap at each k
    abs_k  = np.abs(k_arr)

    # "near Dirac" = small |k|; "away" = large |k|
    k_cut_low  = np.percentile(abs_k, 30)   # bottom 30% of |k| values
    k_cut_high = np.percentile(abs_k, 70)   # top 30% of |k| values

    low_mask  = abs_k <= k_cut_low
    high_mask = abs_k >= k_cut_high

    gap_low  = gap[low_mask].mean()
    gap_high = gap[high_mask].mean()

    return float(gap_low - gap_high)   # positive = anomalous widening


def objective(x: np.ndarray, base_params: Params) -> float:
    """Objective for scipy optimizers (minimise → negate score)."""
    p = Params(
        vF=base_params.vF,
        lambdaVZ=base_params.lambdaVZ,
        lambdaR=base_params.lambdaR,
        lambdaKM=x[0],
        Delta=x[1],
        lambdaPIA_A=x[2],
        lambdaPIA_B=x[3],
    )
    return -gap_slope_score(p)


# ── Strategy 1: Differential Evolution (global, evolutionary) ─────────────────

def run_differential_evolution(
    base_params: Params,
    max_iter: int = 150,
    popsize: int = 12,
    physical_bound: float = 5.0,
    callback: Callable = None,
) -> Tuple[Params, float, OptimizeResult]:
    """
    Use scipy differential_evolution to search (λ_KM, Δ, λ_PIA_A, λ_PIA_B)
    within physically reasonable bounds (±physical_bound meV).

    Parameters
    ----------
    base_params    : fixed parameters (vF, λ_VZ, λ_R kept constant)
    max_iter       : maximum generations
    popsize        : population multiplier (actual pop = popsize × 4 params)
    physical_bound : hard bound on all searched parameters (meV)
    callback       : optional fn(xk, convergence) called each generation

    Returns
    -------
    best_params : Params with optimal searched values
    best_score  : gap slope score (meV, positive = anomalous)
    result      : raw scipy OptimizeResult
    """
    bounds = [(-physical_bound, physical_bound)] * 4

    print(f"\n{'='*60}")
    print(f"  Differential Evolution Search")
    print(f"  Searching: λ_KM, Δ, λ_PIA_A, λ_PIA_B  ∈ [±{physical_bound} meV]")
    print(f"  Population: {popsize * 4} individuals, max {max_iter} generations")
    print(f"{'='*60}")

    gen_log: List[Dict] = []

    def _cb(xk, convergence):
        score = -objective(xk, base_params)
        gen_log.append({
            "generation": len(gen_log),
            "lambdaKM": xk[0],
            "Delta": xk[1],
            "lambdaPIA_A": xk[2],
            "lambdaPIA_B": xk[3],
            "score": score,
            "convergence": convergence,
        })
        if len(gen_log) % 10 == 0:
            print(f"  Gen {len(gen_log):4d} | score={score:+.4f} meV | "
                  f"λ_KM={xk[0]:+.3f} Δ={xk[1]:+.3f} | conv={convergence:.4f}")
        if callback:
            callback(xk, convergence)

    t0 = time.time()
    result = differential_evolution(
        objective,
        bounds,
        args=(base_params,),
        maxiter=max_iter,
        popsize=popsize,
        tol=1e-5,
        mutation=(0.5, 1.5),
        recombination=0.7,
        seed=42,
        callback=_cb,
        polish=True,
        workers=1,
    )
    elapsed = time.time() - t0

    best_x = result.x
    best_score = -result.fun

    best_params = Params(
        vF=base_params.vF,
        lambdaVZ=base_params.lambdaVZ,
        lambdaR=base_params.lambdaR,
        lambdaKM=best_x[0],
        Delta=best_x[1],
        lambdaPIA_A=best_x[2],
        lambdaPIA_B=best_x[3],
    )

    print(f"\n  ✓ Finished in {elapsed:.1f}s | {len(gen_log)} generations")
    print(f"  Best gap slope score : {best_score:+.4f} meV")
    print(f"  Best parameters found:")
    print(f"    λ_KM    = {best_x[0]:+.4f} meV")
    print(f"    Δ       = {best_x[1]:+.4f} meV")
    print(f"    λ_PIA_A = {best_x[2]:+.4f} meV")
    print(f"    λ_PIA_B = {best_x[3]:+.4f} meV")
    print(f"  Convergence: {'SUCCESS' if result.success else 'PARTIAL'}")

    return best_params, best_score, result, gen_log


# ── Strategy 2: Monte Carlo random walk ───────────────────────────────────────

def run_monte_carlo(
    base_params: Params,
    n_iter: int = 5000,
    physical_bound: float = 5.0,
    temperature: float = 0.5,
    step_size: float = 0.3,
    seed: int = 42,
) -> Tuple[Params, float, List[Dict]]:
    """
    Metropolis Monte Carlo search over (λ_KM, Δ, λ_PIA_A, λ_PIA_B).

    Uses simulated annealing: temperature decreases over iterations,
    allowing broad exploration early and fine-tuning late.

    Parameters
    ----------
    physical_bound : hard bound on parameter magnitudes (meV)
    temperature    : initial MC temperature (controls acceptance of bad moves)
    step_size      : initial perturbation size (meV)
    """
    rng = np.random.default_rng(seed)

    # Start from baseline searched params
    x = np.array([base_params.lambdaKM, base_params.Delta,
                  base_params.lambdaPIA_A, base_params.lambdaPIA_B])

    def make_params(x):
        return Params(vF=base_params.vF, lambdaVZ=base_params.lambdaVZ,
                      lambdaR=base_params.lambdaR,
                      lambdaKM=x[0], Delta=x[1],
                      lambdaPIA_A=x[2], lambdaPIA_B=x[3])

    current_score = gap_slope_score(make_params(x))
    best_x        = x.copy()
    best_score    = current_score

    history: List[Dict] = []
    accepted = 0

    print(f"\n{'='*60}")
    print(f"  Monte Carlo Search  ({n_iter} iterations)")
    print(f"  Bounds: ±{physical_bound} meV | T₀={temperature} | step={step_size}")
    print(f"{'='*60}")

    for i in range(n_iter):
        # Anneal temperature and step size
        frac = i / n_iter
        T    = temperature * np.exp(-3 * frac)
        step = step_size * (1 - 0.7 * frac) + 0.02

        # Propose new point
        x_new = x + rng.normal(0, step, size=4)
        x_new = np.clip(x_new, -physical_bound, physical_bound)

        new_score = gap_slope_score(make_params(x_new))
        delta     = new_score - current_score

        # Metropolis acceptance
        if delta > 0 or rng.random() < np.exp(delta / (T + 1e-9)):
            x             = x_new
            current_score = new_score
            accepted     += 1

        if new_score > best_score:
            best_score = new_score
            best_x     = x_new.copy()

        if i % 500 == 0:
            print(f"  Iter {i:5d} | best score={best_score:+.4f} | "
                  f"current={current_score:+.4f} | accept rate={accepted/(i+1):.2f}")

        history.append({
            "iteration": i,
            "score": current_score,
            "best_score": best_score,
            "lambdaKM": x[0],
            "Delta": x[1],
        })

    print(f"\n  ✓ Done | Best score: {best_score:+.4f} meV")
    print(f"  Best: λ_KM={best_x[0]:+.4f}  Δ={best_x[1]:+.4f}  "
          f"λ_PIA_A={best_x[2]:+.4f}  λ_PIA_B={best_x[3]:+.4f}")

    return make_params(best_x), best_score, history
