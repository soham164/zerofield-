"""
main.py
=======
Runs all 4 steps and exports publication-quality matplotlib figures.

Usage:
    python main.py

Outputs (saved to ./outputs/):
    step1_baseline_dispersion.png
    step2_unreasonable_scenario.png
    step3_optimizer_results.png
    step4_charge_puddles.png
    step4_puddle_landscape.png
    optimizer_log.csv
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import csv

from hamiltonian import Params, dispersion, inner_band_gap
from optimizer   import run_differential_evolution, run_monte_carlo, gap_slope_score
from charge_puddles import (generate_puddle_landscape,
                             puddle_broadened_dispersion,
                             compute_effective_gap)

os.makedirs("outputs", exist_ok=True)

# ── Shared plot style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0a0a0f",
    "axes.facecolor":    "#0d0d1a",
    "axes.edgecolor":    "#2a2a4a",
    "axes.labelcolor":   "#aabbcc",
    "xtick.color":       "#667788",
    "ytick.color":       "#667788",
    "text.color":        "#ccd",
    "grid.color":        "#1a1a2e",
    "grid.linewidth":    0.5,
    "lines.linewidth":   2.0,
    "font.family":       "monospace",
    "axes.titlesize":    11,
    "axes.labelsize":    9,
})

BAND_COLORS = ["#00d4ff", "#ff6b6b", "#00ff9d", "#ffd700"]
BAND_LABELS = ["Band 1 (↑)", "Band 2 (↑↓)", "Band 3 (↓↑)", "Band 4 (↓)"]

K_ARRAY = np.linspace(-0.07, 0.07, 200)


def plot_dispersion(ax, bands, k_array, title="", highlight_gap=True, alpha=1.0):
    for i in range(4):
        ax.plot(k_array * 1000, bands[:, i],
                color=BAND_COLORS[i], label=BAND_LABELS[i],
                lw=2.5 if i in (1, 2) else 1.5, alpha=alpha)
    if highlight_gap:
        ax.fill_between(k_array * 1000, bands[:, 1], bands[:, 2],
                        color="#ff6b6b", alpha=0.12, label="Spin gap")
    ax.axhline(0, color="#334466", lw=0.8, ls="--")
    ax.axvline(0, color="#334466", lw=0.8, ls="--")
    ax.set_xlabel("k (10⁻³ Å⁻¹)")
    ax.set_ylabel("Energy (meV)")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(fontsize=7, loc="upper right",
              facecolor="#080812", edgecolor="#222244")


def plot_gap_vs_energy(ax, bands, k_array, label="", color="#00d4ff"):
    e_centers, gaps = compute_effective_gap(bands, k_array)
    valid = ~np.isnan(gaps)
    ax.plot(e_centers[valid], gaps[valid], color=color, lw=2, label=label)
    ax.axvspan(0, 30, color="#ffd700", alpha=0.07, label="Charge puddle region")
    ax.set_xlabel("Energy (meV)")
    ax.set_ylabel("Inner-band gap (meV)")
    ax.set_title("Gap vs Energy (anomaly probe)")
    ax.grid(True)
    ax.legend(fontsize=7, facecolor="#080812", edgecolor="#222244")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Baseline Model
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 1 — Baseline Hamiltonian")
print("═"*60)

baseline = Params()
bands_baseline = dispersion(K_ARRAY, baseline)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Step 1 — Baseline Continuum Hamiltonian  "
             f"(λ_VZ={baseline.lambdaVZ}, λ_R={baseline.lambdaR} meV)",
             fontsize=12, y=1.01)

plot_dispersion(axes[0], bands_baseline, K_ARRAY,
                title="Energy Dispersion E(k)")
plot_gap_vs_energy(axes[1], bands_baseline, K_ARRAY,
                   label="Baseline", color="#00d4ff")

# Print key values
gap_at_dirac = inner_band_gap(K_ARRAY, baseline)
k_near_dirac = np.argmin(np.abs(K_ARRAY))
print(f"  Inner gap at k=0   : {gap_at_dirac[k_near_dirac]:.4f} meV")
print(f"  Baseline score     : {gap_slope_score(baseline):+.4f} meV")

plt.tight_layout()
plt.savefig("outputs/step1_baseline_dispersion.png", dpi=150,
            bbox_inches="tight", facecolor="#0a0a0f")
plt.close()
print("  ✓ Saved: outputs/step1_baseline_dispersion.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Unreasonable ~20 meV Scenario
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 2 — Unreasonable ~20 meV Scenario")
print("═"*60)

unreasonable = Params.unreasonable()
bands_unreasonable = dispersion(K_ARRAY, unreasonable)

# Sweep λ_KM from 0 → 22 meV to show mechanism
sweep_vals = [0.12, 2.0, 5.0, 10.0, 15.0, 20.0]
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Step 2 — How λ_KM Bends the Band Structure\n"
             "Sweeping Kane-Mele coupling 0.12 → 20 meV", fontsize=12)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

for idx, lkm_val in enumerate(sweep_vals):
    ax = fig.add_subplot(gs[idx // 3, idx % 3])
    p  = Params(); p.lambdaKM = lkm_val; p.Delta = lkm_val * 0.9
    b  = dispersion(K_ARRAY, p)
    for i in range(4):
        ax.plot(K_ARRAY * 1000, b[:, i],
                color=BAND_COLORS[i], lw=1.8 if i in (1, 2) else 1.2)
    ax.fill_between(K_ARRAY * 1000, b[:, 1], b[:, 2],
                    color="#ff6b6b", alpha=0.15)
    ax.axhline(0, color="#334466", lw=0.6, ls="--")
    score = gap_slope_score(p)
    color = "#ff6b6b" if lkm_val > 5 else "#00ff9d"
    ax.set_title(f"λ_KM = {lkm_val} meV   score={score:+.2f}",
                 color=color, fontsize=9)
    ax.set_xlabel("k (10⁻³ Å⁻¹)", fontsize=8)
    ax.set_ylabel("E (meV)", fontsize=8)
    ax.grid(True)
    ax.tick_params(labelsize=7)

plt.savefig("outputs/step2_unreasonable_scenario.png", dpi=150,
            bbox_inches="tight", facecolor="#0a0a0f")
plt.close()

print(f"  Unreasonable score : {gap_slope_score(unreasonable):+.4f} meV")
print("  ✓ Saved: outputs/step2_unreasonable_scenario.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Optimizer
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 3 — Optimization Search")
print("═"*60)

# Run Differential Evolution
best_de, score_de, result_de, gen_log = run_differential_evolution(
    baseline, max_iter=80, popsize=10, physical_bound=5.0)

# Run Monte Carlo
best_mc, score_mc, mc_history = run_monte_carlo(
    baseline, n_iter=2000, physical_bound=5.0)

# Pick overall best
if score_de >= score_mc:
    best_overall, best_score, best_name = best_de, score_de, "Differential Evolution"
else:
    best_overall, best_score, best_name = best_mc, score_mc, "Monte Carlo"

bands_best    = dispersion(K_ARRAY, best_overall)
bands_best_mc = dispersion(K_ARRAY, best_mc)

# Save optimizer log
with open("outputs/optimizer_log.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["iteration", "score", "best_score", "lambdaKM", "Delta"])
    w.writeheader()
    w.writerows(mc_history)
print("  ✓ Saved: outputs/optimizer_log.csv")

# Plot results
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Step 3 — Optimization Results\n"
             "Searching physically reasonable parameters (±5 meV)",
             fontsize=12)

# MC convergence
scores_mc = [h["best_score"] for h in mc_history]
axes[0, 0].plot(scores_mc, color="#00ff9d", lw=1.5)
axes[0, 0].axhline(0, color="#ff6b6b", lw=0.8, ls="--", label="Baseline (flat gap)")
axes[0, 0].set_title("MC: Best Score vs Iteration")
axes[0, 0].set_xlabel("Iteration"); axes[0, 0].set_ylabel("Gap slope (meV)")
axes[0, 0].legend(fontsize=7, facecolor="#080812", edgecolor="#222244")
axes[0, 0].grid(True)

# DE convergence
if gen_log:
    axes[0, 1].plot([g["score"] for g in gen_log], color="#ffd700", lw=2)
    axes[0, 1].axhline(0, color="#ff6b6b", lw=0.8, ls="--")
    axes[0, 1].set_title("DE: Best Score vs Generation")
    axes[0, 1].set_xlabel("Generation"); axes[0, 1].set_ylabel("Gap slope (meV)")
    axes[0, 1].grid(True)

# Parameter scatter from MC
lkm_vals   = [h["lambdaKM"] for h in mc_history[::10]]
delta_vals = [h["Delta"]    for h in mc_history[::10]]
sc_vals    = [h["score"]    for h in mc_history[::10]]
sc = axes[0, 2].scatter(lkm_vals, delta_vals, c=sc_vals,
                         cmap="plasma", s=8, alpha=0.6)
plt.colorbar(sc, ax=axes[0, 2], label="Score (meV)")
axes[0, 2].set_xlabel("λ_KM (meV)"); axes[0, 2].set_ylabel("Δ (meV)")
axes[0, 2].set_title("Parameter Landscape")
axes[0, 2].grid(True)

# Dispersion: baseline vs best found
plot_dispersion(axes[1, 0], bands_baseline, K_ARRAY, title="Baseline Dispersion")
plot_dispersion(axes[1, 1], bands_best,     K_ARRAY,
                title=f"Best Found ({best_name})\nScore={best_score:+.4f} meV")

# Gap comparison
axes[1, 2].set_facecolor("#0d0d1a")
ec_b, g_b = compute_effective_gap(bands_baseline, K_ARRAY)
ec_o, g_o = compute_effective_gap(bands_best,     K_ARRAY)
ec_u, g_u = compute_effective_gap(bands_unreasonable, K_ARRAY)
axes[1, 2].plot(ec_b, g_b, color="#00d4ff",  lw=2, label="Baseline")
axes[1, 2].plot(ec_o, g_o, color="#00ff9d",  lw=2, label=f"Optimised (physical)")
axes[1, 2].plot(ec_u, g_u, color="#ff6b6b",  lw=1.5, ls="--", label="~20 meV (unreasonable)")
axes[1, 2].axvspan(0, 30, color="#ffd700", alpha=0.08)
axes[1, 2].set_title("Gap vs Energy Comparison")
axes[1, 2].set_xlabel("Energy (meV)"); axes[1, 2].set_ylabel("Gap (meV)")
axes[1, 2].legend(fontsize=7, facecolor="#080812", edgecolor="#222244")
axes[1, 2].grid(True)

plt.tight_layout()
plt.savefig("outputs/step3_optimizer_results.png", dpi=150,
            bbox_inches="tight", facecolor="#0a0a0f")
plt.close()
print("  ✓ Saved: outputs/step3_optimizer_results.png")

print(f"\n  Summary:")
print(f"    Baseline score         : {gap_slope_score(baseline):+.4f} meV")
print(f"    Optimised (DE)  score  : {score_de:+.4f} meV  "
      f"(λ_KM={best_de.lambdaKM:+.3f}, Δ={best_de.Delta:+.3f})")
print(f"    Optimised (MC)  score  : {score_mc:+.4f} meV  "
      f"(λ_KM={best_mc.lambdaKM:+.3f}, Δ={best_mc.Delta:+.3f})")
print(f"    Unreasonable scenario  : {gap_slope_score(unreasonable):+.4f} meV")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Charge Puddles
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 4 — Charge Puddle Simulation")
print("═"*60)

# Generate puddle landscape visualisation
x, y, V = generate_puddle_landscape(
    n_sites=150, box_size=80, n_puddles=25,
    puddle_strength=6.0, puddle_radius=8.0)

fig, ax = plt.subplots(figsize=(7, 6))
norm = TwoSlopeNorm(vmin=V.min(), vcenter=0, vmax=V.max())
im = ax.contourf(x, y, V, levels=40, cmap="RdBu_r", norm=norm)
ax.contour(x, y, V, levels=10, colors="white", alpha=0.15, linewidths=0.5)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Local potential V(r) [meV]")
ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
ax.set_title("Step 4 — Charge Puddle Landscape\n"
             "Random impurity potential near the Dirac point")
plt.tight_layout()
plt.savefig("outputs/step4_puddle_landscape.png", dpi=150,
            bbox_inches="tight", facecolor="#0a0a0f")
plt.close()
print("  ✓ Saved: outputs/step4_puddle_landscape.png")

# Disorder-averaged band structure
print("  Computing disorder-averaged dispersion (this takes ~20s)...")
bands_clean, bands_disordered = puddle_broadened_dispersion(
    K_ARRAY, baseline, puddle_strength=4.0, n_samples=30)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Step 4 — Effect of Charge Puddle Disorder on Band Structure",
             fontsize=12)

plot_dispersion(axes[0], bands_clean,      K_ARRAY,
                title="Clean Dispersion (no puddles)")
plot_dispersion(axes[1], bands_disordered, K_ARRAY,
                title="Disorder-Averaged Dispersion\n(charge puddles, W=4 meV)")

# Gap comparison: does disorder create apparent widening?
ec_c,  g_c  = compute_effective_gap(bands_clean,      K_ARRAY)
ec_d,  g_d  = compute_effective_gap(bands_disordered, K_ARRAY)

axes[2].plot(ec_c, g_c, color="#00d4ff", lw=2.5, label="Clean")
axes[2].plot(ec_d, g_d, color="#ff9f43", lw=2.5, label="With puddles (W=4 meV)")
axes[2].axvspan(0, 30, color="#ffd700", alpha=0.10, label="Charge puddle region")
axes[2].set_xlabel("Energy (meV)"); axes[2].set_ylabel("Inner-band gap (meV)")
axes[2].set_title("Gap vs Energy\nDoes disorder fake the anomaly?")
axes[2].legend(fontsize=8, facecolor="#080812", edgecolor="#222244")
axes[2].grid(True)

# Annotation
e_p30 = np.percentile(ec_c, 30)
e_p70 = np.percentile(ec_c, 70)
gap_low_clean     = g_c[ec_c <= e_p30].mean()
gap_low_disorder  = g_d[ec_d <= e_p30].mean()
gap_hi_clean      = g_c[ec_c >= e_p70].mean()
gap_hi_disorder   = g_d[ec_d >= e_p70].mean()

print(f"\n  Gap near Dirac point (E < 30 meV):")
print(f"    Clean     : {gap_low_clean:.3f} meV")
print(f"    Disordered: {gap_low_disorder:.3f} meV  (Δ = {gap_low_disorder - gap_low_clean:+.3f})")
print(f"\n  Gap away from Dirac (E > 50 meV):")
print(f"    Clean     : {gap_hi_clean:.3f} meV")
print(f"    Disordered: {gap_hi_disorder:.3f} meV  (Δ = {gap_hi_disorder - gap_hi_clean:+.3f})")

verdict = "SUPPORTS artefact hypothesis" if (gap_low_disorder - gap_low_clean) > 0.3 else "does NOT explain anomaly"
print(f"\n  Charge puddle effect: {verdict}")

plt.tight_layout()
plt.savefig("outputs/step4_charge_puddles.png", dpi=150,
            bbox_inches="tight", facecolor="#0a0a0f")
plt.close()
print("  ✓ Saved: outputs/step4_charge_puddles.png")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  ALL STEPS COMPLETE — Output files:")
for f in ["step1_baseline_dispersion.png",
          "step2_unreasonable_scenario.png",
          "step3_optimizer_results.png",
          "step4_puddle_landscape.png",
          "step4_charge_puddles.png",
          "optimizer_log.csv"]:
    path = f"outputs/{f}"
    size = os.path.getsize(path) // 1024
    print(f"    {path}  ({size} KB)")
print("═"*60)
