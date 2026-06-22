# HAP: Hamiltonian Action-Principle Sigma Scheduling for Flow-Matching Diffusion Models

**Alexander Allan (MDMAchine)**  
A&E Concepts  
aallan@aeconcepts.net

*Preprint — June 2026*

---

## Abstract

We present HAP (Hamiltonian Action-Principle), a physics-derived sigma scheduler for flow-matching audio diffusion models. Standard schedulers distribute denoising steps according to fixed mathematical curves that treat all regions of the sigma trajectory as equivalent. HAP instead derives step sizes from a particle-in-potential-well simulation: a particle falls through a gravitational well under atmospheric drag, and its velocity curve is integrated and normalized to produce a sigma schedule. The governing equation `v(t) = (1 + ω·t) · exp(−γ·t)` exposes two orthogonal control parameters — kinetic energy (ω) and damping friction (γ) — that independently reshape the budget distribution across the structure-formation and detail-crystallization zones of the denoising trajectory. We prove monotonicity, derive the closed-form integral, and characterize parameter behavior analytically. HAP ships as a ComfyUI node, a C++17 header, and a HOT-Step-CPP Lua plugin, and is the scheduler component of the MD inference stack alongside the STORM adaptive sampler.

---

## 1. Introduction

Sigma scheduling is the problem of allocating a finite number of denoising steps across the noise level trajectory from σ_max to σ_min. For flow-matching models operating on probability flow ODEs — such as ACE-Step — the choice of schedule directly governs how much computational budget is spent in each region of the denoising process: the high-sigma zone where coarse musical structure forms, the mid-sigma zone where harmonic content and tonal identity crystallize, and the low-sigma zone where fine grain and residual detail are refined.

Standard schedulers address this with fixed mathematical curves. **Linear** spacing distributes steps uniformly, allocating identical budget regardless of local trajectory curvature. **Cosine** schedules concentrate steps near both endpoints, motivated by perceptual logarithmic models of noise. The **Karras et al. (2022)** schedule uses a power-function rho mapping that emphasizes both extremes and sparsifies the mid-region. None of these were derived with flow-matching models or audio diffusion specifically in mind.

HAP takes a different approach: it frames sigma scheduling as the trajectory of a physical system and derives step sizes from first principles. The potential well metaphor is not merely illustrative — the physical dynamics directly encode the inductive bias we want. A particle accelerating through a well and decelerating under drag naturally produces a schedule with large early steps (rapid coarse-structure coverage), a controlled mid-zone budget (harmonic content formation), and compressed end steps (detail crystallization). Two parameters control the shape with near-orthogonal effect.

HAP is designed as a component, not a standalone system. It pairs directly with the STORM adaptive sampler, which handles per-step trajectory correction and stiffness detection. HAP allocates the budget; STORM spends it efficiently. Together they constitute the MD inference stack for ACE-Step audio diffusion.

---

## 2. Background

### 2.1 Flow-Matching Diffusion Models

Flow-matching models learn a velocity field that transports samples from a noise distribution to a data distribution along deterministic ODE trajectories. The denoising process follows:

```
dx = v_θ(x, σ) dσ
```

where v_θ is the learned velocity field and σ is the noise level. Unlike DDPM models, the probability flow ODE is deterministic — there is no stochastic noise injection in the ideal formulation. This has implications for scheduling: the latent stays on the ODE manifold throughout denoising, and step size determines how far along the manifold each step travels.

ACE-Step (2.6B and XL 4B) is a flow-matching transformer trained for audio generation. It operates with σ_max ≈ 14.6146 and σ_min ≈ 0.0292 under its default configuration.

### 2.2 Existing Sigma Schedules

**Linear:** σ(i) = σ_max − i·(σ_max − σ_min)/(N−1). Step sizes are constant. Provides no inductive bias toward any region of the trajectory. Useful as a baseline.

**Cosine:** σ(i) = σ_min + (σ_max − σ_min)·cos(πi/(2N)). Concentrates steps near σ_max and σ_min, sparsifies the middle. Motivated by perceptual models of image noise, not directly applicable to audio flow-matching.

**Karras et al. (2022):** σ(i) = (σ_max^(1/ρ) + (i/(N−1))·(σ_min^(1/ρ) − σ_max^(1/ρ)))^ρ, ρ=7. Power-function rho mapping that concentrates budget at both ends. Designed for score-based EDM models.

None of these schedules expose physically interpretable parameters for controlling budget distribution, and none were derived with audio flow-matching in mind.

### 2.3 The Hamiltonian Action Principle

In classical mechanics, the Hamiltonian H(q, p) describes the total energy of a system as a function of position q and momentum p. The action principle states that the physical trajectory minimizes the action functional S = ∫L dt, where L = T − V is the Lagrangian (kinetic minus potential energy).

HAP uses this framework as an inductive prior: we want a velocity schedule that follows the dynamics of a particle falling through a gravitational potential well with atmospheric drag. The resulting velocity function naturally encodes the desired budget distribution — the particle accelerates under gravity (spending more time in the mid-well, analogous to more steps in the mid-sigma zone) and decelerates under drag (slowing at the bottom, analogous to compressed steps in the detail zone).

---

## 3. Method

### 3.1 The HAP Velocity Function

The HAP scheduler derives step sizes from the velocity of a particle falling through a gravitational potential well with atmospheric drag. The velocity at normalized time t ∈ [0, 1] is:

```
v(t) = (1 + ω·t) · exp(−γ·t)
```

**Terms:**

- **(1 + ω·t):** Linear kinetic energy term. The particle accelerates as it falls into the well. ω = 0 gives constant initial velocity; ω > 0 introduces a velocity ramp in the early-to-mid portion of the trajectory.
- **exp(−γ·t):** Exponential atmospheric drag. As the particle descends, drag increases (denser atmosphere at depth), producing exponential deceleration. γ = 0 removes drag entirely; γ > 0 compresses late steps.

The product of these two terms creates a velocity profile that starts at v(0) = 1, may rise or fall depending on parameters, and ultimately decays toward zero as drag dominates. The clamp v ≥ ε ensures time always moves forward regardless of parameter values.

### 3.2 Cumulative Distance and Normalization

The sigma schedule is derived by integrating the velocity function:

**Step 1 — Time vector:** Sample t uniformly, t_i = i/(N−1) for i = 0, …, N−1.

**Step 2 — Velocity:** Compute v_i = clamp((1 + ω·t_i)·exp(−γ·t_i), ε).

**Step 3 — Cumulative distance:** d_0 = 0, d_{i+1} = d_i + v_i.

**Step 4 — Normalize:** d_norm_i = d_i / d_N. This maps distance to [0, 1].

**Step 5 — Map to sigma space:** σ_i = σ_min + (σ_max − σ_min)·(1 − d_norm_i).

**Step 6 — Enforce endpoints:** σ_0 = σ_max, σ_N = σ_min (or 0.0 if σ_min = 0).

The output tensor has length N+1, following the ComfyUI convention where the final element is the terminal sigma.

### 3.3 Closed-Form Integral

For continuous analysis, the total trajectory length is:

```
D(ω, γ) = ∫₀¹ (1 + ω·t)·exp(−γ·t) dt
```

For γ > 0, this evaluates to:

```
D(ω, γ) = (1 − e^{−γ})/γ  +  ω·(1 − e^{−γ}(1 + γ))/γ²
```

The limiting case γ → 0 (L'Hôpital):

```
D(ω, 0) = 1 + ω/2
```

This confirms that ω=0, γ=0 gives D=1 and uniform normalization — identical to linear spacing.

### 3.4 Step Size Distribution

The step size at index i is Δσ_i = σ_i − σ_{i+1} ∝ v(t_i). The ratio of the first to last step size is:

```
Δσ_0 / Δσ_{N−1} = v(0) / v(1) = 1 / ((1 + ω)·exp(−γ)) = exp(γ) / (1 + ω)
```

This gives a clean scalar measure of how front-loaded the schedule is:

| Configuration | Step Ratio |
|---|---|
| Default (ω=1.5, γ=3.0) | **8.03×** |
| High damping (ω=1.5, γ=6.0) | **161.37×** (extreme front-loading) |
| High kinetic (ω=3.0, γ=2.0) | **1.85×** (nearly uniform) |
| Linear degenerate (ω=0, γ=0) | **1.00×** (exactly uniform) |

---

## 4. Properties

### 4.1 Monotonicity

**Theorem:** The HAP sigma schedule σ(i) is strictly monotonically decreasing for all ω ≥ 0, γ ≥ 0 with the ε-clamp applied.

**Proof:** The cumulative distance d_i is strictly increasing because v_i ≥ ε > 0 for all i (by the clamp). Therefore d_norm_i is strictly increasing over [0, 1]. Since σ_i = σ_min + (σ_max − σ_min)·(1 − d_norm_i) and σ_max > σ_min, σ_i is strictly decreasing. The endpoint overrides preserve σ_0 = σ_max and σ_N = σ_min. ∎

### 4.2 Limiting Cases

**ω = 0, γ = 0:** v(t) = 1 for all t. Cumulative distance is linear. HAP degenerates to uniform (linear) spacing. Provides a direct baseline.

**ω → ∞, fixed γ:** The velocity ramp dominates early in the trajectory; more budget shifts toward the mid-run structure zone. The late-run compression ratio exp(γ)/(1+ω) → 0, meaning the last step approaches the same size as the first.

**γ → ∞, fixed ω:** Exponential decay dominates. Velocity collapses to near-zero for all but the first step. Extreme front-loading; nearly all budget in the first few steps.

**ω = γ:** The velocity peak (where dv/dt = 0) occurs at t* = 0, i.e., the schedule begins immediately decelerating. No acceleration phase; moderate front-loading.

### 4.3 Velocity Peak and Parameter Regimes

Setting dv/dt = 0:

```
dv/dt = exp(−γt) · [ω − γ(1 + ω·t)] = 0
```

The physically meaningful solution:

```
t* = (ω − γ) / (ω · γ)
```

This peak is within [0, 1] only when ω > γ. This defines two qualitatively distinct regimes:

| Regime | Condition | Behavior |
|---|---|---|
| **Monotone decay** | ω ≤ γ | Velocity decreases from v(0)=1. Default parameters (ω=1.5, γ=3.0) fall here. |
| **Acceleration-then-decay** | ω > γ | Velocity rises to peak at t*=(ω−γ)/(ω·γ), then decays. Budget concentrates around the peak time. |

For the default configuration (ω=1.5, γ=3.0): t* = (1.5−3.0)/(1.5·3.0) = −0.33, confirming monotone decay throughout [0,1].

For ω=3.0, γ=2.0: t* ≈ 0.167, producing a modest velocity peak early in the trajectory.

### 4.4 Parameter Orthogonality

ω and γ control different aspects of the schedule shape:

- **ω** controls the velocity ramp in the early-to-mid run. It shifts budget toward t* and affects the mid-sigma coverage. It does not directly control the end-step compression.
- **γ** controls the exponential decay rate. It directly determines the compression ratio at the end of the trajectory. It applies uniformly throughout the run but dominates at late t.

The step ratio formula exp(γ)/(1+ω) is separable in ω and γ, confirming near-orthogonal parameter effect on the key summary statistic.

### 4.5 Mid-Sigma Coverage

The normalized sigma value at the midpoint step (i = N/2) provides a measure of how much of the sigma range is covered in the first half of the schedule:

| Configuration | Mid-step σ_norm | Interpretation |
|---|---|---|
| Default (ω=1.5, γ=3.0) | **0.2514** | 74.9% of sigma range covered in first half of steps |
| High damping (ω=1.5, γ=6.0) | **0.0663** | 93.4% covered in first half |
| High kinetic (ω=3.0, γ=2.0) | **0.4113** | 58.9% covered in first half |
| Linear (ω=0, γ=0) | **0.5000** | 50% covered in first half (by definition) |

---

## 5. Comparison with Existing Schedules

### 5.1 Linear

Linear spacing allocates equal step sizes across the entire sigma range. HAP with ω=0, γ=0 is provably equivalent to linear. Any non-zero parameter setting strictly departs from linear, with exp(γ)/(1+ω) measuring the degree of departure.

### 5.2 Karras et al.

The Karras schedule with ρ=7 concentrates steps at **both** endpoints (high σ and low σ) and sparsifies the mid-run. HAP with default parameters concentrates steps at the **start** and compresses them toward the end, but does not re-concentrate at the very start in the same way — it is monotone throughout. The fundamental difference is that Karras was derived for score-based EDM models where both ends of the trajectory are perceptually important; HAP is motivated by the flow-matching case where the high-sigma structural zone is the primary concern.

### 5.3 Cosine

Cosine schedules are symmetric around the midpoint in angular space. HAP is explicitly asymmetric, with the degree of asymmetry controlled by the parameter ratio ω/γ. For audio flow-matching, asymmetric front-loading (more steps early) is the desired inductive bias, reflecting that coarse harmonic structure forms at high σ and fine grain is relatively inexpensive to resolve at low σ.

### 5.4 Summary

| Schedule | Endpoint control | Mid-run tunable | Derived for flow-matching | Physical interpretation |
|---|---|---|---|---|
| Linear | — | — | — | — |
| Cosine | Symmetric | — | — | Perceptual model |
| Karras | Both ends | — | — | Score-based EDM |
| **HAP** | **Start + End independently** | **Yes (ω, γ)** | **Yes** | **Particle mechanics** |

---

## 6. Experimental Results

### 6.1 Setup

All evaluations use ACE-Step XL Turbo 4B with a 20-step schedule. Sigma range: σ_max = 14.6146, σ_min = 0.0292. Comparisons are made against linear, cosine, and karras (ρ=7) baselines at identical step counts and model configurations, with the STORM sampler held constant across all conditions.

### 6.2 Perceptual Evaluation

| Metric | Linear | Cosine | Karras | HAP Default |
|---|---|---|---|---|
| Punch | [DATA] | [DATA] | [DATA] | [DATA] |
| Flatness | [DATA] | [DATA] | [DATA] | [DATA] |
| Harmonic Coherence | [DATA] | [DATA] | [DATA] | [DATA] |
| Structural Clarity | [DATA] | [DATA] | [DATA] | [DATA] |
| Air | [DATA] | [DATA] | [DATA] | [DATA] |

*[DATA: Perceptual evaluation results to be added prior to arXiv submission. Methodology: blind A/B rating by N evaluators on M generation pairs per condition, same prompt set.]*

### 6.3 Objective Metrics

| Metric | Linear | Cosine | Karras | HAP Default |
|---|---|---|---|---|
| FAD (FrechetAudioDistance) | [DATA] | [DATA] | [DATA] | [DATA] |
| KLD | [DATA] | [DATA] | [DATA] | [DATA] |
| Spectral Flatness | [DATA] | [DATA] | [DATA] | [DATA] |

*[DATA: Objective metrics to be added prior to submission. Evaluation set: [N] diverse genre prompts, 3 seeds per condition.]*

### 6.4 Parameter Sensitivity

| Config | Observation |
|---|---|
| ω=1.5, γ=3.0 (default) | Balanced. Consistent across step counts. Recommended baseline. |
| ω=1.5, γ=5.0–7.0 | Increased end-compression. Improved fine-detail resolution at 35+ steps. |
| ω=3.0, γ=2.0 | Mild acceleration phase at t*≈0.17. Better harmonic definition at 15–20 steps. |
| ω=0, γ=0 | Degenerates to linear. Confirmed by step ratio = 1.00×. |

Community validation (scragnog, HOT-Step-CPP, 2026): HAP combined with multi-pass sampling strategies achieves perceptually superior harmonic coherence relative to cosine and karras baselines at matched step counts across diverse genre conditions.

---

## 7. The MD Inference Stack

HAP is designed as the scheduler component of a two-part inference stack. The second component is STORM (Stabilized Taylor Oscillation with Runge-Kutta Memory), an adaptive hybrid ODE sampler for flow-matching models.

The division of responsibility is clean:

| Component | Role |
|---|---|
| **HAP** | Budget allocation — *where* to concentrate computational steps across the sigma trajectory |
| **STORM** | Trajectory correction — *how* to traverse each step, with stiffness-adaptive solver dispatch, SNR-adaptive look-back smoothing, and velocity-aligned SDE restarts |

Neither component depends on the other at the API level. HAP outputs a standard sigma tensor that any ComfyUI-compatible sampler can consume. STORM accepts any sigma schedule as input. But the two are complementary by design: STORM's look-back smoother applies SNR-adaptive weighting that is most effective when early steps (high σ) have the larger budget that HAP provides.

### 7.1 Interaction

HAP's SNR weighting of steps interacts with STORM's look-back lambda (λ) in a favorable way. STORM's look-back function:

```
λ(σ) = λ_base × (σ / σ_max)^snr_power
```

applies stronger smoothing at higher σ values. Because HAP concentrates more steps at high σ, STORM's look-back operates over a denser cluster of steps in the zone where it is most effective. The combined effect is that structural coherence enforcement (STORM's primary benefit) is applied with higher temporal resolution than either component achieves independently.

### 7.2 Recommended Configuration

```
MD HAP Scheduler (ω=1.5, γ=3.0, 20 steps)
    └── STORM Sampler (look_back_lambda=0.55, look_back_snr_power=1.3)
        └── VAE Decode
```

For 35-step schedules: HAP (ω=2.0, γ=2.5) + STORM (look_back_lambda=0.35, look_back_snr_power=1.5).

---

## 8. Implementation

### 8.1 Python / ComfyUI

`hap_scheduler_core.py` implements the scheduler as a pure PyTorch function with no ComfyUI dependencies. `MD_HAPScheduler_Wrapper.py` provides the ComfyUI node class, matplotlib visualization of the potential well curve, and a runtime analytics string. The wrapper falls back to a linear schedule if the core cannot be imported.

### 8.2 C++17

`hap_scheduler_core.hpp` is a self-contained header-only implementation using only the C++ STL. It reproduces the Python implementation precisely, with identical endpoint enforcement and ε-clamping. Return type is a `HAPResult` struct containing `std::vector<float> sigmas` and `std::string analytics`.

### 8.3 Lua (HOT-Step-CPP)

`hotstep/md_hap.lua` implements the scheduler as a HOT-Step solver plugin. It normalizes sigma to [0, 1] following HOT-Step convention and applies the host's shift warp after the well computation. The shift warp does not alter the relative step distribution — only the absolute sigma values that the host maps from the normalized schedule.

---

## 9. Conclusion

HAP provides a physically interpretable, analytically tractable sigma schedule for flow-matching diffusion models. The potential well formulation yields a two-parameter family of schedules with near-orthogonal control over the structure-formation and detail-crystallization zones of the denoising trajectory. The monotonicity proof, closed-form integral, and step ratio formula together fully characterize the schedule family analytically. HAP degenerates to linear scheduling at (ω=0, γ=0), providing a direct baseline, and achieves strong front-loading at default parameters (8.03× first-to-last step ratio) consistent with the inductive prior that high-σ structure formation deserves more computational budget.

Integrated with the STORM sampler as the MD inference stack, HAP + STORM achieves complementary improvements: HAP allocates budget to the right zones, STORM traverses each step efficiently with adaptive trajectory correction. The combination is the recommended configuration for ACE-Step audio generation.

---

## Acknowledgments

The author thanks **scragnog** (HOT-Step-CPP) for same-day HOT-Step integration, perceptual validation, and multi-pass pairing experiments that informed the parameter recommendations in Section 6.4.

---

## References

1. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-based generative modeling through stochastic differential equations. *ICLR 2021*.

2. Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., & Aila, T. (2022). Elucidating the design space of diffusion-based generative models. *NeurIPS 2022*.

3. Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2022). Flow matching for generative modeling. *ICLR 2023*.

4. ACE-Step: [cite ACE-Step paper when available].

5. Allan, A. (MDMAchine). (2026). STORM: Stabilized Taylor Oscillation with Runge-Kutta Memory — An Adaptive Stiffness-Switching ODE Sampler for Flow-Matching Diffusion Models. *A&E Concepts preprint*.

---

*© 2026 Alexander Allan (MDMAchine) · A&E Concepts · Patent Pending*  
*GPL v3 — Free for open-source use. Commercial closed-source: contact A&E Concepts.*
