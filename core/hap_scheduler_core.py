# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# █▓▒░                                                                     ░▒▓█
# █▓▒░ ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬ hap_scheduler_core v1.0.0 ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬ ░▒▓█
# █▓▒░                                                                     ░▒▓█
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ╠═ © 2026 Alexander Allan (MDMAchine) | A&E Concepts
# ╠═ License: GNU General Public License v3.0 (GPLv3)
# ║
# ║  This program is free software: you can redistribute it and/or modify
# ║  it under the terms of the GNU General Public License as published by
# ║  the Free Software Foundation, either version 3 of the License, or
# ║  (at your option) any later version.
# ║
# ║  GPL v3 — community free · closed-source commercial: contact for license
# ╠════════════════════════════════════════════════════════════════════════════
# ║ CORE RESPONSIBILITIES:
# ║   • Hamiltonian Action-Principle simulation (Kinetic Energy + Potential Well).
# ║   • Calculates exponential velocity decay (damping friction).
# ║   • Stateless processing (pure tensor math).
# ╚════════════════════════════════════════════════════════════════════════════

import torch

CONST_EPSILON = 1e-6


def calculate_hap_sigmas(steps, damping_friction, kinetic_energy, sigma_max, sigma_min):
    """
    Distributes sigma steps based on Hamiltonian mechanics (particle in a potential well).

    Args:
        steps:            Number of sampling steps.
        damping_friction: Atmospheric drag that slows the particle (compresses steps at the end).
        kinetic_energy:   Initial velocity curve (stretches steps in the middle).
        sigma_max:        Starting noise level (top of the gravity well).
        sigma_min:        Ending noise level (bottom of the well).

    Returns:
        tuple: (sigmas_tensor, analytics_string)
    """
    # 1. Safety Validation
    steps = max(1, steps)
    sigma_max = max(sigma_max, 0.01)
    safe_sigma_min = max(sigma_min, 0.0)

    # 2. Simulate Time Vector
    # t represents the normalized generation time from 0 to 1
    t = torch.linspace(0.0, 1.0, steps, dtype=torch.float32)

    # 3. Calculate Particle Velocity (Step Size Curve)
    # v(t) = (1 + kinetic_energy * t) * exp(-damping_friction * t)
    # Simulates a particle accelerating due to gravity, then hitting atmospheric drag.
    velocity = (1.0 + kinetic_energy * t) * torch.exp(-damping_friction * t)

    # Ensure velocity never goes negative or hits absolute zero (time must move forward)
    velocity = torch.clamp(velocity, min=CONST_EPSILON)

    # 4. Integrate Velocity → Cumulative Distance
    distance = torch.cumsum(velocity, dim=0)

    # 5. Prepend 0 for the exact start position
    distance = torch.cat([torch.tensor([0.0], dtype=torch.float32), distance])

    # 6. Normalize Distance to [0.0, 1.0]
    distance_norm = distance / distance[-1]

    # 7. Map to Sigma Space (Inverted: sigma goes from Max down to Min)
    sigmas = safe_sigma_min + (sigma_max - safe_sigma_min) * (1.0 - distance_norm)

    # 8. Enforce Exact Endpoints to prevent floating-point drift
    sigmas[0]  = sigma_max
    sigmas[-1] = 0.0 if safe_sigma_min == 0.0 else safe_sigma_min

    # Analytics Payload
    mid_idx  = steps // 2
    log_data = (
        f"HAP_POTENTIAL_WELL | "
        f"Damping: {damping_friction:.2f} | "
        f"Kinetic: {kinetic_energy:.2f} | "
        f"Mid-Sigma: {sigmas[mid_idx].item():.4f}"
    )

    return sigmas, log_data


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# Embedded Unit Tests
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
if __name__ == "__main__":
    print("🧪 Running Self-Tests for HAP Core...")

    # Test 1: correct output length
    s, log = calculate_hap_sigmas(20, 3.0, 1.5, 1.0, 0.0292)
    assert len(s) == 21, f"Sigma tensor length mismatch: got {len(s)}"

    # Test 2: final sigma endpoint — use tolerance to account for float32 precision
    # float32(0.0292) ≈ 0.029200000688..., not exactly == 0.0292 (float64)
    assert abs(s[-1].item() - 0.0292) < 1e-4, \
        f"Final sigma mismatch: got {s[-1].item():.6f}, expected ~0.0292"

    # Test 3: first sigma is sigma_max exactly
    assert s[0].item() == 1.0, \
        f"Initial sigma mismatch: got {s[0].item()}"

    # Test 4: sigma_min=0.0 path produces exact zero endpoint
    s_zero, _ = calculate_hap_sigmas(10, 3.0, 1.5, 1.0, 0.0)
    assert s_zero[-1].item() == 0.0, \
        f"Zero sigma_min path failed: got {s_zero[-1].item()}"

    # Test 5: physical mechanics — high damping compresses end steps vs start steps
    dt = s[:-1] - s[1:]
    assert dt[-1] < dt[0], "HAP failed to apply damping friction at the end of the well."

    # Test 6: all step deltas are positive (sigma is monotonically decreasing)
    assert torch.all(dt > 0), "Sigma schedule is not monotonically decreasing."

    print(f"✅ All Tests Passed | {log}")
