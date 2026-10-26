/**
 * hap_scheduler_core.hpp
 * MD HAP Scheduler — Hamiltonian Action-Principle
 * Header-only C++17 port of hap_scheduler_core.py v1.0.0
 *
 * © 2026 Alexander Allan (MDMAchine) | A&E Concepts
 * GPL v3 — community free · closed-source commercial: contact for license
 *
 * WHAT THIS DOES:
 *   Simulates a particle falling through a gravitational potential well with
 *   atmospheric drag. Maps particle velocity to sigma step sizes.
 *
 *   velocity(t) = (1 + kinetic_energy * t) * exp(-damping_friction * t)
 *
 *   - kinetic_energy:   initial boost — stretches steps in the middle (structure zone)
 *   - damping_friction: atmospheric drag — compresses steps at the end (detail zone)
 *
 *   distance = cumsum(velocity) → normalize → map to sigma space [sigma_max → sigma_min]
 */

#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace md_hap {

static constexpr float EPSILON = 1e-6f;

struct HAPResult {
    std::vector<float> sigmas;   // length = steps + 1
    std::string        analytics;
};

/**
 * calculate_hap_sigmas
 *
 * @param steps           Number of sampling steps
 * @param damping_friction Atmospheric drag (compresses end steps)
 * @param kinetic_energy  Initial velocity boost (stretches mid steps)
 * @param sigma_max       Starting noise level (top of gravity well)
 * @param sigma_min       Ending noise level (bottom of well), 0.0 = fully clean
 * @return HAPResult with sigmas vector (size steps+1) and analytics string
 */
inline HAPResult calculate_hap_sigmas(
    int   steps,
    float damping_friction,
    float kinetic_energy,
    float sigma_max,
    float sigma_min)
{
    // 1. Safety validation
    steps     = std::max(1, steps);
    sigma_max = std::max(sigma_max, 0.01f);
    float safe_sigma_min = std::max(sigma_min, 0.0f);

    // 2. Simulate time vector t ∈ [0, 1]
    std::vector<float> t(steps);
    for (int i = 0; i < steps; ++i) {
        t[i] = (steps > 1) ? static_cast<float>(i) / static_cast<float>(steps - 1) : 0.0f;
    }

    // 3. Calculate particle velocity
    // v(t) = (1 + kinetic_energy * t) * exp(-damping_friction * t)
    std::vector<float> velocity(steps);
    for (int i = 0; i < steps; ++i) {
        float v = (1.0f + kinetic_energy * t[i]) * std::exp(-damping_friction * t[i]);
        velocity[i] = std::max(v, EPSILON);
    }

    // 4. Integrate velocity → cumulative distance (prepend 0)
    std::vector<float> distance(steps + 1, 0.0f);
    for (int i = 0; i < steps; ++i) {
        distance[i + 1] = distance[i] + velocity[i];
    }

    // 5. Normalize distance to [0, 1]
    float total = distance[steps];
    if (total < EPSILON) total = EPSILON;
    for (auto& d : distance) d /= total;

    // 6. Map to sigma space (inverted: sigma goes max → min)
    std::vector<float> sigmas(steps + 1);
    for (int i = 0; i <= steps; ++i) {
        sigmas[i] = safe_sigma_min + (sigma_max - safe_sigma_min) * (1.0f - distance[i]);
    }

    // 7. Enforce exact endpoints
    sigmas[0]     = sigma_max;
    sigmas[steps] = (sigma_min <= 0.0f) ? 0.0f : safe_sigma_min;

    // Analytics
    int mid_idx = steps / 2;
    std::ostringstream ss;
    ss << "HAP_POTENTIAL_WELL | Damping: " << damping_friction
       << " | Kinetic: " << kinetic_energy
       << " | Mid-Sigma: " << sigmas[mid_idx];

    return HAPResult{ sigmas, ss.str() };
}

} // namespace md_hap
