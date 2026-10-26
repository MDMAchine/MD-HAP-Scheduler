-- ============================================================================
-- SPDX-License-Identifier: GPL-3.0-or-later
-- Copyright (C) 2026 Alexander Allan (MDMAchine) -- A&E Concepts
--
-- This program is free software: you can redistribute it and/or modify
-- it under the terms of the GNU General Public License as published by
-- the Free Software Foundation, either version 3 of the License, or
-- (at your option) any later version.
--
-- This program is distributed in the hope that it will be useful,
-- but WITHOUT ANY WARRANTY; without even the implied warranty of
-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
-- GNU General Public License for more details: https://www.gnu.org/licenses/
-- ============================================================================

-- MD HAP Scheduler v1.0 — Hamiltonian Action-Principle
-- MDMAchine | A&E Concepts © 2026
--
-- Port of hap_scheduler_core.py calculate_hap_sigmas() to HOT-Step-CPP Lua.
--
-- WHAT THIS DOES:
--   Simulates a particle falling through a gravitational potential well with
--   atmospheric drag. Maps the particle's velocity to sigma step sizes.
--
--   velocity(t) = (1 + kinetic_energy * t) * exp(-damping_friction * t)
--
--   - kinetic_energy:   initial boost — stretches steps in the middle of the run
--     (particle accelerates as it falls into the well)
--   - damping_friction: atmospheric drag — compresses steps at the end
--     (particle slows as drag increases with velocity)
--
--   distance = cumsum(velocity) → normalize → map to sigma space
--
--   HIGH kinetic_energy:   more steps in the mid-sigma zone (structure formation)
--   HIGH damping_friction: more steps compressed toward the end (detail refinement)
--
-- This is the HAP component of the HT scheduler (used standalone here).
-- ============================================================================

scheduler = {
    name        = "md_hap",
    display     = "MD HAP (Hamiltonian Potential Well)",
    description = "Particle-in-potential-well sigma schedule. Kinetic energy stretches "
               .. "mid steps, damping friction compresses end steps. "
               .. "Port of hap_scheduler_core v1.0.",
    params      = {
        {
            key     = "kinetic_energy",
            type    = "slider",
            label   = "Kinetic Energy",
            default = 1.5,   -- matches Python/C++ default
            min     = 0.0,
            max     = 5.0,
            step    = 0.1,
            hint    = "Initial velocity boost. Stretches steps in the middle of the "
                   .. "trajectory (structure formation zone).",
        },
        {
            key     = "damping_friction",
            type    = "slider",
            label   = "Damping Friction",
            default = 3.0,   -- matches Python/C++ default
            min     = 0.0,
            max     = 8.0,
            step    = 0.1,
            hint    = "Atmospheric drag. Compresses steps toward the end (detail "
                   .. "refinement zone). Higher = more end compression.",
        },
    },
}

local EPSILON = 1e-6

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

function schedule(output, num_steps, shift)
    -- Fallback defaults match scheduler.params defaults above
    local ke = (params and params.kinetic_energy)  or 1.5
    local df = (params and params.damping_friction) or 3.0

    -- Compute velocity at each normalized time point
    local velocity = {}
    for i = 0, num_steps - 1 do
        local t = i / math.max(num_steps - 1, 1)
        local v = (1.0 + ke * t) * math.exp(-df * t)
        velocity[i] = math.max(v, EPSILON)  -- never negative
    end

    -- Integrate: cumulative distance
    local distance = {}
    distance[0] = 0.0
    local running = 0.0
    for i = 0, num_steps - 1 do
        running = running + velocity[i]
        distance[i + 1] = running
    end

    -- Normalize and map to sigma [1.0 → 0.0]
    local total = distance[num_steps]
    if total < EPSILON then total = EPSILON end

    local sigmas = {}
    for i = 0, num_steps do
        sigmas[i] = 1.0 - (distance[i] / total)
    end

    -- Enforce exact endpoints
    sigmas[0]         = 1.0
    sigmas[num_steps] = 0.0

    -- Shift warp
    if shift ~= 1.0 then
        for i = 0, num_steps do
            local t = sigmas[i]
            sigmas[i] = shift * t / (1.0 + (shift - 1.0) * t)
        end
    end

    for i = 0, num_steps - 1 do
        output[i] = sigmas[i]
    end
end
