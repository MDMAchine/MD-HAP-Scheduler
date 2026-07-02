# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes / HAPScheduler — WRAPPER v1.0.1 (GPLv3) ████▓▒░
# © 2026 MDMAchine / A&E Concepts
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

import io
import logging
import os
import sys
import time

import torch

# =============================================================================
# == Core Discovery
# =============================================================================
def _find_core_paths():
    """Locates the core/ directory regardless of install nesting depth."""
    current = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current, "core"),
        os.path.join(current, "..", "core"),
        os.path.join(current, "..", "..", "core"),
    ]
    return list(dict.fromkeys([
        os.path.abspath(c) for c in candidates if os.path.exists(c)
    ]))

for _loc in _find_core_paths():
    if _loc not in sys.path:
        sys.path.insert(0, _loc)

HAP_CORE_LOADED = False
HAP_CORE_TYPE   = "NONE"
hap_core        = None

try:
    import hap_scheduler_core_bin as hap_core  # compiled binary (optional)
    HAP_CORE_LOADED = True
    HAP_CORE_TYPE   = "BINARY"
except ImportError:
    try:
        import hap_scheduler_core as hap_core  # pure-Python fallback
        HAP_CORE_LOADED = True
        HAP_CORE_TYPE   = "SOURCE"
    except ImportError:
        HAP_CORE_LOADED = False

# =============================================================================
# == Optional Visualization Dependencies
# =============================================================================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    VIS_AVAILABLE = True
except ImportError:
    VIS_AVAILABLE = False

CONST_PLOT_DPI        = 120
CONST_WAVEFORM_COLOR  = "#87CEEB"         # sky blue — sigma path
CONST_VELOCITY_COLOR  = "mediumseagreen"  # velocity overlay


# =============================================================================
# == Performance Profiler (MD_Nodes Standard)
# =============================================================================
class PerformanceProfiler:
    def __init__(self, enabled: bool = True):
        self.enabled     = enabled
        self.timings     = {}
        self.start_times = {}

    def start(self, op: str):
        if self.enabled:
            self.start_times[op] = time.perf_counter()

    def stop(self, op: str):
        if self.enabled and op in self.start_times:
            elapsed = time.perf_counter() - self.start_times.pop(op)
            self.timings.setdefault(op, []).append(elapsed)

    def print_report(self):
        if not self.enabled or not self.timings:
            return
        print("\n⏱️  PERFORMANCE:")
        for op, times in sorted(self.timings.items()):
            print(f"    • {op:<20}: {sum(times)/len(times):.4f}s avg")


# =============================================================================
# == Node
# =============================================================================
class MD_HAPScheduler:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps":            ("INT",   {"default": 20,      "min": 1,   "max": 1000}),
                "damping_friction": ("FLOAT", {"default": 3.0,     "min": 0.0, "max": 10.0, "step": 0.1}),
                "kinetic_energy":   ("FLOAT", {"default": 1.5,     "min": 0.0, "max": 10.0, "step": 0.1}),
                "sigma_max":        ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1000.0, "step": 0.1}),
                "sigma_min":        ("FLOAT", {"default": 0.0292,  "min": 0.0, "max": 10.0,  "step": 0.0001}),
                "debug_mode":       (["0 - Silent", "1 - Info", "2 - Verbose"], {"default": "1 - Info"}),
                "enable_profiling": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES  = ("SIGMAS", "IMAGE", "STRING")
    RETURN_NAMES  = ("sigmas", "visual_curve", "analytics")
    FUNCTION      = "generate_sigmas"
    CATEGORY      = "MD_Nodes/Schedulers"

    # -------------------------------------------------------------------------
    def _plot_hap_curve(self, sigmas: torch.Tensor, damping: float, kinetic: float) -> torch.Tensor:
        """Renders the HAP potential well as a dark-background matplotlib image."""
        if not VIS_AVAILABLE:
            return torch.zeros((1, 64, 64, 3))
        try:
            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(10, 4))
            s_np      = sigmas.cpu().numpy()
            steps_idx = np.arange(len(s_np))

            # Sigma decay (the potential well)
            ax.plot(steps_idx, s_np, color=CONST_WAVEFORM_COLOR, linewidth=2.5,
                    label="Sigma Path")
            ax.fill_between(steps_idx, s_np, alpha=0.15, color=CONST_WAVEFORM_COLOR)

            # Velocity (Hamiltonian flux) on a twin axis
            velocity = np.abs(np.diff(s_np))
            ax2 = ax.twinx()
            ax2.plot(steps_idx[:-1], velocity, color=CONST_VELOCITY_COLOR,
                     linestyle="--", alpha=0.6, label="Kinetic Velocity")

            ax.set_title(f"HAP Potential Well  |  γ={damping}  ω={kinetic}",
                         fontsize=12, color="white", pad=20)
            ax.set_xlabel("Step Index", fontsize=9)
            ax.set_ylabel("Sigma Intensity",  fontsize=9, color=CONST_WAVEFORM_COLOR)
            ax2.set_ylabel("Flux Velocity",   fontsize=9, color=CONST_VELOCITY_COLOR)
            ax.grid(True, which="both", linestyle=":", alpha=0.2)
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=CONST_PLOT_DPI)
            plt.close(fig)
            buf.seek(0)

            img = Image.open(buf).convert("RGB")
            return torch.from_numpy(
                np.array(img).astype(np.float32) / 255.0
            ).unsqueeze(0)

        except Exception as e:
            logging.error(f"[MD_HAP] Visualization failed: {e}")
            return torch.zeros((1, 64, 64, 3))

    # -------------------------------------------------------------------------
    def generate_sigmas(
        self,
        steps:            int,
        damping_friction: float,
        kinetic_energy:   float,
        sigma_max:        float,
        sigma_min:        float,
        debug_mode:       str,
        enable_profiling: bool,
    ):
        debug_level = int(debug_mode.split(" ")[0])
        profiler    = PerformanceProfiler(enabled=(enable_profiling or debug_level >= 1))
        profiler.start("total_compute")

        sigmas    = None
        used_mode = "None"
        log_data  = "NONE"

        # 1. Local core
        if HAP_CORE_LOADED:
            try:
                sigmas, log_data = hap_core.calculate_hap_sigmas(
                    steps, damping_friction, kinetic_energy, sigma_max, sigma_min
                )
                used_mode = f"Local ({HAP_CORE_TYPE})"
            except Exception as e:
                logging.error(f"[MD_HAP] Core error: {e}")

        # 2. Failsafe linear
        if sigmas is None:
            logging.warning("[MD_HAP] Core unavailable — falling back to linear schedule.")
            sigmas    = torch.linspace(sigma_max, sigma_min, steps + 1)
            used_mode = "Failsafe (Linear)"

        # 3. Visualization
        profiler.start("visual_render")
        visual_image = self._plot_hap_curve(sigmas, damping_friction, kinetic_energy)
        profiler.stop("visual_render")

        profiler.stop("total_compute")

        if debug_level >= 1:
            print(f"\n{'='*60}")
            print(f"📊 [MD_HAP] ANALYTICS REPORT")
            print(f"{'='*60}")
            print(f"🪐 Mode: {used_mode}  |  γ: {damping_friction}  |  ω: {kinetic_energy}")
            profiler.print_report()
            print("=" * 60)

        render_t  = profiler.timings.get("visual_render", [0])[0]
        analytics = f"✨ [HAP] {used_mode}\n{log_data}\nRender: {render_t:.3f}s"
        return (sigmas, visual_image, analytics)


# =============================================================================
NODE_CLASS_MAPPINGS        = {"MD_HAPScheduler": MD_HAPScheduler}
NODE_DISPLAY_NAME_MAPPINGS = {"MD_HAPScheduler": "MD: HAP Scheduler (Hamiltonian) 🪐"}
