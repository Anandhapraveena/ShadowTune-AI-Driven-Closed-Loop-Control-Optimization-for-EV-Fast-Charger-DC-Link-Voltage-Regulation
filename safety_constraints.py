"""
Safety and Uncertainty Constraints for ShadowTune
Enforces:
1. Strict, single-valued epistemic uncertainty gating (threshold = 1.60)
2. Expanded parameter boundaries avoiding boundary-limited optimization
3. Explicit separation of voltage overshoot (rise above ref) and undershoot (sag below ref)
4. Full constraint verification including IAE and settling limits
5. Analytical Routh-Hurwitz control-theoretic stability checking
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any


# Single global uncertainty threshold for consistency across the entire codebase
MAX_ALLOWED_UNCERTAINTY: float = 1.60


@dataclass
class SafetyThresholds:
    # Expanded physical gain feasibility bounds (avoids boundary-limited optimization)
    kp_min: float = 0.05
    kp_max: float = 12.00        # Expanded from 8.0 to guarantee interior optimum discovery
    ki_min: float = 1.0
    ki_max: float = 200.0

    # Electrical safety thresholds
    max_overshoot_pct: float = 7.50      # Maximum permissible voltage rise above reference (%)
    max_undershoot_pct: float = 15.0     # Maximum permissible dynamic voltage sag below reference (%)
    max_settling_time_s: float = 0.045   # Maximum allowable settling time (45 ms)
    max_voltage_ripple_v: float = 2.20   # Maximum steady-state ripple (Volts peak-to-peak)
    max_iae: float = 4.00                # Maximum permissible Integral Absolute Error (V*s)

    # Epistemic prediction uncertainty threshold (harmonized to 1.60)
    max_allowed_uncertainty: float = MAX_ALLOWED_UNCERTAINTY

    # Minimum required Routh-Hurwitz stability margin
    min_routh_stability_margin: float = 0.05


class SafetyFilter:
    """
    Evaluates candidate gain predictions and actual simulated performance
    against electrical boundaries, Routh stability, and GP confidence levels.
    """

    def __init__(self, thresholds: Optional[SafetyThresholds] = None):
        self.cfg = thresholds or SafetyThresholds()

    def check_parameter_bounds(self, kp: float, ki: float) -> Tuple[bool, Optional[str]]:
        if not (self.cfg.kp_min <= kp <= self.cfg.kp_max):
            return False, f"Kp={kp:.3f} outside valid boundary [{self.cfg.kp_min}, {self.cfg.kp_max}]"
        if not (self.cfg.ki_min <= ki <= self.cfg.ki_max):
            return False, f"Ki={ki:.3f} outside valid boundary [{self.cfg.ki_min}, {self.cfg.ki_max}]"
        return True, None

    def check_routh_stability(
        self,
        kp: float,
        ki: float,
        C_dc: float = 0.0022,
        R_esr: float = 0.05,
        tau_i: float = 0.0008,
    ) -> Tuple[bool, float, str]:
        """
        Routh-Hurwitz analytical closed-loop stability verification.
        Characteristic polynomial: a3*s^3 + a2*s^2 + a1*s + a0 = 0
        """
        a3 = tau_i * C_dc
        a2 = C_dc * (1.0 + kp * R_esr)
        a1 = kp + ki * C_dc * R_esr
        a0 = ki

        if a3 <= 0 or a2 <= 0 or a1 <= 0 or a0 <= 0:
            return False, -1.0, "Negative characteristic polynomial coefficients (unstable)"

        # Routh array condition: (a2 * a1 - a3 * a0) > 0
        b1_num = a2 * a1 - a3 * a0
        margin = b1_num / (a2 * a1)

        if margin < self.cfg.min_routh_stability_margin:
            return False, margin, f"Insufficient Routh-Hurwitz stability margin ({margin:.4f} < {self.cfg.min_routh_stability_margin})"

        return True, margin, f"Stable with Routh margin {margin:.4f}"

    def evaluate_candidate(
        self,
        kp: float,
        ki: float,
        pred_cost_mean: float,
        pred_cost_std: float,
        pred_overshoot: Optional[float] = None,
        pred_undershoot: Optional[float] = None,
        pred_settling_time: Optional[float] = None,
        pred_ripple: Optional[float] = None,
        pred_iae: Optional[float] = None,
        C_dc: float = 0.0022,
        R_esr: float = 0.05,
    ) -> Tuple[bool, str]:
        """
        Validates if candidate (kp, ki) satisfies all safety constraints,
        analytical stability, and confidence requirements.
        """
        # 1. Parameter boundary check
        valid_bounds, reason = self.check_parameter_bounds(kp, ki)
        if not valid_bounds:
            return False, f"REJECTED: {reason}"

        # 2. Epistemic uncertainty check (strictly harmonized to 1.60)
        if pred_cost_std > self.cfg.max_allowed_uncertainty:
            return (
                False,
                f"REJECTED: Excessive GP uncertainty (std={pred_cost_std:.3f} > max={self.cfg.max_allowed_uncertainty:.2f})"
            )

        # 3. Routh-Hurwitz Analytical Stability Verification
        is_stable, margin, stab_reason = self.check_routh_stability(kp, ki, C_dc=C_dc, R_esr=R_esr)
        if not is_stable:
            return False, f"REJECTED: Analytical instability ({stab_reason})"

        # 4. Voltage overshoot check (rise above reference)
        if pred_overshoot is not None and pred_overshoot > self.cfg.max_overshoot_pct:
            return (
                False,
                f"REJECTED: Voltage overshoot risk ({pred_overshoot:.2f}% > {self.cfg.max_overshoot_pct:.2f}%)"
            )

        # 5. Voltage undershoot check (sag below reference)
        if pred_undershoot is not None and pred_undershoot > self.cfg.max_undershoot_pct:
            return (
                False,
                f"REJECTED: Excessive voltage sag ({pred_undershoot:.2f}% > {self.cfg.max_undershoot_pct:.2f}%)"
            )

        # 6. Settling time check
        if pred_settling_time is not None and pred_settling_time > self.cfg.max_settling_time_s:
            return (
                False,
                f"REJECTED: Sluggish settling time ({pred_settling_time*1000:.1f}ms > {self.cfg.max_settling_time_s*1000:.1f}ms)"
            )

        # 7. Voltage ripple check
        if pred_ripple is not None and pred_ripple > self.cfg.max_voltage_ripple_v:
            return (
                False,
                f"REJECTED: Excessive voltage ripple ({pred_ripple:.2f}V > {self.cfg.max_voltage_ripple_v:.2f}V)"
            )

        # 8. IAE constraint check (now explicitly verified)
        if pred_iae is not None and pred_iae > self.cfg.max_iae:
            return (
                False,
                f"REJECTED: Excessive Integral Absolute Error ({pred_iae:.3f} > {self.cfg.max_iae:.3f} V*s)"
            )

        return True, "ACCEPTED: Safe, stable and confident"

    def verify_simulated_metrics(self, metrics: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Tier 2 Post-Simulation Certification:
        Validates actual simulated metrics before gains can be certified and deployed.
        """
        if not metrics.get("is_stable", False):
            return False, "REJECTED_POST_SIM: Simulation indicated control loop instability"

        if not metrics.get("is_settled", True):
            return False, "REJECTED_POST_SIM: DC-link voltage failed to settle within tolerance band"

        if metrics.get("overshoot_pct", 0.0) > self.cfg.max_overshoot_pct:
            return False, f"REJECTED_POST_SIM: Actual overshoot ({metrics['overshoot_pct']:.2f}% > {self.cfg.max_overshoot_pct:.2f}%)"

        if metrics.get("undershoot_pct", 0.0) > self.cfg.max_undershoot_pct:
            return False, f"REJECTED_POST_SIM: Actual voltage sag ({metrics['undershoot_pct']:.2f}% > {self.cfg.max_undershoot_pct:.2f}%)"

        settling = metrics.get("settling_time_s")
        if settling is not None and settling > self.cfg.max_settling_time_s:
            return False, f"REJECTED_POST_SIM: Actual settling time ({settling*1000:.1f}ms > {self.cfg.max_settling_time_s*1000:.1f}ms)"

        if metrics.get("voltage_ripple_v", 0.0) > self.cfg.max_voltage_ripple_v:
            return False, f"REJECTED_POST_SIM: Actual voltage ripple ({metrics['voltage_ripple_v']:.2f}V > {self.cfg.max_voltage_ripple_v:.2f}V)"

        if metrics.get("iae", 0.0) > self.cfg.max_iae:
            return False, f"REJECTED_POST_SIM: Actual IAE ({metrics['iae']:.3f} > {self.cfg.max_iae:.3f} V*s)"

        return True, "CERTIFIED: All actual simulation metrics cleared safety envelope"
