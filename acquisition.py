"""
Acquisition Function and Candidate Prioritization Module
Implements Constrained Lower Confidence Bound (cLCB) with uncertainty gating
and multi-objective Pareto prioritization.
"""

import math
from typing import List, Tuple, Dict, Any, Optional
from gp_model import GaussianProcessRegressor
from safety_constraints import SafetyFilter, SafetyThresholds


class CandidatePrioritizer:
    """
    Selects and prioritizes untested PI gain candidates (Kp, Ki)
    using GP surrogate predictions, uncertainty estimates, and safety constraints.
    """

    def __init__(
        self,
        gp_model: GaussianProcessRegressor,
        safety_filter: SafetyFilter,
        beta: float = 1.8,  # Exploration trade-off coefficient
    ):
        self.gp = gp_model
        self.safety_filter = safety_filter
        self.beta = beta

    def generate_candidate_grid(
        self,
        kp_points: int = 25,
        ki_points: int = 25,
    ) -> List[Tuple[float, float]]:
        """Generate grid of candidate (Kp, Ki) gain pairs across search space."""
        cfg = self.safety_filter.cfg
        kp_step = (cfg.kp_max - cfg.kp_min) / (kp_points - 1)
        ki_step = (cfg.ki_max - cfg.ki_min) / (ki_points - 1)

        grid = []
        for i in range(kp_points):
            kp = cfg.kp_min + i * kp_step
            for j in range(ki_points):
                ki = cfg.ki_min + j * ki_step
                grid.append((round(kp, 3), round(ki, 2)))
        return grid

    def evaluate_and_rank_candidates(
        self,
        candidates: List[Tuple[float, float]],
        sub_models: Optional[Dict[str, GaussianProcessRegressor]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate candidate gains using GP surrogate and rank them.
        Filters out candidates that violate safety constraints or exhibit excessive uncertainty.
        """
        ranked = []

        for kp, ki in candidates:
            # 1. GP prediction for composite performance cost J
            pred_cost_mu, pred_cost_std = self.gp.predict([kp, ki])

            # Optional individual metric predictions if specialized sub-GPs are active
            pred_overshoot = None
            pred_settling = None
            pred_ripple = None
            pred_iae = None

            if sub_models:
                if "overshoot" in sub_models:
                    pred_overshoot, _ = sub_models["overshoot"].predict([kp, ki])
                if "settling" in sub_models:
                    pred_settling, _ = sub_models["settling"].predict([kp, ki])
                if "ripple" in sub_models:
                    pred_ripple, _ = sub_models["ripple"].predict([kp, ki])
                if "iae" in sub_models:
                    pred_iae, _ = sub_models["iae"].predict([kp, ki])

            # 2. Safety and uncertainty gating (includes IAE check)
            is_acceptable, reason = self.safety_filter.evaluate_candidate(
                kp=kp,
                ki=ki,
                pred_cost_mean=pred_cost_mu,
                pred_cost_std=pred_cost_std,
                pred_overshoot=pred_overshoot,
                pred_settling_time=pred_settling,
                pred_ripple=pred_ripple,
                pred_iae=pred_iae,
            )

            # 3. Constrained Acquisition Score:
            # We want to minimize cost J. Lower Confidence Bound (LCB):
            # LCB(x) = mu(x) - beta * sigma(x)
            if is_acceptable:
                acq_score = pred_cost_mu - (self.beta * pred_cost_std)
                confidence_penalty = 0.2 * pred_cost_std
                effective_score = acq_score + confidence_penalty
            else:
                effective_score = 9999.0 + pred_cost_mu

            ranked.append({
                "kp": kp,
                "ki": ki,
                "pred_mean_cost": round(pred_cost_mu, 4),
                "pred_std_cost": round(pred_cost_std, 4),
                "acquisition_score": round(effective_score, 4),
                "is_acceptable": is_acceptable,
                "gating_status": reason,
            })

        # Sort: Acceptable candidates first, ordered by lowest acquisition score (best)
        ranked.sort(key=lambda item: (not item["is_acceptable"], item["acquisition_score"]))
        return ranked
