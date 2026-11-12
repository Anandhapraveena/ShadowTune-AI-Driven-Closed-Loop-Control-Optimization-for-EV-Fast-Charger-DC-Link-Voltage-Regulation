"""
ShadowTune Gaussian Process Optimization Engine
Enhanced with:
1. Two-Tier Candidate Acceptance (GP Pre-Screening + Post-Simulation Certification)
2. Harmonized 4-decimal precision on all cost values
3. Sub-GP for IAE constraint prediction
4. Expanded search boundaries preventing boundary-limited optimization
5. Real-time RLS Parameter Estimation & Digital Twin Model Correction
6. Cryptographic SHA-256 Hash-Chained Audit Ledger
7. Automatic Fallback & Rollback Protection
8. Rate-Limited Gain Transitions (Bumpless Transfer)
"""

from typing import List, Dict, Any, Tuple, Optional
from gp_model import GaussianProcessRegressor
from safety_constraints import SafetyFilter, SafetyThresholds, MAX_ALLOWED_UNCERTAINTY
from acquisition import CandidatePrioritizer
from digital_twin_simulator import DCLinkDigitalTwin, DCLinkPlantConfig, RateLimiter
from rls_estimator import RLSParameterEstimator
from audit_log import HashChainedAuditLedger


class ShadowTuneOptimizer:
    """
    Production-grade Cyber-Physical Optimizer for ShadowTune DC-link PI control.
    """

    def __init__(
        self,
        safety_thresholds: Optional[SafetyThresholds] = None,
        digital_twin: Optional[DCLinkDigitalTwin] = None,
        exploration_beta: float = 1.6,
        enable_audit_ledger: bool = True,
    ):
        self.safety_filter = SafetyFilter(safety_thresholds or SafetyThresholds())
        self.digital_twin = digital_twin or DCLinkDigitalTwin()
        self.exploration_beta = exploration_beta

        # RLS Parameter Estimator
        self.rls = RLSParameterEstimator(
            nominal_C=self.digital_twin.cfg.C_dc,
            nominal_R=self.digital_twin.cfg.R_esr,
            sample_time=self.digital_twin.cfg.dt,
        )

        # Cryptographic Audit Ledger
        self.ledger = HashChainedAuditLedger() if enable_audit_ledger else None

        # Composite Cost GP Surrogate
        self.gp_cost = GaussianProcessRegressor(
            length_scales=(1.2, 18.0),
            signal_variance=1.0,
            noise_variance=0.01,
        )

        # Specialized Sub-GPs for granular metric constraints (now includes IAE)
        self.sub_gps = {
            "overshoot": GaussianProcessRegressor(length_scales=(1.2, 18.0), noise_variance=0.02),
            "settling": GaussianProcessRegressor(length_scales=(1.2, 18.0), noise_variance=0.01),
            "ripple": GaussianProcessRegressor(length_scales=(1.5, 20.0), noise_variance=0.005),
            "iae": GaussianProcessRegressor(length_scales=(1.2, 18.0), noise_variance=0.01),
        }

        self.prioritizer = CandidatePrioritizer(
            gp_model=self.gp_cost,
            safety_filter=self.safety_filter,
            beta=self.exploration_beta,
        )

        # Historical database of evaluated experiments
        self.history: List[Dict[str, Any]] = []

        # Current active gains and certified rollback checkpoint
        self.current_active_gains = {"kp": 0.40, "ki": 10.0}
        self.last_certified_gains = {"kp": 0.40, "ki": 10.0, "cost": 3.3500}
        self.tuning_paused = False
        self.rollback_history: List[Dict[str, Any]] = []

    def load_initial_history(self, initial_records: List[Dict[str, Any]]):
        """Seed the optimizer with previously tested PI gains and measured metrics."""
        for idx, record in enumerate(initial_records):
            if "iteration" not in record:
                record["iteration"] = idx + 1
            # Ensure 4-decimal precision consistency
            record["composite_cost"] = round(record["composite_cost"], 4)
            self.history.append(record)

            if self.ledger:
                self.ledger.append_event(
                    event_type="SEED_EVALUATION",
                    data={
                        "iteration": record["iteration"],
                        "kp": record["kp"],
                        "ki": record["ki"],
                        "composite_cost": f"{record['composite_cost']:.4f}",
                        "decision": "CERTIFIED_BASELINE",
                        "status": "INITIAL_SEED",
                    }
                )

        self._retrain_models()

        # Update last certified gains
        best_seed = min(self.history, key=lambda r: r["composite_cost"])
        self.last_certified_gains = {
            "kp": best_seed["kp"],
            "ki": best_seed["ki"],
            "cost": round(best_seed["composite_cost"], 4),
        }
        self.current_active_gains = {"kp": best_seed["kp"], "ki": best_seed["ki"]}

    def update_rls_from_telemetry(self, telemetry_samples: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Process physical converter telemetry through RLS estimator and apply
        model correction to the adaptive shadow digital twin.
        """
        last_C, last_R = self.digital_twin.cfg.C_dc, self.digital_twin.cfg.R_esr
        for s in telemetry_samples:
            last_C, last_R = self.rls.update(
                v_dc=s["v_dc"],
                i_conv=s["i_conv"],
                i_load=s["i_load"],
                t=s["t"],
            )

        # Apply model correction to the Digital Twin
        self.digital_twin.apply_model_correction(C_est=last_C, R_est=last_R)

        est_info = self.rls.get_estimated_parameters()
        if self.ledger:
            self.ledger.append_event(
                event_type="RLS_MODEL_CORRECTION",
                data={
                    "estimated_parameters": est_info,
                    "applied_to_digital_twin": True,
                    "model_corrected": True,
                }
            )
        return est_info

    def _retrain_models(self):
        """Retrain composite GP and sub-GPs using all accumulated history."""
        if not self.history:
            return

        X = [[r["kp"], r["ki"]] for r in self.history]
        y_cost = [r["composite_cost"] for r in self.history]
        self.gp_cost.fit(X, y_cost)

        y_overshoot = [r.get("overshoot_pct", 5.0) for r in self.history]
        y_settling = [r.get("settling_time_s") if r.get("settling_time_s") is not None else 0.080 for r in self.history]
        y_ripple = [r.get("voltage_ripple_v", 1.0) for r in self.history]
        y_iae = [r.get("iae", 1.0) for r in self.history]

        self.sub_gps["overshoot"].fit(X, y_overshoot)
        self.sub_gps["settling"].fit(X, y_settling)
        self.sub_gps["ripple"].fit(X, y_ripple)
        self.sub_gps["iae"].fit(X, y_iae)

    def predict_candidate(self, kp: float, ki: float) -> Dict[str, Any]:
        """Predict performance and uncertainty for candidate (Kp, Ki)."""
        mu_cost, std_cost = self.gp_cost.predict([kp, ki])
        mu_ov, _ = self.sub_gps["overshoot"].predict([kp, ki])
        mu_set, _ = self.sub_gps["settling"].predict([kp, ki])
        mu_rip, _ = self.sub_gps["ripple"].predict([kp, ki])
        mu_iae, _ = self.sub_gps["iae"].predict([kp, ki])

        is_safe, reason = self.safety_filter.evaluate_candidate(
            kp=kp,
            ki=ki,
            pred_cost_mean=mu_cost,
            pred_cost_std=std_cost,
            pred_overshoot=mu_ov,
            pred_settling_time=mu_set,
            pred_ripple=mu_rip,
            pred_iae=mu_iae,
            C_dc=self.digital_twin.cfg.C_dc,
            R_esr=self.digital_twin.cfg.R_esr,
        )

        return {
            "kp": kp,
            "ki": ki,
            "predicted_cost": round(mu_cost, 4),
            "uncertainty_cost_std": round(std_cost, 4),
            "predicted_overshoot_pct": round(mu_ov, 2),
            "predicted_settling_time_s": round(mu_set, 4),
            "predicted_ripple_v": round(mu_rip, 3),
            "predicted_iae": round(mu_iae, 4),
            "is_acceptable": is_safe,
            "gating_status": reason,
        }

    def suggest_next_candidate(self, grid_resolution: int = 25) -> Optional[Dict[str, Any]]:
        """Scan candidate grid across expanded search boundaries and prioritize safe candidates."""
        if self.tuning_paused:
            return None

        tested_pairs = {(round(r["kp"], 2), round(r["ki"], 1)) for r in self.history}
        candidates = self.prioritizer.generate_candidate_grid(kp_points=grid_resolution, ki_points=grid_resolution)
        untested = [c for c in candidates if (round(c[0], 2), round(c[1], 1)) not in tested_pairs]

        if not untested:
            return None

        ranked = self.prioritizer.evaluate_and_rank_candidates(candidates=untested, sub_models=self.sub_gps)
        for cand in ranked:
            if cand["is_acceptable"]:
                return cand

        return ranked[0] if ranked else None

    def execute_automatic_rollback(self, trigger_reason: str) -> Dict[str, Any]:
        """
        Executes immediate rollback to the last certified safe gain pair.
        Records an immutable rollback block in the cryptographic audit ledger.
        """
        reverted_from = dict(self.current_active_gains)
        self.current_active_gains = {
            "kp": self.last_certified_gains["kp"],
            "ki": self.last_certified_gains["ki"],
        }
        self.tuning_paused = True

        rollback_record = {
            "event": "AUTOMATIC_ROLLBACK",
            "timestamp_index": len(self.rollback_history) + 1,
            "reverted_from": reverted_from,
            "restored_gains": self.current_active_gains,
            "certified_cost": f"{self.last_certified_gains['cost']:.4f}",
            "trigger_reason": trigger_reason,
            "tuning_status": "PAUSED_FOR_SAFETY",
        }
        self.rollback_history.append(rollback_record)

        if self.ledger:
            self.ledger.append_event(
                event_type="AUTOMATIC_ROLLBACK",
                data=rollback_record,
            )

        return rollback_record

    def run_optimization_step(self, enable_rate_limiting: bool = True) -> Dict[str, Any]:
        """
        Execute one active learning optimization step with Two-Tier Acceptance Protocol:
        Tier 1: GP Pre-Screening
        Tier 2: Post-Simulation Certification
        """
        candidate = self.suggest_next_candidate()
        if not candidate:
            return {"status": "NO_CANDIDATE_OR_TUNING_PAUSED"}

        kp, ki = candidate["kp"], candidate["ki"]

        # Tier 1 Pre-Screening: Safety & Confidence Gating Check
        if not candidate["is_acceptable"]:
            rejection_info = {
                "status": "REJECTED_BEFORE_SIMULATION",
                "candidate": candidate,
                "reason": candidate["gating_status"],
            }
            if self.ledger:
                self.ledger.append_event(
                    event_type="SAFETY_REJECTION",
                    data={
                        "kp": kp,
                        "ki": ki,
                        "decision": "REJECTED_PRE_SIM",
                        "predicted_cost": f"{candidate['pred_mean_cost']:.4f}",
                        "uncertainty": f"{candidate['pred_std_cost']:.4f}",
                        "reason": candidate["gating_status"],
                    }
                )
            return rejection_info

        # Digital Twin Simulation with rate-limited bumpless transfer
        initial_kp = self.current_active_gains["kp"] if enable_rate_limiting else None
        initial_ki = self.current_active_gains["ki"] if enable_rate_limiting else None

        sim_metrics = self.digital_twin.simulate(
            kp=kp,
            ki=ki,
            initial_kp=initial_kp,
            initial_ki=initial_ki,
            extract_telemetry_for_rls=True,
        )

        # Update RLS from simulation telemetry to keep Digital Twin adapted
        if "telemetry" in sim_metrics:
            self.update_rls_from_telemetry(sim_metrics["telemetry"])

        # Tier 2 Post-Simulation Certification:
        # Check actual simulated performance against all safety constraints
        actual_passed, cert_reason = self.safety_filter.verify_simulated_metrics(sim_metrics)

        if not actual_passed:
            # Candidate looked promising to GP, but failed physical simulation tests!
            rollback_res = self.execute_automatic_rollback(
                trigger_reason=f"Post-simulation verification failed: {cert_reason}"
            )
            return {
                "status": "REJECTED_POST_SIMULATION",
                "reason": cert_reason,
                "rollback_details": rollback_res,
                "evaluated_metrics": sim_metrics,
            }

        # Actual metrics strictly passed Tier 2 Certification -> Accept and Deploy!
        actual_cost = round(sim_metrics["composite_cost"], 4)
        pred_cost = round(candidate["pred_mean_cost"], 4)
        pred_error = round(abs(pred_cost - actual_cost), 4)

        eval_record = {
            "iteration": len(self.history) + 1,
            "kp": kp,
            "ki": ki,
            **sim_metrics,
            "composite_cost": actual_cost,
            "pred_mean_cost": pred_cost,
            "pred_std_cost": candidate["pred_std_cost"],
            "prediction_error": pred_error,
            "rate_limited": enable_rate_limiting,
            "tier2_certified": True,
        }
        self.history.append(eval_record)
        self.current_active_gains = {"kp": kp, "ki": ki}

        # Update certified safe gains if new cost is superior
        if actual_cost < self.last_certified_gains["cost"]:
            self.last_certified_gains = {
                "kp": kp,
                "ki": ki,
                "cost": actual_cost,
            }

        # Append to Cryptographic Audit Ledger
        if self.ledger:
            self.ledger.append_event(
                event_type="OPTIMIZATION_STEP",
                data={
                    "iteration": eval_record["iteration"],
                    "kp": kp,
                    "ki": ki,
                    "actual_cost": f"{actual_cost:.4f}",
                    "predicted_cost": f"{pred_cost:.4f}",
                    "prior_uncertainty": f"{candidate['pred_std_cost']:.4f}",
                    "overshoot_pct": sim_metrics["overshoot_pct"],
                    "undershoot_pct": sim_metrics.get("undershoot_pct", 0.0),
                    "settling_time_s": sim_metrics["settling_time_s"],
                    "voltage_ripple_v": sim_metrics["voltage_ripple_v"],
                    "iae": sim_metrics["iae"],
                    "decision": "ACCEPTED_AND_CERTIFIED",
                    "model_corrected": sim_metrics.get("model_corrected", False),
                }
            )

        # Continuously retrain GP models with newly evaluated result
        self._retrain_models()

        return {
            "status": "EVALUATED_AND_UPDATED",
            "iteration": eval_record["iteration"],
            "evaluated_kp": kp,
            "evaluated_ki": ki,
            "metrics": sim_metrics,
            "prediction_comparison": {
                "predicted_cost": pred_cost,
                "actual_cost": actual_cost,
                "prior_uncertainty": candidate["pred_std_cost"],
                "error": pred_error,
            },
        }

    def get_best_tested_gains(self) -> Optional[Dict[str, Any]]:
        if not self.history:
            return None
        return min(self.history, key=lambda r: r["composite_cost"])
