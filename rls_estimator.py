"""
Precision Recursive Least Squares (RLS) Parameter Estimator for DC-Link Dynamics
Formulated in balanced discrete-time regression:
    Δv_dc(k) = v_dc(k) - v_dc(k-1) = θ_1 * i_c(k) + θ_2 * Δi_c(k)
where:
    θ_1 = Δt / C_dc
    θ_2 = R_esr
Both parameters are normalized to the same order of magnitude (~0.01 - 0.10),
guaranteeing accurate, unbiased estimation of BOTH Capacitance and ESR.
"""

import math
from typing import Tuple, Dict, Any, Optional


class RLSParameterEstimator:
    """
    Numerically balanced RLS estimator for DC-link capacitance (C_dc)
    and Equivalent Series Resistance (R_esr).
    """

    def __init__(
        self,
        nominal_C: float = 0.0022,       # 2.2 mF nominal
        nominal_R: float = 0.050,        # 50 mΩ nominal
        sample_time: float = 0.00005,    # 50 μs
        forgetting_factor: float = 0.990, # λ parameter
    ):
        self.nominal_C = nominal_C
        self.nominal_R = nominal_R
        self.dt = sample_time
        self.lam = forgetting_factor

        # Parameter vector theta = [theta_1, theta_2] = [dt / C_dc, R_esr]
        self.theta = [self.dt / nominal_C, nominal_R]

        # Covariance matrix P initialized with balanced scaling
        self.P = [
            [1.0, 0.0],
            [0.0, 10.0]  # Higher variance on ESR to track faster transients
        ]

        # Prior sample buffers
        self.prev_v_dc: Optional[float] = None
        self.prev_i_c: Optional[float] = None
        self.prev_t: Optional[float] = None
        self.updates_count = 0

    def reset(self, nominal_C: Optional[float] = None, nominal_R: Optional[float] = None):
        """Reset estimator state."""
        if nominal_C: self.nominal_C = nominal_C
        if nominal_R: self.nominal_R = nominal_R
        self.theta = [self.dt / self.nominal_C, self.nominal_R]
        self.P = [[1.0, 0.0], [0.0, 10.0]]
        self.prev_v_dc = None
        self.prev_i_c = None
        self.prev_t = None
        self.updates_count = 0

    def update(
        self,
        v_dc: float,
        i_conv: float,
        i_load: float,
        t: float,
    ) -> Tuple[float, float]:
        """
        Ingest telemetry sample and perform RLS parameter update.
        Returns estimated (C_dc_estimated, R_esr_estimated).
        """
        i_c = i_conv - i_load

        if self.prev_v_dc is None or self.prev_i_c is None or self.prev_t is None:
            self.prev_v_dc = v_dc
            self.prev_i_c = i_c
            self.prev_t = t
            C_init = self.dt / max(1e-8, self.theta[0])
            return C_init, self.theta[1]

        # Dynamically adapt sample time if timestamps are provided
        if t > self.prev_t:
            sample_dt = t - self.prev_t
            if sample_dt > 1e-7:
                self.dt = sample_dt

        # Compute discrete increments
        delta_v = v_dc - self.prev_v_dc
        delta_i = i_c - self.prev_i_c

        self.prev_v_dc = v_dc
        self.prev_i_c = i_c
        self.prev_t = t

        # Regressor vector phi = [ i_c(k), Δi_c(k) ]
        phi = [i_c, delta_i]
        y = delta_v

        # Deadband: update only when dynamic current excitation is present
        excitation = abs(i_c) + 2.0 * abs(delta_i)
        if excitation < 0.20:
            C_est = self.dt / max(1e-8, self.theta[0])
            return C_est, self.theta[1]

        # P * phi
        P00, P01 = self.P[0][0], self.P[0][1]
        P10, P11 = self.P[1][0], self.P[1][1]

        P_phi = [
            P00 * phi[0] + P01 * phi[1],
            P10 * phi[0] + P11 * phi[1]
        ]

        # Denominator: denom = λ + phi^T * P * phi
        phi_P_phi = phi[0] * P_phi[0] + phi[1] * P_phi[1]
        denom = self.lam + phi_P_phi
        if abs(denom) < 1e-12:
            denom = 1e-12

        # Kalman gain vector: K = (P * phi) / denom
        K = [P_phi[0] / denom, P_phi[1] / denom]

        # Innovation / prediction error: e = y - phi^T * theta
        y_hat = phi[0] * self.theta[0] + phi[1] * self.theta[1]
        e = y - y_hat

        # Update parameter estimates: theta = theta + K * e
        self.theta[0] += K[0] * e
        self.theta[1] += K[1] * e

        # Enforce physical constraints:
        # C_dc in [0.5 mF, 5.0 mF] => theta_1 in [1e-5, 1e-4]
        # ESR in [0.010, 0.300] Ω
        min_theta1 = self.dt / 0.0050  # 5 mF
        max_theta1 = self.dt / 0.0005  # 0.5 mF
        self.theta[0] = max(min_theta1, min(max_theta1, self.theta[0]))
        self.theta[1] = max(0.010, min(0.300, self.theta[1]))

        # Update covariance matrix: P = (P - K * phi^T * P) / λ
        new_P00 = (P00 - K[0] * P_phi[0]) / self.lam
        new_P01 = (P01 - K[0] * P_phi[1]) / self.lam
        new_P10 = (P10 - K[1] * P_phi[0]) / self.lam
        new_P11 = (P11 - K[1] * P_phi[1]) / self.lam

        self.P = [[new_P00, new_P01], [new_P10, new_P11]]
        self.updates_count += 1

        C_est = self.dt / max(1e-8, self.theta[0])
        R_est = self.theta[1]
        return C_est, R_est

    def get_estimated_parameters(self) -> Dict[str, float]:
        """Return currently estimated plant capacitance and ESR."""
        C_est = self.dt / max(1e-8, self.theta[0])
        R_est = self.theta[1]
        return {
            "C_dc_estimated": round(C_est, 6),
            "R_esr_estimated": round(R_est, 5),
            "C_drift_pct": round(((C_est - self.nominal_C) / self.nominal_C) * 100.0, 2),
            "R_drift_pct": round(((R_est - self.nominal_R) / self.nominal_R) * 100.0, 2),
            "updates_count": self.updates_count,
        }
