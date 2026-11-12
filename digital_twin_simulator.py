"""
Digital Twin Simulator for DC-Link Voltage Regulation Loop
Enhanced with:
1. True capacitor physics: core voltage state dv_C/dt = i_c/C_dc, terminal v_dc = v_C + R_esr * i_c
2. Detailed carrier-based PWM switching ripple model
3. Distinct Overshoot (rise above ref) and Undershoot (sag below ref) metrics
4. Verified settling-time calculation with explicit unsettled detection
5. Analytical Routh-Hurwitz stability margin calculation
6. Environmental robustness: parameter variation, grid sag, and sensor noise
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List


@dataclass
class DCLinkPlantConfig:
    Vdc_ref: float = 400.0             # Reference DC-link voltage (V)
    C_dc: float = 0.0022               # True DC bus capacitance (2.2 mF nominal)
    R_esr: float = 0.050               # True capacitor ESR (50 mΩ nominal)
    f_sw: float = 10000.0              # PWM switching frequency (10 kHz)
    tau_current_loop: float = 0.0008   # Inner current loop time constant (0.8 ms)
    i_load_initial: float = 10.0       # Initial load current (A)
    i_load_step: float = 30.0          # Load step disturbance (A)
    t_step: float = 0.020              # Time of load disturbance (20 ms)
    t_sim: float = 0.080               # Total simulation horizon (80 ms)
    dt: float = 0.00005                # Simulation step size (50 us)

    # Grid-voltage sag configuration
    enable_grid_sag: bool = False
    t_grid_sag: float = 0.035          # Grid sag event time (35 ms)
    grid_sag_depth: float = 0.20       # 20% voltage sag (drops to 0.8 p.u.)

    # Sensor noise configuration
    enable_sensor_noise: bool = False
    sensor_noise_std: float = 0.40     # Voltage sensor noise standard deviation (0.4V)
    adc_resolution_v: float = 0.10     # ADC voltage quantization step (0.1V)

    # Rate limiting on gains (bumpless transfer)
    enable_rate_limiting: bool = False
    max_kp_rate_per_sec: float = 20.0  # Max Kp change rate (units/s)
    max_ki_rate_per_sec: float = 400.0 # Max Ki change rate (units/s)


class RateLimiter:
    """Enforces slew-rate limiting on PI gains to ensure bumpless transfer."""

    def __init__(self, max_kp_rate: float = 20.0, max_ki_rate: float = 400.0):
        self.max_kp_rate = max_kp_rate
        self.max_ki_rate = max_ki_rate
        self.current_kp: Optional[float] = None
        self.current_ki: Optional[float] = None

    def initialize(self, kp: float, ki: float):
        self.current_kp = kp
        self.current_ki = ki

    def step(self, target_kp: float, target_ki: float, dt: float) -> Tuple[float, float]:
        if self.current_kp is None or self.current_ki is None:
            self.current_kp = target_kp
            self.current_ki = target_ki
            return target_kp, target_ki

        max_delta_kp = self.max_kp_rate * dt
        max_delta_ki = self.max_ki_rate * dt

        delta_kp = max(-max_delta_kp, min(max_delta_kp, target_kp - self.current_kp))
        delta_ki = max(-max_delta_ki, min(max_delta_ki, target_ki - self.current_ki))

        self.current_kp += delta_kp
        self.current_ki += delta_ki

        return self.current_kp, self.current_ki


class DCLinkDigitalTwin:
    """
    Physical converter simulator & Adaptive Shadow Digital Twin.
    """

    def __init__(self, config: Optional[DCLinkPlantConfig] = None):
        self.cfg = config or DCLinkPlantConfig()
        self.model_corrected = False
        self.correction_log: Dict[str, Any] = {}

    def apply_model_correction(self, C_est: float, R_est: float):
        """
        Applies RLS parameter estimation to correct model mismatch in the Digital Twin.
        """
        old_C = self.cfg.C_dc
        old_R = self.cfg.R_esr
        self.cfg.C_dc = C_est
        self.cfg.R_esr = R_est
        self.model_corrected = True
        self.correction_log = {
            "old_C_dc": old_C,
            "corrected_C_dc": C_est,
            "old_R_esr": old_R,
            "corrected_R_esr": R_est,
            "C_correction_pct": round(((C_est - old_C) / old_C) * 100.0, 2),
            "R_correction_pct": round(((R_est - old_R) / old_R) * 100.0, 2),
        }

    def compute_routh_stability(self, kp: float, ki: float) -> Tuple[bool, float]:
        """Calculates closed-loop analytical Routh-Hurwitz stability margin."""
        a3 = self.cfg.tau_current_loop * self.cfg.C_dc
        a2 = self.cfg.C_dc * (1.0 + kp * self.cfg.R_esr)
        a1 = kp + ki * self.cfg.C_dc * self.cfg.R_esr
        a0 = ki

        b1_num = a2 * a1 - a3 * a0
        if a2 * a1 <= 0:
            return False, -1.0

        margin = b1_num / (a2 * a1)
        is_stable = (a3 > 0) and (a2 > 0) and (a1 > 0) and (a0 > 0) and (b1_num > 0)
        return is_stable, margin

    def simulate(
        self,
        kp: float,
        ki: float,
        initial_kp: Optional[float] = None,
        initial_ki: Optional[float] = None,
        extract_telemetry_for_rls: bool = False,
    ) -> Dict[str, Any]:
        """
        Simulate DC-link converter dynamics with rigorous physics, PWM switching ripple,
        overshoot/undershoot separation, and verified settling-time detection.
        """
        dt = self.cfg.dt
        steps = int(self.cfg.t_sim / dt)

        # Capacitor state: v_C is the internal core capacitor voltage
        v_C = self.cfg.Vdc_ref
        v_dc = self.cfg.Vdc_ref
        i_conv = self.cfg.i_load_initial
        # Steady-state pre-charge: initialize PI integrator so initial i_ref matches initial load current
        integral_error = self.cfg.i_load_initial / max(0.1, ki)

        v_dc_history = []
        t_history = []
        telemetry_samples = []

        iae = 0.0
        max_voltage = self.cfg.Vdc_ref
        min_voltage = self.cfg.Vdc_ref

        # Rate limiter setup
        rate_limiter = None
        current_kp = kp
        current_ki = ki
        if self.cfg.enable_rate_limiting and initial_kp is not None and initial_ki is not None:
            rate_limiter = RateLimiter(
                max_kp_rate=self.cfg.max_kp_rate_per_sec,
                max_ki_rate=self.cfg.max_ki_rate_per_sec,
            )
            rate_limiter.initialize(initial_kp, initial_ki)

        # Routh-Hurwitz stability pre-check
        is_routh_stable, routh_margin = self.compute_routh_stability(kp, ki)

        rng = random.Random(42)
        T_sw = 1.0 / self.cfg.f_sw

        for step in range(steps):
            t = step * dt
            t_history.append(t)

            # 1. Gain Rate Limiting
            if rate_limiter:
                current_kp, current_ki = rate_limiter.step(kp, ki, dt)

            # 2. Load disturbance profile
            i_load = self.cfg.i_load_step if t >= self.cfg.t_step else self.cfg.i_load_initial

            # 3. Grid Voltage Sag injection
            if self.cfg.enable_grid_sag and t >= self.cfg.t_grid_sag:
                grid_sag_factor = 1.0 - self.cfg.grid_sag_depth
                i_load += 3.5 * (1.0 - grid_sag_factor)

            # 4. Sensor measurement with optional noise & ADC quantization
            v_dc_measured = v_dc
            if self.cfg.enable_sensor_noise:
                noise = rng.gauss(0.0, self.cfg.sensor_noise_std)
                v_quantized = round((v_dc + noise) / self.cfg.adc_resolution_v) * self.cfg.adc_resolution_v
                v_dc_measured = v_quantized

            # 5. PI Controller action
            error = self.cfg.Vdc_ref - v_dc_measured
            iae += abs(self.cfg.Vdc_ref - v_dc) * dt

            integral_error += error * dt
            integral_error = max(-200.0, min(200.0, integral_error))

            i_ref = current_kp * error + current_ki * integral_error
            i_ref = max(-100.0, min(100.0, i_ref))

            # Inner current loop dynamics (first-order lag)
            d_iconv = (i_ref - i_conv) / self.cfg.tau_current_loop
            i_conv += d_iconv * dt

            # 6. Core capacitor physical differential integration:
            i_c = i_conv - i_load
            dv_C = (i_c / self.cfg.C_dc) * dt
            v_C += dv_C

            # 7. Physical Carrier-Based PWM Switching Ripple Model:
            # Converter inductor switching ripple current: delta_I_sw = (Vdc * D * (1 - D)) / (L_conv * f_sw)
            duty = max(0.10, min(0.90, 0.50 + (i_conv / 150.0)))
            delta_I_sw = (self.cfg.Vdc_ref * duty * (1.0 - duty)) / (0.0015 * self.cfg.f_sw)
            t_in_period = t % T_sw
            tri_norm = (4.0 * t_in_period / T_sw - 1.0) if (t_in_period < 0.5 * T_sw) else (3.0 - 4.0 * t_in_period / T_sw)
            # Physical voltage ripple = ESR ohmic component + capacitive charge component
            v_ripple_pwm = (self.cfg.R_esr * delta_I_sw * 0.5) * tri_norm + (delta_I_sw / (8.0 * self.cfg.C_dc * self.cfg.f_sw)) * tri_norm

            # Terminal voltage: core capacitor voltage + ESR ohmic drop + high-frequency PWM switching ripple
            v_dc = v_C + self.cfg.R_esr * i_c + v_ripple_pwm
            v_dc_macro = v_C + self.cfg.R_esr * i_c

            # Track peak rise (overshoot) and peak drop (undershoot) after disturbance
            if t >= self.cfg.t_step:
                if v_dc > max_voltage:
                    max_voltage = v_dc
                if v_dc < min_voltage:
                    min_voltage = v_dc

            # Record telemetry for RLS parameter estimation
            if extract_telemetry_for_rls:
                telemetry_samples.append({
                    "t": t,
                    "v_dc": v_dc_macro,
                    "i_conv": i_conv,
                    "i_load": i_load,
                })

            # Instability check
            if math.isnan(v_dc) or abs(v_dc) > 2000.0:
                return {
                    "overshoot_pct": 100.0,
                    "undershoot_pct": 100.0,
                    "settling_time_s": None,
                    "is_settled": False,
                    "voltage_ripple_v": 50.0,
                    "iae": 999.0,
                    "composite_cost": 999.0,
                    "is_stable": False,
                    "routh_margin": -1.0,
                    "model_corrected": self.model_corrected,
                }

            v_dc_history.append(v_dc)

        # 8. Distinct Metric Calculations:
        # A. Overshoot (% rise above V_ref)
        overshoot_pct = max(0.0, ((max_voltage - self.cfg.Vdc_ref) / self.cfg.Vdc_ref) * 100.0)

        # B. Undershoot / Sag (% drop below V_ref)
        undershoot_pct = max(0.0, ((self.cfg.Vdc_ref - min_voltage) / self.cfg.Vdc_ref) * 100.0)

        # C. Settling-Time Verification (Must settle and REMAIN within tolerance band)
        settling_band = 0.015 * self.cfg.Vdc_ref  # 1.5% band = 6.0V
        step_idx = int(self.cfg.t_step / dt)
        settling_time = None
        is_settled = True

        # Check if terminal voltage at end of simulation is inside the band
        last_window_steps = int(0.005 / dt)
        final_samples = v_dc_history[-last_window_steps:]
        if any(abs(v - self.cfg.Vdc_ref) > settling_band for v in final_samples):
            # Did not settle by end of simulation!
            is_settled = False
            settling_time = None
        else:
            # Scan backwards to find the exact last exit from the band
            for i in range(len(v_dc_history) - 1, step_idx, -1):
                if abs(v_dc_history[i] - self.cfg.Vdc_ref) > settling_band:
                    settling_time = (i - step_idx) * dt
                    break
            if settling_time is None:
                settling_time = 0.001  # Instantaneous / settled immediately

        # D. Steady-State Switching Ripple (Peak-to-Peak from physical PWM over last 10ms)
        last_window = int(0.010 / dt)
        recent = v_dc_history[-last_window:]
        voltage_ripple = max(recent) - min(recent)

        # E. Composite Cost Function
        w_overshoot = 0.30
        w_undershoot = 0.15
        w_settling = 0.25
        w_ripple = 0.15
        w_iae = 0.15

        norm_overshoot = overshoot_pct / 2.0
        norm_undershoot = undershoot_pct / 8.0
        norm_ripple = voltage_ripple / 0.8
        norm_iae = iae / 0.5

        if is_settled and settling_time is not None:
            norm_settling = (settling_time * 100.0) / 2.0
        else:
            # Heavy penalty if unsettled
            norm_settling = 15.0

        composite_cost = (
            w_overshoot * norm_overshoot
            + w_undershoot * norm_undershoot
            + w_settling * norm_settling
            + w_ripple * norm_ripple
            + w_iae * norm_iae
        )

        overall_stability = is_routh_stable and is_settled

        res = {
            "overshoot_pct": round(overshoot_pct, 3),
            "undershoot_pct": round(undershoot_pct, 3),
            "settling_time_s": round(settling_time, 4) if settling_time is not None else None,
            "is_settled": is_settled,
            "voltage_ripple_v": round(voltage_ripple, 3),
            "iae": round(iae, 4),
            "composite_cost": round(composite_cost, 4),
            "is_stable": overall_stability,
            "routh_margin": round(routh_margin, 4),
            "model_corrected": self.model_corrected,
            "C_dc_used": self.cfg.C_dc,
            "R_esr_used": self.cfg.R_esr,
        }

        if extract_telemetry_for_rls:
            res["telemetry"] = telemetry_samples

        return res
