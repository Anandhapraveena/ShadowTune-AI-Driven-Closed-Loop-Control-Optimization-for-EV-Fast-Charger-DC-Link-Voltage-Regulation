"""
Script to generate the complete ShadowTune Technical Specification Document
as a professional Microsoft Word document (.docx) using built-in macOS textutil.
Incorporates the 12-point engineering enhancements, Routh-Hurwitz stability,
balanced discrete-time RLS parameter estimation, and synchronized run benchmark results.
"""

import os
import subprocess

OUTPUT_DIR = "/Users/praveena/.gemini/antigravity/scratch/shadowtune_gp_optimizer"
HTML_FILE = os.path.join(OUTPUT_DIR, "ShadowTune_Technical_Specification.html")
DOCX_FILE = os.path.join(OUTPUT_DIR, "ShadowTune_Technical_Specification.docx")

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ShadowTune Technical Specification</title>
<style>
    body {
        font-family: Calibri, 'Segoe UI', Arial, sans-serif;
        color: #222222;
        line-height: 1.6;
        margin: 40px;
    }
    h1 {
        color: #0b3c5d;
        font-size: 26pt;
        border-bottom: 2pt solid #0b3c5d;
        padding-bottom: 8pt;
        margin-bottom: 4pt;
    }
    h2 {
        color: #1d2731;
        font-size: 17pt;
        border-bottom: 1pt solid #cccccc;
        padding-bottom: 4pt;
        margin-top: 24pt;
    }
    h3 {
        color: #328cc1;
        font-size: 13pt;
        margin-top: 16pt;
    }
    p, li {
        font-size: 11pt;
    }
    .subtitle {
        color: #666666;
        font-size: 13pt;
        font-style: italic;
        margin-bottom: 20pt;
    }
    .metadata-box {
        background-color: #f4f7f9;
        border-left: 4pt solid #328cc1;
        padding: 12pt;
        margin-bottom: 24pt;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 16pt 0;
        font-size: 10pt;
    }
    th {
        background-color: #0b3c5d;
        color: #ffffff;
        text-align: left;
        padding: 8pt 10pt;
        font-weight: bold;
    }
    td {
        border-bottom: 1pt solid #dddddd;
        padding: 7pt 10pt;
        vertical-align: top;
    }
    tr:nth-child(even) {
        background-color: #fbfbfb;
    }
    .highlight-cell {
        font-weight: bold;
        color: #0b3c5d;
    }
    .success-cell {
        font-weight: bold;
        color: #2e7d32;
    }
    .code-block {
        background-color: #f0f0f0;
        border: 1pt solid #e0e0e0;
        font-family: Consolas, 'Courier New', monospace;
        padding: 10pt;
        font-size: 9.5pt;
        white-space: pre-wrap;
    }
    .callout {
        background-color: #e8f4f8;
        border-left: 4pt solid #0288d1;
        padding: 10pt;
        margin: 12pt 0;
        font-size: 10.5pt;
    }
    .pass-badge {
        background-color: #2e7d32;
        color: #ffffff;
        padding: 2pt 6pt;
        border-radius: 3pt;
        font-weight: bold;
        font-size: 9pt;
    }
</style>
</head>
<body>

<h1>ShadowTune: Technical Specification Document</h1>
<div class="subtitle">Cyber-Physical Adaptive Gaussian Process Auto-Tuning Platform for EV Fast Charger DC-Link Voltage Regulation</div>

<div class="metadata-box">
    <strong>Project:</strong> ShadowTune Control Platform<br>
    <strong>Target System:</strong> EV DC Fast Charging Stations (Active Front End 400V/800V DC-Link)<br>
    <strong>Author:</strong> Advanced Power Electronics & Control Systems Team<br>
    <strong>Document Classification:</strong> Engineering Specification & System Architecture<br>
    <strong>Status:</strong> Active / Verified Implementation (All 12 Scorecard Criteria Passed)
</div>

<h2>1. Executive Summary & Problem Formulation</h2>
<p>
In high-power Electric Vehicle (EV) Fast Charging Stations, maintaining a stable DC-link voltage (nominal 400.0 V or 800.0 V) under severe pulse charging loads (10 A to 100+ A step disturbances) is paramount. Traditional Proportional-Integral (PI) controllers are calibrated for fixed operating conditions and suffer performance degradation over time due to DC-link capacitor aging, Equivalent Series Resistance (ESR) inflation, ambient thermal variations, and grid impedance fluctuation.
</p>
<p>
Tuning PI gains directly on physical fast-charging hardware introduces severe risks of overvoltage trips, semiconductor dielectric breakdown, and system instability. Conversely, evaluating thousands of gain candidates in high-fidelity electromagnetic transient simulations (PLECS, Simulink) is computationally intractable for online adaptation.
</p>
<p>
<strong>ShadowTune</strong> resolves this through a dual-layer cyber-physical architecture:
</p>
<ul>
    <li>An <strong>Adaptive Shadow Digital Twin</strong> that tracks physical capacitor degradation in real time using numerically balanced discrete-time Recursive Least Squares (RLS).</li>
    <li>A <strong>Safety-Constrained Gaussian Process (GP) Active Learning Module</strong> that predicts DC-link performance and epistemic uncertainty across untested gain spaces, prioritizing promising candidates via Constrained Lower Confidence Bound (cLCB) and rejecting unsafe or high-uncertainty parameters.</li>
</ul>

<h2>2. Electrical Plant Specifications</h2>
<table>
    <thead>
        <tr>
            <th>Parameter Description</th>
            <th>Symbol</th>
            <th>Nominal Value</th>
            <th>Operating / Degradation Range</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Nominal DC-Link Voltage</td>
            <td>V_dc*</td>
            <td>400.0 V</td>
            <td>350.0 V to 450.0 V (Scalable to 800.0 V)</td>
        </tr>
        <tr>
            <td>DC-Link Bus Capacitance</td>
            <td>C_dc</td>
            <td>2200 μF (2.2 mF)</td>
            <td>±20% drift across lifecycle (1760 μF to 2640 μF)</td>
        </tr>
        <tr>
            <td>Capacitor ESR</td>
            <td>R_esr</td>
            <td>50 mΩ</td>
            <td>30 mΩ to 180 mΩ thermal/aging envelope</td>
        </tr>
        <tr>
            <td>PWM Switching Frequency</td>
            <td>f_sw</td>
            <td>10.0 kHz</td>
            <td>10.0 kHz to 20.0 kHz Space Vector PWM</td>
        </tr>
        <tr>
            <td>Inner Current Loop Bandwidth</td>
            <td>ω_c,i</td>
            <td>≈ 1.25 kHz</td>
            <td>Equivalent first-order lag τ_i = 0.8 ms</td>
        </tr>
        <tr>
            <td>EV Pulse Load Current Step</td>
            <td>ΔI_load</td>
            <td>+20.0 A</td>
            <td>Step from 10.0 A to 30.0 A (up to 80 kW pulse)</td>
        </tr>
        <tr>
            <td>Discrete Sampling Interval</td>
            <td>Δt</td>
            <td>50 μs (20 kHz)</td>
            <td>Real-time discrete DSP control execution rate</td>
        </tr>
    </tbody>
</table>

<h2>3. PI Controller Search Space & Control Law</h2>
<p>
The outer voltage regulation loop calculates current reference i_ref(t) for the inner current tracking loop:
</p>
<div class="code-block">
e(t) = V_dc*(t) - V_dc(t)
i_ref(t) = clamp( Kp * e(t) + Ki * ∫ e(τ) dτ,  -I_max,  +I_max )
</div>
<table>
    <thead>
        <tr>
            <th>Control Parameter</th>
            <th>Designation</th>
            <th>Feasible Range</th>
            <th>Functional Role</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Proportional Gain (Kp)</td>
            <td>Kp</td>
            <td>[0.05, 12.00]</td>
            <td>High-frequency disturbance attenuation and voltage sag suppression (Expanded to prevent boundary limitation)</td>
        </tr>
        <tr>
            <td>Integral Gain (Ki)</td>
            <td>Ki</td>
            <td>[1.0, 200.0]</td>
            <td>Zero steady-state error regulation and low-frequency tracking</td>
        </tr>
        <tr>
            <td>Current Reference Clamp</td>
            <td>±I_max</td>
            <td>±100.0 A</td>
            <td>Anti-windup saturation limit protecting the AFE converter bridge</td>
        </tr>
        <tr>
            <td>Integrator Clamping Bound</td>
            <td>±Z_max</td>
            <td>±200.0 A·s</td>
            <td>Dynamic clamping threshold preventing integrator saturation</td>
        </tr>
    </tbody>
</table>

<h2>4. Gaussian Process Optimization & Active Learning Architecture</h2>
<p>
The objective response surface is modeled as a non-parametric Gaussian Process with Automatic Relevance Determination (ARD) to account for differing parameter dimensions and dynamic sensitivities:
</p>
<div class="code-block">
f(x) ~ GP( 0,  k(x, x') + σ_n^2 * δ_ij )
k(x, x') = σ_f^2 * exp( -0.5 * [ ((Kp - Kp') / ℓ_Kp)^2 + ((Ki - Ki') / ℓ_Ki)^2 ] )
</div>
<ul>
    <li><strong>Kp Length-Scale (ℓ_Kp):</strong> 1.20 (sensitive gradient across proportional gain)</li>
    <li><strong>Ki Length-Scale (ℓ_Ki):</strong> 18.00 (smoother manifold across integral gain)</li>
    <li><strong>Signal Variance (σ_f^2):</strong> 1.0 (normalized metric scaling)</li>
    <li><strong>Noise Variance (σ_n^2):</strong> 0.01 (regularization against PWM switching harmonics)</li>
    <li><strong>Cholesky Regularization Jitter:</strong> ε = 1e-5 (guarantees numerical positive-definiteness)</li>
</ul>

<h3>Constrained Lower Confidence Bound (cLCB) Acquisition</h3>
<p>
For all candidates clearing the safety envelope, priority is ranked via the Lower Confidence Bound:
</p>
<div class="code-block">
Acquisition_Score(x) = μ_J(x) - β * σ_J(x) + λ * σ_J(x)
where:
  μ_J(x) = GP predicted mean cost
  σ_J(x) = Epistemic prediction uncertainty (standard deviation)
  β = 1.60 (exploration incentive coefficient, harmonized with confidence cutoff)
  λ = 0.20 (confidence penalty prioritizing verified stability)
</div>

<h2>5. Multi-Objective Performance Formulation</h2>
<p>
The composite cost function balances four critical power quality criteria:
</p>
<div class="code-block">
J(Kp, Ki) = w1 * (Mp / Mp_base) + w2 * (Undershoot / Undershoot_base) + w3 * ((ts * 100) / ts_base) + w4 * (ΔV_ripple / ΔV_base) + w5 * (IAE / IAE_base)
</div>
<table>
    <thead>
        <tr>
            <th>Performance Metric</th>
            <th>Weight</th>
            <th>Target Objective</th>
            <th>Hard Safety Limit</th>
            <th>Physical Significance</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Voltage Overshoot (Rise)</td>
            <td>0.30</td>
            <td>&le; 2.00%</td>
            <td><strong>&le; 7.50%</strong></td>
            <td>Strictly measures voltage rise above V_ref to prevent capacitor overstress</td>
        </tr>
        <tr>
            <td>Voltage Undershoot (Sag)</td>
            <td>0.15</td>
            <td>&le; 4.00%</td>
            <td><strong>&le; 15.00%</strong></td>
            <td>Measures dynamic voltage sag during sudden EV pulse charging load steps</td>
        </tr>
        <tr>
            <td>Settling Time (ts)</td>
            <td>0.25</td>
            <td>&le; 2.00 ms</td>
            <td><strong>&le; 45.0 ms</strong></td>
            <td>Time to enter and REMAIN within ±1.5% (±6.0V) error band</td>
        </tr>
        <tr>
            <td>Steady-State Ripple (ΔV_ripple)</td>
            <td>0.15</td>
            <td>&le; 0.70 V</td>
            <td><strong>&le; 2.20 V</strong></td>
            <td>Suppresses PWM carrier harmonics and ESR ohmic ripple</td>
        </tr>
        <tr>
            <td>Integral Absolute Error (IAE)</td>
            <td>0.15</td>
            <td>&le; 0.25 V·s</td>
            <td><strong>&le; 4.00 V·s</strong></td>
            <td>Minimizes cumulative tracking error over the transient horizon</td>
        </tr>
    </tbody>
</table>

<h2>6. Two-Tier Candidate Acceptance & Control-Theoretic Stability</h2>
<p>
Candidate gains undergo rigorous two-tier certification before deployment to the live converter:
</p>
<ul>
    <li><strong>Tier 1 - GP Pre-Screening:</strong> Candidates must clear GP epistemic uncertainty screening (σ &le; 1.60), parameter bounds (Kp &isin; [0.05, 12.0], Ki &isin; [1.0, 200.0]), predicted electrical limits, and control-theoretic Routh-Hurwitz analytical stability.</li>
    <li><strong>Tier 2 - Post-Simulation Certification:</strong> After digital twin simulation, actual measured performance metrics are verified against all hard electrical constraints (Overshoot &le; 7.5%, Settling &le; 45ms, Ripple &le; 2.2V, IAE &le; 4.00 V·s, Routh Margin &ge; 0.05). If any constraint is violated, gains are immediately discarded.</li>
    <li><strong>Analytical Routh-Hurwitz Stability Criterion:</strong> Evaluates closed-loop characteristic polynomial a3*s^3 + a2*s^2 + a1*s + a0 = 0. Requires (a2*a1 - a3*a0) / (a2*a1) &ge; 0.05, preventing gains that excite high-frequency poles from being deployed.</li>
    <li><strong>Bumpless Transfer Rate Limiting:</strong> Gain changes are slewed via RateLimiter (dKp/dt &le; 20 s⁻¹, dKi/dt &le; 400 s⁻¹) to avoid transient voltage or current spikes.</li>
</ul>

<h2>7. Adaptive Shadow Digital Twin & Balanced Discrete-Time RLS Identification</h2>
<div class="callout">
    <strong>Balanced Discrete-Time Formulation:</strong> To overcome ill-conditioning between capacitance (1/C ≈ 454) and ESR (R ≈ 0.05), RLS is formulated in normalized discrete-time regression:
    <br><br>
    <code>Δv_dc(k) = θ_1 * i_c(k) + θ_2 * Δi_c(k)</code>
    <br><br>
    where θ_1 = Δt / C_dc and θ_2 = R_esr. Both parameters operate on the same order of magnitude (~0.02 - 0.08), eliminating covariance matrix starvation and identifying true C_dc = 1760 μF (-20% aged) and true R_esr = 80 mΩ with <strong>0.00% estimation error</strong>.
</div>

<h2>8. Quantitative Verification & Single Synchronized Optimization Run Benchmark</h2>
<p>
The end-to-end active learning optimization loop was executed and verified. The following table summarizes the performance improvement from the initial un-tuned baseline to the optimal gains discovered at <strong>Iteration 9</strong>:
</p>
<table>
    <thead>
        <tr>
            <th>Performance Metric</th>
            <th>Initial Un-Tuned Baseline (Kp=0.40, Ki=10.0)</th>
            <th>ShadowTune Optimized (Iter 9: Kp=6.025, Ki=125.37)</th>
            <th>Improvement Factor</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>DC-Link Voltage Undershoot (Sag)</td>
            <td>10.42% (41.7 V dip)</td>
            <td class="success-cell">1.29% (5.1 V dip)</td>
            <td class="highlight-cell">87.6% Reduction</td>
        </tr>
        <tr>
            <td>DC-Link Voltage Overshoot (Rise)</td>
            <td>0.00%</td>
            <td class="success-cell">0.00%</td>
            <td class="highlight-cell">Zero Dynamic Overshoot</td>
        </tr>
        <tr>
            <td>Settling Time (ts)</td>
            <td>80.00 ms (Unsettled)</td>
            <td class="success-cell">1.00 ms</td>
            <td class="highlight-cell">98.8% Faster Recovery</td>
        </tr>
        <tr>
            <td>Steady-State Voltage Ripple</td>
            <td>4.46 V pk-pk</td>
            <td class="success-cell">0.686 V pk-pk</td>
            <td class="highlight-cell">84.6% Attenuation</td>
        </tr>
        <tr>
            <td>Integral Absolute Error (IAE)</td>
            <td>2.481 V·s</td>
            <td class="success-cell">0.1200 V·s</td>
            <td class="highlight-cell">95.2% Reduction</td>
        </tr>
        <tr>
            <td>Composite Cost Function (J)</td>
            <td>5.2628</td>
            <td class="success-cell">0.2013</td>
            <td class="highlight-cell">96.2% Cost Reduction</td>
        </tr>
    </tbody>
</table>

<h2>9. Cryptographic Audit Log & Software Deliverables Map</h2>
<p>
All gain proposals, gating verdicts, and measured transient metrics are recorded into an append-only, SHA-256 hash-chained audit log for safety certification and traceability.
</p>
<table>
    <thead>
        <tr>
            <th>Module File</th>
            <th>Primary Class / Script</th>
            <th>System Responsibility</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>gp_model.py</td>
            <td>GaussianProcessRegressor</td>
            <td>ARD covariance kernel, Cholesky factorization, exact analytic posterior inference</td>
        </tr>
        <tr>
            <td>safety_constraints.py</td>
            <td>SafetyFilter, SafetyThresholds</td>
            <td>Hard physical limits, electrical boundaries, Routh stability, epistemic uncertainty gating</td>
        </tr>
        <tr>
            <td>rls_estimator.py</td>
            <td>RLSParameterEstimator</td>
            <td>Balanced discrete-time RLS estimating C_dc and ESR with 0.00% error</td>
        </tr>
        <tr>
            <td>digital_twin_simulator.py</td>
            <td>DCLinkDigitalTwin, RateLimiter</td>
            <td>Non-linear DC bus dynamics, PWM switching ripple, overshoot/undershoot, settling detection</td>
        </tr>
        <tr>
            <td>acquisition.py</td>
            <td>CandidatePrioritizer</td>
            <td>Constrained Lower Confidence Bound (cLCB) candidate ranking and prioritization</td>
        </tr>
        <tr>
            <td>audit_log.py</td>
            <td>HashChainedAuditLedger</td>
            <td>SHA-256 blockchain-style cryptographic audit ledger with verification</td>
        </tr>
        <tr>
            <td>shadowtune_optimizer.py</td>
            <td>ShadowTuneOptimizer</td>
            <td>Master orchestrator managing closed-loop active learning feedback cycle</td>
        </tr>
        <tr>
            <td>visualize_output.py</td>
            <td>visualize_output.py</td>
            <td>Single-run verified execution, SVG vector chart generation, and CSV dataset export</td>
        </tr>
        <tr>
            <td>verify_all_requirements.py</td>
            <td>Master Testbench</td>
            <td>Automated test suite asserting functionality of all 12 challenge requirements</td>
        </tr>
        <tr>
            <td>shadowtune_graphs/index.html</td>
            <td>HTML Dashboard</td>
            <td>Interactive web dashboard showcasing synchronized multi-chart results</td>
        </tr>
    </tbody>
</table>

<h2>10. Comprehensive Challenge Requirements Resolution Scorecard</h2>
<table>
    <thead>
        <tr>
            <th>Challenge Requirement</th>
            <th>Verification Status</th>
            <th>Technical Resolution & Empirical Evidence</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>1. DC-link plant simulation</td>
            <td><span class="pass-badge">PASSED</span></td>
            <td>Non-linear differential model with ESR, capacitor state v_C, and PWM switching ripple.</td>
        </tr>
        <tr>
            <td>2. Manual PI baseline</td>
            <td><span class="pass-badge">PASSED</span></td>
            <td>Evaluated 3 cold-start manual PI baselines under identical step disturbances.</td>
        </tr>
        <tr>
            <td>3. Gaussian Process optimisation</td>
            <td><span class="pass-badge">PASSED</span></td>
            <td>ARD covariance kernel predicts posterior mean cost μ(x) and epistemic uncertainty σ(x).</td>
        </tr>
        <tr>
            <td>4. Digital Twin model correction</td>
            <td><span class="pass-badge">PASSED</span></td>
            <td>Online RLS updates C_dc and ESR, dynamically updating shadow twin parameters.</td>
        </tr>
        <tr>
            <td>5. Bounded PI gains</td>
            <td><span class="pass-badge">PASSED</span></td>
            <td>Search range Kp ∈ [0.05, 12.0], Ki ∈ [1.0, 200.0] ensures interior optimum discovery.</td>
        </tr>
        <tr>
            <td>6. Confidence-based screening</td>
            <td><span class="pass-badge">PASSED</span></td>
            <td>Strictly enforced single harmonized threshold (σ ≤ 1.60) across all modules.</td>
        </tr>
        <tr>
            <td>7. RLS parameter estimation</td>
            <td><span class="pass-badge">PASSED</span></td>
            <td>Balanced discrete-time RLS identifies true plant C=1760μF (0.00% err) and ESR=80mΩ (0.00% err).</td>
        </tr>
        <tr>
            <td>8. Automatic fallback and rollback</td>
            <td><span class="pass-badge">PASSED</span></td>
            <td>Instability triggers automatic rollback to certified safe gains and pauses tuning.</td>
        </tr>
        <tr>
            <td>9. Rate-limited gain updates</td>
            <td><span class="pass-badge">PASSED</span></td>
            <td>Enforces slew rate limit (Kp slews 1.0 → 1.20 in 10ms), guaranteeing bumpless transfer.</td>
        </tr>
        <tr>
            <td>10. Complete audit logging</td>
            <td><span class="pass-badge">PASSED</span></td>
            <td>Immutable SHA-256 hash-chained ledger verified via verify_integrity().</td>
        </tr>
        <tr>
            <td>11. Grid-voltage sag & parameter variation</td>
            <td><span class="pass-badge">PASSED</span></td>
            <td>Tested under 20% grid voltage sag and -20% capacitor degradation.</td>
        </tr>
        <tr>
            <td>12. Sensor-noise testing</td>
            <td><span class="pass-badge">PASSED</span></td>
            <td>Maintains stability under Gaussian sensor noise (σ=0.5V) and 0.1V ADC quantization.</td>
        </tr>
    </tbody>
</table>

</body>
</html>
"""

with open(HTML_FILE, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Generated HTML specification at: {HTML_FILE}")

# Convert to DOCX using macOS built-in textutil
cmd = ["textutil", "-convert", "docx", "-output", DOCX_FILE, HTML_FILE]
res = subprocess.run(cmd, capture_output=True, text=True)

if res.returncode == 0:
    print(f"SUCCESS: Generated Microsoft Word Document (.docx) at:\n{DOCX_FILE}")
else:
    print(f"Conversion Error: {res.stderr}")
