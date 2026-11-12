# ShadowTune Gaussian Process Optimization Module

A specialized, safety-constrained Gaussian Process (GP) active learning and Bayesian optimization module for tuning PI controller gains ($K_p$, $K_i$) regulating DC-link voltage in power converters and electric drive systems.

---

## 1. What the Solution Does

In power converter DC-link regulation (e.g. Active Front End rectifiers, EV motor drives, grid-tied solar/wind inverters):
- Evaluating arbitrary gains directly on physical equipment risks **overvoltage trips, capacitor dielectric stress, and converter instability**.
- Running exhaustive grid searches in high-fidelity switching digital twins (transient ODEs, PLECS, Simulink) is **computationally slow and resource-heavy**.

This module solves the problem through **Safety-Constrained Active Learning**:
1. **Learns from Historical Tests**: Trains Gaussian Process surrogates on past $(K_p, K_i)$ trials and their measured DC-link performance.
2. **Predicts Untested Gains & Epistemic Uncertainty**: Computes mean predicted response $\mu(K_p, K_i)$ and posterior standard deviation $\sigma(K_p, K_i)$.
3. **Multi-Objective Cost Formulation**: Jointly optimizes:
   - **Voltage Overshoot ($M_p$)**
   - **Settling Time ($t_s$)**
   - **Voltage Ripple ($\Delta V_{ripple}$)**
   - **Integral Absolute Error (IAE)**
4. **Safety & Confidence Gating**:
   - Rejects gains predicted to violate voltage overshoot, ripple, or settling limits.
   - Rejects gains with **excessive prediction uncertainty** ($\sigma > \sigma_{threshold}$).
5. **Continuous Model Updating**: Automatically incorporates each digital twin evaluation into the GP memory, refining accuracy and pruning search spaces over time.

---

## 2. File Structure

```
shadowtune_gp_optimizer/
├── gp_model.py                # Pure-Python Gaussian Process Regressor with ARD kernel and Cholesky solver
├── safety_constraints.py      # Hard electrical thresholds & uncertainty rejection gating
├── digital_twin_simulator.py  # Physics-based DC-link converter simulation model
├── acquisition.py             # Constrained Lower Confidence Bound (cLCB) prioritizer
├── shadowtune_optimizer.py    # Master coordinator managing active learning loop
├── demo.py                    # Runnable end-to-end verification and performance report
└── README.md                  # System architecture and integration documentation
```

---

## 3. How to Run the Demo

From the project directory:
```bash
python3 demo.py
```
*(No external third-party dependencies required. Compatible with Python 3.9+)*
