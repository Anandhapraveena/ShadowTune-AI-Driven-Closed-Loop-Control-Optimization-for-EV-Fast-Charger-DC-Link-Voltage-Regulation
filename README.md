# ShadowTune ⚡

### AI-Driven, Safety-Constrained PI Controller Optimization

ShadowTune is an intelligent, Python-based optimization framework for EV fast-charger DC-link voltage regulation. It combines **Gaussian Process Regression, Active Learning, Constrained Optimization, and Digital Twin Simulation** to intelligently identify effective PI controller gains (\(K_p,K_i\)).

Unlike conventional trial-and-error tuning, ShadowTune learns from previous simulation results, predicts the performance of new gain combinations, estimates uncertainty, and selects promising candidates while respecting predefined safety constraints.

## Intelligent AI Capabilities

- **Learning:** Gaussian Process Regression learns the relationship between PI gains and controller performance.
- **Prediction:** Estimates expected optimization cost and prediction uncertainty.
- **Decision-Making:** Active Learning selects promising candidates for evaluation.
- **Safety-Aware Optimization:** Rejects candidates that violate predefined safety limits.
- **Adaptive Feedback:** Uses digital-twin simulation results to improve subsequent optimization decisions.

## Key Features

- AI-driven PI gain optimization
- Gaussian Process-based performance prediction
- Uncertainty-aware candidate selection
- Safety-constrained optimization
- Digital twin simulation of DC-link voltage response
- Performance evaluation and visualization
- Audit logging and verification utilities

## Project Structure

```text
ShadowTune/
│
├── acquisition.py
├── audit_log.py
├── challenge_scorecard.json
├── demo.py
├── digital_twin_simulator.py
├── generate_word_doc.py
├── gp_model.py
├── rls_estimator.py
├── run_complete_summary.py
├── safety_constraints.py
├── shadowtune_optimizer.py
├── verify_all_requirements.py
├── visualize_output.py
│
├── shadowtune_graphs/
├── README.md
└── .gitignore
```

## Technologies Used

Python | Machine Learning | Gaussian Processes | Active Learning | Digital Twins | Control Systems

## Execution

Run the main demonstration:

```bash
python3 demo.py
```

Generate visualizations:

```bash
python3 visualize_output.py
```

## Project Goal

To develop an intelligent, adaptive, and safety-aware optimization framework that improves PI controller tuning for EV fast-charging applications.

**Implementation Status:** Currently demonstrated through software-based digital-twin simulation. Hardware validation is a future step.

**AI Approach:** Intelligent prediction, uncertainty estimation, iterative learning, and safety-aware decision-making.
