from digital_twin_simulator import DCLinkDigitalTwin
from safety_constraints import SafetyThresholds
from shadowtune_optimizer import ShadowTuneOptimizer


def run_demo():
    print("=" * 80)
    print(" SHADOWTUNE GAUSSIAN PROCESS OPTIMIZATION ENGINE")
    print(" DC-Link Voltage PI Controller Auto-Tuning Loop")
    print("=" * 80)

    # 1. Instantiate Digital Twin & Safety Constraints
    digital_twin = DCLinkDigitalTwin()
    safety = SafetyThresholds(
        max_overshoot_pct=7.5,
        max_settling_time_s=0.045,
        max_voltage_ripple_v=2.2,
        max_allowed_uncertainty=1.6,
    )

    optimizer = ShadowTuneOptimizer(
        safety_thresholds=safety,
        digital_twin=digital_twin,
        exploration_beta=1.5,
    )

    # 2. Seed with historical tested gain records
    print("\n[Step 1] Loading previously tested historical PI gain evaluations...")
    initial_gain_experiments = [
        (0.40, 10.0),   # Very sluggish, conservative baseline
        (1.50, 40.0),   # Moderate initial tuning
        (3.00, 90.0),   # More aggressive tuning
    ]

    initial_records = []
    for kp, ki in initial_gain_experiments:
        metrics = digital_twin.simulate(kp, ki)
        initial_records.append({
            "kp": kp,
            "ki": ki,
            **metrics,
        })
        print(f"  Historical Gain (Kp={kp:4.2f}, Ki={ki:5.1f}) -> "
              f"Overshoot: {metrics['overshoot_pct']:5.2f}%, "
              f"Settling: {metrics['settling_time_s']*1000:4.1f}ms, "
              f"Ripple: {metrics['voltage_ripple_v']:4.2f}V, "
              f"IAE: {metrics['iae']:5.3f}, "
              f"Cost: {metrics['composite_cost']:5.3f}")

    optimizer.load_initial_history(initial_records)
    print("✓ Gaussian Process surrogates fitted on historical baseline.")

    # 3. Demonstration of Untested Gain Prediction & Gating
    print("\n[Step 2] Testing GP Prediction & Safety/Uncertainty Gating on Sample Untested Gains:")
    test_proposals = [
        (1.20, 35.0, "Moderate Untested Candidate"),
        (7.50, 140.0, "Over-Aggressive Candidate (Safety Risk)"),
        (0.08, 2.0, "Extreme Boundary Candidate"),
    ]

    for kp, ki, label in test_proposals:
        pred = optimizer.predict_candidate(kp, ki)
        status = "PASSED" if pred["is_acceptable"] else "FILTERED"
        print(f"\n  • Candidate: {label} (Kp={kp:.2f}, Ki={ki:.1f})")
        print(f"    Predicted Cost: {pred['predicted_cost']:.3f} ± {pred['uncertainty_cost_std']:.3f}")
        print(f"    Predicted Overshoot: {pred['predicted_overshoot_pct']:.2f}%, Settling: {pred['predicted_settling_time_s']*1000:.1f}ms")
        print(f"    Gating Result: [{status}] -> {pred['gating_status']}")

    # 4. Iterative ShadowTune Active Learning Optimization
    print("\n[Step 3] Running Iterative ShadowTune Optimization with Digital Twin Feedback:")
    print("-" * 80)
    print(f"{'Iter':<5} | {'Kp':<6} | {'Ki':<6} | {'Pred Cost':<10} | {'Act Cost':<9} | {'Pred Err':<9} | {'Overshoot':<9} | {'Settling':<9} | {'Ripple':<7} | {'IAE':<6}")
    print("-" * 80)

    num_iterations = 6
    for i in range(num_iterations):
        res = optimizer.run_optimization_step()
        if res["status"] == "EVALUATED_AND_UPDATED":
            m = res["metrics"]
            pc = res["prediction_comparison"]
            print(f"{res['iteration']:<5} | "
                  f"{res['evaluated_kp']:<6.2f} | "
                  f"{res['evaluated_ki']:<6.1f} | "
                  f"{pc['predicted_cost']:<10.3f} | "
                  f"{pc['actual_cost']:<9.3f} | "
                  f"{pc['error']:<9.3f} | "
                  f"{m['overshoot_pct']:<8.2f}% | "
                  f"{m['settling_time_s']*1000:<7.1f}ms | "
                  f"{m['voltage_ripple_v']:<6.2f}V | "
                  f"{m['iae']:<6.3f}")

    # 5. Final Summary and Best Gains
    best = optimizer.get_best_tested_gains()
    print("-" * 80)
    print("\n[Step 4] Optimization Complete! Best Discovered PI Controller Parameters:")
    print(f"  ★ Optimal Kp:            {best['kp']:.3f}")
    print(f"  ★ Optimal Ki:            {best['ki']:.2f}")
    print(f"  ★ DC-Link Overshoot:     {best['overshoot_pct']:.2f} % (Safe < 7.5%)")
    print(f"  ★ Settling Time:         {best['settling_time_s']*1000:.1f} ms")
    print(f"  ★ Voltage Ripple:        {best['voltage_ripple_v']:.2f} V pk-pk")
    print(f"  ★ Integral Abs Error:    {best['iae']:.4f} V*s")
    print(f"  ★ Minimum Composite Cost:{best['composite_cost']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
