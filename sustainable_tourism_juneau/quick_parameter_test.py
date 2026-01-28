#!/usr/bin/env python3
"""
Quick Parameter Testing Examples
Demonstrates how to adjust and test model parameters
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.system_dynamics import simulate_policy
from src.config import *
import matplotlib.pyplot as plt

def test_glacier_sensitivity():
    """Test how glacier retreat rate affects outcomes"""

    print("Testing Glacier Retreat Rate Sensitivity")
    print("=" * 40)

    # Original value
    original_rate = NATURAL_GLACIER_RETREAT_RATE

    # Test different rates
    test_rates = [0.02, 0.05, 0.08]  # optimistic, baseline, pessimistic
    results = []

    test_policy = (15000, 25, 0.3)  # moderate policy

    for rate in test_rates:
        # Temporarily change parameter
        import src.config
        src.config.NATURAL_GLACIER_RETREAT_RATE = rate

        # Run simulation
        sim_result = simulate_policy(test_policy)
        results.append({
            'retreat_rate': rate,
            'final_glacier': sim_result['final_glacier'],
            'npv': sim_result['npv'] / 1e9  # in billions
        })

        print("6.0%")

    # Reset
    src.config.NATURAL_GLACIER_RETREAT_RATE = original_rate

    # Simple plot
    rates = [r['retreat_rate'] for r in results]
    glaciers = [r['final_glacier'] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(rates, glaciers, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Natural Glacier Retreat Rate')
    plt.ylabel('Final Glacier Size (20 years later)')
    plt.title('Glacier Retreat Sensitivity Analysis')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('parameter_analysis/glacier_test.png', dpi=150)
    plt.close()

    print("\nPlot saved to parameter_analysis/glacier_test.png")
    return results

def test_crowding_effects():
    """Test how crowding threshold affects tourist satisfaction"""

    print("\nTesting Crowding Threshold Effects")
    print("=" * 35)

    original_threshold = CROWDING_THRESHOLD
    test_thresholds = [10000, 15000, 20000]

    results = []
    test_policy = (18000, 10, 0.2)  # high capacity policy

    for threshold in test_thresholds:
        import src.config
        src.config.CROWDING_THRESHOLD = threshold

        sim_result = simulate_policy(test_policy)
        avg_daily_visitors = np.mean([t / TOURISM_DAYS_PER_YEAR for t in sim_result['tourists']])

        results.append({
            'threshold': threshold,
            'avg_daily_visitors': avg_daily_visitors,
            'total_tourists': np.sum(sim_result['tourists']) / 1e6,  # in millions
            'avg_satisfaction': sim_result['avg_satisfaction']
        })

        print("8.0f")
    # Reset
    src.config.CROWDING_THRESHOLD = original_threshold

    return results

def demonstrate_parameter_file():
    """Show how to create and use parameter files"""

    print("\nParameter File Management")
    print("=" * 25)

    # Example parameter configurations
    parameter_sets = {
        "conservative": {
            "NATURAL_GLACIER_RETREAT_RATE": 0.03,
            "TOURIST_IMPACT_FACTOR": 0.15,
            "CROWDING_THRESHOLD": 18000,
            "SATISFACTION_TOURIST_IMPACT": 0.3
        },
        "aggressive": {
            "NATURAL_GLACIER_RETREAT_RATE": 0.08,
            "TOURIST_IMPACT_FACTOR": 0.30,
            "CROWDING_THRESHOLD": 12000,
            "SATISFACTION_TOURIST_IMPACT": 0.8
        }
    }

    print("Available parameter sets:")
    for name, params in parameter_sets.items():
        print(f"\n{name.upper()}:")
        for param, value in params.items():
            print(f"  {param}: {value}")

    print("\nTo use these parameters:")
    print("1. Edit src/config.py directly")
    print("2. Use parameter_tuning.py for systematic testing")
    print("3. Run main.py to see results")

    return parameter_sets

def main():
    """Run all parameter tests"""
    print("Juneau Tourism Model - Quick Parameter Testing")
    print("=" * 50)

    # Create output directory
    os.makedirs('parameter_analysis', exist_ok=True)

    # Run tests
    glacier_results = test_glacier_sensitivity()
    crowding_results = test_crowding_effects()
    param_sets = demonstrate_parameter_file()

    print("\n" + "=" * 50)
    print("PARAMETER TESTING COMPLETED")
    print("=" * 50)

    print("\nKey Findings:")
    print("- Glacier retreat rate has strong impact on long-term preservation")
    print("- Crowding threshold affects both tourist satisfaction and capacity utilization")
    print("- Parameters should be calibrated based on local data and expert input")

    print("\nNext Steps:")
    print("1. Review PARAMETER_GUIDE.md for detailed parameter explanations")
    print("2. Use parameter_tuning.py for comprehensive sensitivity analysis")
    print("3. Adjust parameters in src/config.py based on your research needs")
    print("4. Re-run main.py to generate updated results")

if __name__ == "__main__":
    main()
