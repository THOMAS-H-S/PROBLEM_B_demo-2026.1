#!/usr/bin/env python3
"""
Demo script showing how to run the Sustainable Tourism Model
This demonstrates the key components without requiring full optimization
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from system_dynamics import simulate_policy
from utils import format_policy_recommendation, calculate_policy_metrics
from visualization import plot_time_series
import matplotlib.pyplot as plt

def demo_policies():
    """Demonstrate different policy scenarios"""

    print("Sustainable Tourism Management: Juneau, Alaska")
    print("=" * 50)

    # Define three representative policies
    policies = {
        'Economic Priority': (18000, 5, 0.12),    # High capacity, low fee, low investment
        'Balanced': (14000, 25, 0.35),            # Moderate values
        'Environmental Priority': (11000, 40, 0.48)  # Low capacity, high fee, high investment
    }

    results = {}

    for name, policy in policies.items():
        print(f"\nTesting {name} Policy:")
        print("-" * 30)

        # Run simulation
        sim_results = simulate_policy(policy)
        results[name] = sim_results

        # Calculate metrics
        metrics = calculate_policy_metrics(sim_results)

        # Display results
        recommendation = format_policy_recommendation({
            'daily_cap': policy[0],
            'entrance_fee': policy[1],
            'eco_investment_ratio': policy[2],
            'npv': sim_results['npv'],
            'final_glacier': sim_results['final_glacier'],
            'avg_satisfaction': sim_results['avg_satisfaction']
        })

        print(recommendation)
        print(f"Total tourists (20 years): {metrics['total_tourists_20yr']:,.0f}")
        print(f"Average annual revenue: ${metrics['avg_annual_revenue']/1e6:.1f}M")

    # Create comparison visualization
    print("\n" + "=" * 50)
    print("Generating Time Series Comparison...")

    policy_list = [(results[name], name) for name in policies.keys()]

    # Simple time series plot for one policy
    plot_time_series(results['Balanced'], 'Balanced Policy Demo',
                    'results/demo_time_series.png')

    print("Demo completed! Check 'results/demo_time_series.png' for visualization.")

    return results

if __name__ == "__main__":
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    demo_policies()
