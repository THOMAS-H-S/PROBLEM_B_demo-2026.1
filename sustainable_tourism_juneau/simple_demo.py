#!/usr/bin/env python3
"""
Simple demo script without visualization dependencies
Shows the core functionality of the tourism model
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from system_dynamics import simulate_policy
from utils import format_policy_recommendation, calculate_policy_metrics

def main():
    """Main demo function"""

    print("=" * 70)
    print("MCM 2025 Problem B: Sustainable Tourism in Juneau, Alaska")
    print("SYSTEM DYNAMICS MODEL DEMONSTRATION")
    print("=" * 70)

    # Define three policy scenarios
    policies = {
        'Economic Priority': {
            'policy': (18000, 5, 0.12),  # High capacity, low fee, low investment
            'description': 'Maximize tourism revenue with minimal regulations'
        },
        'Balanced Approach': {
            'policy': (14000, 25, 0.35),  # Moderate values
            'description': 'Balance economic, environmental, and social objectives'
        },
        'Environmental Priority': {
            'policy': (11000, 40, 0.48),  # Low capacity, high fee, high investment
            'description': 'Prioritize glacier preservation and resident quality of life'
        }
    }

    print("\nPOLICY SCENARIOS ANALYSIS (2025-2045)")
    print("-" * 50)

    results_summary = []

    for name, config in policies.items():
        print(f"\n{name.upper()}")
        print(f"Description: {config['description']}")
        print("-" * 40)

        # Run simulation
        policy_vars = config['policy']
        sim_results = simulate_policy(policy_vars)

        # Calculate metrics
        metrics = calculate_policy_metrics(sim_results)

        # Format policy recommendation
        policy_data = {
            'daily_cap': policy_vars[0],
            'entrance_fee': policy_vars[1],
            'eco_investment_ratio': policy_vars[2],
            'npv': sim_results['npv'],
            'final_glacier': sim_results['final_glacier'],
            'avg_satisfaction': sim_results['avg_satisfaction']
        }

        recommendation = format_policy_recommendation(policy_data)
        print(f"Policy: {recommendation}")

        # Key metrics
        print("Key Metrics:")
        print(f"  NPV (20-year): ${sim_results['npv']/1e6:.1f}M")
        print(f"  Total tourists: {metrics['total_tourists_20yr']:,.0f}")
        print(f"  Glacier preservation: {sim_results['final_glacier']*100:.1f}%")
        print(f"  Avg satisfaction: {sim_results['avg_satisfaction']*100:.1f}%")
        print(f"  Avg annual revenue: ${metrics['avg_annual_revenue']/1e6:.1f}M")
        # Store results for comparison
        results_summary.append({
            'name': name,
            'npv': sim_results['npv'] / 1e6,  # in millions
            'glacier': sim_results['final_glacier'] * 100,  # percentage
            'satisfaction': sim_results['avg_satisfaction'] * 100,  # percentage
            'total_tourists': metrics['total_tourists_20yr'] / 1e6  # in millions
        })

    # Comparative analysis
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS")
    print("=" * 70)

    print("\nTRADE-OFF SUMMARY:")
    print(f"{'Policy':<20} {'NPV':<8} {'Glacier':<8} {'Satisf.':<8} {'Tourists':<8}")
    print("-" * 50)

    for result in results_summary:
        print(f"{result['name']:<20} ${result['npv']:<7.1f} {result['glacier']:<7.1f}% {result['satisfaction']:<7.1f}% {result['total_tourists']:<7.1f}M")

    print("\nKEY INSIGHTS:")
    print("-" * 15)
    print("- Economic and environmental goals show strong negative correlation")
    print("- Tourist capacity has the largest impact on all objectives")
    print("- Environmental investment provides diminishing returns")
    print("- Entrance fees affect demand elasticity but not glacier retreat")
    print("- Optimal policies require balancing all three objectives")

    print("\nMODEL CHARACTERISTICS:")
    print("-" * 22)
    print("- Time horizon: 20 years (2025-2045)")
    print("- Annual time steps with seasonal adjustment")
    print("- Feedback loops: attraction -> demand -> crowding -> satisfaction")
    print("- Glacier dynamics: natural + tourist-accelerated retreat")
    print("- Economic model: capacity constraints + crowding penalties")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETED")
    print("=" * 70)
    print("\nTo run full optimization with NSGA-II:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run main script: python main.py")
    print("3. Check results/ directory for visualizations and Pareto solutions")

if __name__ == "__main__":
    main()
