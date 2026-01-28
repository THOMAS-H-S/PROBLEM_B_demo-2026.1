#!/usr/bin/env python3
"""
Run the optimized model and save outputs to a new results directory.
This script overrides output paths by passing explicit filenames to visualization functions.
"""

import os
from pathlib import Path
import sys

# Ensure src is importable
sys.path.append(str(Path(__file__).parent / 'src'))

from src.optimization import run_nsga_ii_optimization
from src.visualization import (
    plot_pareto_front_3d, plot_statistical_distributions, plot_regression_analysis,
    plot_probability_distributions, plot_correlation_matrix, plot_correlation_network,
    plot_decision_variable_importance, plot_policy_efficiency_frontier,
    plot_cost_benefit_analysis, plot_time_series_decomposition, plot_trend_analysis,
    plot_scenario_comparison, plot_interactive_dashboard
)
from src.config import *


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def main():
    target_dir = 'results_optimized'
    ensure_dir(target_dir)

    print('Running optimized NSGA-II...')
    pareto_df = run_nsga_ii_optimization(population_size=NSGA_II_POPULATION_SIZE,
                                        generations=NSGA_II_GENERATIONS,
                                        seed=NSGA_II_SEED)

    pareto_csv = f"{target_dir}/pareto_solutions.csv"
    pareto_df.to_csv(pareto_csv, index=False)
    print(f'Pareto solutions saved to {pareto_csv}')

    # Generate a core set of advanced visualizations into the new folder
    plot_pareto_front_3d(pareto_df, filename=f"{target_dir}/pareto_front_3d.png")
    plot_statistical_distributions(pareto_df, filename=f"{target_dir}/statistical_distributions.png")
    plot_regression_analysis(pareto_df, filename=f"{target_dir}/regression_analysis.png")
    plot_probability_distributions(pareto_df, filename=f"{target_dir}/probability_distributions.png")
    plot_correlation_matrix(pareto_df, filename=f"{target_dir}/correlation_matrix.png")
    plot_correlation_network(pareto_df, filename=f"{target_dir}/correlation_network.png")
    plot_decision_variable_importance(pareto_df, filename=f"{target_dir}/variable_importance.png")
    plot_policy_efficiency_frontier(pareto_df, filename=f"{target_dir}/efficiency_frontier.png")
    plot_cost_benefit_analysis(pareto_df, filename=f"{target_dir}/cost_benefit_analysis.png")

    # Scenario comparison using baseline scenarios (simple representative series)
    scenarios = {
        'Conservative': {'npv': 2500000000, 'final_glacier': 0.9, 'avg_satisfaction': 0.6},
        'Baseline': {'npv': 3000000000, 'final_glacier': 0.8, 'avg_satisfaction': 0.5},
        'Aggressive': {'npv': 2000000000, 'final_glacier': 0.95, 'avg_satisfaction': 0.7}
    }
    scenario_results = {}
    for name, vals in scenarios.items():
        scenario_results[name] = {
            'years': list(range(SIMULATION_START_YEAR, SIMULATION_END_YEAR + 1)),
            'tourists': [BASE_TOURISTS_2023 // SIMULATION_YEARS] * SIMULATION_YEARS,
            'revenue': [vals['npv'] / SIMULATION_YEARS] * SIMULATION_YEARS,
            'glacier': [vals['final_glacier']] * SIMULATION_YEARS,
            'satisfaction': [vals['avg_satisfaction']] * SIMULATION_YEARS,
            'npv': vals['npv']
        }

    plot_scenario_comparison(scenario_results, filename=f"{target_dir}/scenario_comparison.png")

    # Interactive dashboard (if available)
    try:
        plot_interactive_dashboard(pareto_df, filename=f"{target_dir}/interactive_dashboard.html")
    except Exception as e:
        print('Interactive dashboard skipped:', e)

    print('Optimized run complete. Outputs in', target_dir)


if __name__ == '__main__':
    main()


