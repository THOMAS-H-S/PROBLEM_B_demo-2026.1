#!/usr/bin/env python3
"""
Run optimized model variant v2 (with improved math) and save outputs to results_optimized_v2/
"""

import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent / 'src'))

from src.optimization import run_nsga_ii_optimization
from src.visualization import (
    plot_pareto_front_3d, plot_statistical_distributions, plot_probability_distributions,
    plot_correlation_matrix, plot_correlation_network, plot_decision_variable_importance,
    plot_cost_benefit_analysis, plot_scenario_comparison
)
from src.config import *


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def main():
    target = 'results_optimized_v2'
    ensure_dir(target)

    print('Running NSGA-II (optimized v2)...')
    pareto_df = run_nsga_ii_optimization(population_size=NSGA_II_POPULATION_SIZE,
                                        generations=NSGA_II_GENERATIONS,
                                        seed=NSGA_II_SEED)

    pareto_df.to_csv(f"{target}/pareto_solutions.csv", index=False)
    print('Saved pareto_solutions.csv')

    plot_pareto_front_3d(pareto_df, filename=f"{target}/pareto_front_3d.png")
    plot_statistical_distributions(pareto_df, filename=f"{target}/statistical_distributions.png")
    plot_probability_distributions(pareto_df, filename=f"{target}/probability_distributions.png")
    plot_correlation_matrix(pareto_df, filename=f"{target}/correlation_matrix.png")
    plot_correlation_network(pareto_df, filename=f"{target}/correlation_network.png")
    plot_decision_variable_importance(pareto_df, filename=f"{target}/variable_importance.png")
    plot_cost_benefit_analysis(pareto_df, filename=f"{target}/cost_benefit_analysis.png")

    # Basic scenario comparison based on summary stats
    scenarios = {
        'Conservative': {'npv': pareto_df['npv'].median() * 0.9, 'final_glacier': pareto_df['final_glacier'].median() * 1.05, 'avg_satisfaction': pareto_df['avg_satisfaction'].median() * 1.05},
        'Baseline': {'npv': pareto_df['npv'].median(), 'final_glacier': pareto_df['final_glacier'].median(), 'avg_satisfaction': pareto_df['avg_satisfaction'].median()},
        'Aggressive': {'npv': pareto_df['npv'].median() * 1.1, 'final_glacier': pareto_df['final_glacier'].median() * 0.95, 'avg_satisfaction': pareto_df['avg_satisfaction'].median() * 0.95}
    }

    scenario_struct = {}
    for name, v in scenarios.items():
        scenario_struct[name] = {
            'years': list(range(SIMULATION_START_YEAR, SIMULATION_END_YEAR + 1)),
            'tourists': [BASE_TOURISTS_2023 // SIMULATION_YEARS] * SIMULATION_YEARS,
            'revenue': [v['npv'] / SIMULATION_YEARS] * SIMULATION_YEARS,
            'glacier': [v['final_glacier']] * SIMULATION_YEARS,
            'final_glacier': v['final_glacier'],
            'satisfaction': [v['avg_satisfaction']] * SIMULATION_YEARS,
            'avg_satisfaction': v['avg_satisfaction'],
            'npv': v['npv']
        }

    plot_scenario_comparison(scenario_struct, filename=f"{target}/scenario_comparison.png")

    print('Optimized v2 run complete. Outputs in', target)


if __name__ == '__main__':
    main()


