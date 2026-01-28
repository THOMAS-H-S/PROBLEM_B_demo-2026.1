#!/usr/bin/env python3
"""
Parameter Tuning and Sensitivity Analysis Tool
for Juneau Sustainable Tourism Management Model

This script helps users adjust model parameters and analyze their impact
on simulation results and optimization outcomes.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.system_dynamics import simulate_policy
from src.optimization import run_nsga_ii_optimization
from src.config import *

class ParameterTuner:
    """Parameter tuning and sensitivity analysis tool"""

    def __init__(self):
        self.base_params = self._load_base_parameters()
        self.results_dir = "parameter_analysis"
        os.makedirs(self.results_dir, exist_ok=True)

    def _load_base_parameters(self) -> Dict:
        """Load current parameters from config"""
        return {
            'natural_glacier_retreat': NATURAL_GLACIER_RETREAT_RATE,
            'tourist_impact_factor': TOURIST_IMPACT_FACTOR,
            'crowding_threshold': CROWDING_THRESHOLD,
            'whale_success_rate': WHALE_WATCHING_SUCCESS_RATE,
            'discount_rate': DISCOUNT_RATE,
            'satisfaction_impact': SATISFACTION_TOURIST_IMPACT,
            'population': POPULATION_JUNEAU,
            'tourism_days': TOURISM_DAYS_PER_YEAR
        }

    def sensitivity_analysis(self, param_name: str, param_range: List[float],
                           test_policy: Tuple = (15000, 25, 0.3)) -> pd.DataFrame:
        """
        Perform sensitivity analysis for a single parameter

        Parameters:
        - param_name: parameter to test
        - param_range: list of values to test
        - test_policy: (daily_cap, entrance_fee, eco_ratio) to test with

        Returns:
        - DataFrame with results for each parameter value
        """
        print(f"Running sensitivity analysis for {param_name}")
        print(f"Testing range: {param_range}")

        results = []

        for param_value in param_range:
            # Temporarily modify parameter
            self._set_parameter(param_name, param_value)

            # Run simulation
            sim_results = simulate_policy(test_policy)

            # Store results
            result = {
                param_name: param_value,
                'npv': sim_results['npv'],
                'final_glacier': sim_results['final_glacier'],
                'avg_satisfaction': sim_results['avg_satisfaction'],
                'total_tourists': np.sum(sim_results['tourists'])
            }
            results.append(result)

            print(".2f")

        # Reset to base parameters
        self._set_parameter(param_name, self.base_params[param_name])

        return pd.DataFrame(results)

    def _set_parameter(self, param_name: str, value: float):
        """Temporarily set a parameter value"""
        # This is a simplified version - in practice, you'd modify the config
        # For demonstration, we'll use global variable modification
        if param_name == 'natural_glacier_retreat':
            global NATURAL_GLACIER_RETREAT_RATE
            NATURAL_GLACIER_RETREAT_RATE = value
        elif param_name == 'tourist_impact_factor':
            global TOURIST_IMPACT_FACTOR
            TOURIST_IMPACT_FACTOR = value
        elif param_name == 'crowding_threshold':
            global CROWDING_THRESHOLD
            CROWDING_THRESHOLD = value
        elif param_name == 'discount_rate':
            global DISCOUNT_RATE
            DISCOUNT_RATE = value
        elif param_name == 'satisfaction_impact':
            global SATISFACTION_TOURIST_IMPACT
            SATISFACTION_TOURIST_IMPACT = value
        elif param_name == 'population':
            global POPULATION_JUNEAU
            POPULATION_JUNEAU = int(value)
        elif param_name == 'tourism_days':
            global TOURISM_DAYS_PER_YEAR
            TOURISM_DAYS_PER_YEAR = int(value)

    def plot_sensitivity_results(self, results_df: pd.DataFrame, param_name: str):
        """Plot sensitivity analysis results"""
        param_col = results_df.columns[0]  # First column is the parameter

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Sensitivity Analysis: {param_name.replace("_", " ").title()}', fontsize=16)

        # NPV sensitivity
        axes[0,0].plot(results_df[param_col], results_df['npv']/1e9, 'o-', linewidth=2, markersize=6)
        axes[0,0].set_ylabel('NPV ($B)')
        axes[0,0].set_title('Economic Impact')
        axes[0,0].grid(True, alpha=0.3)

        # Glacier preservation
        axes[0,1].plot(results_df[param_col], results_df['final_glacier'], 's-', linewidth=2, markersize=6, color='green')
        axes[0,1].set_ylabel('Final Glacier Size')
        axes[0,1].set_title('Environmental Impact')
        axes[0,1].grid(True, alpha=0.3)

        # Resident satisfaction
        axes[1,0].plot(results_df[param_col], results_df['avg_satisfaction'], '^-', linewidth=2, markersize=6, color='orange')
        axes[1,0].set_xlabel(param_name.replace('_', ' ').title())
        axes[1,0].set_ylabel('Average Satisfaction')
        axes[1,0].set_title('Social Impact')
        axes[1,0].grid(True, alpha=0.3)

        # Total tourists
        axes[1,1].plot(results_df[param_col], results_df['total_tourists']/1e6, 'd-', linewidth=2, markersize=6, color='purple')
        axes[1,1].set_xlabel(param_name.replace('_', ' ').title())
        axes[1,1].set_ylabel('Total Tourists (M)')
        axes[1,1].set_title('Tourism Volume')
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/sensitivity_{param_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Sensitivity plot saved: {self.results_dir}/sensitivity_{param_name}.png")

    def scenario_analysis(self, scenarios: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Run multiple scenarios with different parameter combinations

        Parameters:
        - scenarios: dict of scenario_name -> parameter_dict

        Returns:
        - DataFrame with results for each scenario
        """
        print("Running scenario analysis...")
        results = []

        for scenario_name, params in scenarios.items():
            print(f"Testing scenario: {scenario_name}")

            # Set all parameters for this scenario
            for param_name, value in params.items():
                self._set_parameter(param_name, value)

            # Run optimization (simplified - just run one policy for demo)
            test_policy = (15000, 25, 0.3)
            sim_results = simulate_policy(test_policy)

            result = {
                'scenario': scenario_name,
                'npv': sim_results['npv'],
                'final_glacier': sim_results['final_glacier'],
                'avg_satisfaction': sim_results['avg_satisfaction'],
                'total_tourists': np.sum(sim_results['tourists']),
                **params  # Include parameter values
            }
            results.append(result)

            # Reset parameters
            for param_name in params.keys():
                self._set_parameter(param_name, self.base_params[param_name])

        return pd.DataFrame(results)

    def save_parameter_config(self, filename: str = "custom_parameters.json"):
        """Save current parameter configuration"""
        with open(f'{self.results_dir}/{filename}', 'w') as f:
            json.dump(self.base_params, f, indent=2)
        print(f"Parameter configuration saved to {self.results_dir}/{filename}")

    def load_parameter_config(self, filename: str = "custom_parameters.json"):
        """Load parameter configuration"""
        try:
            with open(f'{self.results_dir}/{filename}', 'r') as f:
                loaded_params = json.load(f)

            for param_name, value in loaded_params.items():
                if param_name in self.base_params:
                    self._set_parameter(param_name, value)
                    self.base_params[param_name] = value

            print(f"Parameter configuration loaded from {self.results_dir}/{filename}")
            return True
        except FileNotFoundError:
            print(f"Parameter file not found: {self.results_dir}/{filename}")
            return False

def main():
    """Main parameter tuning interface"""
    tuner = ParameterTuner()

    print("=" * 60)
    print("Juneau Sustainable Tourism Model - Parameter Tuning Tool")
    print("=" * 60)

    while True:
        print("\nAvailable operations:")
        print("1. Sensitivity analysis for single parameter")
        print("2. Scenario analysis with multiple parameters")
        print("3. Save current parameter configuration")
        print("4. Load parameter configuration")
        print("5. View current parameters")
        print("6. Exit")

        choice = input("\nSelect operation (1-6): ").strip()

        if choice == '1':
            # Sensitivity analysis
            print("\nAvailable parameters for sensitivity analysis:")
            params = {
                '1': ('natural_glacier_retreat', 'Natural Glacier Retreat Rate'),
                '2': ('tourist_impact_factor', 'Tourist Impact Factor'),
                '3': ('crowding_threshold', 'Crowding Threshold'),
                '4': ('discount_rate', 'Discount Rate'),
                '5': ('satisfaction_impact', 'Satisfaction Impact Factor')
            }

            for key, (param, desc) in params.items():
                print(f"{key}. {desc}")

            param_choice = input("Select parameter (1-5): ").strip()
            if param_choice in params:
                param_name, desc = params[param_choice]

                # Define test ranges based on parameter
                ranges = {
                    'natural_glacier_retreat': [0.02, 0.03, 0.05, 0.08, 0.10],
                    'tourist_impact_factor': [0.10, 0.15, 0.20, 0.25, 0.30],
                    'crowding_threshold': [10000, 12000, 15000, 18000, 20000],
                    'discount_rate': [0.02, 0.03, 0.05, 0.07, 0.10],
                    'satisfaction_impact': [0.2, 0.3, 0.5, 0.7, 0.9]
                }

                results = tuner.sensitivity_analysis(param_name, ranges[param_name])
                tuner.plot_sensitivity_results(results, param_name)

                print("
Results saved to parameter_analysis/ directory"
                print(f"✓ Sensitivity analysis completed for {desc}")

        elif choice == '2':
            # Scenario analysis
            scenarios = {
                'business_as_usual': {
                    'natural_glacier_retreat': 0.05,
                    'tourist_impact_factor': 0.20,
                    'crowding_threshold': 15000
                },
                'sustainable_growth': {
                    'natural_glacier_retreat': 0.03,
                    'tourist_impact_factor': 0.15,
                    'crowding_threshold': 18000
                },
                'aggressive_conservation': {
                    'natural_glacier_retreat': 0.02,
                    'tourist_impact_factor': 0.10,
                    'crowding_threshold': 20000
                }
            }

            results = tuner.scenario_analysis(scenarios)
            results.to_csv(f'{tuner.results_dir}/scenario_analysis.csv', index=False)
            print("
Scenario analysis results:"            print(results.to_string(index=False))
            print(f"\nDetailed results saved to {tuner.results_dir}/scenario_analysis.csv")

        elif choice == '3':
            tuner.save_parameter_config()

        elif choice == '4':
            tuner.load_parameter_config()

        elif choice == '5':
            print("\nCurrent Parameter Configuration:")
            print("-" * 40)
            for param, value in tuner.base_params.items():
                print("30")

        elif choice == '6':
            print("Exiting parameter tuning tool.")
            break

        else:
            print("Invalid choice. Please select 1-6.")

if __name__ == "__main__":
    main()
