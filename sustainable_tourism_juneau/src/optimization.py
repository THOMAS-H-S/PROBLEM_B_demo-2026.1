"""
Multi-objective optimization using NSGA-II algorithm
Couples with system dynamics model to find Pareto optimal tourism policies
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from platypus import NSGAII, Problem, Real, nondominated
try:
    from .system_dynamics import simulate_policy
    from .config import *
    from .utils import normalize_objectives
except ImportError:
    from system_dynamics import simulate_policy
    from config import *
    from utils import normalize_objectives


def tourism_optimization_problem():
    """
    Define the multi-objective optimization problem for tourism management.

    Returns:
    - problem: Platypus Problem object
    """
    def objective_function(vars):
        """
        Objective function for NSGA-II.
        We minimize: [-NPV, -final_glacier, -avg_satisfaction]
        """
        policy = (vars[0], vars[1], vars[2])  # daily_cap, entrance_fee, eco_investment_ratio
        results = simulate_policy(policy)

        # Objectives to minimize:
        # 1. Negative NPV (maximize NPV)
        # 2. Negative final glacier size (maximize glacier preservation)
        # 3. Negative average satisfaction (maximize resident satisfaction)
        objectives = [
            -results['npv'],                    # minimize -NPV = maximize NPV
            -results['final_glacier'],          # minimize -glacier = maximize glacier
            -results['avg_satisfaction']        # minimize -satisfaction = maximize satisfaction
        ]

        return objectives

    # Define decision variables
    problem = Problem(3, 3)  # 3 variables, 3 objectives

    # Daily capacity: 10,000 to 20,000
    problem.types[0] = Real(DECISION_VARIABLE_RANGES['daily_cap'][0],
                           DECISION_VARIABLE_RANGES['daily_cap'][1])

    # Entrance fee: $0 to $50
    problem.types[1] = Real(DECISION_VARIABLE_RANGES['entrance_fee'][0],
                           DECISION_VARIABLE_RANGES['entrance_fee'][1])

    # Eco-investment ratio: 10% to 50%
    problem.types[2] = Real(DECISION_VARIABLE_RANGES['eco_investment_ratio'][0],
                           DECISION_VARIABLE_RANGES['eco_investment_ratio'][1])

    problem.function = objective_function

    return problem


def run_nsga_ii_optimization(population_size: int = NSGA_II_POPULATION_SIZE,
                           generations: int = NSGA_II_GENERATIONS,
                           seed: int = NSGA_II_SEED) -> pd.DataFrame:
    """
    Run NSGA-II optimization to find Pareto optimal tourism policies.

    Parameters:
    - population_size: size of the population
    - generations: number of generations to run
    - seed: random seed for reproducibility

    Returns:
    - pareto_solutions: DataFrame with Pareto optimal solutions
    """
    print("Starting NSGA-II optimization...")
    print(f"Population size: {population_size}, Generations: {generations}")

    # Initialize the problem
    problem = tourism_optimization_problem()

    # Initialize NSGA-II algorithm
    algorithm = NSGAII(problem, population_size=population_size)

    # Set random seed
    np.random.seed(seed)

    # Run optimization
    solutions = []
    for i in range(generations):
        algorithm.step()
        if (i + 1) % 50 == 0:
            print(f"Generation {i + 1}/{generations} completed")

    # Extract Pareto optimal solutions
    pareto_front = nondominated(algorithm.result)

    print(f"Found {len(pareto_front)} Pareto optimal solutions")

    # Convert solutions to DataFrame
    results_data = []
    for solution in pareto_front:
        daily_cap = solution.variables[0]
        entrance_fee = solution.variables[1]
        eco_ratio = solution.variables[2]

        # Run simulation to get detailed results
        policy = (daily_cap, entrance_fee, eco_ratio)
        sim_results = simulate_policy(policy)

        result_dict = {
            'daily_cap': daily_cap,
            'entrance_fee': entrance_fee,
            'eco_investment_ratio': eco_ratio,
            'npv': sim_results['npv'],
            'final_glacier': sim_results['final_glacier'],
            'avg_satisfaction': sim_results['avg_satisfaction'],
            'total_tourists_20yr': np.sum(sim_results['tourists']),
            'avg_annual_revenue': np.mean(sim_results['revenue'])
        }
        results_data.append(result_dict)

    pareto_df = pd.DataFrame(results_data)

    return pareto_df


def save_pareto_solutions(pareto_solutions: pd.DataFrame, filename: str = PARETO_SOLUTIONS_FILE):
    """
    Save Pareto optimal solutions to CSV file.

    Parameters:
    - pareto_solutions: DataFrame with Pareto solutions
    - filename: output filename
    """
    pareto_solutions.to_csv(filename, index=False)
    print(f"Pareto solutions saved to {filename}")


def load_pareto_solutions(filename: str = PARETO_SOLUTIONS_FILE) -> pd.DataFrame:
    """
    Load Pareto optimal solutions from CSV file.

    Parameters:
    - filename: input filename

    Returns:
    - pareto_solutions: DataFrame with Pareto solutions
    """
    try:
        pareto_df = pd.read_csv(filename)
        print(f"Loaded {len(pareto_df)} Pareto solutions from {filename}")
        return pareto_df
    except FileNotFoundError:
        print(f"File {filename} not found. Run optimization first.")
        return pd.DataFrame()
