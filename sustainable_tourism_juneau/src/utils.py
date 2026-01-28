"""
Utility functions for the Sustainable Tourism Management Model
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
try:
    from .config import *
except ImportError:
    from config import *


def calculate_npv(cash_flows: List[float], discount_rate: float = DISCOUNT_RATE) -> float:
    """
    Calculate Net Present Value of a cash flow series.

    Parameters:
    - cash_flows: list of annual cash flows
    - discount_rate: annual discount rate

    Returns:
    - npv: net present value
    """
    npv = 0
    for t, cf in enumerate(cash_flows):
        npv += cf / ((1 + discount_rate) ** t)
    return npv


def normalize_objectives(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize objectives for multi-objective optimization.
    For minimization: higher values are worse
    For maximization: lower values are worse

    Parameters:
    - results_df: DataFrame with columns ['npv', 'final_glacier', 'avg_satisfaction']

    Returns:
    - normalized_df: DataFrame with normalized objective values
    """
    # For NSGA-II, we want to minimize all objectives
    # NPV: we want to maximize NPV, so minimize -NPV
    # final_glacier: we want to maximize glacier size, so minimize -final_glacier
    # avg_satisfaction: we want to maximize satisfaction, so minimize -avg_satisfaction

    normalized_df = results_df.copy()

    # Min-max normalization for each objective
    for col in ['npv', 'final_glacier', 'avg_satisfaction']:
        min_val = results_df[col].min()
        max_val = results_df[col].max()

        if max_val > min_val:
            if col == 'npv':
                # For NPV, higher is better, so we minimize -NPV
                normalized_df[f'obj_{col}'] = -(results_df[col] - min_val) / (max_val - min_val)
            else:
                # For glacier and satisfaction, higher is better, so minimize -(value)
                normalized_df[f'obj_{col}'] = -(results_df[col] - min_val) / (max_val - min_val)
        else:
            normalized_df[f'obj_{col}'] = 0

    return normalized_df


def select_representative_policies(pareto_solutions: pd.DataFrame, n_policies: int = 3) -> pd.DataFrame:
    """
    Select representative policies from Pareto front for different objectives.

    Parameters:
    - pareto_solutions: DataFrame with Pareto optimal solutions
    - n_policies: number of representative policies to select

    Returns:
    - representative_policies: DataFrame with selected policies
    """
    # Select policies that maximize each objective while being close to Pareto front

    # Economic priority: maximize NPV
    economic_idx = pareto_solutions['npv'].idxmax()

    # Environmental priority: maximize final_glacier
    environmental_idx = pareto_solutions['final_glacier'].idxmax()

    # Social priority: maximize avg_satisfaction
    social_idx = pareto_solutions['avg_satisfaction'].idxmax()

    # If we need more policies, select balanced ones
    if n_policies > 3:
        # Find policy with best average normalized performance
        normalized = normalize_objectives(pareto_solutions[['npv', 'final_glacier', 'avg_satisfaction']])
        avg_performance = normalized[['obj_npv', 'obj_final_glacier', 'obj_avg_satisfaction']].mean(axis=1)
        balanced_idx = avg_performance.idxmax()
        indices = [economic_idx, environmental_idx, social_idx, balanced_idx]
    else:
        indices = [economic_idx, environmental_idx, social_idx]

    representative = pareto_solutions.loc[indices[:n_policies]].copy()

    # Add policy type labels
    policy_types = ['Economic Priority', 'Environmental Priority', 'Social Priority', 'Balanced']
    representative['policy_type'] = policy_types[:n_policies]

    return representative


def format_policy_recommendation(policy: pd.Series) -> str:
    """
    Format a policy into a readable recommendation string.

    Parameters:
    - policy: Series containing policy variables and objectives

    Returns:
    - recommendation: formatted string
    """
    daily_cap = int(policy['daily_cap'])
    entrance_fee = round(policy['entrance_fee'], 1)
    eco_ratio = round(policy['eco_investment_ratio'] * 100, 1)
    npv = round(policy['npv'] / 1e6, 1)  # in millions
    glacier = round(policy['final_glacier'] * 100, 1)
    satisfaction = round(policy['avg_satisfaction'] * 100, 1)

    return (f"每日限流 {daily_cap:,} 人，门票 ${entrance_fee}，"
            f"环保投入 {eco_ratio}% | "
            f"NPV: ${npv}M，冰川保存: {glacier}%，居民满意度: {satisfaction}%")


def calculate_policy_metrics(policy_results: Dict) -> Dict:
    """
    Calculate key performance metrics for a policy.

    Parameters:
    - policy_results: results dictionary from simulate_policy

    Returns:
    - metrics: dictionary with key performance indicators
    """
    tourists = np.array(policy_results['tourists'])
    revenue = np.array(policy_results['revenue'])
    glacier = np.array(policy_results['glacier'])
    satisfaction = np.array(policy_results['satisfaction'])

    metrics = {
        'total_tourists_20yr': np.sum(tourists),
        'avg_annual_tourists': np.mean(tourists),
        'total_revenue_20yr': np.sum(revenue),
        'avg_annual_revenue': np.mean(revenue),
        'npv_revenue': policy_results['npv'],
        'glacier_loss_percent': (INITIAL_GLACIER_SIZE - policy_results['final_glacier']) * 100,
        'avg_satisfaction': policy_results['avg_satisfaction'],
        'tourist_growth_rate': (tourists[-1] / tourists[0]) ** (1/20) - 1 if tourists[0] > 0 else 0,
        'revenue_volatility': np.std(revenue) / np.mean(revenue) if np.mean(revenue) > 0 else 0
    }

    return metrics
