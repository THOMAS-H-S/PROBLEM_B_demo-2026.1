#!/usr/bin/env python3
"""
测试高级可视化功能
快速验证新添加的可视化函数是否正常工作
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from src.system_dynamics import simulate_policy
from src.optimization import run_nsga_ii_optimization
from src.config import *

def test_basic_visualizations():
    """测试几个核心的新可视化函数"""
    print("Testing advanced visualization functions...")
    print("=" * 50)

    # 运行优化获取数据
    print("Running NSGA-II optimization...")
    pareto_solutions = run_nsga_ii_optimization(population_size=50, generations=100)  # 减少规模以加快测试

    if pareto_solutions.empty:
        print("No Pareto solutions found, skipping visualization tests")
        return

    print(f"Found {len(pareto_solutions)} Pareto solutions")

    # 测试敏感性龙卷风图
    print("Testing sensitivity tornado plot...")
    sensitivity_data = {
        'natural_glacier_retreat': {'low': 0.03, 'high': 0.08, 'baseline': 0.05},
        'tourist_impact_factor': {'low': 0.15, 'high': 0.25, 'baseline': 0.20},
        'crowding_threshold': {'low': 12000, 'high': 18000, 'baseline': 15000}
    }

    try:
        from src.visualization import plot_sensitivity_tornado
        plot_sensitivity_tornado(sensitivity_data)
        print("[OK] Sensitivity tornado plot generated")
    except Exception as e:
        print(f"[FAIL] Sensitivity tornado plot failed: {e}")

    # 测试相关性矩阵
    print("Testing correlation matrix...")
    try:
        from src.visualization import plot_correlation_matrix
        plot_correlation_matrix(pareto_solutions)
        print("[OK] Correlation matrix generated")
    except Exception as e:
        print(f"[FAIL] Correlation matrix failed: {e}")

    # 测试决策变量重要性
    print("Testing decision variable importance...")
    try:
        from src.visualization import plot_decision_variable_importance
        plot_decision_variable_importance(pareto_solutions)
        print("[OK] Decision variable importance plot generated")
    except Exception as e:
        print(f"[FAIL] Decision variable importance plot failed: {e}")

    # 测试成本效益分析
    print("Testing cost-benefit analysis...")
    try:
        from src.visualization import plot_cost_benefit_analysis
        plot_cost_benefit_analysis(pareto_solutions)
        print("[OK] Cost-benefit analysis generated")
    except Exception as e:
        print(f"[FAIL] Cost-benefit analysis failed: {e}")

    # 测试情景比较
    print("Testing scenario comparison...")
    try:
        from src.visualization import plot_scenario_comparison

        # 创建简单的情景数据
        scenarios = {
            'Conservative': {
                'years': list(range(2025, 2046)),
                'tourists': np.random.normal(15000, 1000, 21),
                'revenue': np.random.normal(250, 25, 21),
                'glacier': np.maximum(0, 1.0 - np.linspace(0, 0.1, 21)),
                'satisfaction': np.random.normal(0.5, 0.1, 21),
                'npv': 2500000000
            },
            'Baseline': {
                'years': list(range(2025, 2046)),
                'tourists': np.random.normal(16000, 1200, 21),
                'revenue': np.random.normal(275, 30, 21),
                'glacier': np.maximum(0, 1.0 - np.linspace(0, 0.15, 21)),
                'satisfaction': np.random.normal(0.4, 0.15, 21),
                'npv': 2800000000
            }
        }

        plot_scenario_comparison(scenarios)
        print("[OK] Scenario comparison generated")
    except Exception as e:
        print(f"[FAIL] Scenario comparison failed: {e}")

    # 测试预测不确定性
    print("Testing predictive uncertainty...")
    try:
        from src.visualization import plot_predictive_uncertainty

        # 使用一个简单的测试结果
        test_results = {
            'years': list(range(2025, 2046)),
            'tourists': [16000] * 21,  # 简化数据
            'revenue': [275000000] * 21,
            'glacier': np.maximum(0, 1.0 - np.linspace(0, 0.15, 21)),
            'satisfaction': [0.4] * 21
        }

        plot_predictive_uncertainty(test_results)
        print("[OK] Predictive uncertainty analysis generated")
    except Exception as e:
        print(f"[FAIL] Predictive uncertainty analysis failed: {e}")

    print("\n" + "=" * 50)
    print("Advanced visualization testing completed!")
    print("Check the results/ directory for generated plots")

    # 统计生成的文件
    results_dir = "results"
    if os.path.exists(results_dir):
        files = [f for f in os.listdir(results_dir) if f.endswith('.png')]
        print(f"Generated {len(files)} visualization files:")
        for file in sorted(files):
            print(f"  - {file}")

if __name__ == "__main__":
    # 确保结果目录存在
    os.makedirs("results", exist_ok=True)

    test_basic_visualizations()
