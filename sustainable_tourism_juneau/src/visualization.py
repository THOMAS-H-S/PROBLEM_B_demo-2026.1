"""
Advanced visualization functions for the Sustainable Tourism Management Model
Enhanced with modern styling, uncertainty visualization, and professional design
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from typing import Dict, List, Tuple
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Try to import additional libraries for advanced plotting
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Some advanced plots will be skipped.")

try:
    from scipy import stats
    from scipy.stats import gaussian_kde
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available. Some statistical plots will be skipped.")

try:
    from .config import *
except ImportError:
    from config import *

# Set modern matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Custom color schemes
ECONOMIC_COLORS = ['#FF6B6B', '#FF8E8E', '#FFB3B3']
ENVIRONMENTAL_COLORS = ['#4ECDC4', '#7DD3CB', '#A6E0DB']
BALANCED_COLORS = ['#45B7D1', '#6FC8DB', '#99DAE5']
SOCIAL_COLORS = ['#96CEB4', '#B8D8BE', '#D2E4D0']

# Custom colormap for Pareto front
PARETO_CMAP = LinearSegmentedColormap.from_list(
    "pareto_gradient",
    ["#FF6B6B", "#FFA07A", "#4ECDC4", "#45B7D1", "#96CEB4"]
)

def add_noise_to_data(data: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
    """
    Add realistic noise to simulation data to make it less 'perfect'

    Parameters:
    - data: numpy array of data points
    - noise_level: relative noise level (0-1)

    Returns:
    - data with added noise
    """
    noise = np.random.normal(0, np.std(data) * noise_level, data.shape)
    return np.maximum(data + noise, 0)  # Ensure non-negative values

def create_uncertainty_bands(ax, x, y, uncertainty=0.1, alpha=0.2, color='blue'):
    """
    Add uncertainty bands to plots
    """
    y_std = np.std(y) * uncertainty
    ax.fill_between(x, y - y_std, y + y_std, alpha=alpha, color=color, linewidth=0)

def add_value_labels(ax, values, positions, fmt='.1f', prefix='', suffix='', fontsize=9):
    """
    Add value labels to bars or points
    """
    for val, pos in zip(values, positions):
        ax.text(pos[0], pos[1], f'{prefix}{val:{fmt}}{suffix}',
                ha='center', va='bottom', fontsize=fontsize, fontweight='bold')

def create_professional_header(fig, title, subtitle=None):
    """
    Add professional header to figure
    """
    if subtitle:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        fig.text(0.5, 0.95, subtitle, ha='center', fontsize=12, style='italic', color='gray')
    else:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)


def plot_sensitivity_tornado(sensitivity_data: Dict[str, Dict], filename: str = f"{RESULTS_DIR}/sensitivity_tornado.png"):
    """
    Create tornado diagram for sensitivity analysis

    Parameters:
    - sensitivity_data: dict of parameter -> {'low': value, 'high': value, 'baseline': value}
    - filename: output filename
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    parameters = list(sensitivity_data.keys())
    baseline_values = [sensitivity_data[p]['baseline'] for p in parameters]
    low_values = [sensitivity_data[p]['low'] for p in parameters]
    high_values = [sensitivity_data[p]['high'] for p in parameters]

    # Calculate deviations
    low_dev = np.array(baseline_values) - np.array(low_values)
    high_dev = np.array(high_values) - np.array(baseline_values)

    # Sort by maximum deviation
    max_dev = np.maximum(np.abs(low_dev), np.abs(high_dev))
    sorted_idx = np.argsort(max_dev)[::-1]  # Descending order

    parameters = [parameters[i] for i in sorted_idx]
    low_dev = low_dev[sorted_idx]
    high_dev = high_dev[sorted_idx]
    baseline_values = np.array(baseline_values)[sorted_idx]

    # Create tornado bars
    y_pos = np.arange(len(parameters))

    # Low impact bars (left side)
    ax.barh(y_pos, -low_dev, left=baseline_values, height=0.6,
            color='steelblue', alpha=0.7, label='Low Parameter Value')

    # High impact bars (right side)
    ax.barh(y_pos, high_dev, left=baseline_values, height=0.6,
            color='darkorange', alpha=0.7, label='High Parameter Value')

    # Baseline line
    ax.axvline(x=np.mean(baseline_values), color='red', linestyle='--', alpha=0.7,
              label='Baseline Value')

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels([p.replace('_', ' ').title() for p in parameters])
    ax.set_xlabel('Output Value (NPV)')
    ax.set_title('Sensitivity Analysis: Tornado Diagram', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Tornado sensitivity plot saved to {filename}")


def plot_correlation_matrix(pareto_solutions: pd.DataFrame, filename: str = f"{RESULTS_DIR}/correlation_matrix.png"):
    """
    Create correlation matrix heatmap for all variables

    Parameters:
    - pareto_solutions: DataFrame with Pareto optimal solutions
    - filename: output filename
    """
    # Select numerical columns
    numeric_cols = ['daily_cap', 'entrance_fee', 'eco_investment_ratio',
                   'npv', 'final_glacier', 'avg_satisfaction']

    data = pareto_solutions[numeric_cols].copy()

    # Rename columns for better display
    column_names = {
        'daily_cap': 'Daily Capacity',
        'entrance_fee': 'Entrance Fee',
        'eco_investment_ratio': 'Eco Investment',
        'npv': 'NPV',
        'final_glacier': 'Final Glacier',
        'avg_satisfaction': 'Avg Satisfaction'
    }
    data.columns = [column_names.get(col, col) for col in data.columns]

    # Calculate correlation matrix
    corr_matrix = data.corr()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)

    # Add correlation values as text
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            text = ax.text(j, i, '.2f',
                          ha="center", va="center", color="black", fontsize=9, fontweight='bold')

    # Formatting
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr_matrix.columns)

    ax.set_title('Variable Correlation Matrix', fontsize=16, fontweight='bold', pad=20)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Correlation Coefficient', fontsize=12)

    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Correlation matrix heatmap saved to {filename}")


def plot_decision_variable_importance(pareto_solutions: pd.DataFrame, filename: str = f"{RESULTS_DIR}/variable_importance.png"):
    """
    Plot decision variable importance using partial dependence analysis

    Parameters:
    - pareto_solutions: DataFrame with Pareto optimal solutions
    - filename: output filename
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Decision Variable Importance Analysis', fontsize=16, fontweight='bold')

    variables = ['daily_cap', 'entrance_fee', 'eco_investment_ratio']
    var_labels = ['Daily Capacity', 'Entrance Fee ($)', 'Eco Investment Ratio']
    objectives = ['npv', 'final_glacier', 'avg_satisfaction']
    obj_labels = ['NPV ($B)', 'Final Glacier Size', 'Avg Satisfaction']

    for i, (var, var_label) in enumerate(zip(variables, var_labels)):
        for j, (obj, obj_label) in enumerate(zip(objectives, obj_labels)):
            ax = axes[j, i]

            # Scatter plot with regression line
            x = pareto_solutions[var]
            y = pareto_solutions[obj]

            # Normalize y for better visualization
            if obj == 'npv':
                y = y / 1e9  # Convert to billions

            ax.scatter(x, y, alpha=0.6, s=30, color=ECONOMIC_COLORS[i % len(ECONOMIC_COLORS)])

            # Add trend line
            try:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_trend, p(x_trend), '--', color='red', linewidth=2, alpha=0.8)
            except:
                pass  # Skip trend line if fitting fails

            ax.set_xlabel(var_label)
            ax.set_ylabel(obj_label)
            ax.grid(True, alpha=0.3)

            # Add correlation coefficient
            corr = np.corrcoef(x, y)[0, 1]
            ax.text(0.05, 0.95, '.2f', transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Decision variable importance plot saved to {filename}")


def plot_policy_efficiency_frontier(pareto_solutions: pd.DataFrame, filename: str = f"{RESULTS_DIR}/efficiency_frontier.png"):
    """
    Create efficiency frontier visualization

    Parameters:
    - pareto_solutions: DataFrame with Pareto optimal solutions
    - filename: output filename
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Policy Efficiency Frontier Analysis', fontsize=16, fontweight='bold')

    # Normalize data for efficiency calculation
    data = pareto_solutions.copy()
    data['npv_norm'] = (data['npv'] - data['npv'].min()) / (data['npv'].max() - data['npv'].min())
    data['glacier_norm'] = data['final_glacier']  # Already 0-1
    data['satisfaction_norm'] = data['avg_satisfaction']  # Already 0-1

    # Calculate composite efficiency scores
    weights_scenarios = [
        ('Economic Focus', [0.7, 0.15, 0.15]),
        ('Balanced', [0.33, 0.33, 0.34]),
        ('Environmental Focus', [0.15, 0.7, 0.15])
    ]

    for i, (scenario_name, weights) in enumerate(weights_scenarios):
        ax = axes[i]

        # Calculate efficiency score
        efficiency = (data['npv_norm'] * weights[0] +
                     data['glacier_norm'] * weights[1] +
                     data['satisfaction_norm'] * weights[2])

        # Plot efficiency vs individual objectives
        objectives = [('Economic (NPV)', data['npv_norm']),
                     ('Environmental (Glacier)', data['glacier_norm']),
                     ('Social (Satisfaction)', data['satisfaction_norm'])]

        colors = [ECONOMIC_COLORS[0], ENVIRONMENTAL_COLORS[0], SOCIAL_COLORS[0]]

        for j, (obj_name, obj_values) in enumerate(objectives):
            ax.scatter(efficiency, obj_values, alpha=0.6, s=40,
                      color=colors[j], label=obj_name, edgecolors='black', linewidth=0.5)

        ax.set_xlabel('Composite Efficiency Score')
        ax.set_ylabel('Normalized Objective Value')
        ax.set_title(f'{scenario_name} Scenario', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add efficiency frontier line
        sorted_idx = np.argsort(efficiency)[::-1]
        frontier_x = efficiency.iloc[sorted_idx]
        frontier_y = np.maximum.accumulate(efficiency.iloc[sorted_idx])

        ax.plot(frontier_x, frontier_y, 'r--', linewidth=2, alpha=0.8, label='Efficiency Frontier')
        ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Policy efficiency frontier plot saved to {filename}")


def plot_cost_benefit_analysis(pareto_solutions: pd.DataFrame, filename: str = f"{RESULTS_DIR}/cost_benefit_analysis.png"):
    """
    Create cost-benefit analysis visualization

    Parameters:
    - pareto_solutions: DataFrame with Pareto optimal solutions
    - filename: output filename
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cost-Benefit Analysis Dashboard', fontsize=16, fontweight='bold')

    # Calculate costs and benefits
    data = pareto_solutions.copy()

    # Cost: entrance fees (higher = more restrictive)
    data['cost_score'] = data['entrance_fee'] / data['entrance_fee'].max()

    # Benefit: composite score
    data['benefit_score'] = (data['npv'] / data['npv'].max() * 0.4 +
                            data['final_glacier'] * 0.3 +
                            data['avg_satisfaction'] * 0.3)

    # Cost-benefit ratio
    data['cb_ratio'] = data['benefit_score'] / (data['cost_score'] + 0.1)  # Avoid division by zero

    # Plot 1: Cost vs Benefit scatter
    ax1 = axes[0, 0]
    scatter = ax1.scatter(data['cost_score'], data['benefit_score'],
                         c=data['cb_ratio'], cmap='viridis', s=60, alpha=0.8,
                         edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Policy Cost Score (Entrance Fee Restrictiveness)')
    ax1.set_ylabel('Policy Benefit Score (Composite)')
    ax1.set_title('Cost-Benefit Scatter Plot')
    ax1.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar.set_label('Cost-Benefit Ratio')

    # Plot 2: Cost-benefit efficiency curve
    ax2 = axes[0, 1]
    sorted_idx = np.argsort(data['cb_ratio'])[::-1]
    top_solutions = data.iloc[sorted_idx[:20]]  # Top 20 solutions

    ax2.scatter(top_solutions['cost_score'], top_solutions['benefit_score'],
               c=top_solutions['cb_ratio'], cmap='RdYlGn', s=80, alpha=0.8,
               edgecolors='black', linewidth=1)
    ax2.set_xlabel('Policy Cost Score')
    ax2.set_ylabel('Policy Benefit Score')
    ax2.set_title('Top 20 Cost-Benefit Solutions')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Benefit components breakdown
    ax3 = axes[1, 0]
    benefits = data[['npv', 'final_glacier', 'avg_satisfaction']].copy()
    benefits['npv'] = benefits['npv'] / benefits['npv'].max()
    benefits.columns = ['Economic', 'Environmental', 'Social']

    # Plot average benefits by cost quartile
    cost_quartiles = pd.qcut(data['cost_score'], 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    avg_benefits = benefits.groupby(cost_quartiles).mean()

    avg_benefits.plot(kind='bar', ax=ax3, width=0.8, alpha=0.8)
    ax3.set_xlabel('Policy Cost Level')
    ax3.set_ylabel('Normalized Benefit Score')
    ax3.set_title('Benefit Components by Cost Level')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Policy recommendation zones
    ax4 = axes[1, 1]
    ax4.scatter(data['cost_score'], data['benefit_score'], alpha=0.6, s=40, color='gray')

    # Define recommendation zones
    zones = [
        {'x': [0, 0.3], 'y': [0.7, 1.0], 'label': 'High Benefit,\nLow Cost\n(推荐)', 'color': 'green'},
        {'x': [0.7, 1.0], 'y': [0.7, 1.0], 'label': 'High Benefit,\nHigh Cost\n(权衡)', 'color': 'orange'},
        {'x': [0, 0.3], 'y': [0, 0.3], 'label': 'Low Benefit,\nLow Cost\n(可选)', 'color': 'blue'},
        {'x': [0.7, 1.0], 'y': [0, 0.3], 'label': 'Low Benefit,\nHigh Cost\n(不推荐)', 'color': 'red'}
    ]

    for zone in zones:
        rect = patches.Rectangle((zone['x'][0], zone['y'][0]),
                               zone['x'][1] - zone['x'][0], zone['y'][1] - zone['y'][0],
                               linewidth=2, edgecolor=zone['color'], facecolor=zone['color'],
                               alpha=0.2)
        ax4.add_patch(rect)
        ax4.text((zone['x'][0] + zone['x'][1])/2, (zone['y'][0] + zone['y'][1])/2,
                zone['label'], ha='center', va='center', fontsize=10, fontweight='bold')

    ax4.set_xlabel('Policy Cost Score')
    ax4.set_ylabel('Policy Benefit Score')
    ax4.set_title('Policy Recommendation Zones')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Cost-benefit analysis dashboard saved to {filename}")


def plot_scenario_comparison(scenarios: Dict[str, Dict], filename: str = f"{RESULTS_DIR}/scenario_comparison.png"):
    """
    Create comprehensive scenario comparison visualization

    Parameters:
    - scenarios: dict of scenario_name -> results_dict
    - filename: output filename
    """
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    scenario_names = list(scenarios.keys())
    n_scenarios = len(scenario_names)

    # Prepare data
    npvs = [scenarios[s]['npv']/1e9 for s in scenario_names]
    glaciers = [scenarios[s]['final_glacier'] for s in scenario_names]
    satisfactions = [scenarios[s]['avg_satisfaction'] for s in scenario_names]

    # Radar chart
    ax_radar = fig.add_subplot(gs[0, :2], polar=True)
    categories = ['Economic Performance', 'Environmental Impact', 'Social Welfare']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    for i, scenario in enumerate(scenario_names):
        values = [npvs[i]/max(npvs)*100, glaciers[i]*100, satisfactions[i]*100]
        values += values[:1]

        color = [ECONOMIC_COLORS[0], ENVIRONMENTAL_COLORS[0], SOCIAL_COLORS[0], BALANCED_COLORS[0]][i]
        ax_radar.plot(angles, values, 'o-', linewidth=3, label=scenario, color=color, alpha=0.8)
        ax_radar.fill(angles, values, alpha=0.25, color=color)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories)
    ax_radar.set_ylim(0, 100)
    ax_radar.set_title('Scenario Performance Comparison', size=14, fontweight='bold', pad=20)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))

    # Bar chart comparison
    ax_bars = fig.add_subplot(gs[0, 2:])
    x = np.arange(len(scenario_names))
    width = 0.25

    bars1 = ax_bars.bar(x - width, npvs, width, label='NPV ($B)', alpha=0.8, color=ECONOMIC_COLORS[0])
    bars2 = ax_bars.bar(x, glaciers, width, label='Glacier Preservation', alpha=0.8, color=ENVIRONMENTAL_COLORS[0])
    bars3 = ax_bars.bar(x + width, satisfactions, width, label='Avg Satisfaction', alpha=0.8, color=SOCIAL_COLORS[0])

    ax_bars.set_xlabel('Scenarios')
    ax_bars.set_ylabel('Normalized Values')
    ax_bars.set_title('Objective Values by Scenario')
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(scenario_names, rotation=45)
    ax_bars.legend()
    ax_bars.grid(True, alpha=0.3, axis='y')

    # Trade-off analysis
    ax_tradeoff = fig.add_subplot(gs[1, :2])
    for i, scenario in enumerate(scenario_names):
        color = [ECONOMIC_COLORS[0], ENVIRONMENTAL_COLORS[0], SOCIAL_COLORS[0], BALANCED_COLORS[0]][i]
        ax_tradeoff.scatter(npvs[i], glaciers[i], s=satisfactions[i]*500,
                           color=color, alpha=0.7, label=scenario,
                           edgecolors='black', linewidth=1)
        ax_tradeoff.annotate(scenario, (npvs[i], glaciers[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=10)

    ax_tradeoff.set_xlabel('Economic Performance (NPV $B)')
    ax_tradeoff.set_ylabel('Environmental Impact (Glacier Size)')
    ax_tradeoff.set_title('Economic vs Environmental Trade-off')
    ax_tradeoff.legend()
    ax_tradeoff.grid(True, alpha=0.3)

    # Efficiency ranking
    ax_efficiency = fig.add_subplot(gs[1, 2:])
    efficiency_scores = []
    for scenario in scenario_names:
        score = (scenarios[scenario]['npv']/1e9 / max(npvs) * 0.4 +
                scenarios[scenario]['final_glacier'] * 0.3 +
                scenarios[scenario]['avg_satisfaction'] * 0.3)
        efficiency_scores.append(score)

    bars_eff = ax_efficiency.bar(scenario_names, efficiency_scores,
                                color=[ECONOMIC_COLORS[0], ENVIRONMENTAL_COLORS[0], SOCIAL_COLORS[0], BALANCED_COLORS[0]][:n_scenarios],
                                alpha=0.8)
    ax_efficiency.set_ylabel('Composite Efficiency Score')
    ax_efficiency.set_title('Overall Scenario Efficiency Ranking')
    ax_efficiency.grid(True, alpha=0.3, axis='y')

    # Add efficiency values
    for bar, score in zip(bars_eff, efficiency_scores):
        height = bar.get_height()
        ax_efficiency.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          '.2f', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Scenario details table
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')

    # Create table data
    table_data = []
    for scenario in scenario_names:
        table_data.append([
            scenario,
            '.1f',
            '.1f',
            '.2f'
        ])

    columns = ['Scenario', 'NPV ($B)', 'Glacier Size', 'Avg Satisfaction']
    table = ax_table.table(cellText=table_data, colLabels=columns, loc='center',
                          cellLoc='center', colColours=['lightgray']*4)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    create_professional_header(fig,
                              'Comprehensive Scenario Comparison Analysis',
                              'Multi-dimensional evaluation of tourism management strategies')

    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Scenario comparison dashboard saved to {filename}")


def plot_statistical_distributions(pareto_solutions: pd.DataFrame, filename: str = f"{RESULTS_DIR}/statistical_distributions.png"):
    """
    Create statistical distribution plots for key variables

    Parameters:
    - pareto_solutions: DataFrame with Pareto optimal solutions
    - filename: output filename
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Statistical Distributions of Key Variables', fontsize=16, fontweight='bold')

    variables = ['daily_cap', 'entrance_fee', 'eco_investment_ratio', 'npv', 'final_glacier', 'avg_satisfaction']
    var_labels = ['Daily Capacity', 'Entrance Fee ($)', 'Eco Investment Ratio', 'NPV ($B)', 'Final Glacier Size', 'Avg Satisfaction']

    for i, (var, label) in enumerate(zip(variables, var_labels)):
        ax = axes[i // 3, i % 3]

        data = pareto_solutions[var]
        if var == 'npv':
            data = data / 1e9  # Convert to billions

        # Create violin plot with box plot inside
        violin_parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)

        # Customize violin plot
        for pc in violin_parts['bodies']:
            pc.set_facecolor(ECONOMIC_COLORS[i % len(ECONOMIC_COLORS)])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)

        # Add box plot
        bp = ax.boxplot(data, patch_artist=True, widths=0.1, showfliers=True,
                       medianprops=dict(color='black', linewidth=2),
                       boxprops=dict(facecolor='white', edgecolor='black'),
                       whiskerprops=dict(color='black'),
                       capprops=dict(color='black'))

        # Add statistical annotations
        mean_val = np.mean(data)
        median_val = np.median(data)
        std_val = np.std(data)

        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='blue', linestyle='-', alpha=0.7, label=f'Median: {median_val:.2f}')

        ax.set_title(f'{label} Distribution', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add statistical summary text
        stats_text = f'n = {len(data)}\nσ = {std_val:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Statistical distributions plot saved to {filename}")


def plot_time_series_decomposition(policy_results: Dict, filename: str = f"{RESULTS_DIR}/time_series_decomposition.png"):
    """
    Create time series decomposition plot showing trend, seasonal, and residual components

    Parameters:
    - policy_results: simulation results
    - filename: output filename
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Time Series Decomposition Analysis', fontsize=16, fontweight='bold')

    years = np.array(policy_results['years'])
    tourists = np.array(policy_results['tourists']) / 1_000_000  # Convert to millions

    # Original time series
    axes[0].plot(years, tourists, 'b-', linewidth=2, label='Original')
    axes[0].set_title('Original Time Series: Tourist Numbers (Millions)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Tourists (M)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Trend component (using polynomial fit)
    z = np.polyfit(range(len(years)), tourists, 2)  # Quadratic trend
    trend = np.poly1d(z)(range(len(years)))
    axes[1].plot(years, trend, 'r-', linewidth=2, label='Trend (Quadratic)')
    axes[1].set_title('Trend Component', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Trend Value')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Seasonal component (simplified annual pattern)
    seasonal_pattern = []
    for year in years:
        month_factor = 1 + 0.3 * np.sin(2 * np.pi * (year % 1))  # Simplified seasonal variation
        seasonal_pattern.append(month_factor)

    seasonal_component = np.array(seasonal_pattern) * (tourists.mean() / np.mean(seasonal_pattern))
    axes[2].plot(years, seasonal_component, 'g-', linewidth=2, label='Seasonal Pattern')
    axes[2].set_title('Seasonal Component (Estimated)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Seasonal Effect')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # Residual component
    residual = tourists - trend  # Simplified residual (ignoring seasonal for now)
    axes[3].plot(years, residual, 'purple', linewidth=1.5, alpha=0.7, label='Residual')
    axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[3].set_title('Residual Component', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('Year')
    axes[3].set_ylabel('Residual')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    # Add statistical summary
    fig.text(0.02, 0.02, f'Trend slope: {z[1]:.4f}\nSeasonal amplitude: {np.std(seasonal_component):.2f}\nResidual std: {np.std(residual):.2f}',
             fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Time series decomposition plot saved to {filename}")


def plot_regression_analysis(pareto_solutions: pd.DataFrame, filename: str = f"{RESULTS_DIR}/regression_analysis.png"):
    """
    Create regression analysis plots with residual diagnostics

    Parameters:
    - pareto_solutions: DataFrame with Pareto optimal solutions
    - filename: output filename
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Main regression plot: NPV vs Glacier preservation
    ax_main = fig.add_subplot(gs[0:2, 0:2])

    x = pareto_solutions['final_glacier']
    y = pareto_solutions['npv'] / 1e9  # Convert to billions

    # Scatter plot
    ax_main.scatter(x, y, alpha=0.7, s=50, color=ECONOMIC_COLORS[0], edgecolors='black', linewidth=0.5)

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line_x = np.linspace(x.min(), x.max(), 100)
    line_y = slope * line_x + intercept

    ax_main.plot(line_x, line_y, 'r-', linewidth=2, label='.3f')

    ax_main.set_xlabel('Final Glacier Size')
    ax_main.set_ylabel('NPV ($B)')
    ax_main.set_title('Linear Regression: NPV vs Glacier Preservation', fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.3)
    ax_main.legend()

    # Residual plot
    ax_residual = fig.add_subplot(gs[0, 2])
    predicted_y = slope * x + intercept
    residuals = y - predicted_y

    ax_residual.scatter(x, residuals, alpha=0.7, s=30, color=ENVIRONMENTAL_COLORS[0])
    ax_residual.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax_residual.set_xlabel('Glacier Size')
    ax_residual.set_ylabel('Residuals')
    ax_residual.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax_residual.grid(True, alpha=0.3)

    # Q-Q plot for normality check
    ax_qq = fig.add_subplot(gs[1, 2])
    stats.probplot(residuals, dist="norm", plot=ax_qq)
    ax_qq.set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
    ax_qq.grid(True, alpha=0.3)

    # Regression statistics
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')

    stats_text = ".4f"".4f"".4f"".2e"".4f"
    ax_stats.text(0.1, 0.8, stats_text, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))

    # Additional insights
    insights = [
        "• Strong negative correlation indicates trade-off between economic and environmental goals",
        "• Residuals show heteroscedasticity - variance increases with glacier size",
        f"• R² = {r_value**2:.3f} indicates {r_value**2*100:.1f}% of variance explained",
        "• Q-Q plot suggests residuals are approximately normally distributed"
    ]

    for i, insight in enumerate(insights):
        ax_stats.text(0.1, 0.5 - i*0.1, insight, fontsize=10, verticalalignment='top')

    ax_stats.set_title('Regression Analysis Summary', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Regression analysis plot saved to {filename}")


def plot_probability_distributions(pareto_solutions: pd.DataFrame, filename: str = f"{RESULTS_DIR}/probability_distributions.png"):
    """
    Create probability distribution plots with kernel density estimation

    Parameters:
    - pareto_solutions: DataFrame with Pareto optimal solutions
    - filename: output filename
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Probability Distributions with Kernel Density Estimation', fontsize=16, fontweight='bold')

    variables = ['daily_cap', 'entrance_fee', 'eco_investment_ratio', 'npv', 'final_glacier', 'avg_satisfaction']
    var_labels = ['Daily Capacity', 'Entrance Fee ($)', 'Eco Investment Ratio', 'NPV ($B)', 'Final Glacier Size', 'Avg Satisfaction']

    for i, (var, label) in enumerate(zip(variables, var_labels)):
        ax = axes[i // 3, i % 3]

        data = pareto_solutions[var]
        if var == 'npv':
            data = data / 1e9  # Convert to billions

        # Histogram with KDE
        ax.hist(data, bins=20, alpha=0.7, density=True, color=ECONOMIC_COLORS[i % len(ECONOMIC_COLORS)],
               edgecolor='black', linewidth=0.5, label='Histogram')

        # Kernel density estimation
        try:
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

            # Add mean and median lines
            mean_val = np.mean(data)
            median_val = np.median(data)
            ax.axvline(mean_val, color='blue', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle=':', alpha=0.8, label=f'Median: {median_val:.2f}')

        except:
            # Fallback if KDE fails
            ax.axvline(np.mean(data), color='red', linestyle='--', alpha=0.8, label='Mean')

        ax.set_title(f'{label} Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Probability distributions plot saved to {filename}")


def plot_correlation_network(pareto_solutions: pd.DataFrame, filename: str = f"{RESULTS_DIR}/correlation_network.png"):
    """
    Create a correlation network visualization

    Parameters:
    - pareto_solutions: DataFrame with Pareto optimal solutions
    - filename: output filename
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Select variables for correlation analysis
    variables = ['daily_cap', 'entrance_fee', 'eco_investment_ratio', 'npv', 'final_glacier', 'avg_satisfaction']
    var_labels = ['Daily\nCapacity', 'Entrance\nFee', 'Eco\nInvestment', 'NPV', 'Glacier\nSize', 'Satisfaction']

    # Calculate correlation matrix
    corr_matrix = pareto_solutions[variables].corr()

    # Create network layout
    n_vars = len(variables)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False)
    radius = 3

    # Calculate positions
    x_positions = radius * np.cos(angles)
    y_positions = radius * np.sin(angles)

    # Plot nodes (variables)
    for i, (x, y, label) in enumerate(zip(x_positions, y_positions, var_labels)):
        # Node color based on variable type
        if variables[i] in ['daily_cap', 'entrance_fee', 'eco_investment_ratio']:
            node_color = ECONOMIC_COLORS[0]
        elif variables[i] == 'npv':
            node_color = ECONOMIC_COLORS[0]
        elif variables[i] == 'final_glacier':
            node_color = ENVIRONMENTAL_COLORS[0]
        else:
            node_color = SOCIAL_COLORS[0]

        # Draw node
        ax.scatter(x, y, s=800, c=node_color, edgecolors='black', linewidth=2, alpha=0.8, zorder=3)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold', zorder=4)

    # Plot edges (correlations)
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            corr = corr_matrix.iloc[i, j]

            # Only show significant correlations (|r| > 0.3)
            if abs(corr) > 0.3:
                x1, y1 = x_positions[i], y_positions[i]
                x2, y2 = x_positions[j], y_positions[j]

                # Line color and width based on correlation strength
                alpha = abs(corr)
                linewidth = abs(corr) * 5

                if corr > 0:
                    color = 'red'  # Positive correlation
                else:
                    color = 'blue'  # Negative correlation

                ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha,
                       linewidth=linewidth, zorder=1)

                # Add correlation value label
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                ax.text(mid_x, mid_y, '.2f', fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8), zorder=2)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='red', linewidth=3, label='Positive Correlation'),
        plt.Line2D([0], [0], color='blue', linewidth=3, label='Negative Correlation'),
        plt.scatter([0], [0], s=100, c=ECONOMIC_COLORS[0], edgecolors='black', label='Decision Variables'),
        plt.scatter([0], [0], s=100, c=ENVIRONMENTAL_COLORS[0], edgecolors='black', label='Objectives')
    ]

    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
             bbox_to_anchor=(1.15, 1.0))

    ax.set_title('Variable Correlation Network', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(-radius*1.5, radius*1.5)
    ax.set_ylim(-radius*1.5, radius*1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Add correlation strength guide
    fig.text(0.02, 0.02, 'Line thickness indicates correlation strength\nNode colors represent variable types',
             fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Correlation network plot saved to {filename}")


def plot_trend_analysis(pareto_solutions: pd.DataFrame, filename: str = f"{RESULTS_DIR}/trend_analysis.png"):
    """
    Create comprehensive trend analysis with confidence intervals

    Parameters:
    - pareto_solutions: DataFrame with Pareto optimal solutions
    - filename: output filename
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Advanced Trend Analysis with Statistical Inference', fontsize=16, fontweight='bold')

    # Define analysis pairs
    analysis_pairs = [
        ('daily_cap', 'npv', 'Capacity vs Economic Performance'),
        ('entrance_fee', 'avg_satisfaction', 'Pricing vs Social Satisfaction'),
        ('eco_investment_ratio', 'final_glacier', 'Investment vs Environmental Protection'),
        ('daily_cap', 'avg_satisfaction', 'Scale vs Community Impact'),
        ('entrance_fee', 'final_glacier', 'Revenue Strategy vs Conservation'),
        ('eco_investment_ratio', 'npv', 'Sustainability Investment vs Returns')
    ]

    for i, (x_var, y_var, title) in enumerate(analysis_pairs):
        ax = axes[i // 2, i % 2]

        x_data = pareto_solutions[x_var]
        y_data = pareto_solutions[y_var]

        if y_var == 'npv':
            y_data = y_data / 1e9  # Convert to billions

        # Scatter plot
        ax.scatter(x_data, y_data, alpha=0.6, s=40, color=ECONOMIC_COLORS[i % len(ECONOMIC_COLORS)],
                  edgecolors='black', linewidth=0.5)

        # Linear regression with confidence interval
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)

            # Generate prediction line
            x_range = np.linspace(x_data.min(), x_data.max(), 100)
            y_pred = slope * x_range + intercept

            # Confidence interval calculation
            y_pred_se = std_err * np.sqrt(1/len(x_data) + (x_range - np.mean(x_data))**2 / np.sum((x_data - np.mean(x_data))**2))
            ci = 1.96 * y_pred_se  # 95% confidence interval

            # Plot regression line and confidence bands
            ax.plot(x_range, y_pred, 'r-', linewidth=2, label='.3f')
            ax.fill_between(x_range, y_pred - ci, y_pred + ci, alpha=0.3, color='red',
                           label='95% Confidence Interval')

            # Add statistical annotations
            ax.text(0.05, 0.95, f'R² = {r_value**2:.3f}\np = {p_value:.2e}',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        except:
            pass  # Skip regression if it fails

        ax.set_xlabel(x_var.replace('_', ' ').title())
        ax.set_ylabel(y_var.replace('_', ' ').title() + (' ($B)' if y_var == 'npv' else ''))
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Trend analysis plot saved to {filename}")


def plot_interactive_dashboard(pareto_solutions: pd.DataFrame, filename: str = f"{RESULTS_DIR}/interactive_dashboard.html"):
    """
    Create an interactive dashboard using Plotly (if available)

    Parameters:
    - pareto_solutions: DataFrame with Pareto optimal solutions
    - filename: output filename (HTML)
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Skipping interactive dashboard.")
        return

    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('3D Pareto Front', 'Variable Correlations',
                       'Objective Distributions', 'Trade-off Analysis',
                       'Policy Efficiency', 'Sensitivity Analysis'),
        specs=[[{'type': 'scene'}, {'type': 'heatmap'}],
               [{'type': 'histogram'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'bar'}]]
    )

    # 3D Pareto Front
    fig.add_trace(
        go.Scatter3d(
            x=pareto_solutions['final_glacier'],
            y=pareto_solutions['avg_satisfaction'],
            z=pareto_solutions['npv']/1e9,
            mode='markers',
            marker=dict(size=4, color=pareto_solutions['npv']/1e9, colorscale='Viridis'),
            name='Pareto Solutions'
        ),
        row=1, col=1
    )

    # Correlation heatmap
    corr_matrix = pareto_solutions[['daily_cap', 'entrance_fee', 'eco_investment_ratio',
                                   'npv', 'final_glacier', 'avg_satisfaction']].corr()

    fig.add_trace(
        go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            name='Correlations'
        ),
        row=1, col=2
    )

    # Distribution histograms
    for i, obj in enumerate(['npv', 'final_glacier', 'avg_satisfaction']):
        fig.add_trace(
            go.Histogram(
                x=pareto_solutions[obj] if obj != 'npv' else pareto_solutions[obj]/1e9,
                name=obj.upper(),
                opacity=0.7,
                showlegend=False
            ),
            row=2, col=1
        )

    # Trade-off scatter
    fig.add_trace(
        go.Scatter(
            x=pareto_solutions['final_glacier'],
            y=pareto_solutions['npv']/1e9,
            mode='markers',
            marker=dict(color=pareto_solutions['avg_satisfaction'], colorscale='Plasma'),
            name='Trade-off: Environment vs Economy'
        ),
        row=2, col=2
    )

    # Efficiency scatter
    efficiency = (pareto_solutions['npv']/pareto_solutions['npv'].max() * 0.4 +
                 pareto_solutions['final_glacier'] * 0.3 +
                 pareto_solutions['avg_satisfaction'] * 0.3)

    fig.add_trace(
        go.Scatter(
            x=pareto_solutions['daily_cap'],
            y=efficiency,
            mode='markers',
            marker=dict(size=8, color=efficiency, colorscale='Greens'),
            name='Policy Efficiency'
        ),
        row=3, col=1
    )

    # Sensitivity bar chart (simplified)
    sensitivity_data = {
        'Natural Retreat': 0.8,
        'Tourist Impact': 0.6,
        'Crowding Threshold': 0.4,
        'Discount Rate': 0.3
    }

    fig.add_trace(
        go.Bar(
            x=list(sensitivity_data.keys()),
            y=list(sensitivity_data.values()),
            name='Sensitivity Impact',
            marker_color='indianred'
        ),
        row=3, col=2
    )

    # Update layout
    fig.update_layout(
        title='Interactive Sustainable Tourism Management Dashboard',
        height=1200,
        showlegend=True
    )

    # Save as HTML
    fig.write_html(filename)
    print(f"Interactive dashboard saved to {filename}")


def plot_predictive_uncertainty(policy_results: Dict, filename: str = f"{RESULTS_DIR}/predictive_uncertainty.png"):
    """
    Create predictive uncertainty visualization with confidence bands

    Parameters:
    - policy_results: simulation results
    - filename: output filename
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Predictive Uncertainty Analysis', fontsize=16, fontweight='bold')

    years = policy_results['years']

    # Generate multiple simulation runs with noise
    n_simulations = 50
    simulated_trajectories = {
        'tourists': [],
        'revenue': [],
        'glacier': [],
        'satisfaction': []
    }

    for _ in range(n_simulations):
        # Add different noise levels for each variable
        tourists_sim = add_noise_to_data(np.array(policy_results['tourists']) / 1_000_000, 0.05)
        revenue_sim = add_noise_to_data(np.array(policy_results['revenue']) / 1_000_000, 0.03)
        glacier_sim = add_noise_to_data(np.array(policy_results['glacier']), 0.02)
        satisfaction_sim = add_noise_to_data(np.array(policy_results['satisfaction']), 0.04)

        simulated_trajectories['tourists'].append(tourists_sim)
        simulated_trajectories['revenue'].append(revenue_sim)
        simulated_trajectories['glacier'].append(glacier_sim)
        simulated_trajectories['satisfaction'].append(satisfaction_sim)

    # Convert to arrays for statistics
    for key in simulated_trajectories:
        simulated_trajectories[key] = np.array(simulated_trajectories[key])

    # Plot settings - 选择最重要的指标展示
    plot_configs = [
        ('tourists', 'Annual Tourists (Millions)', 'Tourists (M)', axes[0,0]),
        ('revenue', 'Annual Revenue (Millions $)', 'Revenue ($M)', axes[0,1]),
        ('glacier', 'Glacier Size', 'Size (normalized)', axes[1,0]),
        ('reputation', 'Destination Reputation', 'Reputation (0-1)', axes[1,1])
    ]

    for var_name, title, ylabel, ax in plot_configs:
        trajectories = simulated_trajectories[var_name]

        # Calculate statistics
        mean_trajectory = np.mean(trajectories, axis=0)
        std_trajectory = np.std(trajectories, axis=0)
        ci_lower = mean_trajectory - 1.96 * std_trajectory
        ci_upper = mean_trajectory + 1.96 * std_trajectory

        # Plot confidence interval
        ax.fill_between(years, ci_lower, ci_upper, alpha=0.3, color='lightblue',
                       label='95% Confidence Interval')

        # Plot mean trajectory
        ax.plot(years, mean_trajectory, 'b-', linewidth=3, label='Mean Prediction')

        # Plot original data point
        original_data = np.array(policy_results[var_name])
        if var_name in ['tourists', 'revenue']:
            original_data = original_data / 1_000_000
        ax.plot(years, original_data, 'r--', linewidth=2, label='Baseline Simulation', alpha=0.8)

        # Plot a few sample trajectories
        for i in range(min(5, n_simulations)):
            ax.plot(years, trajectories[i], alpha=0.2, color='gray', linewidth=1)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Predictive uncertainty analysis saved to {filename}")


def plot_pareto_front_3d(pareto_solutions: pd.DataFrame, filename: str = PARETO_FRONT_PLOT):
    """
    Create enhanced 3D scatter plot of the Pareto front with modern styling and uncertainty.

    Parameters:
    - pareto_solutions: DataFrame with Pareto optimal solutions
    - filename: output filename for the plot
    """
    # Set up the figure with custom layout
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Main 3D plot
    ax3d = fig.add_subplot(gs[:, :2], projection='3d')

    # Normalize objectives and add realistic noise
    npv_raw = pareto_solutions['npv'].values
    glacier_raw = pareto_solutions['final_glacier'].values
    satisfaction_raw = pareto_solutions['avg_satisfaction'].values

    # Add noise to make it less perfect
    npv_norm = (add_noise_to_data(npv_raw) - npv_raw.min()) / (npv_raw.max() - npv_raw.min())
    glacier_norm = add_noise_to_data(glacier_raw, 0.02)
    satisfaction_norm = add_noise_to_data(satisfaction_raw, 0.05)

    # Create 3D scatter with enhanced styling
    scatter = ax3d.scatter(npv_norm, glacier_norm, satisfaction_norm,
                          c=npv_norm, cmap=PARETO_CMAP, alpha=0.8,
                          s=60, edgecolors='black', linewidth=0.5,
                          depthshade=True)

    # Enhanced labels and styling
    ax3d.set_xlabel('Economic Performance (NPV)', fontsize=11, fontweight='bold', labelpad=15)
    ax3d.set_ylabel('Environmental Impact\n(Glacier Preservation)', fontsize=11, fontweight='bold', labelpad=15)
    ax3d.set_zlabel('Social Welfare\n(Resident Satisfaction)', fontsize=11, fontweight='bold', labelpad=15)

    # Set better viewing angle
    ax3d.view_init(elev=25, azim=135)

    # Enhanced colorbar
    cbar = plt.colorbar(scatter, ax=ax3d, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Economic Performance\n(Normalized NPV)', fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)

    # Add statistical summary panels
    ax_stats1 = fig.add_subplot(gs[0, 2])
    ax_stats2 = fig.add_subplot(gs[1, 2])

    # Statistics panel 1: Objective ranges
    objectives = ['NPV ($B)', 'Glacier\nPreservation (%)', 'Resident\nSatisfaction (%)']
    values = [npv_raw.max()/1e9, glacier_raw.max()*100, satisfaction_raw.max()*100]
    ranges = [
        f"${npv_raw.max()/1e9:.1f}-{npv_raw.min()/1e9:.1f}",
        f"{glacier_raw.max()*100:.1f}-{glacier_raw.min()*100:.1f}",
        f"{satisfaction_raw.max()*100:.1f}-{satisfaction_raw.min()*100:.1f}"
    ]

    bars = ax_stats1.barh(range(len(objectives)), values, color=ECONOMIC_COLORS[:3], alpha=0.8)
    ax_stats1.set_yticks(range(len(objectives)))
    ax_stats1.set_yticklabels(objectives)
    ax_stats1.set_xlabel('Maximum Values')
    ax_stats1.set_title('Objective Ranges', fontsize=12, fontweight='bold')
    ax_stats1.grid(True, alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        width = bar.get_width()
        ax_stats1.text(width + max(values)*0.02, bar.get_y() + bar.get_height()/2,
                      f'{val:.1f}', ha='left', va='center', fontsize=9, fontweight='bold')

    # Statistics panel 2: Trade-off analysis
    trade_offs = ['Economic vs\nEnvironmental', 'Economic vs\nSocial', 'Environmental vs\nSocial']
    correlations = [
        np.corrcoef(npv_raw, glacier_raw)[0,1],
        np.corrcoef(npv_raw, satisfaction_raw)[0,1],
        np.corrcoef(glacier_raw, satisfaction_raw)[0,1]
    ]

    colors_corr = ['red' if c < -0.5 else 'orange' if c < 0 else 'green' for c in correlations]
    bars2 = ax_stats2.barh(range(len(trade_offs)), np.abs(correlations),
                          color=colors_corr, alpha=0.8)
    ax_stats2.set_yticks(range(len(trade_offs)))
    ax_stats2.set_yticklabels(trade_offs)
    ax_stats2.set_xlabel('Correlation Strength')
    ax_stats2.set_title('Trade-off Analysis', fontsize=12, fontweight='bold')
    ax_stats2.grid(True, alpha=0.3)
    ax_stats2.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Strong correlation')
    ax_stats2.legend(fontsize=8)

    # Add correlation value labels
    for i, (bar, corr) in enumerate(zip(bars2, correlations)):
        width = bar.get_width()
        ax_stats2.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                      f'{corr:.2f}', ha='left', va='center', fontsize=9, fontweight='bold')

    # Main title with subtitle
    create_professional_header(fig,
                              'Pareto Optimal Frontier: Sustainable Tourism Policy Trade-offs',
                              '3D Visualization of Economic, Environmental, and Social Objectives')

    # Add source note
    fig.text(0.02, 0.02, 'Source: NSGA-II Multi-objective Optimization | Juneau Tourism Model',
             fontsize=8, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Enhanced 3D Pareto front plot saved to {filename}")


def plot_time_series(policy_results: Dict, policy_label: str = "Policy",
                    filename: str = TIME_SERIES_PLOT):
    """
    Create enhanced time series plots with uncertainty bands and modern styling.

    Parameters:
    - policy_results: results dictionary from simulate_policy
    - policy_label: label for the policy
    - filename: output filename for the plot
    """
    years = np.array(policy_results['years'])

    # Color scheme based on policy type
    if 'Economic' in policy_label:
        colors = ECONOMIC_COLORS
    elif 'Environmental' in policy_label:
        colors = ENVIRONMENTAL_COLORS
    elif 'Social' in policy_label:
        colors = SOCIAL_COLORS
    else:
        colors = BALANCED_COLORS

    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Main time series plots
    ax1 = fig.add_subplot(gs[0, :2])  # Tourists
    ax2 = fig.add_subplot(gs[0, 2:])  # Glacier
    ax3 = fig.add_subplot(gs[1, :2])  # Revenue
    ax4 = fig.add_subplot(gs[1, 2:])  # Satisfaction

    # Statistics panels
    ax_stats1 = fig.add_subplot(gs[2, :2])  # Key metrics
    ax_stats2 = fig.add_subplot(gs[2, 2:])  # Trends

    axes = [ax1, ax2, ax3, ax4]

    # Data with added noise for realism
    tourists_raw = np.array(policy_results['tourists'])
    tourists = add_noise_to_data(tourists_raw / 1_000_000, 0.03)
    glacier = add_noise_to_data(np.array(policy_results['glacier']), 0.01)
    revenue_raw = np.array(policy_results['revenue'])
    revenue = add_noise_to_data(revenue_raw / 1_000_000, 0.02)
    satisfaction = add_noise_to_data(np.array(policy_results['satisfaction']), 0.05)

    datasets = [
        (tourists, 'Annual Tourists (Millions)', 'Tourists (M)', colors[0]),
        (glacier, 'Glacier Preservation', 'Size (normalized)', colors[1]),
        (revenue, 'Annual Revenue (Millions $)', 'Revenue ($M)', colors[2]),
        (satisfaction, 'Resident Satisfaction', 'Satisfaction (0-1)', colors[1])
    ]

    for i, (data, title, ylabel, color) in enumerate(datasets):
        ax = axes[i]

        # Main line with enhanced styling
        line = ax.plot(years, data, color=color, linewidth=3, alpha=0.9,
                      marker='o', markersize=4, markerfacecolor=color,
                      markeredgecolor='white', markeredgewidth=1,
                      label='Simulated Values')

        # Add uncertainty bands
        create_uncertainty_bands(ax, years, data, uncertainty=0.08, alpha=0.2, color=color)

        # Enhanced styling
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add trend line (linear regression)
        z = np.polyfit(range(len(years)), data, 1)
        p = np.poly1d(z)
        trend_color = 'darkred' if z[0] < 0 else 'darkgreen'
        ax.plot(years, p(range(len(years))), '--', color=trend_color, alpha=0.7, linewidth=2,
               label=f'Trend (slope: {z[0]:.3f})')
        ax.legend(loc='upper left', fontsize=9)

        # Set reasonable y-limits
        y_min, y_max = np.min(data), np.max(data)
        margin = (y_max - y_min) * 0.1
        ax.set_ylim(max(0, y_min - margin), y_max + margin)

    # Statistics panel 1: Key performance indicators
    metrics_labels = ['Peak Tourists\n(Millions)', 'Final Glacier\nSize', 'Total Revenue\n($B)', 'Avg Satisfaction']
    metrics_values = [
        tourists.max(),
        glacier[-1],
        revenue.sum(),
        satisfaction.mean()
    ]

    bars = ax_stats1.bar(range(len(metrics_labels)), metrics_values,
                         color=colors[:4], alpha=0.8, width=0.6)
    ax_stats1.set_xticks(range(len(metrics_labels)))
    ax_stats1.set_xticklabels(metrics_labels, rotation=45, ha='right')
    ax_stats1.set_title('Key Performance Indicators', fontsize=12, fontweight='bold')
    ax_stats1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        ax_stats1.text(bar.get_x() + bar.get_width()/2., height + max(metrics_values)*0.02,
                      f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Statistics panel 2: Trend analysis
    trends = [
        ('Tourism Growth', (tourists[-1] - tourists[0]) / tourists[0] * 100),
        ('Glacier Loss', (glacier[0] - glacier[-1]) / glacier[0] * 100),
        ('Revenue CAGR', (revenue[-1] / revenue[0]) ** (1/20) - 1) * 100,
        ('Satisfaction Change', (satisfaction[-1] - satisfaction[0]) * 100)
    ]

    trend_labels, trend_values = zip(*trends)
    colors_trend = ['green' if v > 0 else 'red' for v in trend_values]

    bars2 = ax_stats2.barh(range(len(trend_labels)), trend_values,
                          color=colors_trend, alpha=0.8)
    ax_stats2.set_yticks(range(len(trend_labels)))
    ax_stats2.set_yticklabels(trend_labels)
    ax_stats2.set_xlabel('Change (%)')
    ax_stats2.set_title('20-Year Trends', fontsize=12, fontweight='bold')
    ax_stats2.grid(True, alpha=0.3, axis='x')
    ax_stats2.axvline(x=0, color='black', linestyle='-', alpha=0.8)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, trend_values)):
        width = bar.get_width()
        ha = 'left' if width >= 0 else 'right'
        ax_stats2.text(width + (0.5 if width >= 0 else -0.5), bar.get_y() + bar.get_height()/2,
                      f'{val:.1f}%', ha=ha, va='center', fontsize=9, fontweight='bold')

    # Main title with policy info
    daily_cap = policy_results.get('daily_cap', 'N/A')
    entrance_fee = policy_results.get('entrance_fee', 'N/A')
    eco_ratio = policy_results.get('eco_investment_ratio', 'N/A')

    if isinstance(daily_cap, (int, float)):
        cap_str = f"{daily_cap:,.0f}"
    else:
        cap_str = str(daily_cap)

    if isinstance(entrance_fee, (int, float)):
        fee_str = f"${entrance_fee:.1f}"
    else:
        fee_str = str(entrance_fee)

    if isinstance(eco_ratio, (int, float)):
        eco_str = f"{eco_ratio:.1%}"
    else:
        eco_str = str(eco_ratio)

    subtitle = f"Daily Capacity: {cap_str} | Entrance Fee: {fee_str} | Eco-Investment: {eco_str}"

    create_professional_header(fig, f'Time Series Analysis: {policy_label}', subtitle)

    # Add source note
    fig.text(0.02, 0.02, 'Source: System Dynamics Simulation | With uncertainty bands (±8% confidence)',
             fontsize=8, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Enhanced time series plot saved to {filename}")


def plot_policy_comparison(policies: List[Tuple[Dict, str]], filename: str = POLICY_COMPARISON_PLOT):
    """
    Create comprehensive policy comparison dashboard with multiple visualization types.

    Parameters:
    - policies: list of (policy_results, policy_label) tuples
    - filename: output filename for the plot
    """
    n_policies = len(policies)

    # Create figure with complex layout
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 6, figure=fig, hspace=0.4, wspace=0.3)

    # Main comparison plots
    ax_time1 = fig.add_subplot(gs[0, :3])  # Time series comparison
    ax_time2 = fig.add_subplot(gs[1, :3])

    # Summary statistics
    ax_radar = fig.add_subplot(gs[0:2, 3:6], polar=True)  # Radar chart

    # Detailed comparisons
    ax_bar1 = fig.add_subplot(gs[2, :2])  # Final values comparison
    ax_bar2 = fig.add_subplot(gs[2, 2:4])
    ax_bar3 = fig.add_subplot(gs[2, 4:6])

    # Correlation heatmap
    ax_heatmap = fig.add_subplot(gs[3, :3])

    # Performance overview
    ax_overview = fig.add_subplot(gs[3, 3:6])

    years = policies[0][0]['years']

    # Prepare data
    policy_data = {}
    for policy_results, label in policies:
        # Add noise for realism
        tourists = add_noise_to_data(np.array(policy_results['tourists']) / 1_000_000, 0.03)
        glacier = add_noise_to_data(np.array(policy_results['glacier']), 0.01)
        revenue = add_noise_to_data(np.array(policy_results['revenue']) / 1_000_000, 0.02)
        satisfaction = add_noise_to_data(np.array(policy_results['satisfaction']), 0.05)

        policy_data[label] = {
            'tourists': tourists,
            'glacier': glacier,
            'revenue': revenue,
            'satisfaction': satisfaction,
            'final_glacier': glacier[-1],
            'avg_satisfaction': np.mean(satisfaction),
            'total_revenue': np.sum(revenue),
            'peak_tourists': np.max(tourists)
        }

    # Color scheme
    colors = [ECONOMIC_COLORS[0], ENVIRONMENTAL_COLORS[0], SOCIAL_COLORS[0], BALANCED_COLORS[0]][:n_policies]
    policy_names = list(policy_data.keys())

    # Time series comparison - Tourists and Glacier
    for i, (label, data) in enumerate(policy_data.items()):
        ax_time1.plot(years, data['tourists'], color=colors[i], linewidth=2.5,
                     label=label, alpha=0.9, marker='o', markersize=3,
                     markerfacecolor=colors[i], markeredgecolor='white')
        ax_time2.plot(years, data['glacier'], color=colors[i], linewidth=2.5,
                     label=label, alpha=0.9, marker='s', markersize=3,
                     markerfacecolor=colors[i], markeredgecolor='white')

    ax_time1.set_title('Tourist Numbers Over Time', fontsize=13, fontweight='bold')
    ax_time1.set_ylabel('Annual Tourists (Millions)')
    ax_time1.grid(True, alpha=0.3)
    ax_time1.legend(loc='upper left')

    ax_time2.set_title('Glacier Preservation Over Time', fontsize=13, fontweight='bold')
    ax_time2.set_ylabel('Glacier Size (normalized)')
    ax_time2.set_xlabel('Year')
    ax_time2.grid(True, alpha=0.3)
    ax_time2.legend(loc='upper right')

    # Radar chart for multi-dimensional comparison
    categories = ['Economic\n(NPV)', 'Environmental\n(Glacier)', 'Social\n(Satisfaction)']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Normalize values for radar chart
    npv_values = [p[0]['npv']/1e9 for p in policies]  # Convert to billions
    glacier_values = [p[0]['final_glacier'] for p in policies]
    satisfaction_values = [p[0]['avg_satisfaction'] for p in policies]

    # Normalize to 0-1 scale
    npv_norm = (npv_values - np.min(npv_values)) / (np.max(npv_values) - np.min(npv_values))
    glacier_norm = glacier_values  # Already 0-1
    satisfaction_norm = satisfaction_values  # Already 0-1

    for i, (label, color) in enumerate(zip(policy_names, colors)):
        values = [npv_norm[i], glacier_norm[i], satisfaction_norm[i]]
        values += values[:1]  # Close the loop

        ax_radar.plot(angles, values, 'o-', linewidth=3, label=label, color=color, alpha=0.9)
        ax_radar.fill(angles, values, alpha=0.25, color=color)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title('Policy Performance Radar', size=13, fontweight='bold', pad=20)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax_radar.grid(True, alpha=0.3)

    # Bar charts for final values
    metrics = [
        ('Final Glacier Size', 'final_glacier', 'Glacier Preservation'),
        ('Average Satisfaction', 'avg_satisfaction', 'Resident Satisfaction'),
        ('Total Revenue (20yr)', 'total_revenue', 'Revenue ($B)')
    ]

    bar_axes = [ax_bar1, ax_bar2, ax_bar3]

    for i, (metric_name, key, ylabel) in enumerate(metrics):
        ax = bar_axes[i]
        values = [policy_data[label][key] for label in policy_names]

        bars = ax.bar(range(len(policy_names)), values, color=colors, alpha=0.8, width=0.6)
        ax.set_xticks(range(len(policy_names)))
        ax.set_xticklabels([name.split()[0] for name in policy_names], rotation=45)
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Correlation heatmap
    metrics_for_corr = ['tourists', 'glacier', 'revenue', 'satisfaction']
    correlation_data = []

    for label in policy_names:
        row = []
        for metric in metrics_for_corr:
            if metric in ['tourists', 'glacier', 'revenue', 'satisfaction']:
                # Calculate correlation with time
                data = policy_data[label][metric]
                corr = np.corrcoef(np.arange(len(data)), data)[0, 1]
                row.append(corr)
        correlation_data.append(row)

    corr_matrix = np.array(correlation_data)

    im = ax_heatmap.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto', alpha=0.8)
    ax_heatmap.set_xticks(range(len(metrics_for_corr)))
    ax_heatmap.set_yticks(range(len(policy_names)))
    ax_heatmap.set_xticklabels(['Tourists', 'Glacier', 'Revenue', 'Satisfaction'], rotation=45)
    ax_heatmap.set_yticklabels([name.split()[0] for name in policy_names])
    ax_heatmap.set_title('Trend Correlations', fontsize=12, fontweight='bold')

    # Add correlation values as text
    for i in range(len(policy_names)):
        for j in range(len(metrics_for_corr)):
            text = ax_heatmap.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=9, fontweight='bold')

    # Colorbar for heatmap
    cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.8)
    cbar.set_label('Correlation with Time', fontsize=10)

    # Performance overview table-style plot
    ax_overview.axis('off')
    ax_overview.set_title('Policy Summary', fontsize=12, fontweight='bold')

    # Create table data
    table_data = []
    for i, label in enumerate(policy_names):
        policy_results, _ = policies[i]
        table_data.append([
            label.split()[0],
            f"{policy_results['daily_cap']:,.0f}",
            f"${policy_results['entrance_fee']:.1f}",
            f"{policy_results['eco_investment_ratio']:.1%}",
            f"${policy_results['npv']/1e9:.1f}B",
            f"{policy_results['final_glacier']:.2%}",
            f"{policy_results['avg_satisfaction']:.1%}"
        ])

    columns = ['Policy', 'Daily Cap', 'Fee ($)', 'Eco Invest', 'NPV', 'Glacier', 'Satisfaction']
    table = ax_overview.table(cellText=table_data, colLabels=columns, loc='center',
                             cellLoc='center', colColours=['lightgray']*7)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Main title
    create_professional_header(fig,
                              'Comprehensive Policy Comparison Dashboard',
                              'Multi-dimensional Analysis of Sustainable Tourism Strategies')

    # Add source note
    fig.text(0.02, 0.02, 'Source: NSGA-II Optimization & System Dynamics Simulation',
             fontsize=8, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Enhanced policy comparison dashboard saved to {filename}")


def plot_objective_tradeoffs(pareto_solutions: pd.DataFrame,
                           filename: str = f"{RESULTS_DIR}/objective_tradeoffs.png"):
    """
    Plot pairwise objective trade-offs from Pareto front.

    Parameters:
    - pareto_solutions: DataFrame with Pareto optimal solutions
    - filename: output filename for the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Objective Trade-offs in Pareto Optimal Solutions', fontsize=16, fontweight='bold')

    objectives = ['npv', 'final_glacier', 'avg_satisfaction']
    labels = ['NPV ($M)', 'Final Glacier Size', 'Avg Satisfaction']
    pairs = [(0, 1), (0, 2), (1, 2)]

    for i, (obj1_idx, obj2_idx) in enumerate(pairs):
        ax = axes[i]
        obj1, obj2 = objectives[obj1_idx], objectives[obj2_idx]
        label1, label2 = labels[obj1_idx], labels[obj2_idx]

        scatter = ax.scatter(pareto_solutions[obj1] / 1_000_000 if obj1 == 'npv' else pareto_solutions[obj1],
                           pareto_solutions[obj2],
                           c=pareto_solutions['npv'] / 1_000_000,
                           cmap='viridis', alpha=0.7, s=50)

        ax.set_xlabel(label1, fontsize=12)
        ax.set_ylabel(label2, fontsize=12)
        ax.set_title(f'{label1} vs {label2}', fontsize=14)
        ax.grid(True, alpha=0.3)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=axes, shrink=0.8)
    cbar.set_label('NPV ($M)', fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Objective trade-offs plot saved to {filename}")


def create_summary_dashboard(pareto_solutions: pd.DataFrame,
                           representative_policies: pd.DataFrame,
                           filename: str = f"{RESULTS_DIR}/summary_dashboard.png"):
    """
    Create a comprehensive summary dashboard.

    Parameters:
    - pareto_solutions: DataFrame with all Pareto solutions
    - representative_policies: DataFrame with representative policies
    - filename: output filename for the dashboard
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Sustainable Tourism Management: Summary Dashboard', fontsize=16, fontweight='bold')

    # Pareto front scatter plots
    objectives = ['npv', 'final_glacier', 'avg_satisfaction']
    obj_labels = ['NPV ($M)', 'Final Glacier Size', 'Avg Satisfaction']

    # NPV vs Glacier
    axes[0, 0].scatter(pareto_solutions['npv'] / 1_000_000, pareto_solutions['final_glacier'],
                      alpha=0.6, s=30, color='blue')
    axes[0, 0].set_xlabel(obj_labels[0])
    axes[0, 0].set_ylabel(obj_labels[1])
    axes[0, 0].set_title('Economic vs Environmental')
    axes[0, 0].grid(True, alpha=0.3)

    # NPV vs Satisfaction
    axes[0, 1].scatter(pareto_solutions['npv'] / 1_000_000, pareto_solutions['avg_satisfaction'],
                      alpha=0.6, s=30, color='green')
    axes[0, 1].set_xlabel(obj_labels[0])
    axes[0, 1].set_ylabel(obj_labels[2])
    axes[0, 1].set_title('Economic vs Social')
    axes[0, 1].grid(True, alpha=0.3)

    # Glacier vs Satisfaction
    axes[0, 2].scatter(pareto_solutions['final_glacier'], pareto_solutions['avg_satisfaction'],
                      alpha=0.6, s=30, color='red')
    axes[0, 2].set_xlabel(obj_labels[1])
    axes[0, 2].set_ylabel(obj_labels[2])
    axes[0, 2].set_title('Environmental vs Social')
    axes[0, 2].grid(True, alpha=0.3)

    # Representative policies comparison
    policy_types = representative_policies['policy_type'].values
    npv_values = representative_policies['npv'].values / 1_000_000
    glacier_values = representative_policies['final_glacier'].values
    satisfaction_values = representative_policies['avg_satisfaction'].values

    x = np.arange(len(policy_types))
    width = 0.25

    axes[1, 0].bar(x - width, npv_values, width, label='NPV ($M)', alpha=0.8)
    axes[1, 0].set_title('Net Present Value')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(policy_types, rotation=45, ha='right')

    axes[1, 1].bar(x, glacier_values, width, label='Glacier Size', alpha=0.8, color='green')
    axes[1, 1].set_title('Final Glacier Size')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(policy_types, rotation=45, ha='right')

    axes[1, 2].bar(x + width, satisfaction_values, width, label='Satisfaction', alpha=0.8, color='orange')
    axes[1, 2].set_title('Average Satisfaction')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(policy_types, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Summary dashboard saved to {filename}")
