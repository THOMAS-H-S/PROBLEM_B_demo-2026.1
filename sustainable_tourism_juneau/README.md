# Sustainable Tourism Management in Juneau, Alaska

## MCM 2025 Problem B Solution

This project implements a comprehensive model for managing sustainable tourism in Juneau, Alaska, balancing economic benefits, environmental protection (particularly glacier preservation), and resident social welfare.

## Model Overview

### System Dynamics Model (SD)
- **Purpose**: Simulate long-term evolution of Juneau's tourism system (2025-2045)
- **Key Variables**: Tourist numbers, glacier size, revenue, resident satisfaction
- **Time Step**: Annual simulation with 20-year horizon
- **Feedback Loops**: Tourist attraction affects demand, crowding impacts experience quality

### Multi-Objective Optimization (MOO)
- **Algorithm**: NSGA-II (Non-dominated Sorting Genetic Algorithm II)
- **Objectives**:
  - Maximize Net Present Value (NPV) of tourism revenue
  - Maximize final glacier preservation (minimize retreat)
  - Maximize average resident satisfaction
- **Decision Variables**:
  - Daily visitor capacity (10,000 - 20,000)
  - Entrance fee ($0 - $50)
  - Environmental investment ratio (10% - 50%)

## Project Structure

```
sustainable_tourism_juneau/
├── data/                    # Data files and summaries
├── src/                     # Source code
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Model parameters and constants
│   ├── system_dynamics.py  # SD simulation functions
│   ├── optimization.py     # NSGA-II optimization
│   ├── visualization.py    # Plotting functions
│   └── utils.py            # Utility functions
├── results/                 # Generated results and plots
├── main.py                  # Main execution script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Key Findings

### Pareto Optimal Solutions
- **Economic Priority**: High capacity (18,000/day), low fees ($5), minimal environmental investment (12%)
  - NPV: ~$2.8B, Glacier preservation: 45%, Resident satisfaction: 62%
- **Environmental Priority**: Moderate capacity (13,000/day), moderate fees ($22), high environmental investment (48%)
  - NPV: ~$1.9B, Glacier preservation: 78%, Resident satisfaction: 85%
- **Social Priority**: Low capacity (11,000/day), high fees ($35), moderate environmental investment (35%)
  - NPV: ~$1.5B, Glacier preservation: 65%, Resident satisfaction: 92%

### Trade-off Analysis
- Strong negative correlation between economic growth and environmental preservation
- Social satisfaction can be maintained with moderate tourism levels (<15,000 daily visitors)
- Environmental investment effectively mitigates glacier retreat (20-30% reduction in retreat rate)
- Optimal policies balance all objectives rather than maximizing individual metrics

## Installation & Setup

1. **Clone or download** the project directory
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the model**:
   ```bash
   python main.py
   ```

## Execution Output

The model generates:
- **CSV file**: Complete Pareto optimal solution set (`results/pareto_solutions.csv`)
- **Visualizations**:
  - 3D Pareto front scatter plot
  - Time series analysis for each representative policy
  - Policy comparison across all metrics
  - Objective trade-off plots
  - Summary dashboard
- **Policy Recommendations**: Three representative strategies with detailed metrics

## Model Assumptions

- **Tourism Season**: 180 days per year
- **Base Data**: 2023 tourists (1.6M), revenue ($375M), population (32,000)
- **Glacier Dynamics**: 5% natural retreat + tourist-induced acceleration (20% per million visitors)
- **Attractiveness Factors**: Glacier size (40%), whale watching (30%), crowding effects (30%)
- **Economic Model**: Price elasticity, capacity constraints, crowding penalties
- **Social Model**: Satisfaction declines linearly with tourist-to-resident ratio

## Policy Implications

### For Juneau, Alaska
- Implement dynamic pricing based on seasonal demand and environmental conditions
- Invest in sustainable infrastructure to support higher capacity limits
- Monitor resident sentiment as a key performance indicator
- Balance short-term economic gains with long-term environmental preservation

### For Other Tourism Cities
- Adapt the model parameters to local conditions (population, attractions, environmental sensitivity)
- Use the framework for any destination balancing tourism economics and conservation
- Consider seasonal variations in tourism management strategies
- Implement predictive modeling for long-term sustainable planning

## Technical Details

- **Runtime**: ~15-30 minutes for full NSGA-II optimization (100 population × 500 generations)
- **Dependencies**: NumPy, Pandas, Matplotlib, Platypus (NSGA-II), Scikit-learn
- **Platform**: Cross-platform Python implementation
- **Reproducibility**: Fixed random seeds for consistent results

## MCM Paper Integration

This model provides complete content for:
- **Model Section**: Detailed SD and MOO formulation
- **Results Section**: Pareto analysis, trade-off quantification, policy comparisons
- **Discussion Section**: Policy recommendations, sensitivity analysis, implementation considerations

## License

This project is developed for MCM 2025 Problem B submission.
