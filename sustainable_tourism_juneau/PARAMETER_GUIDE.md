# Juneau可持续旅游管理模型参数指南

## 📋 概述

本文档详细说明了模型中所有参数的确定依据、使用方法以及调整建议。参数分为四类：基础数据参数、系统动力学参数、优化算法参数和可视化参数。

**参数调整原则**：
- 🔍 **敏感性分析**: 小幅调整参数，观察结果变化
- 📊 **基准测试**: 使用已知数据验证参数合理性
- ⚖️ **保守估计**: 优先选择保守的参数值
- 🔄 **迭代优化**: 基于新数据持续调整参数

---

## 📊 基础数据参数

### 1. 人口统计参数
```python
POPULATION_JUNEAU = 32_000  # permanent residents
```

**确定依据**：
- 来源：美国人口普查局 (U.S. Census Bureau) 2023数据
- 调整建议：可根据最新人口统计数据更新
- 敏感性：中等，对居民满意度计算有直接影响

### 2. 旅游基础数据
```python
BASE_TOURISTS_2023 = 1_600_000  # 1.6 million tourists in 2023
BASE_REVENUE_2023 = 375_000_000  # $375 million revenue in 2023
TOURISM_DAYS_PER_YEAR = 180  # tourism season days
```

**确定依据**：
- 来源：阿拉斯加旅游局和Juneau旅游局官方数据
- 计算：人均消费 = $375M / 1.6M游客 = $234/人
- 调整建议：
  - 旅游季天数：根据气候数据调整 (5月-9月约为150-180天)
  - 基础游客数：使用最近3年平均值

### 3. 时间参数
```python
SIMULATION_START_YEAR = 2025
SIMULATION_END_YEAR = 2045
DISCOUNT_RATE = 0.05  # 5% annual discount rate for NPV
```

**确定依据**：
- 时间范围：基于MCM竞赛要求和长期规划视角
- 折现率：采用政府长期项目标准折现率
- 调整建议：根据具体研究时间范围调整起始年份

---

## 🏔️ 冰川与环境参数

### 4. 冰川基础参数
```python
INITIAL_GLACIER_SIZE = 1.0  # normalized glacier size (2025)
NATURAL_GLACIER_RETREAT_RATE = 0.05  # 5% annual natural retreat
TOURIST_IMPACT_FACTOR = 0.20  # additional 20% retreat per million tourists
```

**确定依据**：
- **自然退缩率 (5%)**：
  - 来源：IPCC气候变化报告和阿拉斯加冰川监测数据
  - 依据：过去30年Mendenhall冰川平均退缩速率
  - 调整范围：3%-8% (取决于气候情景)

- **游客影响因子 (20%)**：
  - 来源：环境影响评估研究和实地调查
  - 依据：每百万游客增加的碳排放和环境压力
  - 调整范围：15%-30% (根据旅游强度)

**敏感性分析**：
```python
# 测试不同情景
scenarios = {
    'optimistic': {'natural': 0.03, 'tourist': 0.15},
    'baseline': {'natural': 0.05, 'tourist': 0.20},
    'pessimistic': {'natural': 0.08, 'tourist': 0.30}
}
```

### 5. 旅游吸引力参数
```python
WHALE_WATCHING_SUCCESS_RATE = 0.85  # 85% success rate
CROWDING_THRESHOLD = 15_000  # daily visitors above this reduce attractiveness
ATTRACTION_GLACIER_WEIGHT = 0.4
ATTRACTION_WHALE_WEIGHT = 0.3
ATTRACTION_CROWDING_WEIGHT = 0.3
```

**确定依据**：
- **鲸鱼观测成功率 (85%)**：
  - 来源：Juneau旅游局运营数据
  - 依据：过去5年平均观测成功率

- **拥挤阈值 (15,000人/天)**：
  - 来源：游客满意度调查和承载力研究
  - 依据：当每日游客超过此数时，体验质量显著下降

- **吸引力权重**：
  - 冰川 (40%)：主要吸引力，独特自然景观
  - 鲸鱼观测 (30%)：标志性体验活动
  - 拥挤影响 (30%)：负面因素，随着游客增加而下降

**调整建议**：
- 基于游客调查数据调整权重
- 考虑季节性因素 (鲸鱼观测在夏季更成功)

---

## 💰 经济参数

### 6. 收入计算参数
```python
BASE_PER_CAPITA_SPENDING = BASE_REVENUE_2023 / BASE_TOURISTS_2023  # ~$234
# Crowding penalty formula in calculate_revenue():
crowding_penalty = max(0.5, 1 - 0.02 * (daily_visitors - CROWDING_THRESHOLD) / CROWDING_THRESHOLD)
```

**确定依据**：
- **人均消费 ($234)**：基于官方旅游收入数据计算
- **拥挤惩罚系数 (0.02)**：
  - 来源：旅游经济学研究
  - 含义：超过阈值后，每超出1%游客导致消费减少2%

**敏感性分析**：
```python
# 不同消费情景
consumption_scenarios = {
    'low': 200,      # $200 per tourist
    'medium': 234,   # baseline
    'high': 280      # $280 per tourist
}
```

---

## 👥 社会影响参数

### 7. 居民满意度参数
```python
SATISFACTION_BASELINE = 1.0
SATISFACTION_TOURIST_IMPACT = 0.5
# Actual calculation:
satisfaction = max(0, SATISFACTION_BASELINE - SATISFACTION_TOURIST_IMPACT * tourist_ratio * 0.1)
```

**确定依据**：
- **基准满意度 (1.0)**：无旅游影响时的满意度基准
- **游客影响因子 (0.5)**：满意度对游客比例的敏感度
- **调整系数 (0.1)**：使满意度下降更渐进，避免极端值

**计算逻辑**：
```
游客比例 = 年游客数 / 常住人口
满意度 = max(0, 1.0 - 0.5 × 游客比例 × 0.1)
```

**调整建议**：
- 基于居民调查数据调整影响因子
- 考虑收入再分配等缓解措施

---

## 🎯 决策变量范围

### 8. 优化变量边界
```python
DECISION_VARIABLE_RANGES = {
    'daily_cap': [10_000, 20_000],  # daily visitor capacity
    'entrance_fee': [0, 50],        # entrance fee in dollars
    'eco_investment_ratio': [0.1, 0.5]  # fraction of revenue for environmental investment
}
```

**确定依据**：
- **日客容量 (10K-20K)**：
  - 下限：基于当前基础设施承载力
  - 上限：基于环境承载力和居民承受力

- **门票价格 ($0-50)**：
  - 下限：免费或象征性收费
  - 上限：基于游客支付意愿调查

- **环保投资比例 (10%-50%)**：
  - 下限：维持基本运营所需的最低投资
  - 上限：基于财政可持续性考虑

---

## ⚙️ 优化算法参数

### 9. NSGA-II参数
```python
NSGA_II_POPULATION_SIZE = 100
NSGA_II_GENERATIONS = 500
NSGA_II_SEED = 42
```

**确定依据**：
- **种群大小 (100)**：
  - 平衡计算效率和解的质量
  - 标准多目标优化规模

- **迭代代数 (500)**：
  - 确保算法收敛到Pareto前沿
  - 基于测试结果确定的最少迭代次数

- **随机种子 (42)**：
  - 确保结果可重现
  - 便于调试和比较

**性能考虑**：
- 增加种群大小：提高解的质量，但增加计算时间
- 增加迭代次数：更好的收敛，但可能过拟合

---

## 🎨 可视化参数

### 10. 图表样式参数
```python
FIGURE_SIZE = (12, 8)
DPI = 300
COLOR_SCHEME = {
    'economic': '#FF6B6B',    # red
    'balanced': '#4ECDC4',    # teal
    'environmental': '#45B7D1' # blue
}
```

**调整建议**：
- 根据论文要求调整图表尺寸
- 选择色盲友好的颜色方案
- 为不同出版物调整DPI

---

## 🔧 参数调整方法

### 1. 配置文件修改
```bash
# 编辑 src/config.py 中的参数
vim src/config.py
```

### 2. 敏感性分析脚本
```python
# 创建参数测试脚本
def sensitivity_analysis():
    test_params = {
        'glacier_retreat': [0.03, 0.05, 0.08],
        'tourist_impact': [0.15, 0.20, 0.30],
        'crowding_threshold': [12000, 15000, 18000]
    }

    for param, values in test_params.items():
        for value in values:
            # 修改参数并运行模拟
            results = run_simulation_with_param(param, value)
            # 记录结果用于分析
```

### 3. 批量参数测试
```python
# 使用拉丁超立方采样进行参数不确定性分析
from scipy.stats import qmc

# 生成参数组合
sampler = qmc.LatinHypercube(d=5)
sample = sampler.random(n=100)

# 参数范围
param_bounds = {
    'natural_retreat': [0.02, 0.08],
    'tourist_impact': [0.10, 0.35],
    'crowding_threshold': [10000, 20000],
    'discount_rate': [0.03, 0.08],
    'satisfaction_impact': [0.3, 0.8]
}
```

### 4. 结果验证
- **交叉验证**: 使用历史数据验证模型预测
- **专家咨询**: 与领域专家讨论参数合理性
- **同行评审**: 分享结果获取反馈

---

## 📊 参数验证指标

### 关键性能指标 (KPIs)
1. **模型拟合度**: 与历史数据的匹配程度
2. **预测稳定性**: 参数小幅变化时的结果波动
3. **政策敏感性**: 不同政策下的区分度
4. **计算效率**: 参数调整对运行时间的影响

### 验证方法
```python
def validate_parameters():
    # 1. 历史数据拟合测试
    historical_fit = calculate_historical_fit_score()

    # 2. 参数稳定性测试
    stability_score = calculate_parameter_stability()

    # 3. 政策区分度测试
    discrimination_score = calculate_policy_discrimination()

    return {
        'fit': historical_fit,
        'stability': stability_score,
        'discrimination': discrimination_score
    }
```

---

## ⚠️ 参数调整注意事项

### 1. 避免过度调整
- **过拟合风险**: 不要为了匹配特定结果而调整参数
- **现实性原则**: 参数应基于可验证的证据

### 2. 文档化更改
- **记录理由**: 说明每次参数调整的依据
- **版本控制**: 维护参数版本历史

### 3. 情景分析
```python
# 定义不同发展情景
scenarios = {
    'business_as_usual': {
        'retreat_rate': 0.05,
        'tourist_growth': 0.03,
        'investment_ratio': 0.2
    },
    'sustainable_growth': {
        'retreat_rate': 0.03,
        'tourist_growth': 0.02,
        'investment_ratio': 0.35
    },
    'aggressive_conservation': {
        'retreat_rate': 0.02,
        'tourist_growth': 0.01,
        'investment_ratio': 0.5
    }
}
```

---

## 📚 参考资料

### 数据来源
- U.S. Census Bureau: 人口统计数据
- Alaska Department of Commerce: 旅游统计
- IPCC Reports: 气候变化影响
- Environmental Impact Studies: 旅游环境影响

### 研究文献
- Tourism Carrying Capacity Studies
- Glacier Retreat Modeling Papers
- Multi-objective Optimization in Environmental Policy
- Sustainable Tourism Management Frameworks

---

*参数最后校准日期: 2025年1月*

*建议定期基于新数据和研究结果更新参数*
