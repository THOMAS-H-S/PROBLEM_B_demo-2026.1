# Juneau可持续旅游模型参数调整详细指南

本文档提供每个参数的详细调整方法，包括调整范围、影响分析、数据来源和实际操作步骤。

## 📊 参数分类与调整总览

### 参数类型
- 🔵 **基础数据参数**: 基于官方统计数据，相对稳定
- 🟡 **模型参数**: 基于研究文献，可根据新证据调整
- 🟠 **决策变量范围**: 定义优化搜索空间
- 🔴 **算法参数**: 影响计算效率和结果质量

---

## 🎯 基础数据参数调整

### 1. 人口统计数据
```python
POPULATION_JUNEAU = 32_000  # 常住人口数量
```

**调整方法**:
```bash
# 方法1: 直接修改config.py
POPULATION_JUNEAU = 32500  # 根据最新人口普查数据更新

# 方法2: 使用参数调优工具
python parameter_tuning.py
# 选择选项5查看当前值，然后选择选项3保存新配置
```

**数据来源**:
- 美国人口普查局 (U.S. Census Bureau)
- 阿拉斯加州政府人口统计
- 当地政府年度人口报告

**调整频率**: 每年更新（基于人口普查数据）

**影响分析**:
- 居民满意度计算: `满意度 = f(游客数/人口)`
- 人口增加 → 满意度对游客更敏感
- 人口减少 → 对游客影响容忍度更高

---

### 2. 旅游基础数据
```python
BASE_TOURISTS_2023 = 1_600_000    # 2023年游客基数
BASE_REVENUE_2023 = 375_000_000   # 2023年收入基数
TOURISM_DAYS_PER_YEAR = 180       # 年旅游天数
```

**调整步骤**:
1. **获取最新数据**
   ```python
   # 从阿拉斯加旅游局获取最新年报数据
   latest_tourists = 1_650_000    # 最新游客数据
   latest_revenue = 390_000_000   # 最新收入数据
   ```

2. **计算人均消费**
   ```python
   # 自动计算，不需要手动调整
   BASE_PER_CAPITA_SPENDING = BASE_REVENUE_2023 / BASE_TOURISTS_2023
   # 结果: $234 per tourist
   ```

3. **调整季节天数**
   ```python
   # 根据气候数据调整
   TOURISM_DAYS_PER_YEAR = 160  # 如果气候变化导致季节缩短
   TOURISM_DAYS_PER_YEAR = 200  # 如果开发新旅游产品延长季节
   ```

**数据来源**:
- 阿拉斯加旅游局年度报告
- 港口管理局游客统计
- 航空公司乘客数据

---

## 🏔️ 冰川与环境参数调整

### 3. 冰川退缩参数
```python
INITIAL_GLACIER_SIZE = 1.0           # 初始冰川大小(标准化)
NATURAL_GLACIER_RETREAT_RATE = 0.05  # 自然退缩率(5%)
TOURIST_IMPACT_FACTOR = 0.20         # 游客影响因子(20%)
```

**自然退缩率调整**:
```python
# 基于IPCC气候情景调整
conservative = 0.03   # 乐观情景: 气候变化较慢
baseline = 0.05       # 基准情景: 当前趋势
severe = 0.08         # 悲观情景: 加速变暖

# 实际调整
NATURAL_GLACIER_RETREAT_RATE = conservative  # 选择情景
```

**游客影响因子调整**:
```python
# 基于实地研究调整
low_impact = 0.15     # 游客活动管理良好
medium_impact = 0.20  # 当前管理水平
high_impact = 0.30    # 管理不善，游客活动无序

# 实际调整
TOURIST_IMPACT_FACTOR = low_impact  # 如果实施了环境保护措施
```

**敏感性测试**:
```bash
# 运行敏感性分析
python quick_parameter_test.py
# 查看 parameter_analysis/glacier_test.png
```

---

### 4. 旅游吸引力参数
```python
WHALE_WATCHING_SUCCESS_RATE = 0.85    # 鲸鱼观测成功率
CROWDING_THRESHOLD = 15_000            # 拥挤阈值(人/天)
ATTRACTION_GLACIER_WEIGHT = 0.4       # 冰川吸引力权重
ATTRACTION_WHALE_WEIGHT = 0.3         # 鲸鱼观测权重
ATTRACTION_CROWDING_WEIGHT = 0.3      # 拥挤影响权重
```

**鲸鱼观测成功率**:
```python
# 基于运营数据调整
WHALE_WATCHING_SUCCESS_RATE = 0.82  # 如果海洋条件变差
WHALE_WATCHING_SUCCESS_RATE = 0.90  # 如果开发了新技术
```

**拥挤阈值调整**:
```python
# 基于游客调查调整
CROWDING_THRESHOLD = 12_000  # 如果游客体验调查显示更低容忍度
CROWDING_THRESHOLD = 18_000  # 如果基础设施改善提高容忍度
```

**吸引力权重调整**:
```python
# 基于市场调研调整
# 情景1: 强调自然景观
ATTRACTION_GLACIER_WEIGHT = 0.5
ATTRACTION_WHALE_WEIGHT = 0.3
ATTRACTION_CROWDING_WEIGHT = 0.2

# 情景2: 强调独特体验
ATTRACTION_GLACIER_WEIGHT = 0.3
ATTRACTION_WHALE_WEIGHT = 0.5
ATTRACTION_CROWDING_WEIGHT = 0.2

# 情景3: 注重游客舒适度
ATTRACTION_GLACIER_WEIGHT = 0.3
ATTRACTION_WHALE_WEIGHT = 0.2
ATTRACTION_CROWDING_WEIGHT = 0.5
```

---

## 💰 经济参数调整

### 5. 折现率与消费参数
```python
DISCOUNT_RATE = 0.05                    # NPV折现率
BASE_PER_CAPITA_SPENDING = 234          # 人均消费(美元)
```

**折现率调整**:
```python
# 基于项目类型调整
government_rate = 0.03    # 政府长期项目
private_rate = 0.08       # 私人投资项目
conservative_rate = 0.10  # 风险较高情景

DISCOUNT_RATE = government_rate  # 选择适用类型
```

**人均消费调整**:
```python
# 基于通胀和消费习惯调整
# 自动计算: BASE_REVENUE_2023 / BASE_TOURISTS_2023
# 如果需要手动覆盖:
BASE_PER_CAPITA_SPENDING = 250  # 考虑通胀调整
```

---

## 👥 社会影响参数调整

### 6. 居民满意度参数
```python
SATISFACTION_BASELINE = 1.0            # 满意度基准
SATISFACTION_TOURIST_IMPACT = 0.5      # 游客影响系数
```

**游客影响系数调整**:
```python
# 基于居民调查调整
low_tolerance = 0.8      # 居民对游客敏感度高
medium_tolerance = 0.5   # 当前基准水平
high_tolerance = 0.3     # 居民适应度高

SATISFACTION_TOURIST_IMPACT = low_tolerance  # 如果居民投诉增加
```

**基准满意度调整**:
```python
# 通常保持为1.0，除非有特殊情况
SATISFACTION_BASELINE = 0.9  # 如果基准期就有不满
SATISFACTION_BASELINE = 1.1  # 如果有积极因素(不推荐超过1.0)
```

---

## 🎯 决策变量范围调整

### 7. 优化变量边界
```python
DECISION_VARIABLE_RANGES = {
    'daily_cap': [10_000, 20_000],        # 日客容量范围
    'entrance_fee': [0, 50],              # 门票价格范围
    'eco_investment_ratio': [0.1, 0.5]   # 环保投资比例范围
}
```

**日客容量范围**:
```python
# 基于基础设施承载力调整
DECISION_VARIABLE_RANGES['daily_cap'] = [8_000, 25_000]    # 如果基础设施改善
DECISION_VARIABLE_RANGES['daily_cap'] = [12_000, 18_000]   # 如果环境限制严格
```

**门票价格范围**:
```python
# 基于支付意愿调查调整
DECISION_VARIABLE_RANGES['entrance_fee'] = [0, 75]        # 如果游客支付意愿高
DECISION_VARIABLE_RANGES['entrance_fee'] = [5, 30]        # 如果要限制高端游客
```

**环保投资比例**:
```python
# 基于财政可持续性调整
DECISION_VARIABLE_RANGES['eco_investment_ratio'] = [0.05, 0.6]  # 如果有更多资金
DECISION_VARIABLE_RANGES['eco_investment_ratio'] = [0.15, 0.4]  # 如果预算紧张
```

---

## ⚙️ 算法参数调整

### 8. NSGA-II参数
```python
NSGA_II_POPULATION_SIZE = 100   # 种群大小
NSGA_II_GENERATIONS = 500       # 迭代代数
NSGA_II_SEED = 42               # 随机种子
```

**种群大小调整**:
```python
# 平衡计算时间和解质量
NSGA_II_POPULATION_SIZE = 50    # 快速测试
NSGA_II_POPULATION_SIZE = 200   # 高质量结果(计算时间长)
```

**迭代代数调整**:
```python
# 确保收敛
NSGA_II_GENERATIONS = 200      # 快速收敛
NSGA_II_GENERATIONS = 1000     # 充分优化(计算时间长)
```

---

## 🎨 可视化参数调整

### 9. 图表样式参数
```python
FIGURE_SIZE = (12, 8)           # 图表尺寸
DPI = 300                       # 分辨率
COLOR_SCHEME = {                # 颜色方案
    'economic': '#FF6B6B',
    'balanced': '#4ECDC4',
    'environmental': '#45B7D1'
}
```

**图表尺寸调整**:
```python
# 根据用途调整
FIGURE_SIZE = (16, 10)  # 演示文稿
FIGURE_SIZE = (8, 6)    # 论文插图
FIGURE_SIZE = (24, 16)  # 海报展示
```

**颜色方案调整**:
```python
# 考虑色盲友好性
COLOR_SCHEME = {
    'economic': '#DC267F',      # 红色(色盲友好)
    'balanced': '#785EF0',      # 紫色
    'environmental': '#648FFF'  # 蓝色
}
```

---

## 🔧 实际调整操作流程

### 步骤1: 备份当前配置
```bash
cp src/config.py src/config_backup.py
```

### 步骤2: 运行敏感性分析
```bash
python quick_parameter_test.py
# 查看哪些参数影响最大
```

### 步骤3: 使用调优工具
```bash
python parameter_tuning.py
# 交互式调整参数
```

### 步骤4: 修改config.py
```bash
vim src/config.py
# 根据分析结果调整参数
```

### 步骤5: 运行完整模型
```bash
python main.py
# 生成新结果
```

### 步骤6: 验证结果
```bash
# 检查 results/ 目录的新图表
# 确保结果符合逻辑
```

---

## 📊 参数验证与测试

### 自动验证脚本
```bash
# 运行参数验证
python -c "
from src.system_dynamics import simulate_policy
result = simulate_policy((15000, 25, 0.3))
print(f'NPV: ${result[\"npv\"]/1e9:.1f}B')
print(f'Final glacier: {result[\"final_glacier\"]:.2%}')
print(f'Avg satisfaction: {result[\"avg_satisfaction\"]:.1%}')
"
```

### 批量情景测试
```python
# 定义多个情景
scenarios = {
    'conservative': {'retreat_rate': 0.03, 'tourist_impact': 0.15},
    'baseline': {'retreat_rate': 0.05, 'tourist_impact': 0.20},
    'aggressive': {'retreat_rate': 0.08, 'tourist_impact': 0.30}
}

# 使用parameter_tuning.py进行批量测试
```

---

## ⚠️ 参数调整注意事项

### 1. 渐进式调整
```python
# 不要一次性调整太多参数
# 每次只调整一个参数族
# 观察对结果的影响
```

### 2. 记录更改理由
```python
# 在config.py中添加注释
NATURAL_GLACIER_RETREAT_RATE = 0.04  # 调整日期: 2024-01-15
                                      # 理由: 基于最新IPCC报告
                                      # 来源: IPCC AR6 Chapter 9
```

### 3. 定期复核
```python
# 每季度复核关键参数
# 每年更新基础数据
# 根据新研究调整模型参数
```

### 4. 版本控制
```python
# 维护参数版本历史
# 为不同情景保存不同配置
# 便于结果重现和比较
```

---

## 📚 推荐数据来源

### 官方数据源
- **美国人口普查局**: https://www.census.gov
- **阿拉斯加旅游局**: https://www.alaskavisit.org
- **NOAA气候数据**: https://www.noaa.gov/climate

### 研究文献
- IPCC气候变化评估报告
- 旅游承载力研究论文
- 冰川监测项目报告

---

*最后更新: 2024年1月*
*建议结合当地最新数据定期调整参数*
