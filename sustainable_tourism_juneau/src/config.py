"""
Juneau可持续旅游管理系统配置文件
定义所有模型参数、常数和取值范围
"""

# 仿真参数
SIMULATION_START_YEAR = 2025      # 仿真开始年份
SIMULATION_END_YEAR = 2045        # 仿真结束年份
SIMULATION_YEARS = SIMULATION_END_YEAR - SIMULATION_START_YEAR + 1  # 仿真总年数
TIME_STEP = 1  # 时间步长(年)

# 旅游季节参数
TOURISM_DAYS_PER_YEAR = 180       # 年旅游天数
BASE_TOURISTS_2023 = 1_600_000    # 2023年基础游客数(160万人)
BASE_REVENUE_2023 = 375_000_000   # 2023年基础收入(3.75亿美元)

# 季节性调整因子
PEAK_SEASON_FACTOR = 1.8          # 夏季高峰期因子(6-8月)
OFF_SEASON_FACTOR = 0.3           # 冬季淡季因子(11-4月)
SHOULDER_SEASON_FACTOR = 0.8      # 过渡季节因子(5月、9-10月)

# 人口与人口统计
POPULATION_JUNEAU = 32_000  # Juneau常住人口
POPULATION_GROWTH_RATE = 0.005   # 年人口增长率(0.5%)

# 冰川参数
INITIAL_GLACIER_SIZE = 1.0        # 初始冰川大小(标准化，2025年)
NATURAL_GLACIER_RETREAT_RATE = 0.05  # 自然退缩率(5%每年)
TOURIST_IMPACT_FACTOR = 0.20      # 游客影响因子(每百万游客额外20%退缩)

# 环境与生态参数
CARBON_EMISSION_FACTOR = 2.5      # 碳排放系数(吨CO2/人/天)
ECOLOGICAL_RECOVERY_RATE = 0.02   # 生态恢复率(2%每年)
POLLUTION_ACCUMULATION_RATE = 0.15 # 污染积累率(15%每年)
BIODIVERSITY_IMPACT_FACTOR = 0.08  # 生物多样性影响因子

# 旅游吸引力参数
WHALE_WATCHING_SUCCESS_RATE = 0.85  # 鲸鱼观测成功率(85%)
CROWDING_THRESHOLD = 15_000         # 拥挤阈值(每日游客超过此数减少吸引力)
ATTRACTION_GLACIER_WEIGHT = 0.4     # 冰川吸引力权重
ATTRACTION_WHALE_WEIGHT = 0.3       # 鲸鱼观测吸引力权重
ATTRACTION_CROWDING_WEIGHT = 0.3    # 拥挤影响权重

# 基础设施参数
HOTEL_CAPACITY = 8_000             # 酒店容量(间/夜)
TRANSPORTATION_CAPACITY = 25_000   # 交通承载力(人/天)
PARKING_CAPACITY = 12_000          # 停车场容量(车位)

# 声誉与口碑参数
REPUTATION_BASELINE = 0.8          # 声誉基准值
REPUTATION_IMPROVEMENT_RATE = 0.1  # 声誉改善率(通过满意游客)
REPUTATION_DECAY_RATE = 0.05       # 声誉衰减率(自然衰减)
REPUTATION_IMPACT_WEIGHT = 0.2     # 声誉对吸引力的影响权重

# 经济参数
DISCOUNT_RATE = 0.05  # NPV折现率(5%每年)
BASE_PER_CAPITA_SPENDING = BASE_REVENUE_2023 / BASE_TOURISTS_2023  # 人均消费(约234美元)
INFLATION_RATE = 0.025                # 通胀率(2.5%每年)
LOCAL_PURCHASE_RATIO = 0.65           # 本地采购比例(65%)
TAX_RATE = 0.08                       # 税收率(8%)

# 就业与经济影响参数
EMPLOYMENT_MULTIPLIER = 1.8           # 就业乘数(1.8个间接就业/直接就业)
WAGE_PREMIUM = 1.15                   # 旅游业工资溢价(15%)
ECONOMIC_SPILLOVER_RATE = 0.25        # 经济外溢效应(25%)

# 居民满意度参数
SATISFACTION_BASELINE = 1.0            # 满意度基准值
SATISFACTION_TOURIST_IMPACT = 0.5      # 游客影响系数(随游客比例降低满意度)
COMMUNITY_BENEFIT_SHARE = 0.15         # 社区福利分享比例(15%)
SERVICE_QUALITY_IMPACT = 0.3           # 服务质量对满意度的影响

# 风险与不确定性参数
NATURAL_DISASTER_PROBABILITY = 0.05    # 自然灾害发生概率(5%每年)
ECONOMIC_SHOCK_PROBABILITY = 0.1       # 经济冲击概率(10%每年)
POLICY_UNCERTAINTY_FACTOR = 0.15       # 政策不确定性因子(15%)
GLOBAL_ECONOMIC_IMPACT = 0.2           # 全球经济影响因子

# 动态学习与网络效应参数
LEARNING_BY_DOING_RATE = 0.08          # 经验学习率(8%每年改善)
NETWORK_EFFECT_STRENGTH = 0.12         # 网络效应强度(口碑传播)
BRAND_LOYALTY_BUILDUP = 0.15           # 品牌忠诚度积累率
SATISFACTION_MEMORY_EFFECT = 0.7       # 满意度记忆效应(70%保留率)

# 决策变量范围
DECISION_VARIABLE_RANGES = {
    'daily_cap': [10_000, 20_000],        # 每日游客容量范围
    'entrance_fee': [0, 50],              # 入园门票价格范围(美元)
    'eco_investment_ratio': [0.1, 0.5]   # 环保投资比例范围(收入占比)
}

# 新增财务/决策平滑与风险参数
PRICE_SMOOTHING_ALPHA = 0.6        # 门票平滑系数(alpha*current + (1-alpha)*prev)
FINANCIAL_STOPLOSS_THRESHOLD = 50_000_000  # 年净收入阈值，低于则触发止损机制
FINANCIAL_STOPLOSS_PENALTY = 0.7  # 止损时净收入乘数（保守估计）
TAX_LAG_YEARS = 1                 # 税收分配滞后年数

# 优化算法参数
NSGA_II_POPULATION_SIZE = 100  # NSGA-II种群大小
NSGA_II_GENERATIONS = 500      # NSGA-II迭代代数
NSGA_II_SEED = 42              # 随机种子(保证结果可重现)

# 可视化参数
FIGURE_SIZE = (12, 8)  # 图表尺寸(宽x高，英寸)
DPI = 300             # 图表分辨率(DPI)
COLOR_SCHEME = {
    'economic': '#FF6B6B',      # 经济政策颜色(红色)
    'balanced': '#4ECDC4',      # 平衡政策颜色(青色)
    'environmental': '#45B7D1'  # 环保政策颜色(蓝色)
}

# 文件路径
RESULTS_DIR = 'results'                                    # 结果目录
PARETO_SOLUTIONS_FILE = f'{RESULTS_DIR}/pareto_solutions.csv'        # Pareto最优解文件
POLICY_COMPARISON_PLOT = f'{RESULTS_DIR}/policy_comparison.png'      # 政策比较图
TIME_SERIES_PLOT = f'{RESULTS_DIR}/time_series.png'                  # 时间序列图
PARETO_FRONT_PLOT = f'{RESULTS_DIR}/pareto_front_3d.png'            # 3D Pareto前沿图
