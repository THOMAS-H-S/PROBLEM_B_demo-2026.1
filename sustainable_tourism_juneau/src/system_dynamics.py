"""
Juneau旅游系统动力学模型
模拟旅游、冰川退缩、收入和居民满意度的长期演化
"""

import numpy as np
from typing import Tuple, Dict, List
try:
    from .config import *
except ImportError:
    from config import *


def calculate_seasonal_factor(year: int, month: int = 7) -> float:
    """
    计算季节性调整因子

    参数:
    - year: 年份
    - month: 月份(用于模拟季节变化)

    返回:
    - seasonal_factor: 季节性调整因子
    """
    # 简化的季节性模型：夏季(6-8月)高峰，冬季(11-4月)淡季
    if month in [6, 7, 8]:  # 夏季高峰
        return PEAK_SEASON_FACTOR
    elif month in [11, 12, 1, 2, 3, 4]:  # 冬季淡季
        return OFF_SEASON_FACTOR
    else:  # 过渡季节
        return SHOULDER_SEASON_FACTOR


def calculate_infrastructure_capacity(daily_visitors: int) -> float:
    """
    计算基础设施容量限制因子

    参数:
    - daily_visitors: 每日游客数量

    返回:
    - capacity_factor: 容量限制因子(0-1)
    """
    # 多维度容量评估
    hotel_utilization = min(1.0, daily_visitors / HOTEL_CAPACITY)
    transport_utilization = min(1.0, daily_visitors / TRANSPORTATION_CAPACITY)
    parking_utilization = min(1.0, daily_visitors / PARKING_CAPACITY)

    # 综合容量因子：最严格的限制
    capacity_factor = min(hotel_utilization, transport_utilization, parking_utilization)

    # 超过容量时呈指数衰减
    if capacity_factor > 0.9:
        capacity_factor = max(0.1, 1.0 - 2.0 * (capacity_factor - 0.9))

    return capacity_factor


def calculate_tourist_attractiveness(glacier_size: float, daily_visitors: int,
                                   reputation: float = 0.8, year: int = 2025) -> float:
    """
    计算旅游吸引力，基于多维度因素

    参数:
    - glacier_size: 标准化冰川大小(0-1)
    - daily_visitors: 每日游客数量
    - reputation: 目的地声誉(0-1)
    - year: 年份(用于动态调整)

    返回:
    - attractiveness: 标准化吸引力评分(0-1)
    """
    # 冰川贡献
    glacier_attraction = ATTRACTION_GLACIER_WEIGHT * glacier_size

    # 鲸鱼观测贡献(考虑季节性和成功率)
    seasonal_month = 7  # 假设7月为代表性月份
    seasonal_whale_factor = calculate_seasonal_factor(year, seasonal_month)
    whale_attraction = ATTRACTION_WHALE_WEIGHT * WHALE_WATCHING_SUCCESS_RATE * seasonal_whale_factor

    # 拥挤效应(超过阈值时负面影响)
    crowding_factor = max(0, (CROWDING_THRESHOLD - daily_visitors) / CROWDING_THRESHOLD)
    crowding_attraction = ATTRACTION_CROWDING_WEIGHT * crowding_factor

    # 声誉效应
    reputation_attraction = REPUTATION_IMPACT_WEIGHT * reputation

    # 基础设施容量效应
    capacity_factor = calculate_infrastructure_capacity(daily_visitors)
    infrastructure_attraction = 0.1 * capacity_factor  # 10%权重

    # 学习与网络效应(随时间改善)
    years_since_start = year - 2025
    learning_effect = min(0.1, years_since_start * LEARNING_BY_DOING_RATE)  # 学习效应
    network_effect = min(0.1, years_since_start * NETWORK_EFFECT_STRENGTH)  # 网络效应

    attractiveness = (glacier_attraction + whale_attraction + crowding_attraction +
                     reputation_attraction + infrastructure_attraction +
                     learning_effect + network_effect)

    return max(0, min(1, attractiveness))  # 限制在0-1范围内


def calculate_tourist_demand(attractiveness: float, entrance_fee: float,
                          reputation: float = 0.8, year: int = 2025) -> int:
    """
    根据吸引力和入园门票计算游客需求（增强版）

    参数:
    - attractiveness: 旅游吸引力(0-1)
    - entrance_fee: 入园门票价格(美元)
    - reputation: 目的地声誉(0-1)
    - year: 年份(用于趋势分析)

    返回:
    - annual_tourists: 年游客数量
    """
    # 基础需求随吸引力变化
    base_demand = BASE_TOURISTS_2023 * (0.5 + 0.5 * attractiveness)

    # 价格弹性效应(考虑通胀调整)
    inflation_adjusted_fee = entrance_fee * ((1 + INFLATION_RATE) ** (year - 2023))
    price_elasticity = 0.3  # 需求随价格降低
    price_effect = max(0.1, 1 - price_elasticity * (inflation_adjusted_fee / 25))

    # 声誉效应
    reputation_effect = 0.8 + 0.4 * reputation  # 声誉好的地方需求更高

    # 经济波动效应
    economic_shock = 1.0
    if np.random.random() < ECONOMIC_SHOCK_PROBABILITY:
        economic_shock = np.random.uniform(0.8, 1.2)  # 经济冲击随机因子

    # 全球经济影响
    global_economic_effect = 1.0 + GLOBAL_ECONOMIC_IMPACT * np.sin(2 * np.pi * (year - 2025) / 5)

    annual_tourists = int(base_demand * price_effect * reputation_effect *
                         economic_shock * global_economic_effect)

    return max(0, annual_tourists)


def calculate_revenue(annual_tourists: int, daily_cap: int, entrance_fee: float) -> float:
    """
    计算年收入，考虑容量限制和拥挤效应

    参数:
    - annual_tourists: 年总游客数
    - daily_cap: 每日游客容量
    - entrance_fee: 每位游客入园门票

    返回:
    - revenue: 年收入(美元)
    """
    # 计算有效每日游客(受容量限制)
    max_annual_visitors = daily_cap * TOURISM_DAYS_PER_YEAR
    effective_annual_visitors = min(annual_tourists, max_annual_visitors)

    # 门票收入
    entrance_revenue = effective_annual_visitors * entrance_fee

    # 额外消费收入(受拥挤影响)
    daily_visitors = effective_annual_visitors / TOURISM_DAYS_PER_YEAR
    crowding_penalty = max(0.5, 1 - 0.02 * (daily_visitors - CROWDING_THRESHOLD) / CROWDING_THRESHOLD)
    spending_revenue = effective_annual_visitors * BASE_PER_CAPITA_SPENDING * crowding_penalty

    total_revenue = entrance_revenue + spending_revenue

    return total_revenue


def calculate_glacier_retreat(glacier_size: float, annual_tourists: int, eco_investment: float) -> float:
    """
    计算年冰川退缩率，考虑自然退缩和游客影响

    参数:
    - glacier_size: 当前冰川大小(0-1)
    - annual_tourists: 本年游客数量
    - eco_investment: 本年环保投资

    返回:
    - retreat_rate: 年退缩率(0-1)
    """
    # 自然退缩
    natural_retreat = NATURAL_GLACIER_RETREAT_RATE

    # 游客影响(每百万游客的额外退缩)
    tourist_impact = TOURIST_IMPACT_FACTOR * (annual_tourists / 1_000_000)

    # 环保投资减少退缩
    investment_effect = min(0.5, eco_investment / (0.1 * BASE_REVENUE_2023))  # 标准化投资

    total_retreat = natural_retreat + tourist_impact - 0.1 * investment_effect

    return max(0, total_retreat)  # 不能为负


def calculate_resident_satisfaction(annual_tourists: int, eco_investment: float = 0,
                             community_benefits: float = 0, year: int = 2025) -> float:
    """
    计算居民满意度，考虑多维度因素

    参数:
    - annual_tourists: 本年游客数量
    - eco_investment: 环保投资金额
    - community_benefits: 社区福利分配金额
    - year: 年份(用于动态调整)

    返回:
    - satisfaction: 标准化满意度(0-1)
    """
    tourist_ratio = annual_tourists / POPULATION_JUNEAU

    # 基础游客影响
    base_impact = SATISFACTION_TOURIST_IMPACT * tourist_ratio * 0.1

    # 社区福利效应
    benefit_per_capita = community_benefits / POPULATION_JUNEAU
    benefit_effect = min(0.2, benefit_per_capita / 1000)  # 每人1000美元福利可提升0.2满意度

    # 环保投资效应(通过改善环境质量)
    eco_effect = min(0.15, eco_investment / (0.1 * BASE_REVENUE_2023)) * 0.1

    # 服务质量效应(基于基础设施投资)
    infrastructure_quality = min(1.0, (year - 2025) * 0.05 + 0.8)  # 随时间改善
    service_effect = SERVICE_QUALITY_IMPACT * infrastructure_quality

    # 就业与经济效应
    employment_effect = min(0.1, annual_tourists * EMPLOYMENT_MULTIPLIER / POPULATION_JUNEAU * 0.05)

    # 综合满意度计算
    satisfaction = (SATISFACTION_BASELINE - base_impact + benefit_effect +
                   eco_effect + service_effect + employment_effect)

    return max(0, min(1, satisfaction))  # 限制在0-1范围内


def simulate_policy(policy: Tuple[float, float, float]) -> Dict:
    """
    在给定政策下模拟旅游系统20年演化(2025-2045)

    参数:
    - policy: 元组(daily_cap, entrance_fee, eco_investment_ratio)
             - daily_cap: 每日游客容量
             - entrance_fee: 入园门票价格
             - eco_investment_ratio: 环保投资比例

    返回:
    - results: 包含仿真结果的字典
    """
    daily_cap, entrance_fee, eco_investment_ratio = policy

    # 初始化状态变量
    years = list(range(SIMULATION_START_YEAR, SIMULATION_END_YEAR + 1))
    glacier_size = [INITIAL_GLACIER_SIZE]
    tourists = []
    revenue = []
    satisfaction = []
    attractiveness_history = []

    # 新增状态变量
    reputation = [REPUTATION_BASELINE]      # 目的地声誉
    pollution_level = [0.0]                 # 污染积累水平
    community_benefits = [0.0]              # 社区福利积累
    economic_spillover = [0.0]              # 经济外溢效应
    population_trend = [POPULATION_JUNEAU]  # 人口变化趋势
    # 财务与平滑状态
    prev_entrance_fee = entrance_fee
    tax_history = [0.0] * (TAX_LAG_YEARS + 1)  # 用于实现税收滞后分配

    # 仿真循环
    for year in years:
        # 门票价格平滑（避免价格剧烈跳动对需求的非理性冲击）
        smoothed_fee = PRICE_SMOOTHING_ALPHA * entrance_fee + (1 - PRICE_SMOOTHING_ALPHA) * prev_entrance_fee
        prev_entrance_fee = smoothed_fee

        # 基于当前状态计算吸引力（使用日均估计游客）
        daily_visitors_estimate = min(daily_cap, BASE_TOURISTS_2023 / TOURISM_DAYS_PER_YEAR)
        attractiveness = calculate_tourist_attractiveness(glacier_size[-1], daily_visitors_estimate,
                                                        reputation[-1], year)
        attractiveness_history.append(attractiveness)

        # 计算游客需求(考虑声誉和经济因素)
        annual_tourists = calculate_tourist_demand(attractiveness, smoothed_fee, reputation[-1], year)
        tourists.append(annual_tourists)

        # 计算收入(考虑通胀和税收)
        annual_revenue = calculate_revenue(annual_tourists, daily_cap, smoothed_fee)
        # 应用通胀调整
        inflation_adjusted_revenue = annual_revenue * ((1 + INFLATION_RATE) ** (year - 2023))
        # 计算税收（应用税率），但分配给社区使用时采用滞后机制
        tax_revenue = inflation_adjusted_revenue * TAX_RATE
        # 当前年可用净收入先不包含税收分配
        net_revenue = inflation_adjusted_revenue * (1 - TAX_RATE)
        revenue.append(net_revenue)

        # 计算居民满意度(考虑社区福利)
        eco_investment_amount = net_revenue * eco_investment_ratio
        # 若净收入低于阈值，触发财务止损机制（限制投资与分配）
        if net_revenue < FINANCIAL_STOPLOSS_THRESHOLD:
            net_revenue = net_revenue * FINANCIAL_STOPLOSS_PENALTY
            eco_investment_amount = net_revenue * eco_investment_ratio

        annual_satisfaction = calculate_resident_satisfaction(annual_tourists,
                                                            eco_investment_amount,
                                                            community_benefits[-1], year)
        satisfaction.append(annual_satisfaction)

        # 计算冰川退缩(考虑污染积累)
        eco_investment = net_revenue * eco_investment_ratio
        retreat_rate = calculate_glacier_retreat(glacier_size[-1], annual_tourists, eco_investment)

        # 考虑污染积累效应
        pollution_impact = pollution_level[-1] * 0.05  # 污染对冰川的额外影响
        total_retreat_rate = retreat_rate + pollution_impact

        # 更新冰川大小
        new_glacier_size = max(0, glacier_size[-1] * (1 - total_retreat_rate))
        glacier_size.append(new_glacier_size)

        # 更新声誉(基于满意度和游客反馈)
        reputation_change = (REPUTATION_IMPROVEMENT_RATE * annual_satisfaction -
                           REPUTATION_DECAY_RATE)
        new_reputation = max(0, min(1, reputation[-1] + reputation_change))
        reputation.append(new_reputation)

        # 更新污染水平
        tourist_pollution = annual_tourists * CARBON_EMISSION_FACTOR / 1000000  # 百万吨CO2
        pollution_recovery = pollution_level[-1] * ECOLOGICAL_RECOVERY_RATE
        new_pollution = max(0, pollution_level[-1] + tourist_pollution - pollution_recovery)
        pollution_level.append(new_pollution)

        # 更新税收历史并按滞后分配到社区福利
        tax_history.append(tax_revenue)
        # 本次分配使用 TAX_LAG_YEARS 年前的税收（若可用）
        lag_index = -1 - TAX_LAG_YEARS
        lagged_tax = tax_history[lag_index] if len(tax_history) >= TAX_LAG_YEARS + 1 else 0.0
        community_allocation = lagged_tax * COMMUNITY_BENEFIT_SHARE
        new_community_benefits = community_benefits[-1] + community_allocation
        community_benefits.append(new_community_benefits)

        # 更新经济外溢效应
        local_spending = net_revenue * LOCAL_PURCHASE_RATIO
        spillover_effect = local_spending * ECONOMIC_SPILLOVER_RATE
        new_spillover = economic_spillover[-1] + spillover_effect
        economic_spillover.append(new_spillover)

        # 更新人口(考虑就业机会)
        employment_opportunity = annual_tourists * EMPLOYMENT_MULTIPLIER / POPULATION_JUNEAU
        population_change = population_trend[-1] * POPULATION_GROWTH_RATE * (1 + employment_opportunity * 0.1)
        new_population = population_trend[-1] + population_change
        population_trend.append(new_population)

    # 移除多余的冰川大小条目
    glacier_size = glacier_size[:-1]

    # 计算收入的NPV
    npv = 0
    for t, rev in enumerate(revenue):
        npv += rev / ((1 + DISCOUNT_RATE) ** t)

    # 计算最终指标
    final_glacier = glacier_size[-1]
    avg_satisfaction = np.mean(satisfaction)

    return {
        'years': years,
        'tourists': tourists,
        'glacier': glacier_size,
        'revenue': revenue,
        'satisfaction': satisfaction,
        'attractiveness': attractiveness_history,
        'reputation': reputation,
        'pollution_level': pollution_level,
        'community_benefits': community_benefits,
        'economic_spillover': economic_spillover,
        'population_trend': population_trend,
        'npv': npv,
        'final_glacier': final_glacier,
        'avg_satisfaction': avg_satisfaction,
        'final_reputation': reputation[-1],
        'final_pollution': pollution_level[-1],
        'total_community_benefits': community_benefits[-1],
        'total_economic_spillover': economic_spillover[-1],
        'population_change': population_trend[-1] - population_trend[0]
    }
