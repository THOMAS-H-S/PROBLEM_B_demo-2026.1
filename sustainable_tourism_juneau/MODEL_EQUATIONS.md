### 模型方程与变量释义（汇总）

本文件把项目中使用的关键数学公式与变量名称集中列出，便于在论文中引用和与代码对照（代码变量名见 `src/config.py`）。

1) 状态-控制向量（连续表述）

设状态向量为 $x(t)\in\mathbb{R}^n$，控制/决策向量为 $u(t)\in\mathbb{R}^m$，参数向量为 $\theta$，系统动力学形式为：
$$
\frac{dx(t)}{dt} = f\big(x(t),\,u(t),\,\theta\big),\qquad x(0)=x_0.
$$

离散化（显式欧拉，年为步长）：
$$
x_{k+1} = x_k + \Delta t\cdot f(x_k,u_k,\theta),\quad \Delta t=1\ \text{年}.
$$

2) 关键子模型（示例、代码实现对应）

- 游客需求（年度游客量）：
$$
T(t) = V_{base}\cdot A(t)\cdot e^{-\epsilon\cdot P(t)}\cdot R(t-\tau)
$$
说明：$A(t)$ 为目的地综合吸引力；$P(t)$ 为票价/价格信号；$R(\cdot)$ 为声誉滞后项（代码中对应 `REPUTATION_*` 和 `PRICE_*` 变量）。

- 收入（年度）与折现（用于 NPV）：
$$
R(t) = T(t)\cdot s(t)\quad,\quad \text{NPV}=\sum_{t=0}^{T-1}\frac{R(t)\,(1-{\rm tax}) - C(t)}{(1+\rho)^t}
$$
其中 $s(t)$ 为人均消费（随通胀和拥挤调整；代码变量 `BASE_PER_CAPITA_SPENDING`, `INFLATION_RATE` 等），$\rho$ 为折现率 `DISCOUNT_RATE`。

- 冰川退缩（年度相对变化）：
$$
\frac{dG}{dt} = -\alpha_{nat} - \beta\left(\frac{T(t)}{C_{cap}}\right)^k + \gamma\ln(1+Inv_{eco}(t)) - \eta\cdot Pollution(t)
$$
说明：$\alpha_{nat}$（`NATURAL_GLACIER_RETREAT_RATE`）为自然退缩；第二项为游客引起的人为加速（`TOURIST_IMPACT_FACTOR`、`HOTEL_CAPACITY`、`TRANSPORTATION_CAPACITY` 相关）；第三项为环保投资的减缓效应（`eco_investment_ratio`）；$\eta$ 是污染对退缩的影响系数（`POLLUTION_ACCUMULATION_RATE`）。

- 居民满意度（年度指标）：
$$
S(t) = \max\Big(0,\ S_0 + \lambda_{econ}\cdot\frac{LocalSpend(t)}{Population} - \lambda_{crowd}\cdot\frac{T(t)}{Population} + \lambda_{eco}\cdotInv_{eco}(t)\Big)
$$
说明：$S_0$ 为基线满意度（`SATISFACTION_BASELINE`），局部支出 `LocalSpend` 与就业溢出影响满意度（对应 `LOCAL_PURCHASE_RATIO`、`EMPLOYMENT_MULTIPLIER`）。

3) 目标函数（MOO）

我们在实现中将三个目标写为（供 NSGA-II 优化）：
- 最大化折现旅游净收益：$\max\ {\rm NPV}$（实现为最小化 $-{\rm NPV}$）
- 最大化终末冰川尺度：$\max\ G_{T}$（实现为最小化 $1-G_T$ 或最小化 $-G_T$）
- 最大化平均居民满意度：$\max\ \frac{1}{T}\sum_{t} S(t)$（实现为最小化 $-{\rm avgS}$）

4) 约束（示例）

- 物理承载力：$T_{daily} \le C_{transport}$（代码：`TRANSPORTATION_CAPACITY`）
- 社会契约：$S(t)\ge S_{min}$（代码：`SATISFACTION_BASELINE` 与阈值）
- 预算约束：$\sum Inv_{eco}(t) \le \kappa\cdot \sum Surplus(t)$（代码中有 `COMMUNITY_BENEFIT_SHARE` 等分配规则）

5) 变量与代码名映射（常用）

- 年度/时间：$t$ ↔ `years`、`SIMULATION_START_YEAR`、`SIMULATION_END_YEAR`。  
- 游客（年度总量）：$T(t)$ ↔ `tourists`、`BASE_TOURISTS_2023`。  
- 日游客容量：$X_1$ ↔ `daily_cap`（决策变量）。  
- 入场门票/生态费：$X_2$ ↔ `entrance_fee`（决策变量）。  
- 环保投资比率：$X_3$ ↔ `eco_investment_ratio`（决策变量）。  
- 冰川大小：$G(t)$ ↔ `glacier` / `INITIAL_GLACIER_SIZE`。  
- 收入：$R(t)$ ↔ `revenue` / `BASE_REVENUE_2023`。  
- 满意度：$S(t)$ ↔ `satisfaction` / `SATISFACTION_BASELINE`。  
- 折现率：$\rho$ ↔ `DISCOUNT_RATE`。  
- 污染/生态恢复：`POLLUTION_ACCUMULATION_RATE`, `ECOLOGICAL_RECOVERY_RATE`。

6) 实用注记（实现细节）

- 行内时间步长为 1 年；若需更细粒度（按日/月），请调整 `TIME_STEP` 并相应离散化公式。  
- 价格与需求：代码实现中对门票做了平滑（`PRICE_SMOOTHING_ALPHA`）和通胀调整，建议在敏感性分析时打开/关闭平滑以观察波动敏感性。  
- 若要严格与论文符号匹配，可用上表中“变量 ↔ 代码名”替换代码中的变量名或在论文的附录里列出对照表。

---  
（如需我把本文件转换为 LaTeX 附录或把每个公式配上代码片段定位，请指示我导出哪几个具体公式与代码文件对应行号。）  

## 变量对照表（Variable glossary）
下面表格汇总了模型中出现的主要变量与代码中对应的名称、含义、单位与默认值（若适用）。建议将此表放入论文附录以便审阅者查证参数来源与实现对应关系。

| 变量/符号 | 代码名 | 含义 / 说明 | 单位 | 默认值 / 备注 |
|---|---|---|---:|---|
| 仿真起始年 | `SIMULATION_START_YEAR` | 仿真开始年份 | 年 | 2025 |
| 仿真结束年 | `SIMULATION_END_YEAR` | 仿真结束年份 | 年 | 2045 |
| 时间步长 | `TIME_STEP` | 仿真时间步长 | 年 | 1 |
| 年旅游天数 | `TOURISM_DAYS_PER_YEAR` | 年内旅游天数（用于年/日换算） | 天/年 | 180 |
| 基线年游客数 | `BASE_TOURISTS_2023` | 2023年已知游客数（基线） | 人/年 | 1,600,000 |
| 基线年收入 | `BASE_REVENUE_2023` | 2023年已知总收入（基线） | 美元/年 | 375,000,000 |
| 夏季因子 | `PEAK_SEASON_FACTOR` | 夏季季节性放大因子（用于观鲸等） | 无量纲 | 1.8 |
| 淡季因子 | `OFF_SEASON_FACTOR` | 冬季季节性因子 | 无量纲 | 0.3 |
| 过渡季因子 | `SHOULDER_SEASON_FACTOR` | 过渡季节性因子 | 无量纲 | 0.8 |
| 常住人口 | `POPULATION_JUNEAU` | Juneau 常住人口 | 人 | 32,000 |
| 人口增长率 | `POPULATION_GROWTH_RATE` | 年人口增长率 | 年比率 | 0.005 |
| 初始冰川大小 | `INITIAL_GLACIER_SIZE` | 冰川存量标准化起点（2025） | 标准化 | 1.0 |
| 自然退缩率 | `NATURAL_GLACIER_RETREAT_RATE` | 无人为影响下的基线退缩率 | 年比率 | 0.05 |
| 游客影响因子 | `TOURIST_IMPACT_FACTOR` | 每百万游客额外对退缩的影响（比例） | 无量纲 | 0.20 |
| 碳排放系数 | `CARBON_EMISSION_FACTOR` | 人均日碳排放（吨CO2/人/天） | 吨/人/天 | 2.5 |
| 生态恢复率 | `ECOLOGICAL_RECOVERY_RATE` | 污染/生态恢复速度 | 年比率 | 0.02 |
| 污染积累率 | `POLLUTION_ACCUMULATION_RATE` | 污染年度积累参数（模型可引用） | 无量纲 | 0.15 |
| 鲸观成功率 | `WHALE_WATCHING_SUCCESS_RATE` | 观鲸成功率（用于吸引力） | 比例 | 0.85 |
| 拥挤阈值 | `CROWDING_THRESHOLD` | 日游客超过该值后出现明显拥挤效应 | 人/日 | 15,000 |
| 冰川吸引力权重 | `ATTRACTION_GLACIER_WEIGHT` | 冰川对吸引力的权重 | 无量纲 | 0.4 |
| 鲸观吸引力权重 | `ATTRACTION_WHALE_WEIGHT` | 鲸观对吸引力的权重 | 无量纲 | 0.3 |
| 拥挤吸引力权重 | `ATTRACTION_CROWDING_WEIGHT` | 拥挤对吸引力的权重（负向） | 无量纲 | 0.3 |
| 酒店容量 | `HOTEL_CAPACITY` | 酒店可接待量（间/夜） | 间/夜 | 8,000 |
| 交通承载力 | `TRANSPORTATION_CAPACITY` | 交通日最大承载人数（人/日） | 人/日 | 25,000 |
| 停车容量 | `PARKING_CAPACITY` | 停车位数量 | 车位 | 12,000 |
| 声誉基线 | `REPUTATION_BASELINE` | 初始目的地声誉值 | 0–1 | 0.8 |
| 声誉改善率 | `REPUTATION_IMPROVEMENT_RATE` | 基于满意度声誉每年提升率 | 年比率 | 0.1 |
| 声誉衰减率 | `REPUTATION_DECAY_RATE` | 自然衰减的声誉损失 | 年比率 | 0.05 |
| 声誉影响权重 | `REPUTATION_IMPACT_WEIGHT` | 声誉对吸引力的权重 | 无量纲 | 0.2 |
| 折现率 | `DISCOUNT_RATE` | NPV 折现率 | 年比率 | 0.05 |
| 人均基准消费 | `BASE_PER_CAPITA_SPENDING` | 基线人均消费（由基线收入/游客计算） | 美元/人 | ~234 |
| 通胀率 | `INFLATION_RATE` | 通胀假设 | 年比率 | 0.025 |
| 本地采购比例 | `LOCAL_PURCHASE_RATIO` | 收入中本地采购占比 | 比例 | 0.65 |
| 税率 | `TAX_RATE` | 对旅游收入征收的税率 | 比例 | 0.08 |
| 就业乘数 | `EMPLOYMENT_MULTIPLIER` | 旅游带动的就业乘数 | 无量纲 | 1.8 |
| 工资溢价 | `WAGE_PREMIUM` | 旅游业工资相对溢价 | 比例 | 1.15 |
| 经济外溢率 | `ECONOMIC_SPILLOVER_RATE` | 本地支出到外溢的比例 | 比例 | 0.25 |
| 满意度基线 | `SATISFACTION_BASELINE` | 居民满意度初始值（尺度化） | 0–1 | 1.0 |
| 游客影响满意度系数 | `SATISFACTION_TOURIST_IMPACT` | 游客对满意度的负面影响系数 | 无量纲 | 0.5 |
| 社区福利分享 | `COMMUNITY_BENEFIT_SHARE` | 税收用于社区福利的比例 | 比例 | 0.15 |
| 服务质量影响 | `SERVICE_QUALITY_IMPACT` | 服务质量对满意度的影响权重 | 无量纲 | 0.3 |
| 自然灾害概率 | `NATURAL_DISASTER_PROBABILITY` | 年度自然灾害发生概率 | 比例 | 0.05 |
| 经济冲击概率 | `ECONOMIC_SHOCK_PROBABILITY` | 年度经济冲击概率 | 比例 | 0.1 |
| 政策不确定性因子 | `POLICY_UNCERTAINTY_FACTOR` | 政策执行不确定性 | 比例 | 0.15 |
| 全球经济影响 | `GLOBAL_ECONOMIC_IMPACT` | 全球经济波动对需求的影响强度 | 比例 | 0.2 |
| 学习率 | `LEARNING_BY_DOING_RATE` | 经验学习带来的吸引力改善率 | 年比率 | 0.08 |
| 网络效应强度 | `NETWORK_EFFECT_STRENGTH` | 口碑传播的强度 | 年比率 | 0.12 |
| 品牌忠诚积累 | `BRAND_LOYALTY_BUILDUP` | 品牌忠诚度积累速率 | 年比率 | 0.15 |
| 满意度记忆效应 | `SATISFACTION_MEMORY_EFFECT` | 满意度的时间记忆保留率 | 无量纲 | 0.7 |
| 决策变量范围 | `DECISION_VARIABLE_RANGES` | daily_cap / entrance_fee / eco_investment_ratio 的合法区间 | — | see config |
| 价格平滑系数 | `PRICE_SMOOTHING_ALPHA` | 当年票价与上一年票价平滑权重 | 无量纲 | 0.6 |
| 财务止损阈值 | `FINANCIAL_STOPLOSS_THRESHOLD` | 年净收入低于该值触发止损 | 美元 | 50,000,000 |
| 财务止损惩罚 | `FINANCIAL_STOPLOSS_PENALTY` | 止损触发后净收入乘数 | 比例 | 0.7 |
| 税收滞后年数 | `TAX_LAG_YEARS` | 分配到社区福利的税收滞后年数 | 年 | 1 |
| NSGA-II 种群大小 | `NSGA_II_POPULATION_SIZE` | 优化算法种群大小 | 个体 | 100 |
| NSGA-II 代数 | `NSGA_II_GENERATIONS` | 优化代数 | 代 | 500 |
| 随机种子 | `NSGA_II_SEED` | 随机数种子（复现） | — | 42 |
| 图像尺寸 | `FIGURE_SIZE` | 绘图尺寸 | 英寸 | (12,8) |
| 图像 DPI | `DPI` | 输出分辨率 | DPI | 300 |
| 结果目录 | `RESULTS_DIR` | 保存输出图表与 CSV 的目录 | 路径 | 'results' |
| Pareto 文件 | `PARETO_SOLUTIONS_FILE` | Pareto 解 CSV 路径 | 路径 | results/pareto_solutions.csv |

以上为主要参数和模型变量的摘要表；若需我把此表导出为 CSV（`parameters_table.csv`）并包含每个参数的敏感性等级与文献来源列，请回复“导出参数 CSV”，我将生成并提交到项目目录。

---  
7) 公式对应代码实现位置（文件与近似行号）

下面列出本文件中主要公式在项目代码中的实现位置（基于当前代码版本，行号为文件中的近似位置，可能随后续编辑略有偏移）。建议在论文附录中引用这些位置以便审阅者复现。

- 连续/离散动力学框架（仿真主循环与离散化实现）  
  - 文件：`src/system_dynamics.py`，函数：`simulate_policy`，近似行号：L238–L386（仿真循环与状态更新实现，使用显式欧拉风格的逐年更新）。  

- 季节性与基础设施容量（用于吸引力与需求的中间计算）  
  - `calculate_seasonal_factor(year, month)` — 文件：`src/system_dynamics.py`，近似行号：L14–L31。  
  - `calculate_infrastructure_capacity(daily_visitors)` — 文件：`src/system_dynamics.py`，近似行号：L34–L56。  

- 目的地吸引力 $A(t)$（合成吸引力评分）  
  - 实现：`calculate_tourist_attractiveness(glacier_size, daily_visitors, reputation, year)` — 文件：`src/system_dynamics.py`，近似行号：L59–L101。该函数将冰川、鲸鱼观测、拥挤、声誉、基础设施容量、学习效应与网络效应组合成 [0,1] 的吸引力评分。  

- 游客需求函数 $T(t)$（价格弹性、声誉、经济冲击）  
  - 实现：`calculate_tourist_demand(attractiveness, entrance_fee, reputation, year)` — 文件：`src/system_dynamics.py`，近似行号：L104–L140。该实现包含通胀调整、价格弹性、随机经济冲击与周期性全球经济影响。  

- 年度收入 $R(t)$ 与人均消费调整  
  - 实现：`calculate_revenue(annual_tourists, daily_cap, entrance_fee)` — 文件：`src/system_dynamics.py`，近似行号：L143–L169。该函数计算门票收入与基于拥挤调整的人均消费支出。NPV 的计算在 `simulate_policy` 中完成（见下）。  
  - NPV 计算：在 `simulate_policy` 中累计折现（`DISCOUNT_RATE`），近似行号：L357–L361。  

- 冰川退缩速率模型 $\frac{dG}{dt}$（自然退缩 + 游客影响 - 投资缓解 + 污染）  
  - 实现：`calculate_glacier_retreat(glacier_size, annual_tourists, eco_investment)` — 文件：`src/system_dynamics.py`，近似行号：L172–L195。污染与退缩耦合在 `simulate_policy` 内完成（污染更新见 L327–L331，退缩合并见 L313–L319）。  

- 居民满意度 $S(t)$（基线 + 经济溢出 - 拥挤 + 环保投资 + 服务质量）  
  - 实现：`calculate_resident_satisfaction(annual_tourists, eco_investment, community_benefits, year)` — 文件：`src/system_dynamics.py`，近似行号：L198–L235。该函数使用 `SATISFACTION_BASELINE`、`LOCAL_PURCHASE_RATIO`、`EMPLOYMENT_MULTIPLIER` 等参数计算标准化满意度并截断在 [0,1]。  

- 价格平滑、税收滞后与财务止损（仿真中的政策机制）  
  - 价格平滑（`PRICE_SMOOTHING_ALPHA`）：在 `simulate_policy` 中进行，近似行号：L272–L276（平滑计算）。  
  - 税收滞后（`TAX_LAG_YEARS`）：税收历史与滞后分配在 `simulate_policy` 中实现，近似行号：L268–L339（税收记录与社区分配逻辑）。  
  - 财务止损（`FINANCIAL_STOPLOSS_THRESHOLD`、`FINANCIAL_STOPLOSS_PENALTY`）：在 `simulate_policy` 中对净收入进行判断并调整，近似行号：L297–L303。  

- 优化器与目标函数（NSGA-II）  
  - 问题定义与目标封装：`src/optimization.py`，函数：`tourism_optimization_problem()`，近似行号：L20–L64（目标函数 `objective_function` 在 L27–L45）。  
  - NSGA-II 执行与 Pareto 抽取：`src/optimization.py`，函数：`run_nsga_ii_optimization()`，近似行号：L67–L130（算法运行、代数循环、非支配解抽取与结果收集）。  

- 参数定义与全局常量（配置）  
  - 文件：`src/config.py`，近似行号：L6–L118（包含仿真年份、人口、冰川参数、吸引力权重、容量参数、经济参数、优化参数与结果路径等）。  

- 可视化与后处理（图表生成）  
  - 文件：`src/visualization.py`（绘图函数实现，如 `plot_pareto_front_3d`、`plot_time_series`、`plot_policy_comparison` 等），近似行号块视具体函数而定（请在需要时让我为某个图定位精确行号）。  

注：行号为当前代码快照的近似位置；若你计划提交论文或代码审核，我可以把每个函数对应的更精确行号和相关代码片段摘录到附录中并生成一个映射表（包括函数签名和关键实现行），以便审阅者直接定位实现。是否需要我现在生成此更精确的映射表（包括代码片段引用）？  


