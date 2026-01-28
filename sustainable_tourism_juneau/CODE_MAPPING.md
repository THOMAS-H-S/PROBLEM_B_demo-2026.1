## CODE_MAPPING 初稿
本文件为 `MODEL_EQUATIONS.md` 中公式与代码实现的初步映射，包含函数名、文件位置与关键实现片段引用，便于审稿人定位实现。

1) 游客吸引力与需求
- 公式：$A(t)$ 与 $T(t)$  
- 实现：`src/system_dynamics.py`  
  - 函数：`calculate_tourist_attractiveness` （近似 L59–L101）  
  - 函数：`calculate_tourist_demand` （近似 L104–L140）  
  - 关键变量：`ATTRACTION_GLACIER_WEIGHT`、`WHALE_WATCHING_SUCCESS_RATE`、`CROWDING_THRESHOLD`、`INFLATION_RATE`

2) 收入与 NPV
- 公式：$R_t$, $NPV$  
- 实现：`src/system_dynamics.py`  
  - 函数：`calculate_revenue`（近似 L143–L169）  
  - NPV 累积位于 `simulate_policy`（近似 L357–L361）  

3) 冰川退缩
- 公式：$r_t$ 与 $G_{t+1}$  
- 实现：`src/system_dynamics.py`  
  - 函数：`calculate_glacier_retreat`（近似 L172–L195）  
  - 更新合并在 `simulate_policy` 中（近似 L309–L319）  

4) 居民满意度
- 公式：$S_t$  
- 实现：`src/system_dynamics.py`  
  - 函数：`calculate_resident_satisfaction`（近似 L198–L235）

5) 优化器（NSGA-II）
- 实现：`src/optimization.py`  
  - Problem 定义：`tourism_optimization_problem()`（近似 L20–L64）  
  - 运行与提取：`run_nsga_ii_optimization()`（近似 L67–L130）

6) 配置与参数
- 文件：`src/config.py`（近似 L6–L118）

注：以上行号为当前代码快照的近似位置。若需要，我可以把每个映射条目的代码片段（5–15 行）提取并嵌入此文件的对应附录节中，或者导出为单独的 `CODE_SNIPPETS.md` 文件。请确认是否继续自动提取并写入代码片段（回复“提取代码片段”）。


