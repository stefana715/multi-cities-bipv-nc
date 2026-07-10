# Phase-3 D5 口径统一 — 决策存档

**归档日期**: 2026-07-10
**本次统一 commit**: `e86d653`
**适用对象**: 15城 Paper4（Paper4_FDSI_Manuscript）
**目的**: 论文修订时随时调取，记录 D5 口径的最终定义、演进溯源、修正的数据问题、排名变动与写作口径。

---

## 1. 最终 D5 口径（Phase-3）

D5「部署确定性 / 不确定性」维度，采用 3 个子指标，min-max 归一化（成本型：越低越确定越好）后按内部权重线性组合：

| 子指标 | 列名 | 方向 | 内部权重 |
|---|---|---|---|
| PBT 95% CI 宽度 | `d5_2_pbt_ci95_width` | 成本 (↓好) | **0.35** |
| σLCOE（LCOE 标准差） | `mc_lcoe_std` | 成本 (↓好) | **0.30** |
| **电价敏感性（Sobol S₁）** | `sobol_pbt_S1_elec_price_factor` | 成本 (↓好) | **0.35** |

- 第三子指标**由 `interaction_ratio`（`d5_4_interaction_ratio_pbt`）修订而来**，改为 PBT 对电价因子的一阶 Sobol 敏感性 S₁。
- **Rationale**：引入**独立的市场风险维度**——交互比衡量的是模型内部参数耦合，而电价敏感性刻画城市对外部电价波动的暴露度，是一个此前 D5 未覆盖的、与其他子指标不冗余的风险来源。
- 维度间赋权：熵权 + AHP 线性组合，α=0.5（`w_combined = 0.5·w_entropy + 0.5·w_ahp`，归一化）。D5 组合权重 ≈ 0.26。
- 适宜性等级（发表口径）：固定阈值 **High ≥ 0.70 / Medium ≥ 0.50 / Low < 0.50**（注意：与 `05_fdsi_scoring.py::build_suitability_matrix` 的三分位逻辑不同，Paper4 表沿用固定阈值）。

---

## 2. 演进溯源

D5 口径经过三代，改动仅发生在**聚合层**（`scripts/05_fdsi_scoring.py`），**数据生成层**（`scripts/04_energy_simulation.py`）全程沿用同一套子指标计算（MC N=10,000 LHS；Sobol Saltelli N=4,096）：

| 代 | 承载运行 | commit | D5 = pbt_ci95 / σLCOE / 第三项 | 权重 |
|---|---|---|---|---|
| Phase 1 | 旧版 5 城（母仓库） | — | …/…/interaction_ratio | 35/30/35 |
| **Phase 2** | **15 城稿件基线** | `51dc9b0`（fork） | pbt_ci95 / σLCOE / **interaction_ratio** | **40/35/25** |
| **Phase 3** | **本稿（统一后）** | `ab0539f` 引入口径 | pbt_ci95 / σLCOE / **elec_price S₁** | **35/30/35** |

- **改动点**：`40/35/25 + interaction_ratio` → `35/30/35 + elec_price 敏感性`，发生在 **`ab0539f`（"expand to 39-city" 扩展 commit，2026-04-09）**。
- **时序关系**：`51dc9b0`（15城 fork）是 `ab0539f` 的**祖先**（`git merge-base --is-ancestor 51dc9b0 ab0539f` = YES）。即口径改动在 **fork 之后**引入，随 39城扩展一并落地，并非 fork 时带入。
- 数据生成层未变：`04` 始终计算全部 D5 候选子指标（含 interaction_ratio 与 elec_price S₁）；本次仅在 `05` 重构了**哪三个子指标进入 D5 聚合及其权重**。

---

## 3. 修正的两个 Phase-2 数据问题

在把本稿口径统一到 Phase-3 时，发现 Phase-2 的 15城子指标数据存在两处缺陷，均已修正（仅重跑 `04` 的 **Sobol 步**，未重跑 MC/数据生成层）：

**问题 A — 15城 Sobol 块整体为占位常数。**
Phase-2 的 `table_all_cities_indicators.csv` 中，几乎所有 Sobol 列在 15 城间完全相同（如 `sobol_yield_S1_ghi_factor` = 0.6473 全城一致、`sobol_pbt_S1_ghi_factor` = 0.2305 全城一致），说明该块是单一模板城市的值被复制，非逐城真实计算。
→ **已用真实逐城重跑替换**：对 15 城逐一调用 `04::run_sobol_analysis`（N=4,096，输入 ghi_annual=D1 GHI、elec_price=CITIES 表、mc_params=城市 yaml），写入真实完整 Sobol 块（27 列）。重跑结果与 41城 run 的同城值一致（互证正确）。

**问题 B — 电价敏感性列退化为常数、在打分中失效。**
第三子指标 `sobol_pbt_S1_elec_price_factor` 在 Phase-2 15城数据中为常数 **0.1011**（`nunique=1, std=0`）。经 `05::normalize_indicators` 时命中 `vmax==vmin` 分支，被归一化为**常数 0.5**，对所有城市贡献相同、**完全不参与区分**——即 D5 中 35% 的权重被钉死在一个死值上。
→ **已修**：重跑后该列真实变化（`nunique=10, 范围 0.0571–0.2029`），电价敏感性恢复为有效判别指标。

> 说明：`table_d5_sub_indicators.csv` 保留 `d5_4_interaction_ratio_pbt` 列供**演进对照**，且该列已同步为**真实逐城值**（与 `table_all_cities_indicators.csv` 一致），非 Phase-2 占位假值。

---

## 4. 15城排名变动全表（Phase-2 → Phase-3）

仅 D5 聚合口径改变，D1–D4 维度分不变。适宜性用固定阈值（High≥0.70/Medium≥0.50）。

| 新排名 | 城市 | 气候区 | 旧 FDSI | 新 FDSI | 旧 rank | 新 rank | 旧 suit | 新 suit | Δrank |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Xiamen 厦门 | HSWW | 0.8913 | 0.8237 | 1 | 1 | High | High | 0 |
| 2 | Beijing 北京 | Cold | 0.7626 | 0.7678 | 3 | 2 | High | High | +1 |
| 3 | Jinan 济南 | Cold | 0.7989 | 0.7258 | 2 | 3 | High | High | −1 |
| 4 | Shenyang 沈阳 | Severe Cold | 0.7150 | 0.7171 | 4 | 4 | High | High | 0 |
| 5 | Xian 西安 | Cold | 0.6800 | 0.7050 | 9 | 5 | Medium | **High** | +4 |
| 6 | Kunming 昆明 | Mild | 0.6917 | 0.7005 | 7 | 6 | Medium | **High** | +1 |
| 7 | Wuhan 武汉 | HSCW | 0.6805 | 0.6970 | 8 | 7 | Medium | Medium | +1 |
| 8 | Nanjing 南京 | HSCW | 0.6799 | 0.6392 | 10 | 8 | Medium | Medium | +2 |
| 9 | Harbin 哈尔滨 | Severe Cold | 0.6144 | 0.6365 | 11 | 9 | Medium | Medium | +2 |
| 10 | **Guangzhou 广州** | HSWW | 0.6949 | 0.6316 | 5 | 10 | Medium | Medium | **−5** |
| 11 | Changsha 长沙 | HSCW | 0.6080 | 0.6301 | 12 | 11 | Medium | Medium | +1 |
| 12 | **Shenzhen 深圳** | HSWW | 0.6939 | 0.6161 | 6 | 12 | Medium | Medium | **−6** |
| 13 | Changchun 长春 | Severe Cold | 0.5654 | 0.5750 | 13 | 13 | Medium | Medium | 0 |
| 14 | Chengdu 成都 | Mild | 0.3856 | 0.4492 | 14 | 14 | Low | Low | 0 |
| 15 | Guiyang 贵阳 | Mild | 0.2371 | 0.3238 | 15 | 15 | Low | Low | 0 |

**统计**：10/15 城排名变动；max |Δrank| = 6；**Spearman ρ = 0.839**。High 层由 4 城扩至 6 城（新增西安、昆明）；Top-1 与 Bottom-3 稳定；变动集中在中游（rank 5–12）。

---

## 5. 深圳定位与写作口径

- **排名变动**：深圳 6 → 12（High → Medium）。**机制（真实信号）**：深圳与广州电价最高（0.68 CNY/kWh）→ PBT 对电价最敏感（Sobol S₁ 深圳 0.1853、广州 0.1678，居全样本前列）→ 新第三子指标（成本型）拉低其 D5。
- **深圳仍为核心案例城市**：理由为**数据成熟度 + 研究传承（Paper 1 承接）**，与其 FDSI 排名**无关**。深圳的 OSM 数据完整性（DCS=0.615，全样本最高之一）与既有研究基础使其继续作为方法学演示与深度案例的主城。
- **写作口径**：深圳下滑应表述为**方法学发现**，而非"深圳不适合 BIPV"。核心命题——**高技术潜力 ≠ 低部署风险**：深圳在 D1 气候（0.87）、D4 经济（0.89）上极强，但其对市场电价的高敏感性构成独立的部署风险，被 D5 如实捕捉，故综合 FDSI 落入中游。这正是 FDSI 相对 GHI-only / 单维评价的增量价值。
- **禁止表述**：不得再称深圳为"HSWW 高适宜代表 / 位列前六 / High 适宜"。相关正文/图注/表格清单见 `outputs/audit/shenzhen_guangzhou_review.md`。

---

## 6. 冻结基准与关键 commit

| 事项 | commit |
|---|---|
| 15城 Phase-2 稿件冻结基准 | **`51dc9b0`**（`init: fork from multi-cities-bipv with 15-city results`） |
| D5 口径改动引入点（40/35/25 → 35/30/35） | **`ab0539f`**（`feat: expand to 39-city`） |
| 本次 15城口径统一（4表7图+审计） | **`e86d653`** |
| 前置盘点/溯源（Task A） | `7c87f96` |

- 回滚：本次所有改动的表与图均有 `*.bak_phase2` 备份（未纳入版本库）。
- 复现：`04::run_sobol_analysis`（Saltelli 确定性、可复现）+ `05::select_dimension_indicators/normalize_indicators/compute_dimension_scores/entropy_weight/ahp_weight_d1d5/compute_fdsi`。

---

## 7. 与 41城稿的关系（禁止混引）

- **15城 与 41城是两次独立全量运行，非"筛选子集"**：同城 FDSI 因样本内 min-max 归一化 + 熵权随样本变化而不同（非分数保持的过滤）。时序上 **15城在先（fork 基线 `51dc9b0`）、41城在后（15→39→41 扩展重跑）**。
- **口径不同**：41城稿（NC，`results/fdsi/`）中深圳为 **#14**；本 15城稿深圳为 **#12**。两稿 D5 口径虽同属 Phase-3 家族，但归一化样本不同、排名不可通约。
- **硬约束**：`results/fdsi/`（41城）与所有 `nc_*.md` 正文在本次统一中**保持未改动**；论文写作中**禁止**将 15城排名与 41城排名交叉引用（如"深圳第12"与"深圳第14"混用）。

---

*本存档随 commit 归入版本库，作为 Paper4 D5 口径的权威决策记录。*
