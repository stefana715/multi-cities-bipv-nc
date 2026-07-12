# 模型口径对照表 (Metric Reconciliation Across Model Calibrations)

**生成日期**: 2026-07-13
**目的**: 博士论文整合多篇论文结果，同城同指标在不同模型口径下数值不同（均对，模型边界不同）。本表逐值可溯源，防止答辩/审稿被抓"矛盾"。
**有效口径基准**: Phase-3（FDSI 冻结 commit `e86d653`）+ 核验①真值（comparison commit `855b287`）。
**已排除**（不进有效区）: `archive/` 全部；`results/paper4_summary/`（15 城旧样本）；任何标 RETIRED/ARCHIVED/〔核验〕的值。

---

## 0. 口径定义（模型链元数据）

| 口径 ID | 模型链类型 | 含储能 | 电价/电费 | 成本假设 | MC 样本数 | 所属 commit / 路径 |
|---|---|---|---|---|---|---|
| **A：FDSI 简化概率链** | 确定性 + Monte-Carlo (LHS) 不确定性；纯屋顶 PV 发电，无优化 | ❌ 无（纯屋顶 PV） | 逐城上网电价（深圳 0.680 / 长沙 0.588 CNY/kWh，= 年收益÷比发电量） | capex **3.0 CNY/Wp**；模块效率 20%；衰减 0.5%/yr；寿命 25 yr；折现 6%；O&M 1%/yr | **10,000** LHS | `multi-cities-bipv-nc/results/fdsi/integrated_indicators.csv`（Phase-3 口径，冻结 `e86d653`；数据文件 @`5f014fb`，经济列不受 D5 影响） |
| **B-ma：Paper 4 光储 NSGA-II（形态感知）** | 多目标 NSGA-II（LCOE ↔ 自消纳率 SCR 权衡）；PV+电池，含 SCR 约束；morphology_aware | ✅ 有（PV+电池） | 零售电价 **0.98**（深圳）/ **0.88**（长沙）CNY/kWh | capex ≈ 6.5 CNY/Wp（含电池；由 总投资÷容量 反推，深圳 4497 MCNY/688 MWp） | 风险 MC **1,000** / 验证 MC 10,000；NSGA-II pop 200 × gen 150 | `comparison/results/paper4/`（HEAD `855b287`）：`sci_tables/P4_Table2,5`、`tables/table61,63,64` |
| **B-oa：Paper 4 光储 NSGA-II（统一方案）** | 同上，one_size（不区分形态） | ✅ 有 | 同 B-ma | 同 B-ma | 同 B-ma | 同上（`oa_*` 列 / `one_size` 行） |

> **IRR**：有效口径（A / B）**均未计算 IRR**；IRR 仅存在于已排除的 15 城 Phase-2 `results/paper4_summary/table_npv_irr_co2.csv` → 本表 IRR 一律标 N/A，不可引用旧值。

---

## 1. 主对照表 — 深圳 (Shenzhen)

| 城市/片区 | 口径 | LCOE (CNY/kWh) | NPV | PBT (yr) | 年发电量 (GWh/yr) | 比发电量 (kWh/kWp) | 数据源（可溯源） |
|---|---|---|---|---|---|---|---|
| 深圳（市级） | **A** FDSI | **0.2142** (det) / 0.2362±0.0299 (MC) | 7358 CNY/**kWp** | **3.52** | — | **1298.9** | `results/fdsi/integrated_indicators.csv` 行 shenzhen |
| 深圳/南山 | **B-ma** | 0.7466 (knee) / 0.4374 (Pareto min) | 3961 CNY/**户** | 7.75 | 158.5 | — | `P4_Table2` + `table63` nanshan |
| 深圳/福田 | **B-ma** | 0.7325 (knee) / 0.449 (min) | 3961 CNY/户 | 7.69 | 111.7 | — | `P4_Table2`+`table63` futian |
| 深圳/龙华 | **B-ma** | 0.6579 (knee) / 0.4436 (min) | 4633 CNY/户 | 7.11 | 227.8 | — | `P4_Table2`+`table63` longhua |
| 深圳/宝安 | **B-ma** | 0.7132 (knee) / 0.4346 (min) | 4281 CNY/户 | 7.49 | 239.2 | — | `P4_Table2`+`table63` baoan |
| 深圳（市级合成） | **B-ma / B-oa** | 0.7061 / 0.7514 (composite) | — | 7.46 / 7.79 | 737.2 | — | `P4_Table5` shenzhen（储能 777.9/909.7 MWh） |

## 2. 主对照表 — 长沙 (Changsha)

| 城市/片区 | 口径 | LCOE (CNY/kWh) | NPV | PBT (yr) | 年发电量 (GWh/yr) | 比发电量 (kWh/kWp) | 数据源 |
|---|---|---|---|---|---|---|---|
| 长沙（市级） | **A** FDSI | **0.2394** (det) / 0.2656±0.0367 (MC) | 4926 CNY/**kWp** | **4.59** | — | **1162.1** | `results/fdsi/integrated_indicators.csv` 行 changsha |
| 长沙/岳麓 | **B-ma** | 0.5015 (knee) / 0.4836 (min) | 3796 CNY/户 | 7.23 | 114.9 | — | `P4_Table2`+`table64` yuelu |
| 长沙/天心 | **B-ma** | 0.5813 (knee) / 0.4911 (min) | 3148 CNY/户 | 7.98 | 79.2 | — | `P4_Table2`+`table64` tianxin |
| 长沙/开福 | **B-ma** | 0.5001 (knee) / 0.4963 (min) | 3686 CNY/户 | 7.29 | 95.8 | — | `P4_Table2`+`table64` kaifu |
| 长沙（市级合成） | **B-ma / B-oa** | 0.5229 / 0.8363 (composite) | — | 7.46 / 12.69 | 289.8 | — | `P4_Table5` changsha（储能 37.9/398.1 MWh） |

## 3. FDSI-only 参考区（41 城 Phase-3，仅口径 A，无跨模型对照）

| 城市 | LCOE det / MC | PBT (yr) | NPV CNY/kWp | 比发电量 kWh/kWp | 电价 |
|---|---|---|---|---|---|
| 北京 | 0.2049 / 0.2288±0.0279 | 4.74 | 4680 | 1357.9 | ~0.488 |
| 广州 | 0.2223 / 0.2457±0.0325 | 3.65 | 6968 | 1251.8 | ~0.680 |
| 厦门 | 0.2009 / 0.2208±0.0291 | 3.63 | 7026 | 1385.1 | ~0.618 |
| … | 其余 36 城见 `results/fdsi/integrated_indicators.csv`（41 城全量，口径 A） | | | | |

---

## (a) 同城同指标差异 > 20% 条目及成因

| # | 城市 | 指标 | 口径 A (FDSI) | 口径 B (Paper 4 光储) | 差异 | 成因（一句话） |
|---|---|---|---|---|---|---|
| 1 | 深圳 | **LCOE** | 0.2142 | 0.6579–0.7466 (knee) | **+207%…+248%** | B 含**电池储能**（capex ≈6.5 vs 3.0 CNY/Wp）+ SCR 约束，A 为无储能纯屋顶 PV |
| 2 | 深圳 | **PBT** | 3.52 | 7.11–7.75 | **+102%…+120%** | 储能 capex 拉长回收期；A 无储能且电价口径不同 |
| 3 | 长沙 | **LCOE** | 0.2394 | 0.5001–0.5813 (knee) | **+109%…+143%** | 同 #1（储能 + SCR + NSGA-II 目标权衡） |
| 4 | 长沙 | **PBT** | 4.59 | 7.23–7.98 | **+57%…+74%** | 同 #2 |

> **不可直接比（单位/边界不同，非"矛盾"，须注口径）**：
> - **NPV**：A = CNY/**kWp**（7358 深圳）；B = CNY/**户**（3961–4633）。单位不同，禁止并列比较。
> - **发电量**：A = 比发电量 kWh/**kWp**（1298.9）；B = 片区**总**发电量 GWh/yr。量纲不同。
> - **电价**：A 用上网电价（深圳 0.68）；B 用零售电价（0.98）。LCOE 不受电价影响（成本侧），但 NPV/PBT/收益受其驱动 → 成因 #2/#4 部分来自此。

---

## (b) 旧草稿值残留位置清单（待清除）

核验①（`855b287`）只更新了 `table11` + `P3_Table3` 并归档脚本，**下列 pre-855b287 生成物仍残留退休值**，须清除或重生成：

| # | 残留值 | 位置 | 说明 |
|---|---|---|---|
| 1 | L3 ρ=**0.9231** | `comparison/results/paper3/tables/table10_layer3_rank_correlation.csv:15` | 旧循环 L3（"Paper 2" 12 聚类），已被真值 ρ=0.957/0.792 取代 |
| 2 | L1 合成 60 站指标（nRMSE≈**5.4%**=0.0536 等） | `comparison/results/paper3/tables/table7_layer1_station_metrics.csv`（61 行） | 旧合成自证 L1，已被真实 HKUST +21.4% 取代 |
| 3 | L1 合成月度对比 | `comparison/results/paper3/tables/table8_layer1_monthly_comparison.csv` | 同上 |
| 4 | L2 Rhino/Paper-1 SY（mean\|bias\|=**+3.5%**=3.53） | `comparison/results/paper3/tables/table9_layer2_cross_method.csv` | 旧循环 L2，已被 SCS-2023 +23.88~+45.88% 取代 |
| 5 | L1/L2/L3 旧图（值烧录在 PNG） | `comparison/results/paper3/figures/fig21–fig28*.png`（8 张） | fig21-24(L1合成)/fig25-27(L2 Rhino)/fig28(L3 12聚类) |
| 6 | 草稿"深圳 Paper 4 LCOE **0.482–0.512**" | （文献/正文草稿，非结果文件） | **确认为误记**（疑为**长沙**值）：深圳实际 Pareto min 0.4346–0.449、**knee 0.6579–0.7466**（= 有效值）；0.48–0.51 与长沙 knee（0.50–0.58）吻合 → 深圳勿用 0.48–0.51 |

> **状态**：残留 #1–5（table7-10 + fig21-28）已于 comparison commit `12548d1` 移入 `archive/paper3_validation_retired/`（不重生成，真值在 `outputs/verification/`）。
>
> **⚠ 残留 #6 属写作侧**：错数"深圳 0.482–0.512"未见于结果文件，但**可能残留于大论文/Paper 4 稿正文或图注中** → **写作侧待全文搜查更正**（深圳 Paper 4 LCOE 有效值 = knee **0.66–0.75**；0.48–0.51 若出现在"深圳"语境即为长沙值错置）。

---

## (c) 15 城 Phase-2 ↔ 41 城 Phase-3 同名指标口径警示（严禁交叉引用）

同一城市的 **FDSI 综合分**在 15 城样本与 41 城样本下**不同**（样本内 min-max 归一化 + 熵权随样本变化），**非矛盾，但绝不可交叉引用**：

| 城市 | 15 城样本（`results/paper4_summary/`，**排除**） | 41 城样本（`results/fdsi/`，**有效**） | 差 |
|---|---|---|---|
| 北京 FDSI | **0.7678**（rank 2/15） | **0.6279**（rank 8/41） | −0.140 |
| 深圳 FDSI | 0.6161（rank 12/15） | 0.5737（rank 14/41） | −0.042 |

- ⚠️ **只引用 41 城 Phase-3（`results/fdsi/`）作为 FDSI 综合分的有效值**；15 城样本值仅用于 Paper4_FDSI_Manuscript 内部，不得与 41 城混引（详见 `outputs/audit/41cities_provenance.md` §c）。
- 注：用户示例引 "北京 0.7626" 为 **Task B 前的 Phase-2 旧值**；经 Phase-3 D5 统一（`e86d653`）后 15 城样本现为 **0.7678**。两者均属"15 城样本"，均不可与 41 城 0.6279 交叉引用。
- 经济指标（LCOE/PBT/NPV/发电量）**不受 D5/样本影响**，41 城 `results/fdsi/` 即有效值，无此警示。

---

## 溯源与范围声明

- 每个数值均标注来源文件/行；口径 A 数据 = `multi-cities-bipv-nc/results/fdsi/`（Phase-3，`e86d653` 冻结）；口径 B 数据 = `comparison/results/paper4/`（`855b287`）。
- 验证类指标（L1 +21.4% / L2 +23.88~+45.88% per-m² / L3 ρ=0.957&0.792）唯一有效来源 = `comparison/outputs/verification/`（`855b287`），详见其 `validation_summary.md`（KI-1…6）。
- 本表不含任何 archive/ 或 RETIRED/〔核验〕值作为有效值。
