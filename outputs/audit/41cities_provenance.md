# 41 城溯源报告 (41-City Provenance)

**盘点日期**: 2026-07-10
**结论速览**:
- 41 城运行 = **新版管线（Phase 3 口径）**，不是旧版；且比 15 城稿件口径**又演进了一步**。
- 41 城与 15 城 **不是 "41 审计后经 DCS≥0.40 筛得 15"** 的关系。真实关系与你记忆**相反**：
  **15 城在先、41 城在后**——15 城是从母仓库 `multi-cities-bipv` fork 进来的稿件基线，
  41 城是本仓库内做的 15→39→41 扩展全量重跑。二者是**两次独立全量运行、口径不同**。
- DCS≥0.40 是 **OSM 数据完整性入选门槛**（城市选择阶段用），**不是** 41→15 的筛选器。

---

## a. 41 城的运行记录在哪？

### 脚本
- 扩展执行脚本：`scripts/nc_03_morphology_new_cities.py`（docstring 明写 "批量形态分析 **24 个新城市**"，15+24=39）
- 核心管线（对全 41 城重跑）：`scripts/03_morphology_analysis.py`、`04_energy_simulation.py`、`05_fdsi_scoring.py`
- 新增两城配置：`configs/hongkong.yaml`、`configs/taipei.yaml`（39→41）

### 输出
- `results/fdsi/fdsi_scores.csv` = **41 行**（Lhasa #1=0.7422 … Chongqing #41=0.1752）
- `results/fdsi/integrated_indicators.csv`、`suitability_matrix.csv` = 41 城
- `results/energy/`、`results/morphology/` 逐城文件覆盖 41 城
- `results_nc/*` 全部 NC Phase 2 分析读取 `results/fdsi/`（41 城）——见 `nc_validate_numbers.py`、`nc_directional_bias.py` 的输入路径

### git log 时间点
| commit | 日期 | 事件 |
|---|---|---|
| `51dc9b0` | 2026-04-08 | **init: fork from multi-cities-bipv with 15-city results**（15 城基线进入本仓库） |
| `3f036e6` | 2026-04-08 | fix: shenyang bbox + 面积推断高度 |
| `ab0539f` | 2026-04-09 | **feat: expand to 39-city**（15→39 全量扩展；同时改写 D5 口径→Phase 3） |
| `5f014fb` | 2026-04-09 | **feat: add Hong Kong & Taipei (41-city expansion)** + NC Phase 2（39→41） |
| `66fbfb0` | 2026-04-10 | NC 论文图 Fig 1–4 + 政策成本分析 |
| `161cbba` | 2026-04-10 | fix: Urumqi shift −17；55/55 校验通过 |
| `21e813a` | 2026-04-11 | §3.5 方向性偏差分析（当前 HEAD） |

> **41 城的完整落地 = 提交 `5f014fb`（2026-04-09）**；其后 4 个提交只做分析、修数、出图，未改变 41 城 FDSI 基础评分。

---

## b. 41 城用旧版还是新版口径？（逐项核对）

**结论：新版（Phase 3）。** 与你给的"新版"定义逐项对照：

| 口径项 | 旧版(5城) | 你记的"新版"(15城) | **41 城实际(HEAD)** | 判定 |
|---|---|---|---|---|
| D2 指标数 | 4 项（密度/容积率/高度类型/KD-Tree遮挡） | 5 项（平均高度、密度、平均屋顶面积、紧凑度、容积率） | `{city}_d2_indicators.csv` 列 = `d2_1_height_mean, d2_2_building_density, d2_3_roof_area, d2_4_compactness, d2_5_far` = **5 项** | ✅ 新版 |
| 遮挡归属 | 在 D2（KD-Tree） | 移入 D3 | D3 列含 `d3_2_shading_factor_mean` = **遮挡在 D3** | ✅ 新版 |
| D5 子指标 | — | PBT CI95宽度 / σLCOE / Sobol交互比 | `d5_2_pbt_ci95_width` / `mc_lcoe_std` / **`sobol_pbt_S1_elec_price_factor`** | ⚠ 前二相同，**第三项被替换**（见下 §b.1） |
| D5 内部权重 | 35/30/35 | 40/35/25 | **35/30/35**（`05_fdsi_scoring.py` L263/269/275，注释 "Phase 3: 0.40→0.35 / 0.35→0.30 / 0.25→0.35"） | ⚠ **既非旧版也非15城稿件**，是 Phase 3 新值 |
| 维度赋权 | — | 熵权-AHP 线性组合 α=0.5 | `weight_comparison.csv` 有 w_entropy/w_ahp/w_combined；`05` 内 `entropy_weight()`+`ahp_weight_d1d5()` | ✅ 新版 |
| MC | — | N=10,000 LHS | `integrated_indicators.csv` 有 `mc_n_samples`；`04` 内 LHS N=10,000 | ✅ 新版 |
| Sobol | — | Saltelli 4,096 | `sobol_n_samples` 列 + `{city}_sobol_indices.csv` | ✅ 新版 |
| 辐照数据 | — | PVGIS v5.3 ERA5 | master summary 明写 "PVGIS v5.3 API … ERA5 … all 41 study cities" | ✅ 新版 |
| 形态数据 | — | OSM + 3米层高代理 | `03_morphology_analysis.py` 面积/层高推断 | ✅ 新版 |
| 经济参数 | — | 3000 CNY/kWp、O&M 30、25年、0.5%衰减 | `04_energy_simulation.py` / `{city}_d4_economics.csv` | ✅ 新版 |

**总判定：41 城 = 新版管线血统，无疑。** 但它不是 15 城稿件那一版新版，而是**在其基础上又改了 D5 的第三代（Phase 3）**。

### b.1 D5 三代演进（关键审计发现）

`05_fdsi_scoring.py` 的 git blame 明确记录 D5 内部权重与第三项指标经过三代：

| 代际 | 承载运行 | commit | D5 = pbt_ci95_width / σLCOE / 第三项 | 权重 |
|---|---|---|---|---|
| Phase 1 | 旧版 5 城 | (母仓库) | …/…/交互比 | **35/30/35** |
| **Phase 2** | **15 城稿件** | fork `51dc9b0` | pbt_ci95_width / mc_lcoe_std / **`d5_4_interaction_ratio_pbt`（Sobol交互比）** | **40/35/25** |
| **Phase 3** | **41 城 NC 全量** | `ab0539f` | pbt_ci95_width / mc_lcoe_std / **`sobol_pbt_S1_elec_price_factor`（Sobol PBT-电价敏感性 S1）** | **35/30/35** |

- fork 处（`git show 51dc9b0:scripts/05_fdsi_scoring.py`）D5 = 0.40/0.35/0.25 + 交互比，注释 "Phase 2 revision: 0.35→0.40 / 0.30→0.35 / 0.35→0.25" → **正是你记忆中的"新版 15 城 40/35/25 + Sobol交互比"**。
- `ab0539f`（39 城扩展同一提交）把第三项从 `interaction_ratio` **换成 `elec_price_sensitivity`**，权重改回 35/30/35，注释 "引入独立市场风险维度"。

> **给论文的提醒**：41 城 NC 结果的 D5 口径与 Paper4_FDSI_Manuscript(15 城)的 D5 口径**不完全一致**
> ——第三个 D5 子指标不同、内部权重不同。若两篇稿件要交叉引用彼此 FDSI 数值，需在方法学中明确声明这一代际差异，或就其中一版统一重跑。

---

## c. 41 与 15 的关系：两次独立全量运行，非 DCS 筛选

**你的假设**："41 城审计后经 DCS≥0.40 筛得 15 城"。**证据判定：不成立。**

### 证据 1 — 分数不一致 ⇒ 非子集切片
若 15 是 41 的过滤子集，同城 FDSI 应完全相同。实际：

| 城市 | 15 城 (`paper4_summary`) | 41 城 (`results/fdsi`) |
|---|---|---|
| Beijing | 0.7626 | 0.6279 |
| Xiamen | 0.8913 | 0.6261 |
| Shenzhen | 0.6939 | 0.5737 |
| Guiyang | 0.2371 | 0.3218 |

分数全不同 → 是**两次独立归一化的全量运行**（FDSI 用样本内 min-max + 熵权，样本从 15→41 会整体改变归一化区间与权重），**不是筛选**。

### 证据 2 — 时间顺序相反 ⇒ 15 在先、41 在后
- `results/paper4_summary/` 仅在 **`51dc9b0`（fork, 2026-04-08）** 引入，此后未改。
- fork 提交信息原文："**init: fork from multi-cities-bipv with 15-city results for NC expansion**" → 15 城是**从母仓库带进来的既有稿件结果**，作为 NC 扩展的**起点**。
- `git show 51dc9b0:results/fdsi/fdsi_scores.csv` = **正是 15 城**（Xiamen #1=0.8913），与 `paper4_summary/table_fdsi_ranking.csv` 逐行一致。
- 随后 `ab0539f` 把 `results/fdsi/fdsi_scores.csv` **覆盖为 39 城**、`5f014fb` 再覆盖为 **41 城**。

因此真实链条是：**5 城(母仓库Phase1) → 15 城(母仓库Phase2, 即稿件) → [fork 进本仓库 51dc9b0] → 39 城(ab0539f, Phase3) → 41 城(5f014fb)**。
"41 筛成 15" 的方向不存在。

### 证据 3 — DCS 打分记录的真实用途
`grep DCS` 命中处（`docs/city_selection_final.md`、`scripts/01_osm_audit.py`、`results/osm_audit/audit_results.csv`、若干 `configs/*.yaml`）显示：
- DCS = `0.5·coverage_proxy + 0.3·height_ratio + 0.2·residential_ratio`（`01_osm_audit.py` L295-296），门槛 0.40。
- 它是 **OSM 数据完整性 / 城市入选门槛**，用于**最初 5 城选择的 Layer-3 审计**（`docs/city_selection_final.md`，日期 2026-04-05，"五城市全部通过 DCS≥0.4"）。
- **没有任何** 41 城逐城 DCS 表，也没有 "41→15 按 DCS 过滤" 的脚本或记录。且 15 城稿件里 **Guiyang（DCS=0.376，FAIL）仍被保留**（排名第 15），进一步证明 DCS 不是 15 城名单的筛选器。

---

## d. 未过 DCS 门槛的城市清单（完整）

DCS 门槛只在**城市选择阶段**留有记录（`docs/city_selection_final.md` + `results/osm_audit/audit_results.csv`）。未过 0.40 门槛者：

| 城市 | 研究区 | DCS | 判定 | 处置 |
|---|---|---|---|---|
| Guiyang 贵阳 | 南明区 | **0.376** | ❌ FAIL | 仍进入 15 城稿件（bbox 扩展后使用，排名末位） |
| Kunming 昆明（盘龙区单区） | 盘龙区 | **0.383** | ❌ FAIL | 改用"主城四区 bbox"方案(DCS=0.521 PASS)后入选 |
| Shenyang 沈阳 | 沈河区 | — | ⚠ 查询失败(Nominatim 未地理编码) | 后经 `fix_shenyang_*` 修复 bbox 后纳入 |

其余审计城市均 PASS：Harbin 0.419、Beijing 0.517、Changsha 0.524、Shenzhen 0.615、Kunming(bbox) 0.521、Changchun 0.563、Jinan 0.535、Xian 0.455、Wuhan 0.465、Nanjing 0.482、Guangzhou 0.485、Xiamen 0.421、Chengdu 0.555（`audit_results.csv`）。

> **注**：39/41 城扩展阶段**未保留逐城 DCS 门槛记录**——扩展城市直接进入形态/能源管线，未见独立的 DCS 过滤日志。若论文需声明"所有 41 城均满足数据完整性门槛"，建议对新增 26 城**补跑 `01_osm_audit.py` 并归档 DCS 表**，否则该门槛主张目前只有原始审计的 ~15 城有据可查。

---

## 结论摘要

1. **41 城口径 = 新版（Phase 3）**：D2=5 指标、遮挡在 D3、熵权-AHP、MC 10k、Sobol 4096、PVGIS v5.3 ERA5 全部符合新版；但 **D5 第三项与内部权重已从 15 城稿件的 40/35/25(交互比) 改为 35/30/35(电价敏感性)**——这是需要在论文中显式声明的代际差异。
2. **41 vs 15 = 两次独立全量运行，口径不同**；**15 在先(fork 基线)、41 在后(扩展重跑)**；**非 DCS 筛选、非子集**。
3. **DCS≥0.40 = OSM 数据完整性入选门槛**（5 城选择阶段），未过门槛记录仅 Guiyang(0.376)、Kunming-盘龙(0.383)、Shenyang(查询失败)。
