# 仓库盘点 (Repository Inventory)

**仓库**: `multi-cities-bipv-nc`（"nc" = Nature Communications 投稿分支）
**盘点日期**: 2026-07-10
**HEAD commit**: `21e813a` (2026-04-11, "add directional bias analysis")
**管线代际**: 当前 HEAD 全部脚本为 **Phase 3 新版口径**（D2=5 项形态指标、遮挡移入 D3、
D5=PBT CI95宽度/σLCOE/Sobol、熵权-AHP、MC N=10,000 LHS、Sobol Saltelli 4,096、PVGIS v5.3 ERA5）。
详见 `41cities_provenance.md` 对口径的逐项核对与代际差异。

---

## A. 脚本清单（功能 / 输入 / 输出）

> 路径均相对仓库根目录。"{city}" 表示对每个城市生成一份文件。
> 城市配置统一来自 `configs/{city}.yaml`（44 个 yaml：41 城市 + `_template` + `alternates` + `scenarios`）。

### A.1 核心评价管线 (Steps 01–05)

| 脚本路径 | 功能（分析环节） | 输入数据文件 | 输出文件 |
|---|---|---|---|
| `scripts/01_osm_audit.py` | **城市入选门槛**：OSM 数据完整性审计，计算 DCS（Data Completeness Score = 0.5·coverage + 0.3·height_ratio + 0.2·residential_ratio），门槛 DCS≥0.40 | `configs/{city}.yaml`（OSM Overpass 查询）；`cache/` | `results/osm_audit/audit_results.csv`（15 城审计记录）；`results/osm_audit/audit_comparison.png` |
| `scripts/02_pvgis_download.py` | **D1 数据源**：PVGIS v5.3 (ERA5) TMY + 逐时辐照下载 | `configs/{city}.yaml`（lat/lon） | `results/pvgis/{city}_tmy.csv`、`{city}_hourly.csv`、`{city}_meta.json` |
| `scripts/03_morphology_analysis.py` | **D2 形态 + D3 技术**：D2=5 指标（平均高度、密度、平均屋顶面积、紧凑度、容积率）；D3=屋顶利用率/遮挡因子/有效面积/可部署容量（KD-Tree 遮挡在此，已移出 D2） | OSM 建筑（`cache/`）；`configs/{city}.yaml`；3 米层高代理 | `results/morphology/{city}_d2_indicators.csv`、`{city}_d3_indicators.csv`、`{city}_buildings_classified.gpkg`、`{city}_typology_stats.csv`；`cross_city_d2d3_summary.csv`；`local_density_100m.csv` |
| `scripts/04_energy_simulation.py` | **D1 气候 + D4 经济 + D5 不确定性**：确定性发电模拟；MC N=10,000 LHS；Sobol Saltelli 4,096；经济参数 3000 CNY/kWp、O&M 30、25 年、0.5% 衰减 | `results/pvgis/{city}_*.csv`；`configs/{city}.yaml` | `results/energy/{city}_d1_climate.csv`、`{city}_d4_economics.csv`、`{city}_deterministic.csv`、`{city}_mc_summary.csv`、`{city}_sobol_indices.csv`；`cross_city_d1d4d5.csv` |
| `scripts/05_fdsi_scoring.py` | **FDSI 综合评分**：汇总 D1–D5，min-max 归一化，熵权 + AHP 线性组合（α=0.5），维度权重与 FDSI 排名 | `results/morphology/cross_city_d2d3_summary.csv`；`results/energy/cross_city_d1d4d5.csv` | `results/fdsi/fdsi_scores.csv`、`integrated_indicators.csv`、`suitability_matrix.csv`、`weight_comparison.csv`、`weight_sensitivity.csv`；`figures/fig_*.png` |

### A.2 情景与扩展分析 (Step 06)

| 脚本路径 | 功能 | 输入 | 输出 |
|---|---|---|---|
| `scripts/06_scenario_analysis.py` | **政策情景分析**：4 情景 × 39 城 = 156 次 FDSI 评分（成本下降 / 碳价等） | `results/fdsi/integrated_indicators.csv` | `results/scenarios/scenario_fdsi_matrix.csv`、`scenario_d4_detail.csv`、`suitability_transitions.csv`；`figures/fig_scenario_*.png` |
| `scripts/06_extended_analysis.py` | 扩展经济/环境分析（早期版本） | `results/energy/*`、`results/fdsi/*` | `figures/`、汇总表 |
| `scripts/06_additional_figures.py` | 学位论文附加图 | `results/fdsi/*` | `figures/` |
| `scripts/06b_fix_figures.py` | 修复 3 张损坏图 | `figures/` | `figures/`（覆盖） |

### A.3 统计分析 (Steps 07–10, 输出到 results_nc/)

| 脚本路径 | 功能 | 输入 | 输出 |
|---|---|---|---|
| `scripts/07_clustering_analysis.py` | K-means + 层次聚类（D1–D5，silhouette 定 k） | `results/fdsi/integrated_indicators.csv` | `results_nc/clustering/*.csv`、`*.png` |
| `scripts/07_paper_figures.py` | 生成 16 张论文图（早期版本） | `results/fdsi/*` | `figures/` |
| `scripts/08_regression_analysis.py` | OLS + LASSO：城市特征对 FDSI 的驱动力 | `results/fdsi/integrated_indicators.csv` | `results_nc/regression/ols_summary.txt`、`lasso_coefficients.csv`、`standardized_coefficients.csv`、`vif_check.csv`、`coefficient_plot.png` |
| `scripts/09_spatial_analysis.py` | 空间自相关：Global + Local Moran's I | `results/fdsi/integrated_indicators.csv` | `results_nc/spatial/global_morans_i.csv`、`lisa_*.csv`、`*.png` |
| `scripts/10_bootstrap_ranking.py` | Bootstrap 排名稳定性 + Leave-One-Out | `results/fdsi/integrated_indicators.csv` | `results_nc/bootstrap/ranking_ci.csv`、`leave_one_out.csv`、`*.png` |
| `scripts/11_generate_figures.py` | 生成 NC 论文图（nc_fig1–8, supp1–2） | `results/fdsi/*`、`results_nc/*` | `figures/nc_fig*.png/pdf` |

### A.4 数据修复脚本（Phase 1，一次性）

| 脚本路径 | 功能 | 输入 | 输出 |
|---|---|---|---|
| `scripts/fix_d2_density.py` | D2 密度/容积率修复 | `results/morphology/*` | 覆盖 `{city}_d2_indicators.csv`（留 `.bak_v1`） |
| `scripts/fix_shenyang_osm.py` / `fix_shenyang_v2.py` | 修复沈阳 OSM + 面积推断高度（3.5→16.6m） | OSM、configs | `results/morphology/shenyang_*`（留 `.bak_v1`） |
| `scripts/fix_urumqi.py` | 修复乌鲁木齐 bbox + 高度推断 | OSM、configs | `results/morphology/urumqi_*`（留 `.bak_v1`） |
| `scripts/fix_chengdu.py` | 修复成都高度推断 | OSM、configs | `results/morphology/chengdu_*`（留 `.bak_v1`） |
| `scripts/fix_guiyang.py` | 修复贵阳 bbox + 面积推断高度 | OSM、configs | `results/morphology/guiyang_*`（留 `.bak_v1`） |

### A.5 NC Phase 1–2 专题分析（41 城，输出到 results_nc/）

| 脚本路径 | 功能 | 输入 | 输出 |
|---|---|---|---|
| `scripts/nc_01b_diagnostics.py` | 15 城数据诊断（D5 独立性、维度相关、数据质量旗标） | `results/fdsi/integrated_indicators.csv` | `results_nc/diagnostics/*.csv` |
| `scripts/nc_02a_misclassification.py` | GHI-only 误分类量化（混淆矩阵、rank-shift） | `results/fdsi/*`、`results/energy/*` | `results_nc/misclassification/*.csv`、`misclassification_summary.json` |
| `scripts/nc_02b_robustness.py` | 鲁棒性检验（替代排名、持续误分类） | `results/fdsi/*` | `results_nc/robustness/*.csv`、`robustness_report.json` |
| `scripts/nc_02c_cross_pairs.py` | Cross-pair 控制变量对比（A 形态控制 / B 气候控制 / C GHI 控制） | `results/fdsi/*`、`results/energy/*` | `results_nc/cross_pairs/type_[abc]_*.csv`、`cross_pair_summary.json` |
| `scripts/nc_02d_policy_cost_and_sensitivity.py` | 政策机会成本具体化 + 分类敏感性（分位混淆矩阵） | `results/fdsi/*`、`results/scenarios/*` | `results_nc/policy_cost/*`、`results_nc/sensitivity/*` |
| `scripts/nc_03_morphology_new_cities.py` | 批量形态分析 **24 个新城市**（15→39 扩展的执行脚本） | OSM、`configs/{city}.yaml` | `results/morphology/{city}_*`（24 城） |
| `scripts/nc_directional_bias.py` | §3.5 方向性偏差分析（partial r = −0.58） | `results/fdsi/*`、`results_nc/misclassification/rank_shift_analysis.csv` | `results_nc/directional_bias/directional_bias_report.txt`、`*.png` |
| `scripts/nc_fig2a_changsha_chengdu.py` | Fig 2a 长沙 vs 成都维度对比图 | `results/fdsi/integrated_indicators.csv` | `figures/fig2a_changsha_chengdu.*` |
| `scripts/nc_figs_main.py` | 生成 NC 正文 Fig 1–4 | `results/fdsi/*`、`results_nc/*` | `figures/fig1–4*.png/pdf` |
| `scripts/nc_validate_numbers.py` | 交叉校验论文引用的每个数字（55/55 通过） | `results/fdsi/*`、`results/scenarios/*`、`results_nc/*` | stdout 校验报告 |

### A.6 支撑代码与工具

| 路径 | 功能 |
|---|---|
| `src/suitability/weighting.py` | 熵权/AHP 赋权核心函数 |
| `src/utils/config_loader.py` | 城市 yaml 配置加载 |
| `src/data/`、`src/comparison/` | 数据与对比工具（包骨架） |
| `tools/bipv_lookup.py`、`tools/bipv_lookup_tool.jsx` | BIPV 城市查询工具（Python + JSX） |
| `scripts/init_github.sh` | 仓库初始化脚本 |

---

## B. 城市级结果文件清单（含城市覆盖数）

> **关键**：仓库内存在 **两代不同口径、城市数不同**的结果并存。详见 `41cities_provenance.md`。

### B.1 41 城结果（当前 HEAD，Phase 3 新版口径 — NC 投稿使用）

| 文件 / 目录 | 内容 | 覆盖城市数 |
|---|---|---|
| `results/fdsi/fdsi_scores.csv` | FDSI 排名（Lhasa #1 = 0.7422 … Chongqing #41 = 0.1752） | **41** |
| `results/fdsi/integrated_indicators.csv` | D1–D5 全指标整合宽表 | **41** |
| `results/fdsi/suitability_matrix.csv` | 适宜性矩阵 | **41** |
| `results/fdsi/weight_comparison.csv` | 熵权/AHP/组合权重对比 | 5 维度（全 41 城拟合） |
| `results/fdsi/weight_sensitivity.csv` | 权重敏感性网格 | 41 城 × 权重扫描 |
| `results/energy/{city}_{d1_climate,d4_economics,deterministic,mc_summary,sobol_indices}.csv` | D1/D4/D5 逐城能源模拟 | **41**（部分城市 sobol/mc 文件齐全度不一，见下注） |
| `results/energy/cross_city_d1d4d5.csv` | 跨城 D1/D4/D5 汇总 | **41** |
| `results/morphology/{city}_d2_indicators.csv`、`{city}_d3_indicators.csv` | D2/D3 逐城指标 | **41** |
| `results/morphology/{city}_buildings_classified.gpkg`、`{city}_typology_stats.csv` | 建筑分类/类型统计 | 41 / 部分城市有 typology |
| `results/morphology/cross_city_d2d3_summary.csv` | 跨城 D2/D3 汇总 | **41** |
| `results/pvgis/{city}_tmy.csv`（全部 41）、`{city}_hourly.csv`+`_meta.json`（约 18 早期城） | PVGIS 辐照 | tmy=**41**；hourly/meta 仅早期 ~18 城 |
| `results/scenarios/scenario_fdsi_matrix.csv`、`scenario_d4_detail.csv`、`suitability_transitions.csv` | 政策情景 | **39**（情景脚本基于 39 城，见 provenance 注） |

### B.2 15 城结果（继承自母仓库 `multi-cities-bipv`，Phase 2 口径 — Paper4_FDSI_Manuscript 使用）

| 文件 | 内容 | 覆盖城市数 |
|---|---|---|
| `results/paper4_summary/table_fdsi_ranking.csv` | FDSI 排名（Xiamen #1 = 0.8913 … Guiyang #15 = 0.2371） | **15** |
| `results/paper4_summary/table_d1_d5_scores.csv` | D1–D5 维度分 | **15** |
| `results/paper4_summary/table_d5_sub_indicators.csv` | D5 子指标（含 d5_4_interaction_ratio_pbt） | **15** |
| `results/paper4_summary/table_all_cities_indicators.csv` | 全指标表 | **15** |
| `results/paper4_summary/table_npv_irr_co2.csv`、`table_loo_validation.csv`、`table_robustness_checks.csv` | 经济/验证/鲁棒性 | **15** |
| `results/paper4_summary/table_cashflow_25yr.csv` | 25 年现金流 | 15 城 × 25 年（125 行） |
| `results/paper4_summary/table_monthly_generation.csv` | 逐月发电 | 15 城 × 12 月（180 行） |

> ⚠ **注**：`results/paper4_summary/` 的 15 城 FDSI 分数与 `results/fdsi/` 的 41 城分数**不一致**
> （例：Beijing 15城=0.7626 vs 41城=0.6279；Xiamen 15城=0.8913 vs 41城=0.6261）。
> 二者是**两次独立的全量运行、口径不同**，15 城**不是** 41 城的子集切片。溯源见 `41cities_provenance.md`。

### B.3 NC Phase 2 专题分析结果（41 城，results_nc/）

| 子目录 | 内容 | 覆盖城市数 |
|---|---|---|
| `results_nc/misclassification/` | GHI-only 误分类、rank-shift | **41**（rank_shift_analysis.csv = 41 行） |
| `results_nc/robustness/` | 替代排名、持续误分类 | **41** |
| `results_nc/cross_pairs/` | A/B/C 三类控制变量配对 | 配对子集（含深圳 vs 香港） |
| `results_nc/directional_bias/` | 方向性偏差（41 城中 15 城误分类） | **41** |
| `results_nc/bootstrap/` | 排名 CI + LOO | **41** |
| `results_nc/clustering/`、`regression/`、`spatial/` | 聚类/回归/空间自相关 | **41** |
| `results_nc/policy_cost/`、`sensitivity/` | 政策成本、分类敏感性 | 39/41（部分基于情景 39 城） |
| `results_nc/diagnostics/` | 数据诊断 | 早期 15 城诊断 |

### B.4 OSM 审计（城市入选门槛）

| 文件 | 内容 | 覆盖城市数 |
|---|---|---|
| `results/osm_audit/audit_results.csv` | DCS 打分（15 行：5 主选 + 备选，含 FAIL 记录） | **15**（含 1 例 Nominatim 失败：沈阳） |

---

## C. 目录结构速览

- `configs/` — 44 个 yaml（41 城市配置 + `_template` + `alternates` + `scenarios`）
- `scripts/` — 34 个脚本（见 A 节）
- `src/`、`tools/` — 支撑代码与查询工具
- `results/` — **41 城 Phase 3 主结果**（fdsi/energy/morphology/pvgis/scenarios）+ **15 城 Phase 2 继承结果**（paper4_summary/）+ osm_audit
- `results_nc/` — NC Phase 2 专题分析（41 城）
- `figures/` — 90+ 图（fig_*、nc_fig*、fig1–6*）+ 若干 `.html` 交互表（工作区未提交）
- `docs/` — `city_selection_final.md`（5 城选择 + DCS 门槛）、`d2_density_fix_report.md`
- 根目录 md — `nc_results_master_summary{,_v2,_v2-2}.md`、`nc_gpt_response_report.md`、`paper4_nc_progress_summary.md`（均以 41 城为样本描述）
- `cache/`、`Archive.zip` — OSM/中间缓存与归档（gitignore 大部分）
