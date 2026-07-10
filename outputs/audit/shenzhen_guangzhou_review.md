# 复核清单：深圳/广州排名变动（15城 Paper4，Phase-3 D5 口径）

**生成日期**: 2026-07-10
**触发变更**: 15城 Paper4 的 D5 打分口径统一到 Phase-3（35/30/35，第三子指标 = 电价敏感性
`sobol_pbt_S1_elec_price_factor`，替代 `interaction_ratio`）。为此重跑了 04 的 Sobol 步（仅 Sobol，
未动 MC/数据生成层），获得 15 城**真实**的 per-city 电价敏感性（原 Phase-2 该列为占位常数 0.1011）。

## 核心数值变动

| 城市 | 旧 FDSI | 旧 rank | 旧 suit | 新 FDSI | 新 rank | 新 suit | 变动 |
|---|---|---|---|---|---|---|---|
| **深圳 Shenzhen** | 0.6939 | 6 | Medium | **0.6161** | **12** | Medium | **↓6** |
| **广州 Guangzhou** | 0.6949 | 5 | Medium | **0.6316** | **10** | Medium | **↓5** |
| 西安 Xian | 0.6800 | 9 | Medium | 0.7050 | 5 | **High** | ↑4 |
| 昆明 Kunming | 0.6917 | 7 | Medium | 0.7005 | 6 | **High** | ↑1 |

- **High 层 4城→6城**（新增西安、昆明）；深圳/广州仍 Medium 但跌出中上游。
- Spearman(旧,新)=0.839；10/15 城排名变动；max|Δrank|=6。
- 机制（真实信号）：深圳/广州电价最高（0.68）→ PBT 对电价最敏感 → 新第三子指标（is_benefit=False）拉低其 D5。

---

## ① 已由本次编辑更新（请核对交叉引用是否与新值一致）

| 文件 | 更新内容 | 备份 |
|---|---|---|
| `results/paper4_summary/table_fdsi_ranking.csv` | fdsi_score, rank, suitability | `.bak_phase2` |
| `results/paper4_summary/table_d1_d5_scores.csv` | score_D5 | `.bak_phase2` |
| `results/paper4_summary/table_d5_sub_indicators.csv` | 新增 genuine elec_price 列；保留 interaction_ratio | `.bak_phase2` |
| `results/paper4_summary/table_all_cities_indicators.csv` | score_D5, fdsi_score, genuine elec_price | `.bak_phase2` |

## ② 已重生成的 15城图（Phase-3 数据；各有 `.bak_phase2` 备份）

| 图 | 变化要点 |
|---|---|
| `fig04_radar` | D5 维度轮廓更新（深圳/广州 D5 0.94/0.88→0.64/0.64） |
| `fig05_heatmap` | D5 Certainty 列更新 |
| `fig06_fdsi_ranking` | 新排名 + 标题改为 Phase-3；深圳降至 0.616(#12) |
| `fig07_weight_sensitivity` | 15城 Phase-3 α 扫描（重算） |
| `fig09_sobol_bar` | **用真实全量 Sobol 块**（原 Phase-2 15城 Sobol 全列为占位常数，仅电价列本次重跑变为真实）。深圳电价 S₁=0.1853、广州=0.1678（电价 0.68 最高→敏感性居前，合理） |
| `fig10_d4_vs_d5` | D5 轴位置更新 |

> ⚠ **fig09 注意**：为使该图不误导，(a)(b) 两个面板均改用本次重跑的**真实 per-city Sobol S₁**（5 参数全部）。
> 表格中除电价列外的其它 Sobol 列仍为 Phase-2 占位值（你只授权更新电价列）。若要让表格与 fig09 完全一致，
> 需另行同意把真实全量 Sobol 块写回 `table_all_cities_indicators.csv`。

## ③ 15城 Paper4 手稿正文（在你的 docx — 工具看不到，请按此搜索并手改）

在手稿中定位每处把**深圳/广州描述为高适宜性、HSWW 首选/代表、进入前列**的**段落、表格、图注**：
- 搜索词：`深圳` / `Shenzhen`、`广州` / `Guangzhou`、`HSWW 代表`、`高适宜` / `High suitability`、`前六` / `top`、`排名第5/第6`。
- 典型需改：
  - "深圳作为 HSWW 高适宜代表 / 位列前六 / High" → 现为 **#12, Medium**。
  - "广州排名第5 / 前列" → 现为 **#10**。
  - "High 适宜性城市为 4 个" → 现为 **6 个**（+西安、昆明）。
  - 任何引用旧 FDSI 数值 0.6939(深圳)/0.6949(广州) 的文字。

## ④ 本仓库 41城 NC 正文 — 不受本次影响，切勿改动（列出以防两篇稿混引）

以下为 **41城 NC** 数据（深圳 #14 vs 香港 #32 dream pair），`results/fdsi/` 未动，故不变：
- `nc_results_master_summary.md:61,63`
- `nc_results_master_summary_v2.md:69,71,223`
- `nc_results_master_summary_v2-2.md:69,71,235`
- `nc_gpt_response_report.md:48,57,59,134,192,232`（含 `fig4_dream_pair`）

仅需确认：15城 Paper4 与 41城 NC 的深圳排名（#12 vs #14）没有互相误引。

## ⑤ 代码里"深圳 = HSWW 代表"的硬编码（原5城代表集设计；视叙事需要复核）

非排名断言，但若 15城 叙事依赖"深圳=HSWW 样板"，深圳现为 Medium，需重新斟酌：
- `scripts/06_extended_analysis.py:462` — `representative = {harbin, beijing, changsha, shenzhen, kunming}`
- `scripts/07_paper_figures.py:947` — 同上（雷达代表集）
- `tools/bipv_lookup.py:119` — `"hsww": "shenzhen"`（HSWW 查询默认返回深圳）
- `docs/city_selection_final.md:16`、`docs/d2_density_fix_report.md:38` — 历史文档，含深圳旧分数（陈述性）

---

## 未提交（待你确认后再 commit）

已改：4 表 + 7 图（fig04/05/06/07/09/10/16）。**未动**：`results/fdsi/`（41城）、任何 `.md` 正文、docx、目录文件。
`.bak_phase2` 备份齐全，可一键回滚。
