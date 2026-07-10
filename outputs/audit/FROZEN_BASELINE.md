# 冻结基准版本 (Frozen Baseline)

**记录日期**: 2026-07-10

## 15 城最终结果（Paper4_FDSI_Manuscript 稿件基线）

| 项目 | 值 |
|---|---|
| **冻结基准 commit** | **`51dc9b0`** (`51dc9b0ae7f57604f5f420353426954d4a64b8d5`) |
| 提交信息 | `init: fork from multi-cities-bipv with 15-city results for NC expansion` |
| 日期 | 2026-04-08 |
| 管线口径 | **Phase 2 新版**：D2=5 指标、遮挡在 D3、D5 = PBT-CI95(0.40)/σLCOE(0.35)/Sobol交互比(0.25)、熵权-AHP α=0.5、MC 10k LHS、Sobol 4096、PVGIS v5.3 ERA5 |
| 权威结果文件 | `results/paper4_summary/table_fdsi_ranking.csv`（Xiamen #1=0.8913 … Guiyang #15=0.2371）；等同于 `git show 51dc9b0:results/fdsi/fdsi_scores.csv` |

**依据**：15 城 FDSI 结果在 `51dc9b0` 引入后从未被修改（`git log -- results/paper4_summary/` 仅此一条提交），
且该目录在当前 HEAD 仍原样保留。其后所有提交（`ab0539f`→`5f014fb`→…）是 15→39→41 的 **NC 扩展全量重跑（Phase 3）**，
把 `results/fdsi/` 覆盖成了 41 城、并改动了 D5 口径——因此 15 城基线必须以 `51dc9b0` 冻结，不能以 HEAD 的 `results/fdsi/` 代表。

## 参考：41 城 NC 全量结果（不同论文/不同口径）

| 项目 | 值 |
|---|---|
| 41 城落地 commit | `5f014fb` (`5f014fbfe622016b232285aaeff06f3be1d12da5`, 2026-04-09) |
| 当前 HEAD | `21e813a` (`21e813abe7cf67f3ef7f293b1b13c9ae9f22039d`, 2026-04-11) |
| 管线口径 | **Phase 3 新版**：D5 = PBT-CI95(0.35)/σLCOE(0.30)/Sobol-PBT-电价敏感性(0.35) |
| 权威结果文件 | `results/fdsi/fdsi_scores.csv`（Lhasa #1=0.7422 … Chongqing #41=0.1752） |

> 若 NC 41 城稿件也要一个冻结点，建议用 `5f014fb`（41 城首次完整落地）或论文定稿时的 HEAD。
> 两版 D5 口径不同（见 `41cities_provenance.md` §b.1），交叉引用时须在方法学声明。
