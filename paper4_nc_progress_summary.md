# Paper 4 NC 版本 — 项目完整进度总结

**最后更新**: 2026-04-09
**项目**: multi-cities-bipv-nc
**目标期刊**: Nature Communications / Nature Cities

---

## 一、当前状态总览

| 指标 | 数值 | 状态 |
|------|------|------|
| 城市总数 | 39 | ✓ 完成 |
| CRITICAL 城市 | 0 | ✓ 全部修复 |
| WARNING 城市 | 3 (贵阳/呼和浩特/厦门，OSM限制) | ✓ 不可修，当 limitation |
| OK 城市 | 36 | ✓ |
| Spearman r_s (GHI vs FDSI) | 0.739 | ✓ 核心发现 |
| D5 R² (独立性) | 0.514 | ✓ 目标 <0.7 达成 |
| 回归 R² | 0.929 | ✓ 框架解释力强 |
| Moran's I (FDSI) | 0.214 (p=0.014) | ✓ 空间集聚显著 |
| Bootstrap 不稳定城市 | 0/39 | ✓ 排名稳健 |
| 政策情景 | 4 × 39 = 156 FDSI | ✓ 完成 |

---

## 二、39 城市名单

### 严寒区 Severe Cold (7)
harbin, changchun, shenyang, dalian, hohhot, tangshan, urumqi

### 寒冷区 Cold (13)
beijing, tianjin, jinan, zhengzhou, xian, shijiazhuang, taiyuan, lanzhou, yinchuan, xining, qingdao, wuxi, suzhou

### 夏热冬冷区 HSCW (10)
shanghai, chongqing, wuhan, nanjing, changsha, chengdu, hangzhou, hefei, nanchang, ningbo

### 夏热冬暖区 HSWW (6)
shenzhen, guangzhou, xiamen, fuzhou, nanning, haikou

### 温和区 Mild (3)
kunming, guiyang, lhasa

---

## 三、数据修复历程

| 阶段 | 城市 | 问题 | 修复 | 效果 |
|------|------|------|------|------|
| Phase 1a | 沈阳 | bbox 郊区，均高 3.5m | 换 heping_wide bbox | 均高→16.6m, n=1708 |
| Phase 1a | 成都 | 均高 86.9m 异常 | 换 bbox + outlier filter | 均高→18.5m, FAR→0.616 |
| Phase 1a | 贵阳 | 全市 OSM 覆盖不足 | 优化 bbox | n=707 WARNING (不可修) |
| Phase 2 | 乌鲁木齐 | 均高 4.0m | 换新市区 bbox | 均高→15.8m, n=3113 |
| Phase 2 | 上海 | area=1177km² | 缩到杨浦核心 32km² | FAR 1.27→0.81, density 正常 |
| Phase 2 | 呼和浩特 | area=658km² | 缩到赛罕核心 | density 提升, n=990 WARNING |
| Phase 2 | 银川 | area=577km² | 缩到兴庆核心 | density/FAR 正常 |
| Phase 2 | 南宁 | area=461km² | 缩到青秀核心 | density 正常 |
| Phase 2 | 合肥 | area=694km² | 缩到蜀山核心 | density 正常 |
| Phase 2 | 西安 | FAR=1.09 | 扩大 bbox | FAR→0.843, 排名 #15→#12 |

---

## 四、D5 维度改进

### 问题
MC_PARAMS 全局常量 → D5 退化为 D1+D4 影子（R²=0.962）

### 改进（三轮）

| 轮次 | 改动 | R² |
|------|------|-----|
| 初始 | 全局常量 | 0.962 |
| 第1轮 | 城市特异性 ghi_uncertainty (0.03-0.10) | 0.817 |
| 第2轮 | + electricity_price_factor 差异化 + degradation_rate triangular | 0.708 |
| 第3轮 | + 替换 d5_4 为 sobol_pbt_S1_elec_price_factor + 权重 [0.35,0.30,0.35] | 0.495→0.514 |

### 最终 D5 组成
- pbt_ci95_width (35%): 气候+经济综合不确定性
- mc_lcoe_std (30%): 度电成本不确定性
- sobol_pbt_S1_elec_price_factor (35%): 电价市场风险敞口

---

## 五、统计分析结果

### 5.1 聚类分析
- 最优 k=2（silhouette=0.398）
- C0: 32城 "中等综合型"
- C1: 7城 "盆地低适宜型"（宁波、福州、杭州、苏州、贵阳、成都、重庆）
- PCA: PC1=56.8% (D1/D4/D5), PC2=20.9% (D2)

### 5.2 多元回归
- R²=0.929, Adj-R²=0.904
- Top 3 standardized β: GHI (+0.091***) > FAR (−0.031***) > Shading (+0.030**)
- 电价和建筑高度不显著（被其他变量吸收）

### 5.3 空间自相关
- FDSI: Moran's I=0.214 (p=0.014) — 显著正空间集聚
- D2 形态: 不显著（随机分布） — 形态独立于地理位置
- D4 经济: I=0.249 (p=0.005) — 强集聚（省际电价差异）
- HH hotspot: 银川+呼和浩特（西北）
- LL coldspot: 贵阳（西南孤岛）
- HL 异常: 昆明（自身高分但邻居低分）

### 5.4 Bootstrap 排名稳定性
- 0/39 城市 CI>10
- 最稳定: 拉萨[1,2]、银川[1,2]
- 最不稳定: 合肥 CI=9、青岛 CI=9、南宁 CI=9

---

## 六、政策情景分析结果

### 4 情景 × 39 城 = 156 FDSI

| 情景 | High | Medium | Low | 关键发现 |
|------|------|--------|-----|---------|
| Baseline (2024) | 13 | 13 | 13 | 三等分 |
| Carbon 100 CNY/tCO₂ | 14 | 14 | 11 | 碳价效果极小(+1 High) |
| PV Cost −50% | 36 | 2 | 1 | 成本下降是主要驱动力 |
| Aggressive | 37 | 1 | 1 | 仅重庆仍 Low |

### 关键数字
- "光伏成本下降50%使64%(25/39)城市从当前等级升级"
- "12/13个Low城市升至Medium或High"
- "仅重庆(GHI=1123, 全国最低)在最激进政策下仍为Low"
- 碳价受益最大: 太原(山西碳因子0.884) — 唯一跳变城市
- Aggressive情景下排名变化最大: 西宁(#12→#5, +7), 海口(#5→#13, −8)

---

## 七、NC 论文三大叙事线

### 叙事 1: "适宜性 ≠ 资源"
Spearman r_s=0.739 — GHI只能解释55%的FDSI排名变异。乌鲁木齐(GHI#9→FDSI#31)是"高辐照陷阱"的极端案例：新疆电价0.400 CNY/kWh（全国最低）严重压制BIPV经济性。回归证实GHI是最强预测因子(β=0.091)但FAR(β=−0.031)和遮挡系数(β=0.030)也显著 — 城市规划决策对BIPV适宜性有独立于气候的影响。

### 叙事 2: "降成本是钥匙，碳价不是"
碳价100 CNY/tCO₂几乎无效(仅太原1城跳变)。光伏成本下降50%使64%城市升级。投资于PV制造业降成本的边际收益远高于碳市场。

### 叙事 3: "重庆是物理极限的标尺"
即使最激进政策下重庆仍Low(FDSI=0.376)。GHI=1123是盆地地形的物理极限。并非所有城市都适合BIPV — 需要差异化可再生能源策略。

---

## 八、待完成工作

| 任务 | 优先级 | 预估时间 |
|------|--------|----------|
| Figure 设计与生成 | 高 | 1-2天 |
| 论文正文撰写 | 高 | 2-3周 |
| Supplementary Information | 中 | 1周 |
| GitHub 整理 + 数据归档 | 中 | 1-2天 |
| 投稿前导师审阅 | 高 | 1-2周 |

---

## 九、产出文件清单

### 数据
- `results/fdsi/fdsi_scores.csv` — 39城FDSI排名
- `results/fdsi/suitability_matrix.csv` — D1-D5维度得分
- `results/fdsi/integrated_indicators.csv` — 全部指标
- `results/scenarios/scenario_fdsi_matrix.csv` — 156个情景FDSI
- `results/scenarios/suitability_transitions.csv` — 跳变矩阵

### 分析
- `results_nc/diagnostics/` — GHI vs FDSI诊断
- `results_nc/clustering/` — 聚类分析
- `results_nc/regression/` — 回归分析
- `results_nc/spatial/` — 空间自相关
- `results_nc/bootstrap/` — 排名稳定性

### 图表
- `figures/fig_scenario_fdsi_heatmap.png`
- `figures/fig_scenario_transitions.png`
- `figures/fig_scenario_distribution.png`
- `figures/fig_scenario_rank_bump.png`
- `results_nc/clustering/dendrogram.png`
- `results_nc/clustering/radar_by_cluster.png`
- `results_nc/clustering/pca_scatter.png`
- `results_nc/regression/coefficient_plot.png`
- `results_nc/spatial/moran_scatterplot.png`
- `results_nc/bootstrap/ranking_stability_heatmap.png`
