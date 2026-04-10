# Paper 4 NC — 全部分析结果汇总（写作用）

**冻结日期**: 2026-04-09
**样本**: 41 城市（39 内地 + 香港 + 台北）
**主 Claim**: Using irradiance alone to prioritize urban residential rooftop PV systematically misclassifies city suitability.

---

## 1. 核心误分类数字

| 指标 | 数值 | 用于 |
|------|------|------|
| Spearman r_s (GHI rank vs FDSI rank) | 0.760 | Results §1 |
| GHI 无法捕获的排名变异 | 42.2% | Results §1 |
| 三分位误分类率 | 31.7% (13/41) | Results §1 |
| 中位排名偏移 | 5.0 位 | Results §1 |
| 最大排名偏移 | 22 位 (Urumqi) / 21 位 (Hong Kong) | Results §1 |
| \|shift\| ≥ 5 的城市 | 51.2% (21/41) | Results §1 |
| 持续误分类城市 (全5方案) | 6 城 | Results §2 |

**正文推荐措辞**: "42.2% of rank variation is not captured by irradiance alone."
（不说"来自非资源因素"——避免审稿人质疑未观测因素）

---

## 2. 鲁棒性检验

| 方案 | r_s vs GHI | 核心发现 (< 0.85)? | r_s vs 原始 | 排名稳定 (> 0.80)? |
|------|-----------|-------------------|------------|-------------------|
| Original (entropy+AHP) | 0.760 | PASS | 1.000 | — |
| Equal weight (各0.20) | ~0.72 | PASS | ~0.96 | PASS |
| Drop D5 | ~0.75 | PASS | ~0.93 | PASS |
| Drop D4 | ~0.78 | PASS | ~0.88 | PASS |
| PCA composite (PC1=57%) | ~0.71 | PASS | ~0.91 | PASS |

**6 个持续误分类城市** (所有方案下均被 GHI 错误归类):
Beijing, Changsha, Guangzhou, Qingdao, Harbin, Hong Kong

**正文推荐措辞**: "This finding is robust to alternative index constructions including equal weighting, dimension removal, and data-driven PCA composites (Supplementary Table X)."

**预防审稿人反打** ("去掉D4/D5还稳，为什么需要五维？"):
五维框架不是为了让误分类现象存在（它在四维下也存在），而是为了诊断误分类的来源。D4 识别哪些城市被经济条件拖累，D5 识别哪些城市的评估本身不确定。去掉它们核心发现还在，但失去了解释"为什么错"和"政策应该怎么补"的能力。

---

## 3. 机制型城市配对 (Cross-Pair)

### Type C — 最强证据 (控制 GHI，观察 FDSI 分化)
**长沙 vs 成都**: GHI 仅差 86 kWh/m², FDSI 排名差 23 位
→ 几乎相同的太阳能资源，完全不同的 BIPV 适宜性

### Type B — 形态效应 (同气候区，不同形态)
**南昌 vs 重庆**: 同 HSCW 区，形态差异驱动 FDSI gap = 0.370
→ 城市形态对适宜性的影响独立于地理/气候

### Type A — 气候+经济效应 (相同形态，不同气候)
**重庆 vs 拉萨**: 形态相同但 FDSI 差 0.573
→ 气候和经济条件的联合效应

### Dream Pair — 制度效应
**深圳 vs 香港**: 同纬度 (22.3°N)、同气候 (HSWW)、GHI 差仅 87 kWh/m²

| 维度 | 深圳 (#14) | 香港 (#32) | 解读 |
|------|-----------|-----------|------|
| D1 气候 | 0.635 | 0.713 ↑ | 香港 GHI 略高 |
| D2 形态 | 0.468 | 0.061 ↓↓ | 九龙 FAR=1.667 (全样本最高) → 瓶颈 |
| D3 技术 | 0.660 | 0.259 ↓↓ | 极高密度 → 遮挡严重 |
| D4 经济 | 0.599 | 0.809 ↑↑ | 电价 1.20 CNY → PBT 1.86yr (全样本最短) |
| D5 确定性 | 0.511 | 0.480 | 相近 |
| **FDSI** | **0.574** | **0.458** | 香港经济最优但形态最差 |

→ "经济性无法补偿形态劣势": 香港拥有全样本最短回收期 (1.86yr)，但九龙的极端高密度 (FAR=1.667) 使其 FDSI 排名第32——D2/D3 对 BIPV 的约束比 D4 更根本。

---

## 4. 政策机会成本 (物理量)

### 按 GHI 前1/3 优先部署时被遗漏的潜力

| 指标 | 数值 |
|------|------|
| 被遗漏城市数 | 4 / 13 |
| 被遗漏装机容量 | 1,157 MW |
| 被遗漏年发电量 | 1,522 GWh/yr |
| 被遗漏年减排量 | 837 ktCO₂/yr |
| 被遗漏人口 | 6,700 万人 |
| 占样本总潜力比例 | 9.5% |

**摘要句**: "If the top third of cities are selected for priority deployment based on solar resource alone, 4 cities with 67 million residents and an estimated 1,522 GWh/yr of rooftop PV potential would be systematically overlooked."

---

## 5. 分类敏感性 (SI)

### Panel A: 分组分类

| 方案 | 误分类率 | 严重误分类 (跨≥2组) |
|------|---------|-------------------|
| Tercile (3组) | 31.7% | 2.4% |
| Quartile (4组) | 46.3% | 12.2% |
| Quintile (5组) | 51.2% | 19.5% |

→ 误分类率随粒度单调递增，支持用粗粒度 (High/Med/Low) 做政策分组

### Panel B: 靶向精度

| 策略 | k | 精度 | 遗漏 |
|------|---|------|------|
| Top 10% | 4 | 75% | 1 |
| Top 20% | 8 | 63% | 3 |
| **Top 25%** | **10** | **80%** | **2** |
| Top 33% | 13 | 69% | 4 |
| Top 50% | 20 | 65% | 7 |

→ Top-25% 精度最高 (80%)——自然断点

**正文推荐**: 主文用 tercile (31.7%)，加一句 "Results are qualitatively consistent across alternative classification thresholds (Supplementary Table X)."

---

## 6. 情景分析关键数字 (已有)

| 情景 | High | Medium | Low | 关键发现 |
|------|------|--------|-----|---------|
| Baseline | 13 | 13 | 13 | 三等分 |
| Carbon 100 CNY/tCO₂ | 14 | 14 | 11 | 碳价效果极小 (+1 High) |
| PV Cost −50% | 36 | 2 | 1 | 成本下降是主要驱动力 |
| Aggressive | 37 | 1 | 1 | 仅重庆仍 Low |

- "光伏成本下降50%使64%城市升级"
- "碳价100 CNY/tCO₂几乎无效 (仅太原1城跳变)"
- "重庆是物理极限的标尺 (GHI=1123, 盆地地形)"

---

## 7. 统计分析汇总

| 分析 | 关键结果 | 用于 |
|------|---------|------|
| 聚类 | k=2 最优 (silhouette=0.398); C1="盆地低适宜型" 7城 | Results §3 |
| 回归 | R²=0.929; top β: GHI(+0.091***) > FAR(−0.031***) > Shading(+0.030**) | Results §3 |
| 空间自相关 | FDSI Moran's I=0.214 (p=0.014); D2 不显著 (形态独立于地理) | Results §3 |
| Bootstrap | 0/41 城市 CI>10; 排名完全稳健 | Methods/SI |

---

## 8. Results 正文结构

### §1: Misclassification exists and is quantifiable
- 混淆矩阵 + rank shift 分布 (Fig.2)
- r_s = 0.760, 42.2% unexplained
- 敏感性检验 → SI

### §2: Misclassification is not an artifact of index construction
- 5种替代方案全 PASS (Table 1)
- 6个持续误分类城市
- "为什么五维" 的预防性论证

### §3: Urban morphology and economics independently drive the reordering
- 回归 standardized β
- Cross-pair 对比 (Fig.3): 长沙vs成都, 深圳vs香港, 南昌vs重庆
- Moran's I: D2空间随机 → 形态效应不是地理的影子

### §4: The misclassification has quantifiable policy consequences
- 1,522 GWh/yr + 67M 人口被遗漏 (Fig.4)
- 情景分析: 成本下降是钥匙，碳价不是
- 重庆 = 物理极限标尺

---

## 9. 待确认事项

- [ ] 鲁棒性表中替代方案的精确 r_s 值 (02b 脚本输出，标 ~ 的需替换)
- [ ] 屋顶面积估算方法确认 (building_density × study_area vs 直接字段)
- [ ] 港台的 PVGIS 数据源确认 (ERA5 vs SARAH2)
- [ ] 导师对 41城 vs 42城 (是否加澳门) 的意见
- [ ] Discussion 中港台政治表述的措辞审核
