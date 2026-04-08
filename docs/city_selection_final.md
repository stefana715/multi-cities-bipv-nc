# ============================================================================
# Paper 4 — 代表城市最终确认 (City Selection Final Confirmation)
# 三层筛选结果汇总
# ============================================================================
# 日期: 2026-04-05
# 状态: Layer 3 审计完成，五城市全部通过阈值 (DCS ≥ 0.4)
# ============================================================================

## 最终选定城市

| Climate Zone | City | Study Area | Lat | Lon | DCS | Paper Heritage |
|---|---|---|---|---|---|---|
| Severe Cold (严寒) | Harbin (哈尔滨) | 南岗区 (Nangang) | 45.75°N | 126.65°E | 0.419 | — |
| Cold (寒冷) | Beijing (北京) | 海淀区 (Haidian) | 39.90°N | 116.40°E | 0.517 | — |
| HSCW (夏热冬冷) | Changsha (长沙) | 岳麓区 (Yuelu) | 28.23°N | 112.94°E | 0.524 | Paper 2 |
| HSWW (夏热冬暖) | Shenzhen (深圳) | 福田区 (Futian) | 22.54°N | 114.06°E | 0.615 | Paper 1 |
| Mild (温和) | Kunming (昆明) | 主城四区 bbox | 25.04°N | 102.68°E | 0.521 | — |

## 备选城市审计结果（未选用）

| Climate Zone | City | Study Area | DCS | Reason Not Selected |
|---|---|---|---|---|
| Severe Cold | Changchun (长春) | 朝阳区 | 0.563 | Harbin passed; Harbin more representative |
| Cold | Jinan (济南) | 历下区 | 0.535 | Beijing passed; Beijing is mega-city |
| Cold | Xi'an (西安) | 碑林区 | 0.455 | Beijing passed |
| Mild | Guiyang (贵阳) | 南明区 | 0.376 | FAIL (DCS < 0.4) |
| Severe Cold | Shenyang (沈阳) | 沈河区 | — | Query failed (Nominatim) |

## 昆明补充审计详情

初始查询（盘龙区）DCS=0.383，未通过阈值。补充审计测试四个方案：

| Scheme | Area | Buildings | DCS | Result |
|---|---|---|---|---|
| A: 五华区 | Wuhua District | 2,628 | 0.482 | PASS |
| B: 官渡区 | Guandu District | 3,360 | 0.439 | PASS |
| C: 西山区 | Xishan District | 3,046 | 0.421 | PASS |
| D: 主城四区 bbox | Central 4 districts | 6,074 | 0.521 | **PASS (SELECTED)** |

选择 bbox 方案的理由：
1. 建筑数量最多（6,074栋），样本量充足
2. DCS 最高（0.521）
3. 覆盖四个城区，住宅形态多样性更好

## Layer 2 城市属性对比

| City | Tier | Pop (M) | Dominant Housing | Dev Stage |
|---|---|---|---|---|
| Harbin | Super-large | 10.0 | Mixed | Mixed age |
| Beijing | Mega | 21.5 | High-rise | Mixed age |
| Changsha | Super-large | 10.5 | Mixed | Mixed age |
| Shenzhen | Mega | 17.6 | High-rise | New dominant |
| Kunming | Super-large | 8.5 | Mid-rise | Mixed age |

五城市覆盖：2个超大城市 + 3个特大城市，形态从高层主导到多层主导，
发展阶段从新城主导到新旧混合，具有良好的结构多样性。
