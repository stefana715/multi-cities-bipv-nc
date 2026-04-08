"""
============================================================================
Paper 4 — FDSI 赋权模块
============================================================================
三层赋权体系：
  1. 熵权法 (Entropy Weighting)    → 客观权重，基于数据离散度
  2. AHP (层次分析法)               → 主观权重，基于专家判断
  3. 组合赋权 (Combined Weighting)  → 主客观加权融合
  4. 权重敏感性分析                  → 证明结论对权重选择不敏感

设计原则：
  - 组合赋权公式: w_combined = α * w_entropy + (1-α) * w_ahp
  - 默认 α = 0.5（等权融合），论文中做 α ∈ [0.1, 0.9] 的敏感性分析
  - AHP 一致性比率 CR < 0.1 才接受
  - 所有方法均可独立运行，方便审稿人要求的对比分析

适用层级：
  - Level 1: 五个维度之间的权重 (D1-D5)
  - Level 2: 各维度内部子指标之间的权重

Reference:
  - Shannon, C.E. (1948) 熵权法理论基础
  - Saaty, T.L. (1980) AHP 方法
  - 组合赋权在建筑/能源评价中的应用较成熟，RE/AE 审稿人认可度高
============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import warnings

log = logging.getLogger(__name__)


# ============================================================================
# 1. 熵权法 (Entropy Weighting) — 客观权重
# ============================================================================

def entropy_weight(decision_matrix: np.ndarray,
                   indicator_names: Optional[List[str]] = None,
                   is_benefit: Optional[List[bool]] = None) -> Dict:
    """
    基于信息熵的客观赋权方法。

    原理：指标的离散程度越大，包含的信息量越多，熵值越小，权重越大。
    优点：完全基于数据，无主观偏差。
    缺点：无法体现决策者偏好，可能赋予"噪声大"的指标过高权重。

    Parameters
    ----------
    decision_matrix : np.ndarray, shape (n_samples, n_indicators)
        决策矩阵。行=评价对象（城市×形态组合），列=指标。
    indicator_names : list of str, optional
        指标名称列表。
    is_benefit : list of bool, optional
        各指标是否为正向指标（越大越好）。
        True = 正向 (benefit), False = 负向 (cost)。
        默认全部为正向。

    Returns
    -------
    dict with keys:
        'weights': np.ndarray — 各指标的熵权
        'entropy': np.ndarray — 各指标的信息熵
        'divergence': np.ndarray — 各指标的差异系数 (1-entropy)
        'names': list — 指标名称
        'method': str — "entropy"
    """
    X = np.array(decision_matrix, dtype=float)
    n, m = X.shape

    if indicator_names is None:
        indicator_names = [f"I{i+1}" for i in range(m)]
    if is_benefit is None:
        is_benefit = [True] * m

    assert len(indicator_names) == m
    assert len(is_benefit) == m

    # ── Step 1: 归一化（min-max 标准化到 [0.001, 1]）──
    # 避免 log(0) 问题，下界设为 0.001
    X_norm = np.zeros_like(X)
    for j in range(m):
        col = X[:, j]
        col_min, col_max = col.min(), col.max()

        if col_max == col_min:
            # 所有值相同 → 该指标无区分度 → 均匀分布
            X_norm[:, j] = 1.0 / n
            warnings.warn(f"指标 '{indicator_names[j]}' 所有值相同，熵权将为0")
            continue

        if is_benefit[j]:
            X_norm[:, j] = (col - col_min) / (col_max - col_min)
        else:
            X_norm[:, j] = (col_max - col) / (col_max - col_min)

    # 确保最小值 > 0
    X_norm = np.clip(X_norm, 0.001, None)

    # ── Step 2: 计算各指标下各对象的比重 p_ij ──
    P = X_norm / X_norm.sum(axis=0, keepdims=True)

    # ── Step 3: 计算各指标的信息熵 ──
    k = 1.0 / np.log(n)  # 常数，使 0 ≤ e ≤ 1
    E = np.zeros(m)
    for j in range(m):
        pj = P[:, j]
        pj = pj[pj > 0]  # 安全起见
        E[j] = -k * np.sum(pj * np.log(pj))

    # ── Step 4: 计算差异系数和权重 ──
    D = 1 - E  # 差异系数（divergence）
    if D.sum() == 0:
        weights = np.ones(m) / m
        warnings.warn("所有指标的差异系数为0，返回等权")
    else:
        weights = D / D.sum()

    return {
        "weights": weights,
        "entropy": E,
        "divergence": D,
        "names": indicator_names,
        "method": "entropy",
    }


# ============================================================================
# 2. AHP (层次分析法) — 主观权重
# ============================================================================

# RI 表：随机一致性指标 (Saaty, 1980)
AHP_RI_TABLE = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
}


def ahp_weight(comparison_matrix: np.ndarray,
               indicator_names: Optional[List[str]] = None,
               cr_threshold: float = 0.1) -> Dict:
    """
    层次分析法 (AHP) 主观赋权。

    Parameters
    ----------
    comparison_matrix : np.ndarray, shape (n, n)
        Saaty 标度判断矩阵。
        元素 a_ij 表示指标 i 相对于指标 j 的重要性。
        1=同等重要, 3=稍微重要, 5=明显重要, 7=非常重要, 9=极端重要
        矩阵应满足 a_ji = 1/a_ij（互反性）。
    indicator_names : list of str, optional
    cr_threshold : float
        一致性比率阈值，默认 0.1。CR ≥ 阈值时发出警告。

    Returns
    -------
    dict with keys:
        'weights': np.ndarray — AHP 权重
        'lambda_max': float — 最大特征值
        'CI': float — 一致性指标
        'CR': float — 一致性比率
        'is_consistent': bool — CR < threshold
        'names': list
        'method': str — "ahp"
    """
    A = np.array(comparison_matrix, dtype=float)
    n = A.shape[0]
    assert A.shape == (n, n), "判断矩阵必须是方阵"

    if indicator_names is None:
        indicator_names = [f"I{i+1}" for i in range(n)]

    # ── 检查互反性 ──
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i, j] * A[j, i] - 1.0) > 0.01:
                warnings.warn(
                    f"判断矩阵不满足互反性: A[{i},{j}]×A[{j},{i}] = "
                    f"{A[i, j] * A[j, i]:.3f} ≠ 1"
                )

    # ── 特征值法求权重 ──
    eigenvalues, eigenvectors = np.linalg.eig(A)
    # 取最大特征值对应的特征向量
    idx = np.argmax(eigenvalues.real)
    lambda_max = eigenvalues[idx].real
    w = eigenvectors[:, idx].real
    w = w / w.sum()  # 归一化

    # ── 一致性检验 ──
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    RI = AHP_RI_TABLE.get(n, 1.49)
    CR = CI / RI if RI > 0 else 0.0
    is_consistent = CR < cr_threshold

    if not is_consistent:
        warnings.warn(
            f"AHP 一致性检验未通过: CR = {CR:.4f} ≥ {cr_threshold}。"
            f"请调整判断矩阵。"
        )
    else:
        log.info(f"AHP 一致性检验通过: CR = {CR:.4f} < {cr_threshold}")

    return {
        "weights": w,
        "lambda_max": lambda_max,
        "CI": CI,
        "CR": CR,
        "is_consistent": is_consistent,
        "names": indicator_names,
        "method": "ahp",
    }


# ============================================================================
# Paper 4 默认 AHP 判断矩阵
# ============================================================================

def get_default_ahp_matrix_d1_d5() -> Tuple[np.ndarray, List[str]]:
    """
    五个维度 (D1-D5) 的默认 AHP 判断矩阵。

    专家判断逻辑（可在论文中说明）：
    - D5 确定性维度是本研究核心贡献，权重应适度偏高
    - D1 气候资源是基础前提
    - D3 技术部署是连接理论与实际的桥梁
    - D2 形态和 D4 经济同等重要

    判断矩阵 (Saaty 1-9 标度):
           D1   D2   D3   D4   D5
    D1  [  1    2    1    2    1/2 ]   气候资源
    D2  [ 1/2   1   1/2   1   1/3 ]   城市形态
    D3  [  1    2    1    2    1/2 ]   技术部署
    D4  [ 1/2   1   1/2   1   1/3 ]   经济可行
    D5  [  2    3    2    3    1   ]   确定性/稳健性
    """
    names = ["D1_Climate", "D2_Morphology", "D3_Technical",
             "D4_Economic", "D5_Uncertainty"]

    A = np.array([
        [1,     2,    1,    2,    1/2],
        [1/2,   1,    1/2,  1,    1/3],
        [1,     2,    1,    2,    1/2],
        [1/2,   1,    1/2,  1,    1/3],
        [2,     3,    2,    3,    1  ],
    ])

    return A, names


# ============================================================================
# 3. 组合赋权 (Combined Weighting)
# ============================================================================

def combined_weight(w_entropy: np.ndarray,
                    w_ahp: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    """
    主客观组合赋权。

    w_combined = α × w_entropy + (1 - α) × w_ahp

    Parameters
    ----------
    w_entropy : np.ndarray — 熵权法权重（客观）
    w_ahp : np.ndarray — AHP 权重（主观）
    alpha : float — 客观权重的占比，默认 0.5

    Returns
    -------
    np.ndarray — 组合权重
    """
    assert len(w_entropy) == len(w_ahp)
    assert 0 <= alpha <= 1
    w = alpha * w_entropy + (1 - alpha) * w_ahp
    return w / w.sum()  # 重新归一化（理论上已经归一，保险起见）


def combined_weight_multiplicative(w_entropy: np.ndarray,
                                   w_ahp: np.ndarray) -> np.ndarray:
    """
    乘法组合赋权（备选方案）。

    w_combined_j = (w_entropy_j × w_ahp_j) / Σ(w_entropy × w_ahp)

    优点：同时被两种方法认为重要的指标会被进一步放大。
    """
    w = w_entropy * w_ahp
    return w / w.sum()


# ============================================================================
# 4. 权重敏感性分析
# ============================================================================

def weight_sensitivity_analysis(
    decision_matrix: np.ndarray,
    w_entropy: np.ndarray,
    w_ahp: np.ndarray,
    alpha_range: Optional[np.ndarray] = None,
    indicator_names: Optional[List[str]] = None,
    object_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    权重敏感性分析：扫描 α ∈ [0, 1]，检验 FDSI 排名的稳定性。

    目的：向审稿人证明"结论对权重选择方法不敏感"。

    Parameters
    ----------
    decision_matrix : np.ndarray, shape (n_objects, n_indicators)
        归一化后的决策矩阵（各指标已在 [0,1] 范围内）。
    w_entropy : np.ndarray — 熵权
    w_ahp : np.ndarray — AHP 权重
    alpha_range : np.ndarray, optional
        α 值的扫描范围，默认 [0.0, 0.1, ..., 1.0]
    indicator_names : list, optional
    object_names : list, optional
        评价对象名称（如城市名或"城市×形态"组合名）

    Returns
    -------
    pd.DataFrame
        列: alpha, object_name, fdsi_score, rank
        每行 = 一个 α 值下一个对象的 FDSI 得分和排名
    """
    X = np.array(decision_matrix, dtype=float)
    n_obj, n_ind = X.shape

    if alpha_range is None:
        alpha_range = np.arange(0.0, 1.05, 0.1)  # 0.0, 0.1, ..., 1.0
    if object_names is None:
        object_names = [f"Obj_{i+1}" for i in range(n_obj)]

    records = []
    for alpha in alpha_range:
        w = combined_weight(w_entropy, w_ahp, alpha=alpha)
        scores = X @ w  # 加权求和
        ranks = (-scores).argsort().argsort() + 1  # 排名（1=最好）

        for i in range(n_obj):
            records.append({
                "alpha": round(alpha, 2),
                "object": object_names[i],
                "fdsi_score": round(scores[i], 4),
                "rank": int(ranks[i]),
            })

    df = pd.DataFrame(records)
    return df


def rank_stability_summary(sensitivity_df: pd.DataFrame) -> pd.DataFrame:
    """
    从敏感性分析结果中提取排名稳定性摘要。

    Returns
    -------
    pd.DataFrame with columns:
        object, rank_mean, rank_std, rank_min, rank_max, rank_range,
        most_frequent_rank, rank_change_count
    """
    summary = []
    for obj, grp in sensitivity_df.groupby("object"):
        ranks = grp["rank"].values
        # 排名变化次数（相邻 α 之间排名是否改变）
        changes = np.sum(np.diff(ranks) != 0)
        summary.append({
            "object": obj,
            "rank_mean": round(ranks.mean(), 2),
            "rank_std": round(ranks.std(), 3),
            "rank_min": int(ranks.min()),
            "rank_max": int(ranks.max()),
            "rank_range": int(ranks.max() - ranks.min()),
            "most_frequent_rank": int(pd.Series(ranks).mode().iloc[0]),
            "rank_change_count": int(changes),
        })

    return pd.DataFrame(summary).sort_values("rank_mean")


# ============================================================================
# 5. 便捷接口：一站式赋权流程
# ============================================================================

def run_full_weighting_pipeline(
    decision_matrix: np.ndarray,
    ahp_matrix: np.ndarray,
    indicator_names: List[str],
    object_names: List[str],
    is_benefit: List[bool],
    alpha: float = 0.5,
    run_sensitivity: bool = True,
) -> Dict:
    """
    完整赋权流程：熵权 → AHP → 组合 → (可选)敏感性分析。

    Parameters
    ----------
    decision_matrix : 原始决策矩阵（未归一化）
    ahp_matrix : AHP 判断矩阵
    indicator_names : 指标名称
    object_names : 评价对象名称
    is_benefit : 各指标是否为正向
    alpha : 组合赋权中客观权重占比
    run_sensitivity : 是否执行敏感性分析

    Returns
    -------
    dict with all intermediate and final results
    """
    log.info("=" * 50)
    log.info("FDSI 赋权流程启动")
    log.info("=" * 50)

    # ── 熵权法 ──
    log.info("Step 1: 熵权法（客观权重）")
    entropy_result = entropy_weight(
        decision_matrix, indicator_names, is_benefit
    )
    log.info(f"  熵权: {dict(zip(indicator_names, entropy_result['weights'].round(4)))}")

    # ── AHP ──
    log.info("Step 2: AHP（主观权重）")
    ahp_result = ahp_weight(ahp_matrix, indicator_names)
    log.info(f"  AHP 权重: {dict(zip(indicator_names, ahp_result['weights'].round(4)))}")
    log.info(f"  CR = {ahp_result['CR']:.4f} ({'通过' if ahp_result['is_consistent'] else '未通过'})")

    # ── 组合赋权 ──
    log.info(f"Step 3: 组合赋权 (α={alpha})")
    w_combined = combined_weight(
        entropy_result["weights"], ahp_result["weights"], alpha
    )
    log.info(f"  组合权重: {dict(zip(indicator_names, w_combined.round(4)))}")

    # ── 归一化决策矩阵 ──
    X = np.array(decision_matrix, dtype=float)
    X_norm = np.zeros_like(X)
    for j in range(X.shape[1]):
        col = X[:, j]
        col_min, col_max = col.min(), col.max()
        if col_max == col_min:
            X_norm[:, j] = 0.5
        elif is_benefit[j]:
            X_norm[:, j] = (col - col_min) / (col_max - col_min)
        else:
            X_norm[:, j] = (col_max - col) / (col_max - col_min)

    # ── 计算 FDSI ──
    fdsi_scores = X_norm @ w_combined
    ranks = (-fdsi_scores).argsort().argsort() + 1

    fdsi_df = pd.DataFrame({
        "object": object_names,
        "fdsi_score": fdsi_scores.round(4),
        "rank": ranks.astype(int),
    }).sort_values("rank")

    log.info("Step 4: FDSI 评分结果")
    log.info(f"\n{fdsi_df.to_string(index=False)}")

    result = {
        "entropy": entropy_result,
        "ahp": ahp_result,
        "combined_weights": w_combined,
        "alpha": alpha,
        "normalized_matrix": X_norm,
        "fdsi_scores": fdsi_scores,
        "fdsi_df": fdsi_df,
    }

    # ── 敏感性分析 ──
    if run_sensitivity:
        log.info("Step 5: 权重敏感性分析 (α = 0.0 ~ 1.0)")
        sens_df = weight_sensitivity_analysis(
            X_norm, entropy_result["weights"], ahp_result["weights"],
            object_names=object_names,
        )
        stability = rank_stability_summary(sens_df)
        log.info(f"  排名稳定性摘要:\n{stability.to_string(index=False)}")

        result["sensitivity_df"] = sens_df
        result["stability_summary"] = stability

    return result


# ============================================================================
# Demo / 测试
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 60)
    print("FDSI 赋权模块 — 演示")
    print("=" * 60)

    # 模拟数据：5个城市 × 5个维度
    cities = ["哈尔滨", "北京", "长沙", "深圳", "昆明"]
    dims = ["D1_Climate", "D2_Morphology", "D3_Technical",
            "D4_Economic", "D5_Uncertainty"]
    is_benefit = [True, True, True, True, True]  # D5 确定性越高越好

    # 模拟决策矩阵（示意，非真实数据）
    np.random.seed(42)
    X = np.array([
        [900,  0.45, 120, 0.55, 0.70],   # 哈尔滨：GHI低，密度中，确定性高
        [1100, 0.50, 140, 0.60, 0.75],   # 北京：GHI中高
        [1000, 0.55, 130, 0.65, 0.60],   # 长沙：梅雨影响确定性
        [1150, 0.65, 110, 0.70, 0.55],   # 深圳：GHI高但高密度
        [1300, 0.40, 150, 0.50, 0.80],   # 昆明：GHI最高，确定性最好
    ])

    # AHP 判断矩阵
    A_ahp, _ = get_default_ahp_matrix_d1_d5()

    # 一站式运行
    result = run_full_weighting_pipeline(
        decision_matrix=X,
        ahp_matrix=A_ahp,
        indicator_names=dims,
        object_names=cities,
        is_benefit=is_benefit,
        alpha=0.5,
        run_sensitivity=True,
    )

    print("\n✓ 演示完成")
