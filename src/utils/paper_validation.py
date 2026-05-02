# -*- coding: utf-8 -*-
"""
GENet 论文验证工具
实现论文《How Global is Predictability? The Power of Financial Transfer Learning》中的评估指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats


class PredictiveR2:
    """
    预测性 R² 计算

    论文定义：
    R²_pred = 1 - MSE_model / MSE_baseline

    其中 baseline 通常是历史均值模型
    """

    def __init__(self, baseline: str = "historical_mean"):
        """
        参数：
        - baseline: 基准模型类型
          - "historical_mean": 历史均值模型
          - "naive": 朴素模型（预测0）
          - "market_mean": 市场均值模型
        """
        self.baseline = baseline
        self.baseline_predictions = None

    def fit_baseline(self, y_train: np.ndarray, dates_train: Optional[np.ndarray] = None):
        """拟合基准模型"""
        if self.baseline == "historical_mean":
            # 使用历史均值作为预测
            self.baseline_predictions = np.mean(y_train)
        elif self.baseline == "market_mean":
            # 使用市场均值（需要按日期分组）
            if dates_train is None:
                raise ValueError("market_mean 需要提供日期信息")
            df = pd.DataFrame({'y': y_train, 'date': dates_train})
            self.baseline_predictions = df.groupby('date')['y'].mean().to_dict()
        elif self.baseline == "naive":
            # 朴素模型：预测为0
            self.baseline_predictions = 0.0

    def predict_baseline(self, dates: Optional[np.ndarray] = None) -> np.ndarray:
        """生成基准预测"""
        if self.baseline == "historical_mean" or self.baseline == "naive":
            return np.full_like(np.zeros(1) if dates is None else dates,
                               self.baseline_predictions)
        elif self.baseline == "market_mean":
            if dates is None:
                raise ValueError("market_mean 需要提供日期信息")
            return np.array([self.baseline_predictions.get(d, 0.0) for d in dates])

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 dates: Optional[np.ndarray] = None) -> float:
        """
        计算预测性 R²

        参数：
        - y_true: 真实收益
        - y_pred: 模型预测
        - dates: 日期（用于 market_mean baseline）

        返回：
        - 预测性 R²
        """
        # 确保没有 NaN
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return np.nan

        # 计算模型 MSE
        mse_model = np.mean((y_true - y_pred) ** 2)

        # 计算基准 MSE
        if self.baseline_predictions is None:
            # 如果没有预先拟合，使用当前数据的均值
            y_baseline_pred = np.mean(y_true)
        else:
            y_baseline_pred = self.predict_baseline(dates[mask] if dates is not None else None)

        mse_baseline = np.mean((y_true - y_baseline_pred) ** 2)

        # 计算预测性 R²
        pred_r2 = 1 - mse_model / (mse_baseline + 1e-10)

        return pred_r2

    def calculate_rolling(self, y_true: np.ndarray, y_pred: np.ndarray,
                         window: int = 12, dates: Optional[np.ndarray] = None) -> pd.Series:
        """
        计算滚动预测性 R²

        参数：
        - y_true: 真实收益
        - y_pred: 模型预测
        - window: 滚动窗口大小
        - dates: 日期

        返回：
        - 滚动预测性 R² 序列
        """
        if len(y_true) < window:
            return pd.Series([], dtype=float)

        rolling_r2 = []

        for i in range(window, len(y_true) + 1):
            y_true_window = y_true[i-window:i]
            y_pred_window = y_pred[i-window:i]

            # 使用窗口前期的均值作为 baseline
            baseline_pred = np.mean(y_true_window) if i > window else 0.0

            mse_model = np.mean((y_true_window - y_pred_window) ** 2)
            mse_baseline = np.mean((y_true_window - baseline_pred) ** 2)

            r2 = 1 - mse_model / (mse_baseline + 1e-10)
            rolling_r2.append(r2)

        return pd.Series(rolling_r2)


class GENetPaperValidator:
    """
    GENet 论文验证工具

    验证论文中的关键发现：
    1. 全局成分占比约 94%
    2. 局部参数的收缩特性
    3. 预测性 R² 的提升
    """

    def __init__(self):
        self.results = {}

    def validate_global_dominance(self, theta_g: np.ndarray,
                                   theta_ell_c: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        验证全局成分的主导地位

        论文发现：全局成分占比约 94%

        参数：
        - theta_g: 全局参数
        - theta_ell_c: 各市场局部参数

        返回：
        - 验证结果字典
        """
        results = {}

        # 全局参数统计
        results['theta_g_nonzero_ratio'] = np.sum(np.abs(theta_g) > 1e-6) / len(theta_g)
        results['theta_g_l1_norm'] = np.sum(np.abs(theta_g))

        # 局部参数统计
        for market, theta_ell in theta_ell_c.items():
            nonzero_ratio = np.sum(np.abs(theta_ell) > 1e-6) / len(theta_ell)
            l1_norm = np.sum(np.abs(theta_ell))
            shrinkage_ratio = l1_norm / (results['theta_g_l1_norm'] + 1e-10)

            results[f'{market}_nonzero_ratio'] = nonzero_ratio
            results[f'{market}_l1_norm'] = l1_norm
            results[f'{market}_shrinkage_ratio'] = shrinkage_ratio

        # 验证全局主导地位
        avg_shrinkage = np.mean([results[f'{m}_shrinkage_ratio']
                                for m in theta_ell_c.keys()])
        results['avg_shrinkage_ratio'] = avg_shrinkage

        # 判断是否符合论文预期（全局参数远大于局部参数）
        results['global_dominance_verified'] = avg_shrinkage < 0.2

        return results

    def validate_shrinkage_mechanism(self, X_dict: Dict[str, np.ndarray],
                                     y_dict: Dict[str, np.ndarray],
                                     theta_g: np.ndarray,
                                     theta_ell_c: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        验证收缩机制

        检查：
        1. 局部参数在数据不足时是否收缩到 0
        2. 不同市场的收缩程度是否合理

        参数：
        - X_dict: 各市场特征矩阵
        - y_dict: 各市场标签
        - theta_g: 全局参数
        - theta_ell_c: 局部参数

        返回：
        - 验证结果
        """
        results = {}

        for market in X_dict.keys():
            X = X_dict[market]
            y = y_dict[market]
            theta_ell = theta_ell_c[market]

            n_samples = len(X)

            # 数据量
            results[f'{market}_n_samples'] = n_samples

            # 局部参数的 L1 范数
            results[f'{market}_theta_ell_l1'] = np.sum(np.abs(theta_ell))

            # 如果数据量较少，检查是否充分收缩
            if n_samples < 1000:
                # 期望收缩比例 < 0.1
                shrinkage = results[f'{market}_theta_ell_l1'] / (np.sum(np.abs(theta_g)) + 1e-10)
                results[f'{market}_adequate_shrinkage'] = shrinkage < 0.1

        return results

    def validate_predictive_improvement(self, y_true: np.ndarray,
                                       y_pred_global: np.ndarray,
                                       y_pred_joint: np.ndarray,
                                       dates: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        验证预测性提升

        比较纯全局模型 vs 联合模型的预测性 R²

        参数：
        - y_true: 真实收益
        - y_pred_global: 纯全局模型预测
        - y_pred_joint: 联合模型预测
        - dates: 日期

        返回：
        - 验证结果
        """
        results = {}

        # 计算预测性 R²
        r2_calculator = PredictiveR2(baseline="historical_mean")
        r2_calculator.fit_baseline(y_true)

        results['pred_r2_global'] = r2_calculator.calculate(y_true, y_pred_global, dates)
        results['pred_r2_joint'] = r2_calculator.calculate(y_true, y_pred_joint, dates)
        results['pred_r2_improvement'] = results['pred_r2_joint'] - results['pred_r2_global']

        # 计算传统 R²
        results['r2_global'] = self._calculate_traditional_r2(y_true, y_pred_global)
        results['r2_joint'] = self._calculate_traditional_r2(y_true, y_pred_joint)

        # 统计显著性检验
        results['significance_test'] = self._paired_t_test(
            y_true, y_pred_global, y_pred_joint
        )

        return results

    def _calculate_traditional_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算传统 R²"""
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return np.nan

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        return 1 - ss_res / (ss_tot + 1e-10)

    def _paired_t_test(self, y_true: np.ndarray, y_pred1: np.ndarray,
                      y_pred2: np.ndarray) -> Dict[str, float]:
        """
        配对 t 检验

        检验两个模型的预测误差是否存在显著差异
        """
        errors1 = y_true - y_pred1
        errors2 = y_true - y_pred2
        diff = errors1 - errors2

        # 移除 NaN
        diff = diff[np.isfinite(diff)]

        if len(diff) < 2:
            return {'statistic': np.nan, 'p_value': np.nan}

        statistic, p_value = stats.ttest_rel(diff, np.zeros_like(diff))

        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

    def generate_validation_report(self, global_dominance: Dict,
                                   shrinkage: Dict,
                                   predictive: Dict) -> str:
        """
        生成验证报告

        参数：
        - global_dominance: 全局主导性验证结果
        - shrinkage: 收缩机制验证结果
        - predictive: 预测性提升验证结果

        返回：
        - 报告字符串
        """
        report = []
        report.append("=" * 80)
        report.append("GENet 论文验证报告")
        report.append("=" * 80)

        # 全局主导性
        report.append("\n【1. 全局主导性验证】")
        report.append(f"全局参数非零系数比例: {global_dominance['theta_g_nonzero_ratio']:.2%}")
        report.append(f"全局参数 L1 范数: {global_dominance['theta_g_l1_norm']:.6f}")
        report.append(f"平均收缩比例: {global_dominance['avg_shrinkage_ratio']:.3f}")

        if global_dominance.get('global_dominance_verified', False):
            report.append("✓ 验证通过：全局成分占主导地位（符合论文预期）")
        else:
            report.append("✗ 验证失败：全局成分未占主导地位")

        # 收缩机制
        report.append("\n【2. 收缩机制验证】")
        for market in [k.replace('_n_samples', '') for k in shrinkage.keys() if '_n_samples' in k]:
            if f'{market}_n_samples' in shrinkage:
                n_samples = shrinkage[f'{market}_n_samples']
                theta_ell_l1 = shrinkage.get(f'{market}_theta_ell_l1', 0)
                adequate = shrinkage.get(f'{market}_adequate_shrinkage', True)

                report.append(f"{market}:")
                report.append(f"  样本数: {n_samples}")
                report.append(f"  局部参数 L1: {theta_ell_l1:.6f}")
                report.append(f"  收缩充分: {'✓' if adequate else '✗'}")

        # 预测性提升
        report.append("\n【3. 预测性提升验证】")
        report.append(f"纯全局模型预测性 R²: {predictive.get('pred_r2_global', 0):.6f}")
        report.append(f"联合模型预测性 R²: {predictive.get('pred_r2_joint', 0):.6f}")
        report.append(f"预测性 R² 提升: {predictive.get('pred_r2_improvement', 0):.6f}")
        report.append(f"传统 R² (全局): {predictive.get('r2_global', 0):.6f}")
        report.append(f"传统 R² (联合): {predictive.get('r2_joint', 0):.6f}")

        sig_test = predictive.get('significance_test', {})
        if sig_test.get('significant', False):
            report.append(f"✓ 预测差异显著 (p={sig_test.get('p_value', 1):.4f})")
        else:
            report.append(f"✗ 预测差异不显著 (p={sig_test.get('p_value', 1):.4f})")

        # 总体结论
        report.append("\n【4. 总体结论】")
        global_verified = global_dominance.get('global_dominance_verified', False)
        improvement = predictive.get('pred_r2_improvement', 0) > 0

        if global_verified and improvement:
            report.append("✓ 验证成功：模型符合论文核心发现")
        elif global_verified:
            report.append("⚠ 部分验证：全局主导性符合，但预测提升不明显")
        else:
            report.append("✗ 验证失败：模型不符合论文核心发现")

        report.append("=" * 80)

        return "\n".join(report)


def calculate_predictive_r2(y_true: np.ndarray, y_pred: np.ndarray,
                           baseline: str = "historical_mean",
                           dates: Optional[np.ndarray] = None) -> float:
    """
    便捷函数：计算预测性 R²

    参数：
    - y_true: 真实收益
    - y_pred: 模型预测
    - baseline: 基准模型类型
    - dates: 日期（用于 market_mean baseline）

    返回：
    - 预测性 R²
    """
    calculator = PredictiveR2(baseline=baseline)
    return calculator.calculate(y_true, y_pred, dates)


import warnings
from typing import Dict