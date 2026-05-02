# -*- coding: utf-8 -*-
"""
GENet (Generalized Elastic Net) Implementation
Paper: How Global is Predictability? The Power of Financial Transfer Learning

Core optimization problem:
min_{theta_g, theta_ell,c} sum_c ||Yc - Xc(theta_g + theta_ell,c)||^2 + lambda_g||theta_g||_1 + sum_c lambda_ell,c||theta_ell,c||_1

where beta_c = theta_g + theta_ell,c
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, List, Tuple, Optional
import warnings


class JointGENet:
    """
    Joint optimization GENet implementation (matches paper)

    Core ideas:
    1. Jointly estimate global parameters theta_g and market-specific local parameters theta_ell,c
    2. Global parameters have lighter penalty (lambda_g < lambda_ell,c), encouraging use of global information
    3. Local parameters shrink to 0 when data is insufficient
    """

    def __init__(self, lambda_g: float = 0.001, lambda_l_ratio: float = 10.0,
                 max_iter: int = 100, tol: float = 1e-4, fit_intercept: bool = True,
                 verbose: bool = True):
        """
        Parameters:
        - lambda_g: L1 penalty coefficient for global parameters (usually small)
        - lambda_l_ratio: Ratio of local to global penalty coefficient (lambda_ell = lambda_g * ratio)
                         Paper suggests lambda_ell > lambda_g, default ratio = 10
        - max_iter: Maximum number of iterations
        - tol: Convergence tolerance
        - fit_intercept: Whether to fit intercept
        - verbose: Whether to print training information
        """
        self.lambda_g = lambda_g
        self.lambda_l_ratio = lambda_l_ratio
        self.lambda_l = lambda_g * lambda_l_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.verbose = verbose

        # 模型参数
        self.theta_g = None  # 全局参数
        self.theta_ell_c = {}  # 各市场的局部参数
        self.intercept_g = 0.0  # 全局截距
        self.intercept_ell_c = {}  # 各市场的局部截距
        self.feature_names = None  # 特征名称

        # 训练统计
        self.convergence_history = []
        self.global_contribution_ratio = None

    def fit(self, X_dict: Dict[str, np.ndarray], y_dict: Dict[str, np.ndarray],
            feature_names: Optional[List[str]] = None) -> 'JointGENet':
        """
        训练联合GENet模型

        参数：
        - X_dict: 各市场的特征矩阵，格式 {'market_name': X}
        - y_dict: 各市场的目标变量，格式 {'market_name': y}
        - feature_names: 特征名称列表

        返回：
        - self
        """
        markets = list(X_dict.keys())
        n_features = list(X_dict.values())[0].shape[1]
        self.feature_names = feature_names or [f'f{i}' for i in range(n_features)]

        if self.verbose:
            print("=" * 60)
            print("=== GENet Joint Optimization Training ===")
            print(f"Number of markets: {len(markets)}")
            print(f"Feature dimensions: {n_features}")
            print(f"Global penalty lambda_g: {self.lambda_g}")
            print(f"Local penalty lambda_l: {self.lambda_l} (ratio = {self.lambda_l_ratio})")
            print("=" * 60)

        # 初始化参数
        self.theta_g = np.zeros(n_features)
        self.intercept_g = 0.0
        for market in markets:
            self.theta_ell_c[market] = np.zeros(n_features)
            self.intercept_ell_c[market] = 0.0

        # 交替优化
        for iteration in range(self.max_iter):
            theta_g_old = self.theta_g.copy()

            # 步骤1：更新全局参数 theta_g（固定所有 theta_ell,c）
            self._update_global_parameters(X_dict, y_dict, markets)

            # 步骤2：更新各市场的局部参数 theta_ell,c（固定 theta_g）
            for market in markets:
                self._update_local_parameters(X_dict[market], y_dict[market], market)

            # 计算收敛性
            delta = np.linalg.norm(self.theta_g - theta_g_old) / (np.linalg.norm(theta_g_old) + 1e-10)
            self.convergence_history.append(delta)

            if self.verbose and (iteration % 10 == 0 or iteration == self.max_iter - 1):
                print(f"Iteration {iteration + 1}/{self.max_iter}, Δ = {delta:.6f}")

            if delta < self.tol:
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

        # 计算全局成分贡献比例（验证论文关键发现）
        self.global_contribution_ratio = self._compute_global_contribution(X_dict, y_dict)

        if self.verbose:
            self._print_training_summary()

        return self

    def _update_global_parameters(self, X_dict: Dict[str, np.ndarray],
                                  y_dict: Dict[str, np.ndarray], markets: List[str]):
        """更新全局参数 theta_g"""
        # 构造联合数据：所有市场的残差
        X_combined = []
        y_combined = []

        for market in markets:
            theta_ell = self.theta_ell_c[market]
            intercept_ell = self.intercept_ell_c[market]

            # 移除局部贡献：y' = y - Xtheta_ell,c - bℓ,c
            y_resid = y_dict[market] - X_dict[market] @ theta_ell - intercept_ell
            X_combined.append(X_dict[market])
            y_combined.append(y_resid)

        X_combined = np.vstack(X_combined)
        y_combined = np.concatenate(y_combined)

        # 对 theta_g 进行Lasso回归（惩罚更轻）
        lasso_g = Lasso(alpha=self.lambda_g, max_iter=5000, fit_intercept=self.fit_intercept)
        lasso_g.fit(X_combined, y_combined)

        self.theta_g = lasso_g.coef_
        if self.fit_intercept:
            self.intercept_g = lasso_g.intercept_

    def _update_local_parameters(self, X_market: np.ndarray, y_market: np.ndarray,
                                 market: str):
        """更新单个市场的局部参数 theta_ell,c"""
        # 移除全局贡献：y' = y - Xtheta_g - bg
        y_resid = y_market - X_market @ self.theta_g - self.intercept_g

        # 对 theta_ell,c 进行Lasso回归（惩罚更重，促进收缩）
        lasso_l = Lasso(alpha=self.lambda_l, max_iter=5000, fit_intercept=self.fit_intercept)
        lasso_l.fit(X_market, y_resid)

        self.theta_ell_c[market] = lasso_l.coef_
        if self.fit_intercept:
            self.intercept_ell_c[market] = lasso_l.intercept_

    def _compute_global_contribution(self, X_dict: Dict[str, np.ndarray],
                                     y_dict: Dict[str, np.ndarray]) -> float:
        """
        计算全局成分的贡献比例

        论文发现：全局成分占比约94%
        """
        total_variance = 0.0
        global_variance = 0.0
        n_total = 0

        for market, X in X_dict.items():
            y = y_dict[market]
            n = len(y)
            n_total += n

            # 使用完整参数预测
            beta_total = self.theta_g + self.theta_ell_c[market]
            intercept_total = self.intercept_g + self.intercept_ell_c[market]
            pred_total = X @ beta_total + intercept_total
            resid_total = y - pred_total

            # 仅使用全局参数预测
            pred_global = X @ self.theta_g + self.intercept_g
            resid_global = y - pred_global

            total_variance += np.sum(resid_total ** 2)
            global_variance += np.sum(resid_global ** 2)

        # 全局贡献比例 = 1 - (全局模型残差方差 / 完整模型残差方差)
        # 比例越高，说明局部调整带来的改进越小，全局模型越重要
        ratio = 1 - (global_variance / (total_variance + 1e-10))
        return ratio

    def _print_training_summary(self):
        """打印训练摘要"""
        print("\n" + "=" * 60)
        print("=== GENet 训练摘要 ===")

        # 全局参数统计
        nonzero_g = np.sum(np.abs(self.theta_g) > 1e-6)
        l1_norm_g = np.sum(np.abs(self.theta_g))

        print(f"\n全局参数 theta_g:")
        print(f"  非零系数数量: {nonzero_g} / {len(self.theta_g)} ({nonzero_g/len(self.theta_g)*100:.1f}%)")
        print(f"  L1 范数: {l1_norm_g:.6f}")
        print(f"  最大绝对值: {np.max(np.abs(self.theta_g)):.6f}")

        # 局部参数统计
        print(f"\n局部参数 theta_ell,c:")
        for market, theta_ell in self.theta_ell_c.items():
            nonzero_l = np.sum(np.abs(theta_ell) > 1e-6)
            l1_norm_l = np.sum(np.abs(theta_ell))
            print(f"  {market}:")
            print(f"    非零系数数量: {nonzero_l} / {len(theta_ell)} ({nonzero_l/len(theta_ell)*100:.1f}%)")
            print(f"    L1 范数: {l1_norm_l:.6f}")
            print(f"    收缩比 (||theta_ell|| / ||theta_g||): {l1_norm_l / (l1_norm_g + 1e-10):.3f}")

        # 全局贡献比例
        print(f"\n全局成分贡献比例: {self.global_contribution_ratio * 100:.2f}%")
        if self.global_contribution_ratio >= 0.9:
            print("  [PASS] 验证通过：全局成分占主导地位（论文发现约94%）")
        else:
            print("  [WARNING] 警告：全局成分占比低于论文预期")

        # 惩罚系数比例
        print(f"\n惩罚系数配置:")
        print(f"  lambda_g (全局): {self.lambda_g}")
        print(f"  lambda_ell (局部): {self.lambda_l}")
        print(f"  比例 (lambda_ell/lambda_g): {self.lambda_l_ratio:.1f}x")

        print("=" * 60)

    def predict(self, X: np.ndarray, market: str) -> np.ndarray:
        """
        预测特定市场的收益

        参数：
        - X: 特征矩阵
        - market: 市场名称

        返回：
        - 预测值
        """
        if market not in self.theta_ell_c:
            raise ValueError(f"市场 {market} 不在训练数据中")

        beta = self.theta_g + self.theta_ell_c[market]
        intercept = self.intercept_g + self.intercept_ell_c[market]

        return X @ beta + intercept

    def get_market_coefficients(self, market: str) -> np.ndarray:
        """获取指定市场的完整系数"""
        return self.theta_g + self.theta_ell_c[market]

    def get_global_contribution(self, X: np.ndarray) -> np.ndarray:
        """获取全局成分的预测"""
        return X @ self.theta_g + self.intercept_g

    def get_local_contribution(self, X: np.ndarray, market: str) -> np.ndarray:
        """获取局部成分的预测"""
        return X @ self.theta_ell_c[market] + self.intercept_ell_c[market]


class GENetRegressor:
    """
    GENet 回归器（向后兼容接口）

    包装 JointGENet 提供类似 sklearn 的接口
    """

    def __init__(self, lambda_g: float = 0.001, lambda_l_ratio: float = 10.0,
                 max_iter: int = 100, tol: float = 1e-4, fit_intercept: bool = True,
                 verbose: bool = True):
        self.genet = JointGENet(
            lambda_g=lambda_g,
            lambda_l_ratio=lambda_l_ratio,
            max_iter=max_iter,
            tol=tol,
            fit_intercept=fit_intercept,
            verbose=verbose
        )

    def fit(self, X_train_dict: Dict[str, pd.DataFrame],
            y_train_dict: Dict[str, pd.Series],
            feature_names: Optional[List[str]] = None) -> 'GENetRegressor':
        """训练模型"""
        # 转换为 numpy 数组
        X_dict = {k: v.values if hasattr(v, 'values') else v for k, v in X_train_dict.items()}
        y_dict = {k: v.values.ravel() if hasattr(v, 'values') else v for k, v in y_train_dict.items()}

        self.genet.fit(X_dict, y_dict, feature_names)
        return self

    def predict(self, X: pd.DataFrame, market: str) -> np.ndarray:
        """预测"""
        X_np = X.values if hasattr(X, 'values') else X
        return self.genet.predict(X_np, market)

    @property
    def theta_g(self) -> np.ndarray:
        return self.genet.theta_g

    @property
    def theta_ell_c(self) -> Dict[str, np.ndarray]:
        return self.genet.theta_ell_c

    @property
    def feature_names(self) -> Optional[List[str]]:
        return self.genet.feature_names


def genet_grid_search(X_train_dict: Dict[str, pd.DataFrame],
                     y_train_dict: Dict[str, pd.Series],
                     lambda_g_grid: List[float] = None,
                     lambda_l_ratio_grid: List[float] = None,
                     n_splits: int = 5,
                     metric: str = "r2",
                     verbose: bool = True) -> Tuple[GENetRegressor, Dict]:
    """
    GENet 网格搜索超参数调优

    参数：
    - X_train_dict: 各市场训练数据
    - y_train_dict: 各市场标签
    - lambda_g_grid: 全局惩罚系数搜索网格
    - lambda_l_ratio_grid: 局部惩罚比例搜索网格
    - n_splits: 交叉验证折数
    - metric: 评估指标 ("r2" 或 "mse")
    - verbose: 是否打印详细信息

    返回：
    - best_model: 最优模型
    - best_params: 最优参数和分数
    """
    if lambda_g_grid is None:
        lambda_g_grid = [0.0001, 0.001, 0.01, 0.1]
    if lambda_l_ratio_grid is None:
        lambda_l_ratio_grid = [5.0, 10.0, 20.0, 50.0]

    from sklearn.model_selection import TimeSeriesSplit

    best_score = -np.inf if metric == "r2" else np.inf
    best_params = None
    best_model = None

    markets = list(X_train_dict.keys())

    # 找到最小样本数，确保所有市场都能参与交叉验证
    min_samples = min(len(X_train_dict[market]) for market in markets)

    # 调整交叉验证折数以适应最小样本数
    max_splits = min_samples // 30  # 每折至少30个训练样本
    n_splits = min(n_splits, max_splits)

    if n_splits < 2:
        n_splits = 2  # 至少2折

    # 创建虚拟数据用于生成交叉验证索引
    dummy_data = np.zeros((min_samples, 1))
    ts = TimeSeriesSplit(n_splits=n_splits)

    if verbose:
        print("=" * 60)
        print("=== GENet Grid Search ===")
        print(f"Search space: lambda_g in {lambda_g_grid}, lambda_ell/lambda_g in {lambda_l_ratio_grid}")
        print(f"Cross-validation: {n_splits} folds")
        print(f"Minimum samples across markets: {min_samples}")
        print("=" * 60)

    total_combinations = len(lambda_g_grid) * len(lambda_l_ratio_grid)
    combination = 0

    for lambda_g in lambda_g_grid:
        for lambda_l_ratio in lambda_l_ratio_grid:
            combination += 1
            if verbose:
                print(f"\n[{combination}/{total_combinations}] lambda_g={lambda_g}, lambda_ell/lambda_g={lambda_l_ratio}")

            cv_scores = []

            for fold, (train_idx, val_idx) in enumerate(ts.split(dummy_data)):
                # 构造训练和验证集
                X_train_fold = {}
                y_train_fold = {}
                X_val_fold = {}
                y_val_fold = {}

                for market in markets:
                    X = X_train_dict[market].values
                    y = y_train_dict[market].values
                    X_train_fold[market] = X[train_idx]
                    y_train_fold[market] = y[train_idx]
                    X_val_fold[market] = X[val_idx]
                    y_val_fold[market] = y[val_idx]

                # 训练模型
                model = GENetRegressor(lambda_g=lambda_g, lambda_l_ratio=lambda_l_ratio,
                                     max_iter=50, verbose=False)
                model.fit(X_train_fold, y_train_fold)

                # 在验证集上评估
                fold_scores = []
                for market in markets:
                    y_pred = model.predict(X_val_fold[market], market)
                    y_true = y_val_fold[market]

                    if metric == "r2":
                        score = r2_score(y_true, y_pred)
                    else:
                        score = -mean_squared_error(y_true, y_pred)

                    fold_scores.append(score)

                cv_scores.append(np.mean(fold_scores))

            mean_score = float(np.mean(cv_scores))
            is_better = (metric == "r2" and mean_score > best_score) or \
                       (metric != "r2" and mean_score < best_score)

            if verbose:
                print(f"  CV {metric.upper()}: {mean_score:.6f}")

            if is_better:
                best_score = mean_score
                best_params = {
                    'lambda_g': lambda_g,
                    'lambda_l_ratio': lambda_l_ratio,
                    'lambda_l': lambda_g * lambda_l_ratio,
                    'score': mean_score
                }

    # 用最优参数训练全量数据
    if verbose:
        print("\n" + "=" * 60)
        print("=== 最优参数 ===")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print("=" * 60)

    best_model = GENetRegressor(
        lambda_g=best_params['lambda_g'],
        lambda_l_ratio=best_params['lambda_l_ratio'],
        verbose=verbose
    )
    best_model.fit(X_train_dict, y_train_dict)

    return best_model, best_params