# -*- coding: utf-8 -*-
"""
公共模型定义
用于各个迁移学习模型的共享组件
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler


class GENet:
    """
    广义弹性网络 (GENet) 实现
    用于软迁移学习，通过 v 参数控制对源域参数的依赖程度
    """

    def __init__(self, v=1.0, alpha=1e-3, l1_ratio=0.5, fit_intercept=True,
                 random_state=42, max_iter=5000):
        self.v = float(v)
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.max_iter = max_iter
        self.base = None
        self.theta0_vec = None
        self.coef_ = None
        self.intercept_ = None
        self.scaler = StandardScaler()

    def fit(self, X, y, theta0_vec):
        """
        训练 GENet 模型

        Parameters:
        - X: 特征矩阵
        - y: 目标变量
        - theta0_vec: 源域参数 θ₀
        """
        X_np = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        y_np = y.to_numpy().ravel() if hasattr(y, "to_numpy") else np.ravel(y)
        X_scaled = self.scaler.fit_transform(X_np)
        theta0_vec = np.asarray(theta0_vec, dtype=float)
        self.theta0_vec = theta0_vec

        # 残差目标：y' = y − X_std(vθ0)
        y_resid = y_np - X_scaled.dot(self.v * theta0_vec)

        self.base = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
            random_state=self.random_state,
            max_iter=self.max_iter
        )
        self.base.fit(X_scaled, y_resid)

        # 还原 θ̂ = φ̂ + vθ0
        self.coef_ = self.base.coef_ + self.v * theta0_vec
        self.intercept_ = self.base.intercept_
        return self

    def predict(self, X):
        """
        使用训练好的模型进行预测

        Parameters:
        - X: 特征矩阵

        Returns:
        - 预测值
        """
        X_np = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        X_scaled = self.scaler.transform(X_np)
        return X_scaled.dot(self.coef_) + self.intercept_


def load_theta0_vector(theta_path, feature_names_target):
    """
    从保存的参数文件中加载 θ₀ 向量

    Parameters:
    - theta_path: 参数文件路径
    - feature_names_target: 目标域特征名称列表

    Returns:
    - theta0_vec: θ₀ 向量
    """
    import joblib
    theta_obj = joblib.load(theta_path)
    theta_map = {}
    if isinstance(theta_obj, dict):
        if "theta" in theta_obj and "feature_names" in theta_obj:
            theta_map = dict(zip(theta_obj["feature_names"], np.asarray(theta_obj["theta"], dtype=float)))
        elif "coef_" in theta_obj and "feature_names" in theta_obj:
            theta_map = dict(zip(theta_obj["feature_names"], np.asarray(theta_obj["coef_"], dtype=float)))
    # 对齐为向量
    return np.array([theta_map.get(c, 0.0) for c in feature_names_target], dtype=float)


def align_columns_like(X_target, columns_ref):
    """
    对齐目标域数据列到参考列顺序，缺失列填充0

    Parameters:
    - X_target: 目标域数据
    - columns_ref: 参考列名列表

    Returns:
    - 对齐后的数据
    """
    X_aligned = X_target.copy()
    for col in columns_ref:
        if col not in X_aligned.columns:
            X_aligned[col] = 0.0
    return X_aligned[columns_ref]


def soft_genet_grid_search(X_train, y_train, theta0_vec, v_grid, alpha_grid, l1_grid,
                           n_splits=10, metric="r2"):
    """
    GENet 网格搜索超参数调优

    Parameters:
    - X_train: 训练特征
    - y_train: 训练目标
    - theta0_vec: 源域参数 θ₀
    - v_grid: v 参数搜索网格
    - alpha_grid: alpha 参数搜索网格
    - l1_grid: l1_ratio 参数搜索网格
    - n_splits: 交叉验证折数
    - metric: 评估指标 ("r2" 或 "mse")

    Returns:
    - final: 最优模型
    - best: 最优参数和分数
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import r2_score, mean_squared_error

    ts = TimeSeriesSplit(n_splits=n_splits)
    # r2 越大越好；MSE 越小越好
    best = {"score": -np.inf if metric == "r2" else np.inf, "v": None, "alpha": None, "l1_ratio": None}
    for v in v_grid:
        for alpha in alpha_grid:
            for l1 in l1_grid:
                scores = []
                for tr_idx, va_idx in ts.split(np.arange(len(X_train))):
                    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
                    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
                    model = GENet(v=v, alpha=alpha, l1_ratio=l1)
                    model.fit(X_tr, y_tr, theta0_vec)
                    y_hat = model.predict(X_va)
                    if metric == "r2":
                        scores.append(r2_score(y_va, y_hat))
                    else:
                        scores.append(mean_squared_error(y_va, y_hat))
                mean_score = float(np.mean(scores))
                is_better = (metric == "r2" and mean_score > best["score"]) or (metric != "r2" and mean_score < best["score"])
                if is_better:
                    best.update({"score": mean_score, "v": v, "alpha": alpha, "l1_ratio": l1})
    # 用最优参数拟合全训练集
    final = GENet(v=best["v"], alpha=best["alpha"], l1_ratio=best["l1_ratio"])
    final.fit(X_train, y_train, theta0_vec)
    return final, best