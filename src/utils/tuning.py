# -*- coding: utf-8 -*-
"""
ElasticNet 时间序列超参数调优工具
- 使用按年份逐步扩展训练集的自定义 CV（YearlyExpandingCV）
- 管线：StandardScaler + ElasticNet
- 网格搜索：alpha 与 l1_ratio
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.model_selection import TimeSeriesSplit

class YearlyExpandingCV:
    """
    按年份逐步扩展训练集的时间序列交叉验证：
    - 第 i 折：训练集为前 i 年（从最早年份开始），验证集为第 i+1 年
    - 要求 X.index 为 DatetimeIndex
    - 最少训练年限由 min_train_years 决定
    """
    def __init__(self, min_train_years: int = 3, n_splits: Optional[int] = None):
        if min_train_years < 1:
            raise ValueError("min_train_years 必须 >= 1")
        if n_splits is not None and n_splits < 1:
            raise ValueError("n_splits 必须 >= 1")
        self.min_train_years = min_train_years
        self.n_splits = n_splits

    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, groups: Optional[np.ndarray] = None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("YearlyExpandingCV 需要 X 为 pandas.DataFrame")
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("YearlyExpandingCV 需要 X.index 为 pandas.DatetimeIndex")

        years = np.array(sorted(X.index.year.unique()))
        n_years = len(years)
        if n_years <= self.min_train_years:
            raise ValueError(f"数据年份数不足（{n_years} 年），至少需要 {self.min_train_years + 1} 年用于 CV")

        # 未指定 n_splits：用“最早 min_train_years 年训练 + 后续每年验证”
        if self.n_splits is None:
            for i in range(self.min_train_years, n_years):
                train_years = years[:i]
                val_year = years[i]
                train_idx = X.index.year.isin(train_years)
                val_idx = X.index.year == val_year
                yield np.where(train_idx)[0], np.where(val_idx)[0]
            return

        # 指定固定折数：使用“最近 K 年逐年验证”，训练集始终为验证年前所有年份
        # 可用的最大折数受 min_train_years 和年份总数限制
        max_possible = max(0, n_years - self.min_train_years)
        K = min(self.n_splits, max_possible)
        if K < 1:
            raise ValueError(f"无法创建折数：可用最大折数为 {max_possible}")
        start_i = max(self.min_train_years, n_years - K)
        for i in range(start_i, n_years):
            train_years = years[:i]
            val_year = years[i]
            train_idx = X.index.year.isin(train_years)
            val_idx = X.index.year == val_year
            yield np.where(train_idx)[0], np.where(val_idx)[0]

    def get_n_splits(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None, groups: Optional[np.ndarray] = None) -> int:
        if X is None or not isinstance(X.index, pd.DatetimeIndex):
            return 1
        years = np.array(sorted(X.index.year.unique()))
        n_years = len(years)
        if self.n_splits is None:
            return max(0, n_years - self.min_train_years)
        max_possible = max(0, n_years - self.min_train_years)
        return min(self.n_splits, max_possible)


def tune_elasticnet_ts(
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: Optional[Dict[str, Any]] = None,
    scoring: str = "neg_mean_squared_error",
    n_jobs: int = 1,
    max_iter: int = 5000,
    min_train_years: int = 3,
    random_state: int = 42,
    verbose: int = 1,
    n_splits: Optional[int] = None,
) -> Tuple[Pipeline, Dict[str, Any], Dict[str, Any]]:
    """
    使用时间序列 CV 对 ElasticNet 进行超参数调优。

    参数
    - X: DataFrame，要求 index 为 DatetimeIndex，行与 y 对齐
    - y: Series，目标值（如 'return'）
    - param_grid: 网格搜索空间，默认：
        {
            'elasticnet__alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            'elasticnet__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
    - scoring: 评分指标（默认 MSE 的相反数）
    - n_jobs: 并行数（建议 1，避免内存复制开销）
    - max_iter: ElasticNet 最大迭代次数
    - min_train_years: 最少训练年数
    - random_state: 随机种子
    - verbose: GridSearchCV 的日志等级

    返回
    - best_estimator: 最优管线（StandardScaler + ElasticNet）
    - best_params: 最优超参数
    - cv_results: 完整 CV 结果字典（便于分析）
    """

    # 丢弃缺失行，保持对齐
    aligned = pd.concat([X, y.rename("__y__")], axis=1)
    aligned = aligned.dropna()
    X_clean = aligned.drop(columns=["__y__"])
    y_clean = aligned["__y__"]

    # 默认搜索空间
    if param_grid is None:
        param_grid = {
            "elasticnet__alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            "elasticnet__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }

    # 管线：标准化 + ElasticNet
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("elasticnet", ElasticNet(max_iter=max_iter, random_state=random_state)),
        ]
    )

    # 根据索引类型选择 CV：有 DatetimeIndex 用逐年扩展，否则回退到样本顺序十折
    if isinstance(X_clean.index, pd.DatetimeIndex):
        cv = YearlyExpandingCV(min_train_years=min_train_years, n_splits=n_splits)
    else:
        cv = TimeSeriesSplit(n_splits=n_splits or 10)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=verbose,
        return_train_score=True,
    )
    grid.fit(X_clean, y_clean)

    best_estimator: Pipeline = grid.best_estimator_
    best_params: Dict[str, Any] = grid.best_params_
    cv_results: Dict[str, Any] = grid.cv_results_
    return best_estimator, best_params, cv_results


def extract_hard_transfer_params(model: Pipeline, feature_names: Optional[list] = None) -> Dict[str, Any]:
    """
    从最优管线中提取硬迁移参数（系数、截距、标准化统计）。
    - 便于在目标域（北交所）用同一标准化器 + 线性系数组合做预测。
    """
    if not isinstance(model, Pipeline):
        raise TypeError("model 必须是 sklearn Pipeline")

    scaler: StandardScaler = model.named_steps["scaler"]
    enet: ElasticNet = model.named_steps["elasticnet"]

    params = {
        "coef_": enet.coef_.tolist(),
        "intercept_": float(enet.intercept_),
        "scaler_mean_": scaler.mean_.tolist(),
        "scaler_scale_": scaler.scale_.tolist(),
        "feature_names": feature_names if feature_names is not None else None,
    }
    return params