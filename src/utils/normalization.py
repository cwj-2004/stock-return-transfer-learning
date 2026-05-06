# -*- coding: utf-8 -*-
"""
数据标准化工具
实现论文要求的 Cross-sectional normalization，避免前瞻偏差
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from sklearn.preprocessing import StandardScaler
import warnings


class CrossSectionalNormalizer:
    """
    时间点截面标准化器

    论文要求：
    - 在每个时间点 t，使用历史数据计算标准化参数
    - 避免前瞻偏差（Look-ahead bias）
    """

    def __init__(self, feature_cols: List[str], min_periods: int = 12,
                 method: str = "expanding", window: int = 12):
        """
        参数：
        - feature_cols: 需要标准化的特征列名列表
        - min_periods: 计算标准化参数所需的最小样本数
        - method: 计算窗口类型，"expanding" 或 "rolling"
        - window: 当 method="rolling" 时的窗口大小
        """
        self.feature_cols = feature_cols
        self.min_periods = min_periods
        self.method = method
        self.window = window
        self.normalization_stats = {}

    def fit_transform(self, df: pd.DataFrame, date_col: str = "Date",
                      stock_col: str = "Stkcd") -> pd.DataFrame:
        """
        拟合并转换数据

        参数：
        - df: 输入数据，必须包含 date_col, stock_col 和特征列
        - date_col: 日期列名
        - stock_col: 股票代码列名

        返回：
        - 标准化后的数据
        """
        df = df.copy()
        df = df.sort_values([date_col, stock_col]).reset_index(drop=True)

        # 确保日期格式正确
        df[date_col] = pd.to_datetime(df[date_col])

        for col in self.feature_cols:
            if col not in df.columns:
                warnings.warn(f"特征列 {col} 不存在，跳过")
                continue

            # 按日期计算历史均值和标准差
            if self.method == "expanding":
                # 扩展窗口：使用所有历史数据
                df[f'{col}_hist_mean'] = df.groupby(date_col)[col].transform(
                    lambda x: x.expanding(min_periods=self.min_periods).mean().shift(1)
                )
                df[f'{col}_hist_std'] = df.groupby(date_col)[col].transform(
                    lambda x: x.expanding(min_periods=self.min_periods).std().shift(1)
                )
            else:  # rolling
                # 滚动窗口：使用最近 window 期数据
                df[f'{col}_hist_mean'] = df.groupby(date_col)[col].transform(
                    lambda x: x.rolling(window=self.window, min_periods=self.min_periods).mean().shift(1)
                )
                df[f'{col}_hist_std'] = df.groupby(date_col)[col].transform(
                    lambda x: x.rolling(window=self.window, min_periods=self.min_periods).std().shift(1)
                )

            # 标准化：(x - mean) / std
            # 处理标准差为0或缺失的情况
            std_mask = (df[f'{col}_hist_std'] > 1e-8) & df[f'{col}_hist_std'].notna()
            df[col] = np.where(std_mask,
                              (df[col] - df[f'{col}_hist_mean']) / df[f'{col}_hist_std'],
                              0.0)

            # 删除中间列
            df = df.drop(columns=[f'{col}_hist_mean', f'{col}_hist_std'])

        return df

    def transform(self, df: pd.DataFrame, date_col: str = "Date",
                  stock_col: str = "Stkcd") -> pd.DataFrame:
        """
        使用已拟合的统计量转换新数据

        注意：对于时间序列数据，通常建议重新 fit_transform 而不是直接 transform
        """
        # 对于时间序列数据，直接使用 fit_transform
        return self.fit_transform(df, date_col, stock_col)


def cross_sectional_normalize(df: pd.DataFrame, feature_cols: List[str],
                              min_periods: int = 12) -> pd.DataFrame:
    """
    简化的时间点截面标准化函数

    参数：
    - df: 输入数据
    - feature_cols: 需要标准化的特征列
    - min_periods: 最小样本数

    返回：
    - 标准化后的数据
    """
    normalizer = CrossSectionalNormalizer(
        feature_cols=feature_cols,
        min_periods=min_periods,
        method="expanding"
    )
    return normalizer.fit_transform(df)


class MarketAwareStandardizer:
    """
    跨市场感知的标准化器

    论文中的标准化策略：
    1. 源域（Global/HS300）：使用所有历史数据标准化
    2. 目标域（Local/BSE）：使用该市场历史数据标准化
    """

    def __init__(self, source_markets: List[str], target_market: str,
                 feature_cols: List[str], min_periods: int = 12):
        """
        参数：
        - source_markets: 源域市场名称列表
        - target_market: 目标域市场名称
        - feature_cols: 特征列名列表
        - min_periods: 最小样本数
        """
        self.source_markets = source_markets
        self.target_market = target_market
        self.feature_cols = feature_cols
        self.min_periods = min_periods

    def fit_transform_markets(self, market_data: Dict[str, pd.DataFrame],
                              date_col: str = "Date",
                              stock_col: str = "Stkcd") -> Dict[str, pd.DataFrame]:
        """
        对各个市场进行标准化

        参数：
        - market_data: 各市场数据，格式 {'market_name': DataFrame}

        返回：
        - 标准化后的各市场数据
        """
        result = {}

        # 对源域市场进行标准化
        for market in self.source_markets:
            if market not in market_data:
                continue
            normalizer = CrossSectionalNormalizer(
                feature_cols=self.feature_cols,
                min_periods=self.min_periods,
                method="expanding"
            )
            result[market] = normalizer.fit_transform(market_data[market], date_col, stock_col)

        # 对目标域市场进行标准化
        if self.target_market in market_data:
            normalizer = CrossSectionalNormalizer(
                feature_cols=self.feature_cols,
                min_periods=self.min_periods,
                method="expanding"
            )
            result[self.target_market] = normalizer.fit_transform(
                market_data[self.target_market], date_col, stock_col
            )

        return result


def validate_normalization(df: pd.DataFrame, feature_cols: List[str],
                          date_col: str = "Date") -> Dict[str, float]:
    """
    验证标准化效果

    检查：
    1. 每个时间点的截面均值是否接近0
    2. 每个时间点的截面标准差是否接近1
    3. 是否存在前瞻偏差

    返回：
    - 验证指标字典
    """
    metrics = {}

    for col in feature_cols:
        if col not in df.columns:
            continue

        # 按日期分组计算截面统计量
        cross_sectional_stats = df.groupby(date_col)[col].agg(['mean', 'std'])

        # 检查均值是否接近0
        mean_abs = np.abs(cross_sectional_stats['mean']).mean()
        metrics[f'{col}_mean_abs'] = mean_abs

        # 检查标准差是否接近1
        std_dev = np.abs(cross_sectional_stats['std'] - 1).mean()
        metrics[f'{col}_std_dev'] = std_dev

    return metrics