"""
数据处理器

数据处理流程：
日频数据 -> 月频聚合 -> 时序对齐（t月特征预测t+1月收益）-> 划分训练/测试集
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
import re
from typing import Optional, Tuple


# === 配置路径 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "latest_data")
OUTPUT_PATH = os.path.join("data", "processed", "processed_data.pkl")
TEST_SPLIT_DATE = "2023-01-01"
REAL_RETURNS_PATH = os.path.join(BASE_DIR, "data", "raw", "TRD_Mnth.csv")
CODE_MAP_TXT_PATH = os.path.join(BASE_DIR, "data", "raw", "对照.txt")

# === 关键列名 ===
ID_COLUMNS = ["Stkcd", "Date"]
TARGET_COLUMN = "return"


# ==============================================================================
# 数据加载
# ==============================================================================

def load_market_csv(dir_path: str) -> pd.DataFrame:
    if not os.path.isdir(dir_path):
        return pd.DataFrame(columns=ID_COLUMNS)

    dfs = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith(".csv"):
                p = os.path.join(root, f)
                try:
                    df = pd.read_csv(p, low_memory=False)
                    if "Date" not in df.columns and "Trdmnt" in df.columns:
                        df["Date"] = pd.to_datetime(df["Trdmnt"]).dt.to_period("M").dt.to_timestamp("M")
                    if "Stkcd" not in df.columns:
                        df["Stkcd"] = os.path.splitext(f)[0]
                    dfs.append(df)
                except Exception as e:
                    warnings.warn(f"读取文件 {p} 失败: {e}")

    return pd.concat(dfs, axis=0, ignore_index=True) if dfs else pd.DataFrame(columns=ID_COLUMNS)


# ==============================================================================
# 数据标准化
# ==============================================================================

def standardize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    # year_month 用于月频聚合（period[M] 类型，不会被当作数值特征）
    df["year_month"] = df["Date"].dt.to_period("M")
    return df


def unify_target_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TARGET_COLUMN not in df.columns:
        for col in ["return1", "return2"]:
            if col in df.columns:
                df = df.rename(columns={col: TARGET_COLUMN})
                break
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    exclude = {TARGET_COLUMN, "Stkcd", "return1", "return2"}
    return [c for c in numeric_cols if c not in exclude]


# ==============================================================================
# 特征工程
# ==============================================================================

def convert_daily_to_monthly(df: pd.DataFrame, feature_cols: list, target_col: str = None) -> pd.DataFrame:
    df = df.copy()
    agg_dict = {}

    for col in feature_cols:
        if col in df.columns:
            agg_dict[col] = lambda x: x.iloc[-1] if len(x) > 0 else np.nan

    if target_col and target_col in df.columns:
        agg_dict[target_col] = "mean"

    agg_dict["Date"] = lambda x: x.iloc[-1] if len(x) > 0 else pd.NaT

    df_monthly = df.groupby(["Stkcd", "year_month"]).agg(agg_dict).reset_index()
    df_monthly["year_month"] = df_monthly["year_month"].astype(str)
    df_monthly = df_monthly.sort_values(["Stkcd", "Date"]).reset_index(drop=True)

    return df_monthly


def align_time_series(df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
    if target_col is None or target_col not in df.columns:
        return df

    df = df.copy()
    df = df.sort_values(["Stkcd", "Date"]).reset_index(drop=True)
    df["next_return"] = df.groupby("Stkcd")[target_col].shift(-1)
    df[target_col] = df["next_return"]
    df = df.drop(columns=["next_return"])

    return df


# ==============================================================================
# 真实收益数据处理
# ==============================================================================

def parse_month_column(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.datetime64):
        return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp("M")

    s = s.astype(str).str.strip()
    if s.str.contains("-").any():
        return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    return pd.to_datetime(s, format="%Y%m", errors="coerce").dt.to_period("M").dt.to_timestamp("M")


def load_real_monthly_returns(path: str = REAL_RETURNS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, dtype={"Stkcd": str})

    # 标准化股票代码列
    if "Stkcd" not in df.columns:
        for col in ["Stkcode", "Code", "stkcd"]:
            if col in df.columns:
                df = df.rename(columns={col: "Stkcd"})
                break

    df["Stkcd"] = df["Stkcd"].astype(str).str.strip().str.replace(r"\D", "", regex=True).str.zfill(6)

    # 标准化日期列
    month_col = "Trdmnt" if "Trdmnt" in df.columns else "Date"
    df["Date"] = parse_month_column(df[month_col])

    # 从收盘价计算收益率
    price_col = "Mclsprc" if "Mclsprc" in df.columns else None
    if price_col:
        df = df[["Stkcd", "Date", price_col]].copy()
        df = df.sort_values(["Stkcd", "Date"]).reset_index(drop=True)
        df["real_return"] = df.groupby("Stkcd")[price_col].pct_change()
        df = df.drop(columns=[price_col])

    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    df = df.sort_values(["Stkcd", "Date"]).reset_index(drop=True)

    return df


# ==============================================================================
# 代码映射
# ==============================================================================

def load_code_mapping_from_txt(txt_path: str = CODE_MAP_TXT_PATH) -> Optional[pd.DataFrame]:
    if not os.path.exists(txt_path):
        return None

    rows = []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            nums = re.findall(r"(\d{6})", line)
            if len(nums) >= 2:
                rows.append((nums[0], nums[1]))

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["old_code", "new_code"]).drop_duplicates()
    return df


def apply_code_mapping(df: pd.DataFrame, map_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if map_df is None or map_df.empty:
        return df.copy()

    df = df.copy()
    df["Stkcd"] = df["Stkcd"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)

    mapping = dict(zip(map_df["old_code"], map_df["new_code"]))
    df["Stkcd"] = df["Stkcd"].map(lambda x: mapping.get(x, x))

    return df


def align_real_returns(target_info: pd.DataFrame, real_df: pd.DataFrame) -> pd.Series:
    ti = target_info.copy()
    real_df = real_df.copy()

    ti["Date_month_end"] = pd.to_datetime(ti["Date"]).dt.to_period("M").dt.to_timestamp("M")
    merged = ti.merge(real_df, left_on=["Stkcd", "Date_month_end"], right_on=["Stkcd", "Date"], how="left")

    return merged["real_return"].reset_index(drop=True)


# ==============================================================================
# 主处理流程
# ==============================================================================

def load_market_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_sh = load_market_csv(os.path.join(data_dir, "SHSE"))
    df_sz = load_market_csv(os.path.join(data_dir, "SZSE"))
    df_bj = load_market_csv(os.path.join(data_dir, "BSE"))

    print(f"沪市: {df_sh.shape}, 深市: {df_sz.shape}, 北交所: {df_bj.shape}")
    return df_sh, df_sz, df_bj


def preprocess_market_data(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_date_column(df)
    df = unify_target_column(df)
    return df


def extract_common_features(df_sh: pd.DataFrame, df_sz: pd.DataFrame, df_bj: pd.DataFrame) -> list:
    feats_list = []
    for df in [df_sh, df_sz, df_bj]:
        if not df.empty:
            feats_list.append(get_feature_columns(df))

    if not feats_list:
        return []

    common = set(feats_list[0])
    for feats in feats_list[1:]:
        common = common.intersection(set(feats))

    return sorted(list(common))


def convert_to_monthly_all(
    df_sh: pd.DataFrame,
    df_sz: pd.DataFrame,
    df_bj: pd.DataFrame,
    feature_cols: list
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_sh_monthly = convert_daily_to_monthly(df_sh, feature_cols, TARGET_COLUMN)
    df_sz_monthly = convert_daily_to_monthly(df_sz, feature_cols, TARGET_COLUMN)
    df_bj_monthly = convert_daily_to_monthly(df_bj, feature_cols, TARGET_COLUMN)

    return df_sh_monthly, df_sz_monthly, df_bj_monthly


def align_time_series_all(
    df_sh: pd.DataFrame,
    df_sz: pd.DataFrame,
    df_bj: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_sh = align_time_series(df_sh, TARGET_COLUMN)
    df_sz = align_time_series(df_sz, TARGET_COLUMN)
    df_bj = align_time_series(df_bj, TARGET_COLUMN)

    return df_sh, df_sz, df_bj


def split_target_domain(df: pd.DataFrame, split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_datetime = pd.to_datetime(split_date)
    df_train = df[df["Date"] < split_datetime].copy()
    df_test = df[df["Date"] >= split_datetime].copy()
    return df_train, df_test


def extract_model_data(
    df_source: pd.DataFrame,
    df_target_train: pd.DataFrame,
    df_target_test: pd.DataFrame,
    feature_cols: list
) -> dict:
    return {
        "X_source": df_source[feature_cols],
        "y_source": df_source[TARGET_COLUMN],
        "X_target_train": df_target_train[feature_cols],
        "y_target_train": df_target_train[TARGET_COLUMN],
        "X_target_test": df_target_test[feature_cols],
        "y_target_test": df_target_test[TARGET_COLUMN],
    }


def extract_meta_info(
    df_target_train: pd.DataFrame,
    df_target_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    info_cols = ["Stkcd", "Date", "year_month"]

    train_info = df_target_train[info_cols].copy()
    train_info["Date"] = pd.to_datetime(train_info["Date"], errors="coerce")
    train_info = train_info.dropna(subset=["Date"]).reset_index(drop=True)

    test_info = df_target_test[info_cols].copy()
    test_info["Date"] = pd.to_datetime(test_info["Date"], errors="coerce")
    test_info = test_info.dropna(subset=["Date"]).reset_index(drop=True)

    return train_info, test_info


def generate_processed_data(
    data_dir: str = DATA_DIR,
    output_path: str = OUTPUT_PATH,
    split_date: str = TEST_SPLIT_DATE
) -> dict:
    print("=== 数据处理 ===")

    # 1. 加载数据
    df_sh, df_sz, df_bj = load_market_data(data_dir)

    # 2. 预处理
    df_sh = preprocess_market_data(df_sh)
    df_sz = preprocess_market_data(df_sz)
    df_bj = preprocess_market_data(df_bj)

    # 3. 提取共同特征
    common_features = extract_common_features(df_sh, df_sz, df_bj)
    print(f"共同特征: {len(common_features)}")

    # 4. 日频转月频
    df_sh_monthly, df_sz_monthly, df_bj_monthly = convert_to_monthly_all(
        df_sh, df_sz, df_bj, common_features
    )

    # 5. 时序对齐
    df_sh_monthly, df_sz_monthly, df_bj_monthly = align_time_series_all(
        df_sh_monthly, df_sz_monthly, df_bj_monthly
    )

    # 清理缺失值
    subset_cols = [c for c in (common_features + [TARGET_COLUMN]) if c in df_sh_monthly.columns]
    df_sh_monthly = df_sh_monthly.dropna(subset=subset_cols).reset_index(drop=True)
    df_sz_monthly = df_sz_monthly.dropna(subset=subset_cols).reset_index(drop=True)
    df_bj_monthly = df_bj_monthly.dropna(subset=subset_cols).reset_index(drop=True)

    # 6. 定义源域和目标域
    df_source = pd.concat([df_sh_monthly, df_sz_monthly], axis=0, ignore_index=True)
    df_target = df_bj_monthly

    # 7. 划分训练/测试集
    df_target_train, df_target_test = split_target_domain(df_target, split_date)
    print(f"源域: {len(df_source)}, 目标域训练: {len(df_target_train)}, 测试: {len(df_target_test)}")

    # 8. 提取模型数据
    model_data = extract_model_data(df_source, df_target_train, df_target_test, common_features)

    # 9. 提取元信息
    train_info, test_info = extract_meta_info(df_target_train, df_target_test)

    # 10. 加载代码映射和真实收益
    code_map = load_code_mapping_from_txt(CODE_MAP_TXT_PATH)
    if code_map is None:
        warnings.warn("代码映射文件不存在，请运行: python src/utils/extract_code_mapping.py")

    real_df = load_real_monthly_returns(REAL_RETURNS_PATH)
    real_df = apply_code_mapping(real_df, code_map)
    train_info = apply_code_mapping(train_info, code_map)
    test_info = apply_code_mapping(test_info, code_map)

    # 11. 对齐真实收益
    y_target_train_real = align_real_returns(train_info, real_df)
    y_target_test_real = align_real_returns(test_info, real_df)

    # 12. 保存
    processed_data = {
        **model_data,
        "y_target_test_real": y_target_test_real,
        "y_target_train_real": y_target_train_real,
        "target_test_info": test_info,
        "target_train_info": train_info,
        "feature_names": common_features,
        "split_date": split_date,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(processed_data, output_path)

    print(f"完成: {output_path}, 特征: {len(common_features)}, 测试样本: {len(model_data['X_target_test'])}")

    return processed_data


if __name__ == "__main__":
    generate_processed_data()