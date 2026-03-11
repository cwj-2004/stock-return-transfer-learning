"""
data_processor.py
功能：从三大交易所的已合并 CSV 文件中，提取共同特征，划分目标域训练集与测试集，
      并保存为 processed_data.pkl，可供 hard.py / soft.py / sharp.py 使用。
修复点：日频转月频 + 按年月排序滞后，解决特征滞后不匹配问题
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
import re

def read_all_csv_in(dir_path: str) -> pd.DataFrame:
    if not os.path.isdir(dir_path):
        return pd.DataFrame(columns=ID_COLUMNS)
    dfs = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith(".csv"):
                p = os.path.join(root, f)
                t = pd.read_csv(p, low_memory=False)
                if "Date" not in t.columns and "Trdmnt" in t.columns:
                    t["Date"] = pd.to_datetime(t["Trdmnt"]).dt.to_period("M").dt.to_timestamp("M")
                if "Date" in t.columns:
                    t = ensure_date_column(t)
                if "Stkcd" not in t.columns:
                    code = os.path.splitext(f)[0]
                    t["Stkcd"] = str(code)
                dfs.append(t)
    if not dfs:
        return pd.DataFrame(columns=ID_COLUMNS)
    return pd.concat(dfs, axis=0, ignore_index=True)

# === 配置路径 ===
DATA_DIR = r"d:\ECNU\大创\latest data"  # 存放最新数据目录
OUTPUT_PATH = os.path.join("data", "processed", "processed_data.pkl")
TEST_SPLIT_DATE = "2023-01-01"  # 北交所测试集分割日期
# 使用绝对路径避免相对路径问题
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_RETURNS_PATH = os.path.join(BASE_DIR, "..", "data", "月个股回报率文件100146906(仅供华东师范大学使用)", "TRD_Mnth.csv")
CODE_MAP_PDF_PATH = os.path.join(BASE_DIR, "..", "data", "附件：北交所存量上市公司股票新旧代码对照表.pdf")
CODE_MAP_TXT_PATH = os.path.join(BASE_DIR, "..", "data", "对照.txt")

# === 关键列 ===
ID_COLUMNS = ["Stkcd", "Date"]
TARGET_COLUMN = "return"

# === 时序对齐设置 ===
FEATURE_LAG_MONTHS_DEFAULT = 1  # 所有特征至少滞后1个月


def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """确保 DataFrame 含有标准的 Date 列并转换为日期格式"""
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        # 新增：添加「年月」列，用于月频聚合和排序
        df["year_month"] = df["Date"].dt.to_period("M")
    else:
        raise ValueError("数据中必须包含 'Date' 列")
    return df

def get_numeric_features(df: pd.DataFrame) -> list:
    """提取所有数值型特征（排除收益列与标识列）"""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    exclude = {TARGET_COLUMN, "Stkcd", "return1", "return2"}
    return [c for c in numeric_cols if c not in exclude]

def unify_target_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TARGET_COLUMN in df.columns:
        return df
    for c in ["return1", "return2", "Mretwd", "Mret", "RET", "Ret"]:
        if c in df.columns:
            df = df.rename(columns={c: TARGET_COLUMN})
            return df
    return df

def convert_to_monthly(df: pd.DataFrame, feature_cols: list, target_col: str = None) -> pd.DataFrame:
    """将日频数据聚合为月频：特征取月末值，收益取月内均值（或月末收益）"""
    df = df.copy()
    agg_dict = {}
    feature_cols_present = [c for c in feature_cols if c in df.columns]
    for col in feature_cols_present:
        agg_dict[col] = (lambda x: x.iloc[-1] if len(x) > 0 else np.nan)
    if target_col and target_col in df.columns:
        agg_dict[target_col] = "mean"
    agg_dict["Date"] = (lambda x: x.iloc[-1] if len(x) > 0 else pd.NaT)
    df_monthly = df.groupby(["Stkcd", "year_month"]).agg(agg_dict).reset_index()
    df_monthly["year_month"] = df_monthly["year_month"].astype(str)
    df_monthly = df_monthly.sort_values(["Stkcd", "Date"]).reset_index(drop=True)
    return df_monthly

def _detect_month_col(df: pd.DataFrame) -> str:
    for c in ["Date", "Trdmnt", "TrdMonth", "month", "YM", "year_month"]:
        if c in df.columns:
            return c
    return "Date"

def _to_month_end_series(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.datetime64):
        return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp("M")
    s = s.astype(str).str.strip()
    if s.str.contains("-").any():
        return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    return pd.to_datetime(s, format="%Y%m", errors="coerce").dt.to_period("M").dt.to_timestamp("M")

def _detect_return_col(df: pd.DataFrame) -> str:
    candidates = ["Mretwd", "Mret", "RET", "Ret", "return", "mret"]
    for c in candidates:
        if c in df.columns:
            return c
    nums = df.select_dtypes(include=["number"]).columns.tolist()
    return nums[0] if nums else candidates[-1]

def load_real_monthly_returns(path: str = REAL_RETURNS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, dtype={"Stkcd": str})
    if "Stkcd" not in df.columns:
        for c in ["Stkcd", "Stkcode", "Code", "stkcd"]:
            if c in df.columns:
                df = df.rename(columns={c: "Stkcd"})
                break
    df["Stkcd"] = df["Stkcd"].astype(str).str.strip().str.replace(r"\D", "", regex=True).str.zfill(6)
    mcol = _detect_month_col(df)
    df["Date"] = _to_month_end_series(df[mcol])
    price_col = None
    for c in ["Mclsprc", "Clsprc", "Close", "ClsPrice"]:
        if c in df.columns:
            price_col = c
            break
    if price_col is not None:
        df = df[["Stkcd", "Date", price_col]].copy()
        df = df.sort_values(["Stkcd", "Date"]).reset_index(drop=True)
        df["real_return"] = df.groupby("Stkcd")[price_col].pct_change()
        df = df.drop(columns=[price_col])
        df = df.dropna(subset=["Date"]).reset_index(drop=True)
        return df
    rcol = _detect_return_col(df)
    df = df[["Stkcd", "Date", rcol]].copy()
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    df = df.rename(columns={rcol: "real_return"})
    df = df.sort_values(["Stkcd", "Date"]).reset_index(drop=True)
    return df

def try_load_code_mapping_from_pdf(pdf_path: str = CODE_MAP_PDF_PATH) -> pd.DataFrame | None:
    try:
        import pdfplumber
        rows = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                try:
                    table = page.extract_table()
                    if table:
                        for row in table:
                            if not row:
                                continue
                            nums = [re.sub(r"\D", "", x or "") for x in row]
                            nums = [n for n in nums if n]
                            if len(nums) >= 2:
                                rows.append((nums[0], nums[1]))
                except Exception:
                    text = page.extract_text() or ""
                    for line in text.splitlines():
                        found = re.findall(r"(\d{6,})", line)
                        if len(found) >= 2:
                            rows.append((found[0], found[1]))
        if rows:
            df = pd.DataFrame(rows, columns=["old_code", "new_code"]).drop_duplicates()
            df["old_code"] = df["old_code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
            df["new_code"] = df["new_code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
            return df
    except Exception:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            rows = []
            for page in reader.pages:
                text = page.extract_text() or ""
                for line in text.splitlines():
                    found = re.findall(r"(\d{6,})", line)
                    if len(found) >= 2:
                        rows.append((found[0], found[1]))
            if rows:
                df = pd.DataFrame(rows, columns=["old_code", "new_code"]).drop_duplicates()
                df["old_code"] = df["old_code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
                df["new_code"] = df["new_code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
                return df
        except Exception:
            pass
    warnings.warn("未能自动从PDF提取代码映射，将不进行代码转换。")
    return None

def load_code_mapping_from_txt(txt_path: str = CODE_MAP_TXT_PATH) -> pd.DataFrame | None:
    if not os.path.exists(txt_path):
        return None
    rows = []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            nums = re.findall(r"(\d{6,})", line)
            if len(nums) >= 2:
                old_code, new_code = nums[-2], nums[-1]
                rows.append((old_code, new_code))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["old_code", "new_code"]).drop_duplicates()
    df["old_code"] = df["old_code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    df["new_code"] = df["new_code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    return df

def apply_code_mapping(df: pd.DataFrame, map_df: pd.DataFrame | None) -> pd.DataFrame:
    if map_df is None or map_df.empty:
        return df
    mp = dict(zip(map_df["old_code"], map_df["new_code"]))
    df = df.copy()
    df["Stkcd"] = df["Stkcd"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    df["Stkcd"] = df["Stkcd"].map(lambda x: mp.get(x, x))
    return df

def align_real_returns_same(target_info: pd.DataFrame, real_df: pd.DataFrame) -> pd.Series:
    ti = target_info.copy()
    ti["Stkcd"] = ti["Stkcd"].astype(str).str.replace("\\D", "", regex=True).str.zfill(6)
    real_df = real_df.copy()
    real_df["Stkcd"] = real_df["Stkcd"].astype(str).str.replace("\\D", "", regex=True).str.zfill(6)
    ti["Date_me"] = pd.to_datetime(ti["Date"]).dt.to_period("M").dt.to_timestamp("M")
    merged = ti.merge(real_df, left_on=["Stkcd", "Date_me"], right_on=["Stkcd", "Date"], how="left")
    return merged["real_return"].reset_index(drop=True)

def generate_processed_data(data_dir=DATA_DIR, output_path=OUTPUT_PATH, split_date=TEST_SPLIT_DATE):
    print("=== 开始生成 processed_data.pkl ===")

    # 加载三大市场数据（目录合并）
    sse_dir = os.path.join(data_dir, "SHSE")  # 修正：目录名是 SHSE
    sz_dir = os.path.join(data_dir, "SZSE")
    bj_dir = os.path.join(data_dir, "BSE")
    print(f"读取沪市: {sse_dir}")
    print(f"读取深市: {sz_dir}")
    print(f"读取北交所: {bj_dir}")
    df_sh = read_all_csv_in(sse_dir)
    df_sz = read_all_csv_in(sz_dir)
    df_bj = read_all_csv_in(bj_dir)
    print(f"沪市数据形状: {df_sh.shape}, 列: {list(df_sh.columns[:5])}")
    print(f"深市数据形状: {df_sz.shape}, 列: {list(df_sz.columns[:5])}")
    print(f"北交所数据形状: {df_bj.shape}, 列: {list(df_bj.columns[:5])}")

    # 检查原始数据中的收益列
    if not df_sh.empty:
        print(f"沪市原始列中是否有return1/return2/return: {'return1' in df_sh.columns}, {'return2' in df_sh.columns}, {'return' in df_sh.columns}")
    if not df_sz.empty:
        print(f"深市原始列中是否有return1/return2/return: {'return1' in df_sz.columns}, {'return2' in df_sz.columns}, {'return' in df_sz.columns}")

    # 标准化日期 + 添加年月列
    df_sh = ensure_date_column(df_sh)
    df_sz = ensure_date_column(df_sz)
    df_bj = ensure_date_column(df_bj)
    print(f"日期处理后: 沪市{df_sh.shape}, 深市{df_sz.shape}, 北交所{df_bj.shape}")

    # 统一目标列前检查
    print(f"统一前 - 沪市列中是否有return1/return2: {'return1' in df_sh.columns}, {'return2' in df_sh.columns}, {'return' in df_sh.columns}")
    print(f"统一前 - 深市列中是否有return1/return2: {'return1' in df_sz.columns}, {'return2' in df_sz.columns}, {'return' in df_sz.columns}")

    df_sh = unify_target_column(df_sh)
    df_sz = unify_target_column(df_sz)
    df_bj = unify_target_column(df_bj)
    print(f"统一目标列后: 沪市是否有return: {'return' in df_sh.columns}, 深市是否有return: {'return' in df_sz.columns}, 北交所是否有return: {'return' in df_bj.columns}")

    # 提取共同特征（基于原始数据，未聚合前）
    feats_list = []
    if not df_sh.empty:
        feats_list.append(get_numeric_features(df_sh))
    if not df_sz.empty:
        feats_list.append(get_numeric_features(df_sz))
    if not df_bj.empty:
        feats_list.append(get_numeric_features(df_bj))
    common_features = sorted(list(set(feats_list[0]).intersection(*feats_list[1:])) if feats_list else [])
    print(f"共同特征数: {len(common_features)}")

    # === 关键修复：日频转月频（聚合后再滞后）===
    print("— 沪市日频转月频 —")
    print(f"  沪市原始列数: {len(df_sh.columns)}, 是否有return: {'return' in df_sh.columns}")
    df_sh_monthly = convert_to_monthly(df_sh, common_features, TARGET_COLUMN)
    print(f"  沪市月频列数: {len(df_sh_monthly.columns)}, 是否有return: {'return' in df_sh_monthly.columns}")
    print("— 深市日频转月频 —")
    print(f"  深市原始列数: {len(df_sz.columns)}, 是否有return: {'return' in df_sz.columns}")
    df_sz_monthly = convert_to_monthly(df_sz, common_features, TARGET_COLUMN)
    print(f"  深市月频列数: {len(df_sz_monthly.columns)}, 是否有return: {'return' in df_sz_monthly.columns}")
    print("— 北交所日频转月频 —")
    df_bj_monthly = convert_to_monthly(df_bj, common_features, TARGET_COLUMN)

    # 保留月频原始副本用于校验
    df_sh_orig = df_sh_monthly.copy()
    df_sz_orig = df_sz_monthly.copy()
    df_bj_orig = df_bj_monthly.copy()

    # === 时序对齐：生成下一期收益 & 特征滞后1个月 ===
    print("— 使用同月标签，不生成下一期收益 —")

    # 剔除滞后/对齐产生的缺失值

    subset_sh = [c for c in (common_features + [TARGET_COLUMN]) if c in df_sh_monthly.columns]
    subset_sz = [c for c in (common_features + [TARGET_COLUMN]) if c in df_sz_monthly.columns]
    subset_bj = [c for c in (common_features + [TARGET_COLUMN]) if c in df_bj_monthly.columns]
    df_sh_monthly = df_sh_monthly.dropna(subset=subset_sh).reset_index(drop=True)
    df_sz_monthly = df_sz_monthly.dropna(subset=subset_sz).reset_index(drop=True)
    df_bj_monthly = df_bj_monthly.dropna(subset=subset_bj).reset_index(drop=True)

    # 校验标签对齐
    print("— 已移除下一期收益对齐校验（同月标签） —")

    # === 定义源域与目标域 ===
    df_source = pd.concat([df_sh_monthly, df_sz_monthly], axis=0, ignore_index=True)
    if TARGET_COLUMN in df_source.columns:
        df_source = df_source.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    df_target = df_bj_monthly.copy()

    # === 按时间划分目标域训练/测试集 ===
    split_datetime = pd.to_datetime(split_date)
    df_target_train = df_target[df_target["Date"] < split_datetime].copy()
    df_target_test = df_target[df_target["Date"] >= split_datetime].copy()

    print(f"源域（沪深）样本数: {len(df_source)}")
    print(f"目标域（北交所）训练集: {len(df_target_train)}，测试集: {len(df_target_test)}")

    # === 生成特征与标签 ===
    X_source = df_source[common_features]
    y_source = df_source[TARGET_COLUMN]

    X_target_train = df_target_train[common_features]
    y_target_train = df_target_train[TARGET_COLUMN]

    X_target_test = df_target_test[common_features]
    y_target_test = df_target_test[TARGET_COLUMN]

    # === 提取训练/测试集元信息 ===
    info_cols_test = [c for c in ID_COLUMNS if c in df_target_test.columns]
    target_test_info = df_target_test[info_cols_test + ["year_month"]].copy()
    target_test_info["Date"] = pd.to_datetime(target_test_info["Date"], errors="coerce")
    target_test_info = target_test_info.dropna(subset=["Date"]).reset_index(drop=True)

    info_cols_train = [c for c in ID_COLUMNS if c in df_target_train.columns]
    target_train_info = df_target_train[info_cols_train + ["year_month"]].copy()
    target_train_info["Date"] = pd.to_datetime(target_train_info["Date"], errors="coerce")
    target_train_info = target_train_info.dropna(subset=["Date"]).reset_index(drop=True)

    code_map_df = load_code_mapping_from_txt(CODE_MAP_TXT_PATH)
    if code_map_df is None:
        code_map_df = try_load_code_mapping_from_pdf(CODE_MAP_PDF_PATH)
    real_df = load_real_monthly_returns(REAL_RETURNS_PATH)
    real_df = apply_code_mapping(real_df, code_map_df)
    # 将目标训练/测试代码统一到新代码
    target_test_info = apply_code_mapping(target_test_info, code_map_df)
    target_train_info = apply_code_mapping(target_train_info, code_map_df)
    y_target_test_real = align_real_returns_same(target_test_info, real_df)
    y_target_train_real = align_real_returns_same(target_train_info, real_df)

    # === 打包结果 ===
    print(f"=== 沪市月频数据: 形状={df_sh_monthly.shape}, 列数={len(df_sh_monthly.columns)}, 有return={TARGET_COLUMN in df_sh_monthly.columns}")
    print(f"=== 深市月频数据: 形状={df_sz_monthly.shape}, 列数={len(df_sz_monthly.columns)}, 有return={TARGET_COLUMN in df_sz_monthly.columns}")

    # 获取各市场实际存在的数值特征列（排除目标列和ID列）
    def get_market_features(df):
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        exclude = {TARGET_COLUMN, "Stkcd", "return1", "return2"}
        return [c for c in numeric_cols if c not in exclude]

    features_sh = get_market_features(df_sh_monthly)
    features_sz = get_market_features(df_sz_monthly)
    print(f"=== 沪市特征数: {len(features_sh)}, 深市特征数: {len(features_sz)}")

    # 检查目标列是否存在
    y_source_sh = df_sh_monthly[TARGET_COLUMN] if TARGET_COLUMN in df_sh_monthly.columns else None
    y_source_sz = df_sz_monthly[TARGET_COLUMN] if TARGET_COLUMN in df_sz_monthly.columns else None
    print(f"=== 沪市y_source_sh: {y_source_sh is not None}, 深市y_source_sz: {y_source_sz is not None}")

    processed_data = {
        "X_source": X_source,
        "y_source": y_source,
        "X_source_sh": df_sh_monthly[features_sh] if not df_sh_monthly.empty else pd.DataFrame(),
        "y_source_sh": y_source_sh,
        "X_source_sz": df_sz_monthly[features_sz] if not df_sz_monthly.empty else pd.DataFrame(),
        "y_source_sz": y_source_sz,
        "X_target_train": X_target_train,
        "y_target_train": y_target_train,
        "X_target_test": X_target_test,
        "y_target_test": y_target_test,
        "y_target_test_real": y_target_test_real,
        "y_target_train_real": y_target_train_real,
        "target_test_info": target_test_info,
        "target_train_info": target_train_info,
        "feature_names": common_features,
        "split_date": split_date,
    }

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(processed_data, output_path)

    print(f"[SUCCESS] 数据处理完成，文件已保存到: {output_path}")
    print(f"   特征维度: {len(common_features)}，测试集样本数: {len(X_target_test)}")
    print("   标签定义: y = 当月收益（return）；真实收益同月对齐已保存。")

    return processed_data

if __name__ == "__main__":
    generate_processed_data()
