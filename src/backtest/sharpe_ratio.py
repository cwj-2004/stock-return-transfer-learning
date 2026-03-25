import os
import math
import pandas as pd
import numpy as np

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "output", "models")
MONTHLY_FILES = {
    "hard_transfer_elasticnet": os.path.join(MODELS_DIR, "hard_transfer_elasticnet_monthly_returns.csv"),
    "soft_transfer_genet": os.path.join(MODELS_DIR, "soft_transfer_genet_monthly_returns.csv"),
}
PORTFOLIO_FILES = {
    "hard_transfer_elasticnet": os.path.join(MODELS_DIR, "hard_transfer_elasticnet_long_short_portfolio.csv"),
    "soft_transfer_genet": os.path.join(MODELS_DIR, "soft_transfer_genet_long_short_portfolio.csv"),
}
RF_ANNUAL = 0.0  # 年化无风险，若需要可改为 0.02 等

def load_predictions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Date" not in df.columns or "return" not in df.columns:
        raise ValueError(f"CSV 必须包含 'Date' 与 'return' 列。实际列: {list(df.columns)}")
    pred_cols_pref = ["pred_calib", "pred_raw", "pred_recenter", "pred"]
    pred_col = next((c for c in pred_cols_pref if c in df.columns), None)
    if pred_col is None:
        raise ValueError(f"未找到预测列，需包含任一 {pred_cols_pref}。实际列: {list(df.columns)}")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Date", "return", pred_col]).copy()
    df = df.rename(columns={pred_col: "pred"})
    return df[["Date", "Stkcd", "return", "pred"]].copy()

def export_portfolio(name: str, out_dir: str):
    if name not in PORTFOLIO_FILES:
        return
    src_path = PORTFOLIO_FILES[name]
    if not os.path.isfile(src_path):
        print(f"[WARN] 未找到选股名单文件: {src_path}")
        return
    df_port = pd.read_csv(src_path)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}_long_short_portfolio_for_sharpe.csv")
    df_port.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"保存选股名单: {out_path}")

def main():
    for name, csv_path in MONTHLY_FILES.items():
        if not os.path.isfile(csv_path):
            print(f"[WARN] 未找到月度收益文件: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        if "date" not in df.columns or "long_short_ret" not in df.columns:
            print(f"[WARN] {csv_path} 缺少 'date' 或 'long_short_ret' 列，实际列: {list(df.columns)}")
            continue

        idx = pd.to_datetime(df["date"])
        ret_series = df["long_short_ret"].astype(float)
        ret_series.index = idx

        metrics = compute_sharpe(ret_series, RF_ANNUAL)

        print(f"{name} long-short 组合夏普:")
        print(f"- months={metrics['months']}, start={metrics['start_date']}, end={metrics['end_date']}")
        print(f"- annual_return={metrics['annual_return']:.4f}, vol_annual={metrics['vol_annual']:.4f}, sharpe={metrics['sharpe']:.3f}")

        save_metrics(metrics, MODELS_DIR, f"{name}_long_short")
        export_portfolio(name, MODELS_DIR)

if __name__ == "__main__":
    main()