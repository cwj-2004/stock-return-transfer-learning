import os
import math
import pandas as pd
import numpy as np
import sys

# 添加父目录到路径以导入backtest模块
sys.path.insert(0, os.path.dirname(__file__))
from backtest import (
    load_predictions,
    merge_real_returns_to_predictions,
    REAL_RETURNS_PATH,
    CODE_MAP_TXT_PATH,
)

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "output", "models")

PRED_FILES = {
    "hard_transfer_elasticnet": os.path.join(MODELS_DIR, "hard_transfer_predictions_oos.csv"),
    "soft_transfer_genet": os.path.join(MODELS_DIR, "soft_transfer_predictions_oos.csv"),
}

RF_ANNUAL = 0.0  # 年化无风险利率，如需可改为 0.02 等


def compute_sharpe(ret: pd.Series, rf_annual: float = 0.0) -> dict:
    ret = ret.dropna().copy()
    n = ret.shape[0]
    if n == 0:
        raise ValueError("无有效收益数据。")
    equity = (1.0 + ret).cumprod()
    total_return = (1.0 + ret).prod() - 1.0
    years = (ret.index[-1] - ret.index[0]).days / 365.25
    annual_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else np.nan
    mean_monthly = ret.mean()
    std_monthly = ret.std(ddof=1)
    mean_return_annual = mean_monthly * 12.0
    vol_annual = std_monthly * math.sqrt(12.0)
    excess_annual = mean_return_annual - rf_annual
    sharpe = excess_annual / vol_annual if vol_annual > 0 else np.nan
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    max_drawdown = drawdown.min()
    calmar = (annual_return - rf_annual) / abs(max_drawdown) if max_drawdown != 0 else np.nan
    win_rate = float((ret > 0).mean())
    t_stat = mean_monthly / (std_monthly / math.sqrt(n)) if std_monthly > 0 and n > 1 else np.nan
    return {
        "months": n,
        "start_date": ret.index[0].date(),
        "end_date": ret.index[-1].date(),
        "total_return": total_return,
        "annual_return": annual_return,
        "mean_return_annual": mean_return_annual,
        "vol_annual": vol_annual,
        "win_rate": win_rate,
        "t_stat": t_stat,
        "rf_annual": rf_annual,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
    }


def save_metrics(metrics: dict, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}_sharpe_metrics.csv")
    pd.Series(metrics).to_csv(out_path)
    print(f"保存夏普指标: {out_path}")


def build_decile_portfolios(
    df_pred: pd.DataFrame,
    n_deciles: int = 10,
    min_stocks: int = 30,
    return_portfolio: bool = True,
):
    """
    经典十等分选股逻辑：
    - 每个月按预测值 pred 升序排序
    - 按排序均分为 n_deciles 份（1=最低预测，n=最高预测）
    - 每一份等权，得到每个 decile 的组合收益
    - 同时构造 D10 - D1 的 long-short 组合收益
    """
    df_pred = df_pred.copy()
    df_pred["YM"] = df_pred["Date"].dt.to_period("M")

    rows = []          # 每月各分组收益
    portfolio_rows = [] if return_portfolio else None

    for ym, g in df_pred.groupby("YM"):
        g = g.dropna(subset=["return", "pred"])
        N = len(g)
        if N < max(min_stocks, n_deciles):
            continue

        r = g["return"].astype(float)
        r = r - r.median()
        g = g.copy()
        g["return"] = r

        # 按预测升序排名：rank 1 = 最低预测
        ranks = g["pred"].rank(method="first", ascending=True)
        # 将 rank 映射到 1..n_deciles
        decile = ((ranks - 1) * n_deciles / N).astype(int) + 1
        decile = decile.clip(1, n_deciles)

        g = g.copy()
        g["decile"] = decile

        # 逐 decile 计算等权收益
        for d in range(1, n_deciles + 1):
            gd = g[g["decile"] == d]
            if gd.empty:
                continue
            dec_ret = float(gd["return"].mean())
            rows.append({
                "date": ym.to_timestamp("M"),
                "decile": d,
                "decile_ret": dec_ret,
                "n_stocks": int(len(gd)),
                "n_total": int(N),
            })

        if return_portfolio:
            for _, row in g.iterrows():
                d = int(row["decile"])
                if d == 1:
                    side = "short"       # 经典 long-short: D10 做多, D1 做空
                elif d == n_deciles:
                    side = "long"
                else:
                    side = "neutral"
                portfolio_rows.append({
                    "date": ym.to_timestamp("M"),
                    "Stkcd": str(row["Stkcd"]),
                    "decile": d,
                    "side": side,
                    "pred": float(row["pred"]),
                    "return": float(row["return"]),
                })

    if not rows:
        empty_ret = pd.DataFrame(columns=["date", "decile", "decile_ret", "n_stocks", "n_total"])
        if return_portfolio:
            empty_port = pd.DataFrame(columns=["date", "Stkcd", "decile", "side", "pred", "return"])
            return empty_ret, empty_port
        return empty_ret, None

    ret_df = pd.DataFrame(rows)
    ret_df["date"] = pd.to_datetime(ret_df["date"])
    ret_df = ret_df.sort_values(["date", "decile"]).reset_index(drop=True)

    # 转成宽表：每月 decile1..decile10 以及 D10-D1 long-short
    wide = ret_df.pivot(index="date", columns="decile", values="decile_ret").sort_index(axis=1)
    wide.columns = [f"decile_{int(c)}_ret" for c in wide.columns]
    if f"decile_1_ret" in wide.columns and f"decile_{n_deciles}_ret" in wide.columns:
        wide["long_short_D10_D1_ret"] = wide[f"decile_{n_deciles}_ret"] - wide["decile_1_ret"]
    wide = wide.reset_index()

    if return_portfolio:
        port_df = pd.DataFrame(
            portfolio_rows,
            columns=["date", "Stkcd", "decile", "side", "pred", "return"],
        )
        port_df["date"] = pd.to_datetime(port_df["date"])
        port_df = port_df.sort_values(["date", "decile", "Stkcd"]).reset_index(drop=True)
        return wide, port_df

    return wide, None


def main():
    for name, pred_csv in PRED_FILES.items():
        if not os.path.isfile(pred_csv):
            print(f"[WARN] 未找到预测文件: {pred_csv}")
            continue

        # 1. 读取逐股票预测
        df_pred = load_predictions(pred_csv)

        # 2. 对齐真实月度收益（同月）
        df_pred = merge_real_returns_to_predictions(
            df_pred,
            REAL_RETURNS_PATH,
            CODE_MAP_TXT_PATH,
            shift_next=False,
        )

        # 3. 构建十等分组合
        decile_returns, decile_port = build_decile_portfolios(
            df_pred,
            n_deciles=10,
            min_stocks=30,
            return_portfolio=True,
        )

        if decile_returns.empty:
            print(f"[WARN] {name} 十等分月度收益为空，可能是样本不足或列缺失。")
            continue

        # 4. 保存结果
        os.makedirs(MODELS_DIR, exist_ok=True)

        out_ret_csv = os.path.join(MODELS_DIR, f"{name}_decile_monthly_returns.csv")
        decile_returns.to_csv(out_ret_csv, index=False, encoding="utf-8-sig")
        print(f"保存十等分月度收益: {out_ret_csv}")

        out_port_csv = os.path.join(MODELS_DIR, f"{name}_decile_portfolio.csv")
        decile_port.to_csv(out_port_csv, index=False, encoding="utf-8-sig")
        print(f"保存十等分选股名单: {out_port_csv}")

        # 基于 D10-D1 long-short 组合计算夏普比率
        if "long_short_D10_D1_ret" in decile_returns.columns:
            decile_returns["date"] = pd.to_datetime(decile_returns["date"])
            ret_series = decile_returns.set_index("date")["long_short_D10_D1_ret"].astype(float)
            metrics = compute_sharpe(ret_series, RF_ANNUAL)
            print(f"{name} 十等分 D10-D1 long-short 组合夏普:")
            print(f"- months={metrics['months']}, start={metrics['start_date']}, end={metrics['end_date']}")
            print(f"- annual_return={metrics['annual_return']:.4f}, vol_annual={metrics['vol_annual']:.4f}, sharpe={metrics['sharpe']:.3f}")
            save_metrics(metrics, MODELS_DIR, f"{name}_decile_long_short_D10_D1")


if __name__ == "__main__":
    main()