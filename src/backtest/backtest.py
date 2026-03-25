import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 配置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CSV_PATH = os.path.join(BASE_DIR, "output", "models", "hard_transfer_elasticnet_monthly_returns.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "models")
PLOTS_DIR = os.path.join(BASE_DIR, "output", "plots")
STRATEGY_COL = "long_short_ret"
RF_ANNUAL = 0.0
REAL_RETURNS_PATH = os.path.join(BASE_DIR, "data", "raw", "TRD_Mnth.csv")
CODE_MAP_TXT_PATH = os.path.join(BASE_DIR, "data", "raw", "对照.txt")

def load_returns(csv_path: str, col: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    if "date" not in df.columns or col not in df.columns:
        raise ValueError(f"CSV 必须包含 'date' 和 '{col}' 列。实际列: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    ret = df[col].astype(float)
    ret.index = df["date"]
    ret = ret.dropna()
    return ret

def load_predictions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Date" not in df.columns:
        raise ValueError(f"CSV 必须包含 'Date' 列。实际列: {list(df.columns)}")
    if "Stkcd" not in df.columns:
        for c in ["Stkcd", "stkcd", "Code", "code"]:
            if c in df.columns:
                df = df.rename(columns={c: "Stkcd"})
                break
    if "Stkcd" not in df.columns:
        raise ValueError("预测CSV需包含股票代码列 'Stkcd'")
    pred_cols_pref = ["pred_calib", "pred_raw", "pred_recenter", "pred"]
    pred_col = next((c for c in pred_cols_pref if c in df.columns), None)
    if pred_col is None:
        raise ValueError(f"未找到预测列，需包含任一 {pred_cols_pref}。实际列: {list(df.columns)}")
    df["Date"] = pd.to_datetime(df["Date"])    
    df = df.dropna(subset=["Date", "Stkcd", pred_col]).copy()
    df = df.rename(columns={pred_col: "pred"})
    return df[["Date", "Stkcd", "pred"]].copy()

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
    for c in ["Mretwd", "Mret", "RET", "Ret", "return", "mret"]:
        if c in df.columns:
            return c
    nums = df.select_dtypes(include=["number"]).columns.tolist()
    return nums[0] if nums else "return"

def load_real_monthly_returns(path: str = REAL_RETURNS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, dtype={"Stkcd": str})
    if "Stkcd" not in df.columns:
        for c in ["Stkcd", "Stkcode", "Code", "stkcd"]:
            if c in df.columns:
                df = df.rename(columns={c: "Stkcd"})
                break
    df["Stkcd"] = df["Stkcd"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    mcol = _detect_month_col(df)
    df["Date"] = _to_month_end_series(df[mcol])
    # Try to compute real monthly returns from close price if available
    price_col = None
    for c in ["Mclsprc", "Clsprc", "Close", "ClsPrice"]:
        if c in df.columns:
            price_col = c
            break
    if price_col is not None:
        tmp = df[["Stkcd", "Date", price_col]].copy()
        tmp = tmp.dropna(subset=["Date"]).sort_values(["Stkcd", "Date"]).reset_index(drop=True)
        tmp[price_col] = pd.to_numeric(tmp[price_col], errors="coerce")
        tmp["real_return"] = tmp.groupby("Stkcd")[price_col].pct_change()
        tmp = tmp.drop(columns=[price_col])
        tmp = tmp.dropna(subset=["Date"]).reset_index(drop=True)
        return tmp
    # Fallback to existing numeric return column
    rcol = _detect_return_col(df)
    df = df[["Stkcd", "Date", rcol]].copy()
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    df = df.rename(columns={rcol: "real_return"})
    df = df.sort_values(["Stkcd", "Date"]).reset_index(drop=True)
    return df

def load_code_mapping_from_txt(txt_path: str = CODE_MAP_TXT_PATH) -> pd.DataFrame | None:
    # 优先尝试加载CSV格式的代码映射
    csv_path = os.path.join(BASE_DIR, "data", "raw", "code_mapping.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "old_code" in df.columns and "new_code" in df.columns:
            df["old_code"] = df["old_code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
            df["new_code"] = df["new_code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
            print(f"[回测] 从CSV加载代码映射: {len(df)} 条")
            return df

    # 回退到txt格式
    if not os.path.exists(txt_path):
        return None
    rows = []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            nums = pd.Series(np.array(__import__("re").findall(r"(\d{6,})", line)))
            if nums.size >= 2:
                old_code, new_code = str(nums.iloc[-2]), str(nums.iloc[-1])
                rows.append((old_code, new_code))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["old_code", "new_code"]).drop_duplicates()
    df["old_code"] = df["old_code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    df["new_code"] = df["new_code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    print(f"[回测] 从TXT加载代码映射: {len(df)} 条")
    return df

def apply_code_mapping(df: pd.DataFrame, map_df: pd.DataFrame | None) -> pd.DataFrame:
    if map_df is None or map_df.empty:
        return df
    mp = dict(zip(map_df["old_code"], map_df["new_code"]))
    out = df.copy()
    out["Stkcd"] = out["Stkcd"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    out["Stkcd"] = out["Stkcd"].map(lambda x: mp.get(x, x))
    return out

def normalize_stkcd_prefix(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Stkcd"] = out["Stkcd"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    out["Stkcd"] = out["Stkcd"].str.replace(r"^43", "83", regex=True)
    return out

def merge_real_returns_to_predictions(df_pred: pd.DataFrame,
                                      real_path: str = REAL_RETURNS_PATH,
                                      map_txt_path: str = CODE_MAP_TXT_PATH,
                                      shift_next: bool = True) -> pd.DataFrame:
    # 从统一的月度真实收益文件加载
    real_df = load_real_monthly_returns(real_path)

    # 加载代码映射表（旧代码 -> 新代码）
    map_df = load_code_mapping_from_txt(map_txt_path)

    df = df_pred.copy()
    df["Stkcd"] = df["Stkcd"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    real_df["Stkcd"] = real_df["Stkcd"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)

    # 将预测文件的旧代码映射为新代码，以便与真实收益文件匹配
    if map_df is not None:
        mapping_dict = dict(zip(map_df["old_code"], map_df["new_code"]))
        before_map = df["Stkcd"].unique()
        df["Stkcd"] = df["Stkcd"].map(lambda x: mapping_dict.get(x, x))
        mapped = df["Stkcd"].unique()
        print(f"[回测] 代码映射: 原始 {len(before_map)} 个，映射后 {len(mapped)} 个")

    valid_codes = set(real_df["Stkcd"].unique())
    before = len(df)
    df = df[df["Stkcd"].isin(valid_codes)].copy()
    print(f"[回测] 预测行: {before}，可匹配代码行: {len(df)}，真实收益代码数: {len(valid_codes)}")
    df["Date"] = pd.to_datetime(df["Date"]).dt.to_period("M").dt.to_timestamp("M")
    if shift_next:
        df["Date_real"] = df["Date"] + pd.offsets.MonthEnd(1)
    else:
        df["Date_real"] = df["Date"]
    months_pred = set(df["Date_real"].dt.to_period("M").unique())
    months_real = set(real_df["Date"].dt.to_period("M").unique())
    inter = months_pred & months_real
    print(f"[回测] 可匹配月份数: {len(inter)}")
    merged = df.merge(real_df, left_on=["Stkcd", "Date_real"], right_on=["Stkcd", "Date"], how="left")
    merged["return"] = merged["real_return"].astype(float)
    if "Date_x" in merged.columns:
        merged = merged.rename(columns={"Date_x": "Date"})
    if "Date_y" in merged.columns:
        merged = merged.drop(columns=["Date_y"])
    merged = merged.dropna(subset=["return", "pred"])
    return merged[["Date", "Stkcd", "return", "pred"]].copy()

def build_long_short_returns(df_pred: pd.DataFrame,
                             min_stocks: int = 10,
                             return_portfolio: bool = False):
    df_pred = df_pred.copy()
    df_pred["YM"] = df_pred["Date"].dt.to_period("M")
    rows = []
    portfolio_rows = [] if return_portfolio else None
    for ym, g in df_pred.groupby("YM"):
        g = g.dropna(subset=["return", "pred"])
        if len(g) < min_stocks:
            continue
        r = g["return"].astype(float)
        r = r - r.median()
        g = g.copy()
        g["return"] = r
        ranks = g["pred"].rank(method="first", ascending=True)
        N = len(g)
        R_bar = (N + 1) / 2.0
        z = ranks - R_bar
        if np.allclose(z.values, 0.0):
            continue
        denom = np.abs(z).sum()
        if denom == 0:
            continue
        w = 2.0 * z / denom
        g = g.copy()
        g["w"] = w
        r = g["return"].astype(float)
        w_pos = g["w"].where(g["w"] > 0.0, 0.0)
        w_neg = g["w"].where(g["w"] < 0.0, 0.0)
        long_ret = float((w_pos * r).sum())
        short_ret = float((-w_neg * r).sum())
        long_short_ret = float((g["w"] * r).sum())
        n_long = int((g["w"] > 0.0).sum())
        n_short = int((g["w"] < 0.0).sum())
        rows.append({
            "date": ym.to_timestamp(),
            "long_ret": long_ret,
            "short_ret": short_ret,
            "long_short_ret": long_short_ret,
            "n_long": n_long,
            "n_short": n_short,
            "n_total": int(len(g)),
        })
        if return_portfolio:
            for _, row in g.iterrows():
                if row["w"] == 0.0:
                    continue
                portfolio_rows.append({
                    "date": ym.to_timestamp(),
                    "Stkcd": str(row["Stkcd"]),
                    "side": "long" if row["w"] > 0.0 else "short",
                    "pred": float(row["pred"]),
                    "return": float(row["return"]),
                    "weight": float(row["w"]),
                })
    if not rows:
        empty_ret = pd.DataFrame(columns=[
            "date", "long_ret", "short_ret", "long_short_ret", "n_long", "n_short", "n_total"
        ])
        if return_portfolio:
            empty_port = pd.DataFrame(columns=["date", "Stkcd", "side", "pred", "return", "weight"])
            return empty_ret, empty_port
        return empty_ret
    ret_df = pd.DataFrame(rows)
    ret_df["date"] = pd.to_datetime(ret_df["date"])
    ret_df = ret_df.sort_values("date").reset_index(drop=True)
    if return_portfolio:
        port_df = pd.DataFrame(portfolio_rows, columns=["date", "Stkcd", "side", "pred", "return", "weight"])
        port_df["date"] = pd.to_datetime(port_df["date"])
        port_df = port_df.sort_values(["date", "side", "Stkcd"]).reset_index(drop=True)
        return ret_df, port_df
    return ret_df

def compute_metrics(ret: pd.Series, rf_annual: float = 0.0) -> dict:
    n = ret.shape[0]
    if n == 0:
        raise ValueError("无有效收益数据。")
    # 累计净值曲线
    equity = (1.0 + ret).cumprod()
    # 总体收益与年化收益率（几何）
    total_return = (1.0 + ret).prod() - 1.0
    years = (ret.index[-1] - ret.index[0]).days / 365.25
    annual_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else np.nan
    # 月度均值与波动
    mean_monthly = ret.mean()
    std_monthly = ret.std(ddof=1)
    # 年化平均收益（简单）
    mean_return_annual = mean_monthly * 12.0
    # 年化波动率（按月频，×√12）
    vol_annual = std_monthly * math.sqrt(12)
    # 年化超额收益（扣无风险）
    excess_annual = mean_return_annual - rf_annual
    sharpe = excess_annual / vol_annual if vol_annual > 0 else np.nan
    # 最大回撤
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    max_drawdown = drawdown.min()
    # Calmar 比率
    calmar = (annual_return - rf_annual) / abs(max_drawdown) if max_drawdown != 0 else np.nan
    # 月度胜率（long_short_ret > 0 的比例）
    win_rate = float((ret > 0).mean())
    # 收益 t 统计量（检验均值是否显著非零）
    t_stat = mean_monthly / (std_monthly / math.sqrt(n)) if std_monthly > 0 and n > 1 else np.nan
    return {
        "start_date": ret.index[0].date(),
        "end_date": ret.index[-1].date(),
        "months": n,
        "total_return": total_return,
        "annual_return": annual_return,
        "mean_return_annual": mean_return_annual,
        "vol_annual": vol_annual,
        "win_rate": win_rate,
        "t_stat": t_stat,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
    }, equity, drawdown

def save_report(metrics: dict,
                equity: pd.Series,
                drawdown: pd.Series,
                out_dir: str,
                name: str,
                market_equity: pd.Series | None = None,
                plots_dir: str = None):
    os.makedirs(out_dir, exist_ok=True)
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
    # 指标表
    report_path = os.path.join(out_dir, f"{name}_backtest_report.csv")
    pd.Series(metrics).to_csv(report_path)
    # 净值曲线 - 保存到 plots 目录
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    equity.plot(ax=ax[0], color="tab:blue", label="Strategy")
    if market_equity is not None:
        mk = market_equity.reindex(equity.index).ffill()
        mk.plot(ax=ax[0], color="gray", linestyle="--", alpha=0.8, label="Market")
        ax[0].legend()
    ax[0].set_title(f"{name} Equity Curve")
    ax[0].grid(True, alpha=0.3)
    max_dd_series = drawdown.cummin()
    max_dd_series.plot(ax=ax[1], color="tab:red", label="Max Drawdown")
    ax[1].set_title(f"{name} Max Drawdown Curve")
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    # 使用 plots 目录（如果提供）或默认输出目录
    curve_path = os.path.join(plots_dir or out_dir, f"{name}_equity_curve.png")
    fig.savefig(curve_path, dpi=160)
    plt.close(fig)
    print(f"保存报告: {report_path}")
    print(f"保存净值图: {curve_path}")

def plot_combined_equity_curves(equity_data: dict,
                                out_dir: str,
                                title: str = "Equity Curve Comparison",
                                output_name: str = "combined_comparison.png"):
    """
    在同一坐标系中绘制多个模型的净值曲线

    Args:
        equity_data: dict, {model_name: equity_series}
        out_dir: str, 输出目录
        title: str, 图表标题
        output_name: str, 输出文件名
    """
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    # 定义颜色方案
    color_map = {
        "soft_sh_sz": "tab:blue",
        "two_stage": "tab:red",
        "soft_sh": "tab:green",
        "soft_sz": "tab:orange",
    }

    # 绘制每条曲线
    for name, equity in equity_data.items():
        color = color_map.get(name, None)
        equity.plot(ax=ax, label=name, color=color)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path = os.path.join(out_dir, output_name)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"保存对比图: {output_path}")

def main():
    # 所有模型逐股票预测 CSV（使用相对路径）
    models_dir = os.path.join(BASE_DIR, "output", "models")
    baseline_pred_csv = os.path.join(models_dir, "baseline_bj_predictions_oos.csv")
    hard_sh_pred_csv = os.path.join(models_dir, "hard_sh_predictions_oos.csv")
    hard_sz_pred_csv = os.path.join(models_dir, "hard_sz_predictions_oos.csv")
    hard_pred_csv = os.path.join(models_dir, "hard_transfer_predictions_oos.csv")  # 双市场
    soft_sh_pred_csv = os.path.join(models_dir, "soft_sh_predictions_oos.csv")
    soft_sz_pred_csv = os.path.join(models_dir, "soft_sz_predictions_oos.csv")
    soft_pred_csv = os.path.join(models_dir, "soft_transfer_predictions_oos.csv")  # 双市场
    two_stage_pred_csv = os.path.join(models_dir, "two_stage_predictions_oos.csv")  # 两阶段估计

    # 存储用于对比的净值曲线
    comparison_equity = {}

    # 确保 plots 目录存在
    os.makedirs(PLOTS_DIR, exist_ok=True)

    for name, pred_csv in [
        ("baseline_bj", baseline_pred_csv),
        ("hard_sh", hard_sh_pred_csv),
        ("hard_sz", hard_sz_pred_csv),
        ("hard_sh_sz", hard_pred_csv),  # 双市场
        ("soft_sh", soft_sh_pred_csv),
        ("soft_sz", soft_sz_pred_csv),
        ("soft_sh_sz", soft_pred_csv),  # 双市场
        ("two_stage", two_stage_pred_csv),  # 两阶段估计
    ]:
        if not os.path.isfile(pred_csv):
            print(f"[WARN] 未找到预测文件: {pred_csv}")
            continue
        df_pred = load_predictions(pred_csv)
        df_pred = merge_real_returns_to_predictions(df_pred, REAL_RETURNS_PATH, CODE_MAP_TXT_PATH, shift_next=True)

        x = df_pred["pred"].astype(float)
        y = df_pred["return"].astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        x_fit = x[mask]
        y_fit = y[mask]
        corr = x_fit.corr(y_fit) if x_fit.shape[0] > 1 else np.nan
        r2 = float(corr ** 2) if not np.isnan(corr) else np.nan

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(x_fit, y_fit, s=4, alpha=0.3)
        if x_fit.shape[0] > 1:
            coef = np.polyfit(x_fit.values, y_fit.values, 1)
            x_line = np.linspace(x_fit.min(), x_fit.max(), 100)
            y_line = coef[0] * x_line + coef[1]
            ax.plot(x_line, y_line, color="red", linewidth=1.0, label="OLS fit")
            ax.legend()
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.8)
        title = f"{name} pred vs real return"
        if not np.isnan(corr):
            title += f" (r={corr:.3f}, R2={r2:.4f})"
            print(f"{name} 整体线性相关: r={corr:.3f}, R2={r2:.4f}")
        ax.set_title(title)
        ax.set_xlabel("pred")
        ax.set_ylabel("real_return")
        fig.tight_layout()
        scatter_path = os.path.join(PLOTS_DIR, f"{name}_pred_vs_real_scatter.png")
        fig.savefig(scatter_path, dpi=160)
        plt.close(fig)
        print(f"保存散点图: {scatter_path}")

        df_pred["YM"] = pd.to_datetime(df_pred["Date"]).dt.to_period("M")
        market_ret = df_pred.groupby("YM")["return"].mean()
        market_ret.index = market_ret.index.to_timestamp()
        market_ret = market_ret.sort_index()
        monthly_ret, portfolio_df = build_long_short_returns(df_pred, min_stocks=3, return_portfolio=True)
        if monthly_ret.empty:
            print(f"[WARN] {name} 月度收益为空，可能是每月样本不足或列缺失。")
            continue
        out_csv = os.path.join(OUTPUT_DIR, f"{name}_monthly_returns.csv")
        monthly_ret.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"保存月度收益: {out_csv}")
        portfolio_csv = os.path.join(OUTPUT_DIR, f"{name}_long_short_portfolio.csv")
        portfolio_df.to_csv(portfolio_csv, index=False, encoding="utf-8-sig")
        print(f"保存long-short组合: {portfolio_csv}")
        ret = monthly_ret[STRATEGY_COL].astype(float)
        ret.index = pd.to_datetime(monthly_ret["date"])
        metrics, equity, drawdown = compute_metrics(ret, RF_ANNUAL)
        market_ret_aligned = market_ret.reindex(ret.index).fillna(0.0)
        market_equity = (1.0 + market_ret_aligned).cumprod()
        print(f"{name} 回测指标：")
        for k, v in metrics.items():
            print(f"- {k}: {v}")
        save_report(metrics, equity, drawdown, OUTPUT_DIR, f"{name}_{STRATEGY_COL}", market_equity=market_equity, plots_dir=PLOTS_DIR)

        # 存储净值曲线用于对比
        comparison_equity[name] = equity

    # 生成软迁移 vs 两阶段估计对比图
    if "soft_sh_sz" in comparison_equity and "two_stage" in comparison_equity:
        plot_combined_equity_curves(
            {"soft_sh_sz": comparison_equity["soft_sh_sz"], "two_stage": comparison_equity["two_stage"]},
            PLOTS_DIR,
            title="软迁移 vs 两阶段估计 - 净值曲线对比",
            output_name="soft_vs_two_stage_comparison.png"
        )

if __name__ == "__main__":
    main()