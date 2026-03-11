import os
import joblib
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import sys

# 加载调参工具（使用项目内的 tuning 模块）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from tuning import tune_elasticnet_ts, extract_hard_transfer_params

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PKL_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_data.pkl")
SAVE_DIR = os.path.join(BASE_DIR, "output", "models")

data = joblib.load(PKL_PATH)
X_target_train = data['X_target_train']
y_target_train = data['y_target_train']
X_target_test = data['X_target_test']
y_target_test = data['y_target_test']

# 训练前简单清洗：去除缺失，保证可拟合
mask = X_target_train.notna().all(axis=1) & y_target_train.notna()
X_target_train = X_target_train[mask]
y_target_train = y_target_train[mask]

# 时间索引处理：若没有 DatetimeIndex 或 'Date'，由调参工具回退到 TimeSeriesSplit 十折
if isinstance(X_target_train.index, pd.DatetimeIndex):
    pass
elif 'Date' in X_target_train.columns:
    X_target_train = X_target_train.copy()
    X_target_train['Date'] = pd.to_datetime(X_target_train['Date'])
    X_target_train = X_target_train.set_index('Date')
    y_target_train = y_target_train.loc[X_target_train.index]
# 否则直接用当前索引，调参工具会自动回退到样本顺序十折

# 加载调参工具（使用项目内的 tuning 模块）
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from tuning import tune_elasticnet_ts, extract_hard_transfer_params

# 十折时间序列调参（无日期索引时自动回退到样本顺序 TimeSeriesSplit）
best_estimator, best_params, cv_results = tune_elasticnet_ts(
    X_target_train, y_target_train,
    n_splits=10,
    min_train_years=1,  # 北交所数据量小，最少训练年数设为1
    verbose=1,
    scoring="r2",
)
# 打印最优超参数
print("最优超参数:")
for key, value in best_params.items():
    print(f"  {key}: {value}")

# 保存模型与调参结果
os.makedirs(SAVE_DIR, exist_ok=True)
model_path = os.path.join(SAVE_DIR, "baseline_bj_elasticnet.pkl")
results_path = os.path.join(SAVE_DIR, "baseline_bj_results.pkl")
joblib.dump(best_estimator, model_path)
joblib.dump({'best_params': best_params, 'cv_results': cv_results}, results_path)

baseline_params = extract_hard_transfer_params(best_estimator, feature_names=X_target_train.columns.tolist())
theta_path = os.path.join(SAVE_DIR, "baseline_bj_theta.pkl")
joblib.dump(baseline_params, theta_path)
print(f"模型已保存到: {model_path}")
print(f"调参结果已保存到: {results_path}")
print(f"基线模型参数已保存到: {theta_path}")
print(f"模型与参数已保存到: {SAVE_DIR}")

# 在北交所测试集上评估（特征列对齐后直接用管线预测）
X_bj_test = data['X_target_test'].copy()
y_bj_test = data['y_target_test']
for col in X_target_train.columns:
    if col not in X_bj_test.columns:
        X_bj_test[col] = 0.0
X_bj_test = X_bj_test[X_target_train.columns]

valid_mask = X_bj_test.notna().all(axis=1) & y_bj_test.notna()
# 使用位置索引确保样本一一对应，避免潜在索引错位
valid_pos = np.where(valid_mask.to_numpy().ravel())[0]
X_bj_test_valid = X_bj_test.iloc[valid_pos].copy()
y_bj_test_valid = y_bj_test.iloc[valid_pos].copy()

y_pred_bj = best_estimator.predict(X_bj_test_valid)
r2_bj = r2_score(y_bj_test_valid, y_pred_bj)
mse_bj = mean_squared_error(y_bj_test_valid, y_pred_bj)
print("-" * 40)
print("Baseline（仅北交所数据）表现:")
print(f" - OOS R2: {r2_bj:.6f}")
print(f" - MSE: {mse_bj:.6f}")
print("-" * 40)
info = data["target_test_info"].copy()
info_valid = info.iloc[valid_pos].copy()
df_pred_out = pd.DataFrame({
    "Date": pd.to_datetime(info_valid["Date"]),
    "Stkcd": info_valid["Stkcd"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6),
    "pred_raw": y_pred_bj
})
os.makedirs(SAVE_DIR, exist_ok=True)
out_csv = os.path.join(SAVE_DIR, "baseline_bj_predictions_oos.csv")
df_pred_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"样本外预测已保存到: {out_csv}")