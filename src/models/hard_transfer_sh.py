import os
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import sys

# 加载调参工具（使用项目内的 tuning 模块）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from tuning import tune_elasticnet_ts, extract_hard_transfer_params

# 使用当前目录下的processed_data.pkl
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PKL_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_data.pkl")
SAVE_DIR = os.path.join(BASE_DIR, "output", "models")

data = joblib.load(PKL_PATH)
X_source = data.get('X_source_sh')  # 沪市特征
y_source = data.get('y_source_sh')  # 沪市目标（return）
X_target_test = data['X_target_test']
y_target_test = data['y_target_test']

# 检查数据是否存在
if X_source is None or X_source.empty:
    print(f"[ERROR] 沪市数据不存在或为空！")
    print(f"  X_source_sh: {X_source is not None}")
    if X_source is not None:
        print(f"  X_source shape: {X_source.shape}")
    print(f"  y_source_sh: {y_source is not None}")
    exit(1)

if y_source is None or y_source.empty:
    print(f"[ERROR] 沪市目标数据不存在或为空！")
    exit(1)

print(f"沪市数据加载成功: X shape={X_source.shape}, y shape={y_source.shape}")

# 特征对齐：使用测试集和源域数据的交集特征
target_features = X_target_test.columns.tolist()
source_features = X_source.columns.tolist()
aligned_features = sorted(set(target_features) & set(source_features))
print(f"沪市特征数: {len(source_features)}, 测试集特征数: {len(target_features)}, 对齐后特征数: {len(aligned_features)}")

if len(aligned_features) == 0:
    print(f"[ERROR] 沪市与北交所测试集没有共同特征！")
    print(f"  沪市特征前5个: {source_features[:5] if source_features else 'None'}")
    print(f"  测试集特征前5个: {target_features[:5] if target_features else 'None'}")
    exit(1)

# 使用对齐后的特征
X = X_source[aligned_features]
y = y_source.loc[X.index]  # 确保索引对齐
X_bj_test = X_target_test[aligned_features]

# 训练前简单清洗：去除缺失，保证可拟合
mask = X.notna().all(axis=1) & y.notna()
X = X[mask]
y = y[mask]
print(f"清洗后训练数据: X shape={X.shape}, y shape={y.shape}")

if X.empty:
    print(f"[ERROR] 清洗后训练数据为空！")
    exit(1)

# 时间索引处理：若没有 DatetimeIndex 或 'Date'，由调参工具回退到 TimeSeriesSplit 十折
if isinstance(X.index, pd.DatetimeIndex):
    pass
elif 'Date' in X.columns:
    X = X.copy()
    X['Date'] = pd.to_datetime(X['Date'])
    X = X.set_index('Date')
    y = y.loc[X.index]
# 否则直接用当前索引，调参工具会自动回退到样本顺序十折

# 使用调参工具（默认配置）
best_estimator, best_params, cv_results = tune_elasticnet_ts(X, y)
# 打印最优超参数
print("最优超参数:")
for key, value in best_params.items():
    print(f"  {key}: {value}")

# 保存模型与调参结果
os.makedirs(SAVE_DIR, exist_ok=True)
model_path = os.path.join(SAVE_DIR, "hard_sh_elasticnet.pkl")
results_path = os.path.join(SAVE_DIR, "hard_sh_results.pkl")
joblib.dump(best_estimator, model_path)
joblib.dump({'best_params': best_params, 'cv_results': cv_results}, results_path)

hard_params = extract_hard_transfer_params(best_estimator, feature_names=X.columns.tolist())
theta_path = os.path.join(SAVE_DIR, "theta_hard_sh.pkl")
joblib.dump(hard_params, theta_path)
print(f"模型已保存到: {model_path}")
print(f"调参结果已保存到: {results_path}")
print(f"硬迁移参数已保存到: {theta_path}")
print(f"模型与参数已保存到: {SAVE_DIR}")

# 在北交所测试集上评估
valid_mask = X_bj_test.notna().all(axis=1) & y_target_test.notna()
# 使用位置索引确保样本一一对应，避免潜在索引错位
valid_pos = np.where(valid_mask.to_numpy().ravel())[0]
X_bj_test_valid = X_bj_test.iloc[valid_pos].copy()
y_bj_test_valid = y_target_test.iloc[valid_pos].copy()

y_pred_bj = best_estimator.predict(X_bj_test_valid)
mse_bj = mean_squared_error(y_bj_test_valid, y_pred_bj)
print("-" * 40)
print("Hard Transfer（沪市）迁移至北交所表现:")
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
out_csv = os.path.join(SAVE_DIR, "hard_sh_predictions_oos.csv")
df_pred_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"样本外预测已保存到: {out_csv}")