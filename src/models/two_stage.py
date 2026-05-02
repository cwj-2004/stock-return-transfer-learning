import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import sys

# 数据路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PKL_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_data.pkl")
SAVE_DIR = os.path.join(BASE_DIR, "output", "models")

# 加载调参工具（使用项目内的 tuning 模块）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from tuning import tune_elasticnet_ts, extract_hard_transfer_params

print("=" * 50)
print("=== 两阶段估计模型训练 ===")
print("=" * 50)

# 加载数据
data = joblib.load(PKL_PATH)

# 源域数据（沪深两市合并用于第一阶段）
X_source_sh = data.get('X_source_sh')
y_source_sh = data.get('y_source_sh')
X_source_sz = data.get('X_source_sz')
y_source_sz = data.get('y_source_sz')

# 目标域数据（北交所）
X_target_train = data['X_target_train']
y_target_train = data['y_target_train']
X_target_test = data['X_target_test']
y_target_test = data['y_target_test']

# 检查数据
if X_source_sh is None or X_source_sh.empty:
    print("[ERROR] 沪市数据不存在！")
    exit(1)
if X_source_sz is None or X_source_sz.empty:
    print("[ERROR] 深市数据不存在！")
    exit(1)

print(f"沪市数据: X shape={X_source_sh.shape}, y shape={y_source_sh.shape}")
print(f"深市数据: X shape={X_source_sz.shape}, y shape={y_source_sz.shape}")

# ========================================
# 第一阶段：全局因子载荷估计（沪深两市）
# ========================================
print("\n" + "=" * 50)
print("第一阶段：全局因子载荷估计（沪深两市）")
print("=" * 50)

# 合并沪深两市数据
X_source_all = pd.concat([X_source_sh, X_source_sz], axis=0, ignore_index=True)

# 确保y是Series格式
if isinstance(y_source_sh, pd.DataFrame):
    y_source_sh = y_source_sh.iloc[:, 0]
if isinstance(y_source_sz, pd.DataFrame):
    y_source_sz = y_source_sz.iloc[:, 0]

y_source_all = pd.concat([y_source_sh.reset_index(drop=True), y_source_sz.reset_index(drop=True)], axis=0, ignore_index=True)

# 特征对齐
target_features = X_target_test.columns.tolist()
source_features = X_source_all.columns.tolist()
aligned_features = sorted(set(target_features) & set(source_features))
print(f"对齐后特征数: {len(aligned_features)}")

if len(aligned_features) == 0:
    print("[ERROR] 没有共同特征！")
    exit(1)

# 使用对齐后的特征
X_global = X_source_all[aligned_features]
y_global = y_source_all.loc[X_global.index]

# 清洗缺失值 - 保持DataFrame格式
mask = X_global.notna().all(axis=1) & y_global.notna()
X_global = X_global[mask].reset_index(drop=True)
y_global = y_global[mask].reset_index(drop=True)
print(f"第一阶段训练数据: X shape={X_global.shape}, y shape={y_global.shape}")

if X_global.empty:
    print("[ERROR] 清洗后数据为空！")
    exit(1)

# 标准化 - 转为numpy数组用于训练
global_scaler = StandardScaler()
X_global_scaled = global_scaler.fit_transform(X_global)

# 第一阶段超参数调优
print("第一阶段超参数调优...")
# 确保y_global是Series格式（调参工具需要）
y_global_series = pd.Series(y_global.values if isinstance(y_global, pd.Series) else y_global)
best_estimator, best_params, cv_results = tune_elasticnet_ts(
    pd.DataFrame(X_global_scaled, columns=X_global.columns),  # 转为DataFrame
    y_global_series,
)

print("第一阶段最优超参数:")
for key, value in best_params.items():
    print(f"  {key}: {value}")

# 提取全局因子载荷 theta_G
theta_G = extract_hard_transfer_params(best_estimator, feature_names=aligned_features)
print(f"\n全局因子载荷 theta_G:")
print(f"  非零系数数量: {np.sum(np.abs(theta_G['coef_']) > 1e-6)}")
print(f"  最大系数绝对值: {np.max(np.abs(theta_G['coef_'])):.6f}")

# ========================================
# 第二阶段：北交所回归（仅全局因子）
# ========================================
print("\n" + "=" * 50)
print("第二阶段：北交所回归（仅全局因子）")
print("=" * 50)

# 对齐训练集特征
X_bj_train = X_target_train[aligned_features]

# 清洗训练集
train_mask = X_bj_train.notna().all(axis=1) & y_target_train.notna()
train_pos = np.where(train_mask.to_numpy().ravel())[0]
X_bj_train_valid = X_bj_train.iloc[train_pos].copy()
y_bj_train_valid = y_target_train.iloc[train_pos].copy()

print(f"第二阶段训练数据: X shape={X_bj_train_valid.shape}, y shape={y_bj_train_valid.shape}")

# 标准化训练集（使用全局标准化器）
X_bj_train_scaled = global_scaler.transform(X_bj_train_valid)

# 计算全局因子 F_global = X_bj · theta_G
F_global_train = X_bj_train_scaled @ theta_G['coef_'] + theta_G['intercept_']
print(f"全局因子 F_global (train): shape={F_global_train.shape}")

# 第二阶段模型：用全局因子预测收益（简单线性回归）
second_stage_model = LinearRegression()
F_global_train_reshaped = F_global_train.reshape(-1, 1)
second_stage_model.fit(F_global_train_reshaped, y_bj_train_valid)

print(f"第二阶段系数: {second_stage_model.coef_[0]:.6f}")
print(f"第二阶段截距: {second_stage_model.intercept_:.6f}")

# 对齐测试集特征
X_bj_test = X_target_test[aligned_features]

# 清洗测试集
valid_mask = X_bj_test.notna().all(axis=1) & y_target_test.notna()
valid_pos = np.where(valid_mask.to_numpy().ravel())[0]
X_bj_test_valid = X_bj_test.iloc[valid_pos].copy()
y_bj_test_valid = y_target_test.iloc[valid_pos].copy()

print(f"第二阶段测试数据: X shape={X_bj_test_valid.shape}, y shape={y_bj_test_valid.shape}")

# 标准化测试集（使用全局标准化器）
X_bj_test_scaled = global_scaler.transform(X_bj_test_valid)

# 计算测试集的全局因子
F_global_test = X_bj_test_scaled @ theta_G['coef_'] + theta_G['intercept_']
print(f"全局因子 F_global (test): shape={F_global_test.shape}")

# 最终预测
F_global_test_reshaped = F_global_test.reshape(-1, 1)
y_pred = second_stage_model.predict(F_global_test_reshaped)

# ========================================
# 模型评估
# ========================================
print("\n" + "=" * 50)
print("模型评估结果")
print("=" * 50)

mse = mean_squared_error(y_bj_test_valid, y_pred)

print(f"MSE: {mse:.6f}")

# ========================================
# 保存模型和结果
# ========================================
os.makedirs(SAVE_DIR, exist_ok=True)

# 保存第一阶段模型和标准化器
joblib.dump(best_estimator, os.path.join(SAVE_DIR, "two_stage_global_model.pkl"))
joblib.dump(global_scaler, os.path.join(SAVE_DIR, "two_stage_global_scaler.pkl"))

# 保存第二阶段模型
joblib.dump(second_stage_model, os.path.join(SAVE_DIR, "two_stage_second_model.pkl"))

# 保存参数
two_stage_params = {
    'theta_G': theta_G,
    'first_stage_params': best_params,
    'second_stage_coef': second_stage_model.coef_,
    'second_stage_intercept': second_stage_model.intercept_,
    'aligned_features': aligned_features
}
joblib.dump(two_stage_params, os.path.join(SAVE_DIR, "two_stage_parameters.pkl"))

# 保存预测结果
info = data["target_test_info"].copy()
info_valid = info.iloc[valid_pos].copy()
df_pred_out = pd.DataFrame({
    "Date": pd.to_datetime(info_valid["Date"]),
    "Stkcd": info_valid["Stkcd"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6),
    "pred_raw": y_pred,
    "global_factor": F_global_test
})
out_csv = os.path.join(SAVE_DIR, "two_stage_predictions_oos.csv")
df_pred_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

print(f"\n模型已保存到: {SAVE_DIR}")
print(f"预测结果已保存到: {out_csv}")
print("=" * 50)
print("两阶段估计训练完成！")
print("=" * 50)