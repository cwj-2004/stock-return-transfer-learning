import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import sys

# 数据路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PKL_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_data.pkl")
SAVE_DIR = os.path.join(BASE_DIR, "output", "models")

# 加载调参工具
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from tuning import tune_elasticnet_ts, extract_hard_transfer_params

print("=" * 50)
print("=== 两阶段估计模型（沪市版）===")
print("=" * 50)

# 加载数据
data = joblib.load(PKL_PATH)

# 源域数据（沪市）
X_source_sh = data['X_source_sh']
y_source_sh = data['y_source_sh']

# 目标域数据（北交所）
X_target_train = data['X_target_train']
y_target_train = data['y_target_train']
X_target_test = data['X_target_test']
y_target_test = data['y_target_test']

print(f"沪市源域数据: X shape={X_source_sh.shape}")
print(f"北交所训练数据: X shape={X_target_train.shape}")

# ========================================
# 第一阶段：全局因子载荷估计（沪市源域 + 目标域）
# ========================================
print("\n" + "=" * 50)
print("第一阶段：全局因子载荷估计（沪市源域 + 北交所目标域）")
print("=" * 50)

# 特征对齐
source_features = X_source_sh.columns.tolist()
target_features = X_target_train.columns.tolist()
aligned_features = sorted(set(source_features) & set(target_features))
print(f"对齐后特征数: {len(aligned_features)}")

# 合并源域和目标域训练数据
X_source_aligned = X_source_sh[aligned_features].copy()
X_target_aligned = X_target_train[aligned_features].copy()

# 清洗数据
source_mask = X_source_aligned.notna().all(axis=1) & y_source_sh.notna()
X_source_clean = X_source_aligned[source_mask].reset_index(drop=True)
y_source_clean = y_source_sh[source_mask].reset_index(drop=True)

target_mask = X_target_aligned.notna().all(axis=1) & y_target_train.notna()
X_target_clean = X_target_aligned[target_mask].reset_index(drop=True)
y_target_clean = y_target_train[target_mask].reset_index(drop=True)

print(f"沪市源域有效样本: {len(X_source_clean)}")
print(f"目标域有效样本: {len(X_target_clean)}")

# 合并
X_combined = pd.concat([X_source_clean, X_target_clean], axis=0, ignore_index=True)
y_combined = pd.concat([y_source_clean.reset_index(drop=True), y_target_clean.reset_index(drop=True)], axis=0, ignore_index=True)

print(f"第一阶段训练数据: X shape={X_combined.shape}")

# 标准化（使用合并数据的统计量）
combined_scaler = StandardScaler()
X_combined_scaled = combined_scaler.fit_transform(X_combined)

# 第一阶段超参数调优
print("第一阶段超参数调优...")
y_combined_series = pd.Series(y_combined.values if isinstance(y_combined, pd.Series) else y_combined)
best_estimator, best_params, cv_results = tune_elasticnet_ts(
    pd.DataFrame(X_combined_scaled, columns=aligned_features),
    y_combined_series,
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
# 第二阶段：北交所残差回归（带正则化）
# ========================================
print("\n" + "=" * 50)
print("第二阶段：北交所残差回归（带正则化）")
print("=" * 50)

# 使用目标域训练数据
X_bj_train_valid = X_target_clean.copy()
y_bj_train_valid = y_target_clean.copy()

print(f"第二阶段训练数据: X shape={X_bj_train_valid.shape}")

# 标准化训练集（使用第一阶段合并标准化器）
X_bj_train_scaled = combined_scaler.transform(X_bj_train_valid)

# 计算全局因子 F_global = X_bj · theta_G
F_global_train = X_bj_train_scaled @ theta_G['coef_'] + theta_G['intercept_']
print(f"全局因子 F_global (train): shape={F_global_train.shape}")

# 计算残差: y_error = y_actual - y_pred_global
y_pred_global = F_global_train  # 全局模型预测即为全局因子（线性模型系数=1，截距=0）
y_error = y_bj_train_valid - y_pred_global
print(f"残差统计: 均值={y_error.mean():.6f}, 标准差={y_error.std():.6f}")

# 用北交所特征对残差做带正则化的回归（ElasticNet）
second_stage_model, second_stage_params, _ = tune_elasticnet_ts(
    pd.DataFrame(X_bj_train_scaled, columns=aligned_features),
    pd.Series(y_error.values, index=y_bj_train_valid.index),
)
second_stage_coef = second_stage_model.named_steps["elasticnet"].coef_
second_stage_intercept = second_stage_model.named_steps["elasticnet"].intercept_
print(f"残差模型系数: {np.sum(np.abs(second_stage_coef) > 1e-6)} 个非零")
print(f"残差模型截距: {second_stage_intercept:.6f}")

# 对齐测试集特征
X_bj_test = X_target_test[aligned_features]

# 清洗测试集
valid_mask = X_bj_test.notna().all(axis=1) & y_target_test.notna()
valid_pos = np.where(valid_mask.to_numpy().ravel())[0]
X_bj_test_valid = X_bj_test.iloc[valid_pos].copy()
y_bj_test_valid = y_target_test.iloc[valid_pos].copy()

print(f"第二阶段测试数据: X shape={X_bj_test_valid.shape}")

# 标准化测试集
X_bj_test_scaled = combined_scaler.transform(X_bj_test_valid)

# 计算测试集的全局因子
F_global_test = X_bj_test_scaled @ theta_G['coef_'] + theta_G['intercept_']
print(f"全局因子 F_global (test): shape={F_global_test.shape}")

# 计算测试集残差预测
residual_pred = second_stage_model.predict(pd.DataFrame(X_bj_test_scaled, columns=aligned_features))
# 最终预测 = 全局预测 + 残差预测
y_pred = F_global_test + residual_pred

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
joblib.dump(best_estimator, os.path.join(SAVE_DIR, "two_stage_sh_global_model.pkl"))
joblib.dump(combined_scaler, os.path.join(SAVE_DIR, "two_stage_sh_global_scaler.pkl"))

# 保存第二阶段模型
joblib.dump(second_stage_model, os.path.join(SAVE_DIR, "two_stage_sh_second_model.pkl"))

# 保存参数
two_stage_params = {
    'theta_G': theta_G,
    'first_stage_params': best_params,
    'second_stage_params': second_stage_params,
    'second_stage_coef': second_stage_coef,
    'second_stage_intercept': second_stage_intercept,
    'aligned_features': aligned_features
}
joblib.dump(two_stage_params, os.path.join(SAVE_DIR, "two_stage_sh_parameters.pkl"))

# 保存预测结果
info = data["target_test_info"].copy()
info_valid = info.iloc[valid_pos].copy()
df_pred_out = pd.DataFrame({
    "Date": pd.to_datetime(info_valid["Date"]),
    "Stkcd": info_valid["Stkcd"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6),
    "pred_raw": y_pred,
    "global_factor": F_global_test
})
out_csv = os.path.join(SAVE_DIR, "two_stage_sh_predictions_oos.csv")
df_pred_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

print(f"\n模型已保存到: {SAVE_DIR}")
print(f"预测结果已保存到: {out_csv}")
print("=" * 50)
print("两阶段估计（沪市版）完成！")
print("=" * 50)