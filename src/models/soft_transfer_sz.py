import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from genet import GENet, load_theta0_vector, soft_genet_grid_search

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_PKL = os.path.join(BASE_DIR, "data", "processed", "processed_data.pkl")
MODELS_DIR = os.path.join(BASE_DIR, "output", "models")
SOFT_MODEL_PATH = os.path.join(MODELS_DIR, "soft_sz_genet.pkl")
THETA_HARD_PATH = os.path.join(MODELS_DIR, "theta_hard_sz.pkl")

def main():
    # 加载处理后的数据
    data = joblib.load(PROCESSED_PKL)
    X_bj_train = data["X_target_train"].copy()
    y_bj_train = data["y_target_train"].copy()
    X_bj_test = data["X_target_test"].copy()
    y_bj_test = data["y_target_test"].copy()
    train_mask = X_bj_train.notna().all(axis=1) & y_bj_train.notna()
    X_bj_train = X_bj_train[train_mask]
    y_bj_train = y_bj_train[train_mask]

    # 加载深市硬迁移参数
    theta_sz = joblib.load(THETA_HARD_PATH)
    theta_sz_features = theta_sz.get("feature_names", [])

    # 特征对齐：使用训练集和深市硬迁移参数的交集特征
    train_features = X_bj_train.columns.tolist()
    theta_features = theta_sz_features
    aligned_features = sorted(set(train_features) & set(theta_features))
    print(f"北交所训练集特征数: {len(train_features)}, 深市参数特征数: {len(theta_features)}, 对齐后特征数: {len(aligned_features)}")

    # 使用对齐后的特征
    X_bj_train_aligned = X_bj_train[aligned_features]
    X_bj_test_aligned = X_bj_test[aligned_features]

    theta0_vec = load_theta0_vector(THETA_HARD_PATH, aligned_features)

    # 网格调参：v探索范围，alpha 取 logspace，使用 R^2 指标
    v_grid = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    alpha_grid = np.logspace(-4, 1, 10)
    l1_grid = [0.1, 0.5, 0.9, 1.0]
    soft_model, soft_best = soft_genet_grid_search(
        X_bj_train_aligned, y_bj_train, theta0_vec,
        v_grid=v_grid, alpha_grid=alpha_grid, l1_grid=l1_grid,
        n_splits=10, metric="r2"
    )

    # 对齐测试集列并评估
    valid_mask = X_bj_test_aligned.notna().all(axis=1) & y_bj_test.notna()
    valid_pos = np.where(valid_mask.to_numpy().ravel())[0]
    X_bj_test_valid = X_bj_test_aligned.iloc[valid_pos].copy()
    y_bj_test_valid = y_bj_test.iloc[valid_pos].copy()

    y_pred_bj_soft = soft_model.predict(X_bj_test_valid)
    r2_bj_soft = r2_score(y_bj_test_valid, y_pred_bj_soft)
    mse_bj_soft = mean_squared_error(y_bj_test_valid, y_pred_bj_soft)

    # 保存软迁移模型
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump({
        "coef_": soft_model.coef_,
        "intercept_": soft_model.intercept_,
        "v": soft_model.v,
        "alpha": soft_model.alpha,
        "l1_ratio": soft_model.l1_ratio,
        "theta0_vec": theta0_vec,
        "feature_names": aligned_features,
        "cv_best": soft_best
    }, SOFT_MODEL_PATH)

    print("Soft transfer (GENet) — 深市 — 最优参数：")
    print(f"  v={soft_best['v']}, alpha={soft_best['alpha']}, l1_ratio={soft_best['l1_ratio']}, CV R2={soft_best['score']:.6f}")
    print(f"Soft transfer on BJ — R2: {r2_bj_soft:.6f}, MSE: {mse_bj_soft:.6f}")
    print(f"模型已保存到: {SOFT_MODEL_PATH}")
    info = data["target_test_info"].copy()
    info_valid = info.iloc[valid_pos].copy()
    df_pred_out = pd.DataFrame({
        "Date": pd.to_datetime(info_valid["Date"]),
        "Stkcd": info_valid["Stkcd"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6),
        "pred_calib": y_pred_bj_soft
    })
    os.makedirs(MODELS_DIR, exist_ok=True)
    out_csv = os.path.join(MODELS_DIR, "soft_sz_predictions_oos.csv")
    df_pred_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"预测已保存到: {out_csv}")

if __name__ == "__main__":
    main()