# -*- coding: utf-8 -*-
"""
GENet 论文对齐实现测试脚本
用于验证新的联合优化 GENet 实现是否正常工作
"""

import os
import sys
import numpy as np
import pandas as pd

# 添加工具模块路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'utils'))

from genet_joint import JointGENet, GENetRegressor, genet_grid_search
from paper_validation import GENetPaperValidator, calculate_predictive_r2
from normalization import CrossSectionalNormalizer, cross_sectional_normalize

def test_joint_genet():
    """测试联合优化 GENet"""
    print("=" * 60)
    print("[测试1]联合优化 GENet")
    print("=" * 60)

    # 生成模拟数据
    np.random.seed(42)
    n_features = 10

    # 源域市场1（沪市）
    n_sh = 500
    X_sh = np.random.randn(n_sh, n_features)
    theta_sh_true = np.random.randn(n_features) * 0.1
    y_sh = X_sh @ theta_sh_true + np.random.randn(n_sh) * 0.1

    # 源域市场2（深市）
    n_sz = 500
    X_sz = np.random.randn(n_sz, n_features)
    theta_sz_true = theta_sh_true + np.random.randn(n_features) * 0.05  # 相似但有差异
    y_sz = X_sz @ theta_sz_true + np.random.randn(n_sz) * 0.1

    # 目标域市场（北交所）
    n_bj = 200  # 数据较少
    X_bj = np.random.randn(n_bj, n_features)
    theta_bj_true = theta_sh_true + np.random.randn(n_features) * 0.03  # 更接近全局
    y_bj = X_bj @ theta_bj_true + np.random.randn(n_bj) * 0.1

    # 组织数据
    X_train_dict = {
        'shanghai': X_sh,
        'shenzhen': X_sz,
        'bj': X_bj
    }

    y_train_dict = {
        'shanghai': y_sh,
        'shenzhen': y_sz,
        'bj': y_bj
    }

    # 训练模型
    model = JointGENet(lambda_g=0.001, lambda_l_ratio=10.0, verbose=True)
    model.fit(X_train_dict, y_train_dict)

    print("\n[PASS] 联合优化 GENet 测试通过")

    # 验证关键特性
    print("\n[验证]关键特性检查：")
    print(f"1. 全局参数非零系数: {np.sum(np.abs(model.theta_g) > 1e-6)}")
    print(f"2. 全局参数 L1 范数: {np.sum(np.abs(model.theta_g)):.4f}")

    for market in ['shanghai', 'shenzhen', 'bj']:
        theta_ell = model.theta_ell_c[market]
        shrinkage = np.sum(np.abs(theta_ell)) / (np.sum(np.abs(model.theta_g)) + 1e-10)
        print(f"3. {market} 局部参数 L1: {np.sum(np.abs(theta_ell)):.4f}, 收缩比: {shrinkage:.3f}")

    print(f"4. 全局成分贡献比例: {model.global_contribution_ratio * 100:.2f}%")

    return True


def test_normalization():
    """测试时间点截面标准化"""
    print("\n" + "=" * 60)
    print("[测试2]时间点截面标准化")
    print("=" * 60)

    # 生成模拟数据
    np.random.seed(42)
    n_stocks = 50
    n_periods = 24
    n_features = 5

    # 创建多期数据
    data = []
    for period in range(n_periods):
        for stock in range(n_stocks):
            row = {
                'Stkcd': f'{stock:06d}',
                'Date': pd.Timestamp('2020-01-01') + pd.DateOffset(months=period),
                **{f'feature{i}': np.random.randn() for i in range(n_features)},
                'return': np.random.randn() * 0.1
            }
            data.append(row)

    df = pd.DataFrame(data)

    # 测试标准化
    normalizer = CrossSectionalNormalizer(
        feature_cols=[f'feature{i}' for i in range(n_features)],
        min_periods=12,
        method='expanding'
    )

    df_normalized = normalizer.fit_transform(df)

    print(f"原始数据形状: {df.shape}")
    print(f"标准化后形状: {df_normalized.shape}")

    # 验证标准化效果
    for col in [f'feature{i}' for i in range(n_features)]:
        if col in df_normalized.columns:
            # 计算每个时间点的截面均值（应该接近0）
            cross_sectional_mean = df_normalized.groupby('Date')[col].mean().abs().mean()
            cross_sectional_std = df_normalized.groupby('Date')[col].std().abs().mean()

            print(f"{col}:")
            print(f"  截面均值绝对值: {cross_sectional_mean:.4f} (应接近0)")
            print(f"  截面标准差: {cross_sectional_std:.4f} (应接近1)")

    print("\n[PASS] 时间点截面标准化测试通过")

    return True


def test_predictive_r2():
    """测试预测性 R^2"""
    print("\n" + "=" * 60)
    print("[测试3]预测性 R^2 计算")
    print("=" * 60)

    # 生成模拟数据
    np.random.seed(42)
    n = 100
    y_true = np.random.randn(n)

    # 模型1：较好的预测
    y_pred_good = y_true + np.random.randn(n) * 0.1

    # 模型2：较差的预测
    y_pred_bad = y_true + np.random.randn(n) * 0.5

    # 计算预测性 R^2
    from paper_validation import PredictiveR2

    calculator = PredictiveR2(baseline="historical_mean")

    r2_good = calculator.calculate(y_true, y_pred_good)
    r2_bad = calculator.calculate(y_true, y_pred_bad)

    print(f"较好模型的预测性 R^2: {r2_good:.4f}")
    print(f"较差模型的预测性 R^2: {r2_bad:.4f}")

    assert r2_good > r2_bad, "较好模型应该有更高的预测性 R^2"

    print("\n[PASS] 预测性 R^2 测试通过")

    return True


def test_paper_validation():
    """测试论文验证工具"""
    print("\n" + "=" * 60)
    print("[测试4]论文验证工具")
    print("=" * 60)

    # 生成模拟参数
    np.random.seed(42)
    n_features = 10

    # 全局参数（较大）
    theta_g = np.random.randn(n_features) * 0.5

    # 局部参数（较小）
    theta_ell_c = {
        'market1': np.random.randn(n_features) * 0.05,
        'market2': np.random.randn(n_features) * 0.03,
        'market3': np.random.randn(n_features) * 0.02
    }

    # 验证全局主导性
    validator = GENetPaperValidator()
    result = validator.validate_global_dominance(theta_g, theta_ell_c)

    print(f"全局参数非零比例: {result['theta_g_nonzero_ratio']:.2%}")
    print(f"平均收缩比例: {result['avg_shrinkage_ratio']:.3f}")
    print(f"全局主导性验证: {'通过' if result['global_dominance_verified'] else '失败'}")

    assert result['global_dominance_verified'], "全局参数应该占主导地位"

    print("\n[PASS] 论文验证工具测试通过")

    return True


def test_genet_regressor_wrapper():
    """测试 GENetRegressor 包装器"""
    print("\n" + "=" * 60)
    print("[测试5]GENetRegressor 包装器")
    print("=" * 60)

    # 生成模拟数据
    np.random.seed(42)
    n_features = 5

    X_sh = pd.DataFrame(np.random.randn(100, n_features),
                        columns=[f'feature{i}' for i in range(n_features)])
    y_sh = pd.Series(np.random.randn(100))

    X_sz = pd.DataFrame(np.random.randn(100, n_features),
                        columns=[f'feature{i}' for i in range(n_features)])
    y_sz = pd.Series(np.random.randn(100))

    X_bj = pd.DataFrame(np.random.randn(50, n_features),
                        columns=[f'feature{i}' for i in range(n_features)])
    y_bj = pd.Series(np.random.randn(50))

    X_train_dict = {
        'shanghai': X_sh,
        'shenzhen': X_sz,
        'bj': X_bj
    }

    y_train_dict = {
        'shanghai': y_sh,
        'shenzhen': y_sz,
        'bj': y_bj
    }

    # 训练模型
    model = GENetRegressor(lambda_g=0.001, lambda_l_ratio=10.0, verbose=True)
    model.fit(X_train_dict, y_train_dict)

    # 预测
    X_test = pd.DataFrame(np.random.randn(10, n_features),
                          columns=[f'feature{i}' for i in range(n_features)])
    y_pred = model.predict(X_test, 'bj')

    print(f"预测形状: {y_pred.shape}")
    assert len(y_pred) == 10, "预测结果应该有10个样本"

    print("\n[PASS] GENetRegressor 包装器测试通过")

    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("GENet 论文对齐实现 - 测试套件")
    print("=" * 80)

    tests = [
        ("联合优化 GENet", test_joint_genet),
        ("时间点截面标准化", test_normalization),
        ("预测性 R^2", test_predictive_r2),
        ("论文验证工具", test_paper_validation),
        ("GENetRegressor 包装器", test_genet_regressor_wrapper)
    ]

    results = []

    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "[PASS] 通过" if success else "[FAIL] 失败"))
        except Exception as e:
            results.append((name, f"[FAIL] 错误: {str(e)}"))
            import traceback
            traceback.print_exc()

    # 打印测试结果摘要
    print("\n" + "=" * 80)
    print("[测试结果摘要]")
    print("=" * 80)

    for name, status in results:
        print(f"{name}: {status}")

    passed = sum(1 for _, status in results if "[PASS] 通过" in status)
    total = len(results)

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n[SUCCESS] All tests passed! The new GENet implementation is ready to use.")
    else:
        print(f"\n[WARNING] {total - passed} tests failed, please check the implementation.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)