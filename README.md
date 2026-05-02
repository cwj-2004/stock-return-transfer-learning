# 股票收益预测迁移学习项目

## 项目简介

本项目研究使用迁移学习方法，将沪深两市股票收益预测模型迁移到北交所市场。实现了基线模型、硬迁移、软迁移、两阶段估计等多种方法。

## 项目结构

```
stock-return-transfer-learning/
├── src/                # 源代码
│   ├── data/          # 数据处理
│   │   └── processor.py
│   ├── models/        # 模型实现
│   │   ├── baseline.py         # 基线模型（纯目标域）
│   │   ├── hard_transfer.py    # 硬迁移（沪+深）
│   │   ├── hard_transfer_sh.py # 硬迁移（沪市）
│   │   ├── hard_transfer_sz.py # 硬迁移（深市）
│   │   ├── soft_transfer.py    # 软迁移（沪+深）
│   │   ├── soft_transfer_sh.py # 软迁移（沪市）
│   │   ├── soft_transfer_sz.py # 软迁移（深市）
│   │   ├── two_stage.py        # 两阶段估计
│   │   └── genet_joint_train.py # GENet 联合优化
│   ├── utils/         # 工具函数
│   │   ├── genet.py              # GENet 实现
│   │   ├── genet_joint.py       # 联合优化 GENet
│   │   ├── normalization.py     # 标准化工具
│   │   ├── paper_validation.py  # 论文验证工具
│   │   ├── tuning.py            # 调参工具
│   │   └── extract_code_mapping.py # 代码映射提取
│   └── backtest/      # 回测分析
│       └── backtest.py          # 回测引擎（十等分 Long-Short）
├── tests/             # 测试脚本
│   └── test_genet_alignment.py # GENet 测试套件
├── data/              # 数据文件
│   ├── raw/          # 原始数据
│   └── processed/    # 处理后数据
└── output/            # 输出结果
    ├── models/       # 模型文件和预测
    └── plots/        # 可视化图表
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 提取代码映射（首次使用）

从 PDF 文件提取北交所股票代码映射表：

```bash
python src/utils/extract_code_mapping.py
```

### 2. 数据处理

```bash
python src/data/processor.py
```

### 3. 模型训练

**基线模型**（纯目标域）：
```bash
python src/models/baseline.py
```

**硬迁移**：
```bash
python src/models/hard_transfer.py        # 双市场
python src/models/hard_transfer_sh.py     # 沪市
python src/models/hard_transfer_sz.py     # 深市
```

**软迁移**：
```bash
python src/models/soft_transfer.py        # 双市场
python src/models/soft_transfer_sh.py     # 沪市
python src/models/soft_transfer_sz.py     # 深市
```

**两阶段估计**：
```bash
python src/models/two_stage.py
```

**GENet 联合优化**：
```bash
python src/models/genet_joint_train.py
```

### 4. 回测分析

```bash
python src/backtest/backtest.py
```

回测系统使用**十等分 Long-Short**方法（论文标准）：
- D10 做多，D1 做空
- 每月初调仓
- 市场基准：所有成分股等权重计算

### 5. 运行测试

```bash
python tests/test_genet_alignment.py
```

## 输出文件

### 模型文件
- `output/models/*_model.pkl` - 模型参数
- `output/models/*_predictions_oos.csv` - 样本外预测

### 回测结果
- `output/models/*_decile_monthly_returns.csv` - 十等分月度收益
- `output/models/*_decile_long_short_backtest_report.csv` - 回测报告（夏普比率、最大回撤等）

### 可视化图表
- `output/plots/*_pred_vs_real_scatter.png` - 预测 vs 真实收益散点图
- `output/plots/*_decile_long_short_equity_curve.png` - 净值曲线（包含市场对比）
- `output/plots/decile_comparison.png` - 方法对比图（含市场基准）

## 核心方法说明

### 1. Baseline（基线模型）
- 仅使用北交所训练数据
- 训练 Elastic Net 模型
- 作为迁移学习效果的对比基准

### 2. Hard Transfer（硬迁移）
- 在源市场（沪/深/合并）训练模型
- 直接应用到北交所测试集
- 无需在目标域重新训练

### 3. Soft Transfer（软迁移）
- 使用源市场训练的模型参数作为先验
- 在北交所数据上微调
- GENet 实现通过 `v` 参数控制对源域的依赖程度

### 4. Two-Stage（两阶段估计）
- 第一阶段：估计全局因子载荷
- 第二阶段：在北交所数据上估计因子敏感度

### 5. GENet Joint Optimization（联合优化）
- 同时在多个市场进行联合训练
- 分离全局参数 θg 和局部参数 θℓ,c
- 分离的惩罚系数：λg < λℓ,c
- 验证论文关键发现

## 论文对应

本项目参考论文：
> **How Global is Predictability? The Power of Financial Transfer Learning**

GENet 实现已对齐论文要求：
- ✅ 参数分解：βc = θg + θℓ,c
- ✅ 联合优化：min Σc ||Yc - Xc(θg + θℓ,c)||² + λg||θg||₁ + Σc λℓ,c||θℓ,c||₁
- ✅ 分离惩罚：λg < λℓ,c
- ✅ 时间点截面标准化（避免前瞻偏差）
- ✅ 预测性 R² 评估

## 模型表现

### 预测精度（MSE）

| 模型 | MSE |
|------|-----|
| baseline | 0.03604 |
| hard_sh | 0.03574 |
| hard_sz | 0.03574 |
| hard_sh_sz | 0.03574 |
| soft_sh | 0.03593 |
| soft_sz | 0.03593 |
| soft_sh_sz | 0.03593 |
| two_stage | 0.03574 |

**预测精度分析：**
- 所有迁移学习模型的 MSE 都低于 baseline（0.03604）
- 硬迁移模型 MSE 最低（0.03574），预测精度最高
- 软迁移模型 MSE 略高（0.03593），但提供了更好的稳健性
- 单市场模型（沪市、深市）与双市场模型 MSE 基本一致

### 相关性分析

| 模型 | 预测-真实相关系数 |
|------|------------------|
| baseline | -0.017 |
| hard_sh | 0.049 |
| hard_sz | 0.048 |
| hard_sh_sz | 0.047 |
| soft_sh | 0.048 |
| soft_sz | 0.047 |
| soft_sh_sz | 0.047 |
| two_stage | 0.047 |

**相关性分析：**
- Baseline 模型的预测与真实收益呈微弱负相关（-0.017），几乎无预测能力
- 所有迁移学习模型都呈现弱正相关（0.047~0.049）
- 迁移学习显著改善了预测方向

## 回测结果示例

| 模型 | 年化收益 | 夏普比率 | 最大回撤 | 胜率 |
|------|---------|---------|---------|------|
| baseline | -24.7% | -0.57 | -56.5% | 45.2% |
| hard_sh | 100.6% | 2.37 | -14.8% | 77.4% |
| hard_sz | 89.6% | 1.97 | -21.3% | 74.2% |
| hard_sh_sz | 85.1% | 1.94 | -26.6% | 74.2% |
| soft_sh | 100.9% | 2.37 | -14.8% | 77.4% |
| soft_sz | 89.4% | 1.98 | -20.2% | 71.0% |
| soft_sh_sz | 82.1% | 1.94 | -26.9% | 74.2% |
| two_stage | 85.1% | 1.94 | -26.6% | 74.2% |

**关键发现：**

**预测精度方面：**
- 硬迁移模型 MSE 最低（0.03574），预测精度优于 baseline（0.03604）
- Baseline 模型预测与真实收益几乎无相关（r=-0.017），单独在北交所数据上训练效果差
- 迁移学习模型都呈现弱正相关（r≈0.048），显著改善了预测方向

**回测表现方面：**
- 硬迁移沪市和软迁移沪市表现最好（年化收益~101%，夏普~2.37）
- 沪市单独模型优于深市单独模型
- 迁移学习显著提升了预测能力和实际收益

## License

MIT