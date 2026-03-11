# 股票收益预测迁移学习项目

## 项目简介
本项目研究使用迁移学习方法，将沪深两市股票收益预测模型迁移到北交所市场。实现了基线模型、硬迁移、软迁移和两阶段估计等多种方法。

## 项目结构
```
src/                # 源代码
├── data/          # 数据处理
├── models/        # 模型实现
├── utils/         # 工具函数
└── backtest/      # 回测分析

data/              # 数据文件
├── raw/          # 原始数据
└── processed/    # 处理后数据

output/            # 输出结果
```

## 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 数据处理
```bash
python src/data/processor.py
```

### 2. 模型训练
```bash
# 基线模型
python src/models/baseline.py

# 硬迁移
python src/models/hard_transfer.py        # 双市场
python src/models/hard_transfer_sh.py     # 沪市
python src/models/hard_transfer_sz.py     # 深市

# 软迁移
python src/models/soft_transfer.py        # 双市场
python src/models/soft_transfer_sh.py     # 沪市
python src/models/soft_transfer_sz.py     # 深市

# 两阶段估计
python src/models/two_stage.py
```

### 3. 回测分析
```bash
python src/backtest/backtest.py
```

## 模型对比

| 模型 | 源域 | 方法 | OOS R² |
|------|------|------|--------|
| 基线 | 北交所 | 基线 | -0.000158 |
| 硬迁移 | 沪市 | ElasticNet | 0.001711 |
| 硬迁移 | 深市 | ElasticNet | 0.008155 |
| 硬迁移 | 沪+深 | ElasticNet | 0.008218 |
| 两阶段 | 沪+深 | 两阶段估计 | **0.010761** |
| 软迁移 | 沪市 | GENet | 0.004315 |
| 软迁移 | 深市 | GENet | 0.004461 |
| 软迁移 | 沪+深 | GENet | 0.003904 |

## 许可证
MIT License