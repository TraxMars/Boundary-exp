# 损失归一化消融对比

使用最佳模型的最终 Test Dice，单位为百分数。

| 数据集 | 标注比例 | MT+DS+LossNorm | MT+DS | 差值（百分点） | MT+DS 状态 |
| --- | --- | --- | --- | --- | --- |
| LA | 10% | 91.12 | 87.75 | -3.37 | completed |
| LA | 20% | 91.70 | 90.37 | -1.33 | completed |
| Pancreas | 10% | 81.54 | 77.94 | -3.60 | completed |
| Pancreas | 20% | 83.48 | 81.72 | -1.76 | completed |
| BraTS19 | 10% | 85.23 | 85.54 | +0.31 | completed |
| BraTS19 | 20% | 84.95 | 86.55 | +1.60 | completed |
