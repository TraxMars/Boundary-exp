# MT_DS

当前实验副本位于 `Boundary_exp/MT_DS`。已实现第一轮 A1：保留原网络与 MT/DS 目标，在解码器半分辨率及最终输出增加边界窄带 CE。每个条件配套训练 LA 10%（8 例）与 Pancreas 10%（6 例）。

使用专门的两数据集启动器，以下命令要求在 GPU 主机运行；`full` 会检查数值验证与两数据集真实 patch 验证通过且代码未改变：

```bash
/aita_data/home/aita_duanxuelong/.conda/envs/SSL29/bin/python tests/verify_boundary.py
/aita_data/home/aita_duanxuelong/.conda/envs/SSL29/bin/python -u scripts/run_a1_pair.py --phase smoke --detach
/aita_data/home/aita_duanxuelong/.conda/envs/SSL29/bin/python -u scripts/run_a1_pair.py --phase full --detach
```

启动器拒绝覆盖已有运行。已有运行勿重复执行上述命令。正式输出为 `Results/A1_boundary_supervision_v1/seed_42/{LA_8l,Pancreas_6l}`，汇总与后台进程记录位于 `Results/A1_boundary_supervision_v1/{summary.md,summary.json,supervisor.json}`。

参数与公式见 [A1 实现说明](docs/a1_implementation.md)，阶段安排见 [后续训练安排](docs/next_training_plan.md)。以下内容记录原 Baseline；旧六组启动器不用于当前实验。

从保留的 `../MT_DS_LossNorm` 派生的损失归一化消融版本。
保留基础 EMA Mean Teacher 和四尺度深度监督，移除 `L / L.detach()` 及其参数与函数。

```text
L_sup = mean_4_scales((CrossEntropy + foreground_BinaryDiceLoss) / 2)
L_mt  = mean_4_scales(MSE(softmax(student), stopgrad(softmax(teacher))))
L     = L_sup                         # 前 2000 步
L     = L_sup + 0.3 * L_mt            # 之后
```

Teacher 的 stopgrad 用于隔离目标梯度，继续保留；CE / Dice 中的平均和 Dice 公式属于原始定义。
日志使用 `loss_total` 记录实际优化目标。
不加入原型对齐、混合图像、自适应阈值、置信度掩码或边界模块。

六组实验均从头训练，seed=42、15000 步、恒定学习率 0.01、EMA=0.99、batch=4（2+2）。
启动器逐组读取父版本保存的配置，复制所有训练设置，仅移除损失归一化。
模型、采样器、病例顺序、数据增强、深度监督头位置、Teacher BN 与评估协议均沿用父版本。

| 数据集 | 10% 标注 | 20% 标注 | Patch | 在线选模 | 最终测试 |
| --- | --- | --- | --- | --- | --- |
| LA | 8 例 | 16 例 | 112×112×80 | test.list，20 例 | test.list，20 例 |
| Pancreas | 6 例 | 12 例 | 96×96×96 | test.list，20 例 | test.list，20 例 |
| BraTS19 | 25 例 | 50 例 | 96×96×96 | val.list，25 例 | test.list，60 例 |

在已有 SSL29 环境启动全部六组（最多每张卡同时两组，每组预留 8000 MiB 显存）：

```bash
cd /aita_data/home/aita_duanxuelong/MT_DS
../.conda/envs/SSL29/bin/python -u scripts/run_six.py
# 六个 GPU 参数顺序：LA10、LA20、Pancreas10、Pancreas20、BraTS10、BraTS20
../.conda/envs/SSL29/bin/python -u scripts/run_six.py --gpus 4 4 2 3 5 5
```

训练输出为 `Results/seed_42/{LA_8l,LA_16l,Pancreas_6l,Pancreas_12l,BraTS19_25l,BraTS19_50l}`。
每组保存源码快照、配置、数据划分、原始损失、梯度范数、状态、验证记录和完整恢复断点。
第 800 步起每 200 步验证；全部 15000 步结束后，最佳 Student 自动执行完整测试。

`scripts/run_six.py` 持续监控六组进程，每 30 秒更新 `summary/results.{md,csv,json}`，
以及 `summary/comparison.{md,csv,json}`。完成后的对比使用最终测试结果，缺失结果保留为空。
需要后台运行时应让启动器自身脱离终端；本次实际后台启动记录见 `supervisor.json`。

父版本的六组已完成结果在 `../MT_DS_LossNorm/summary/results.md`。
LA / Pancreas 沿用 test.list 在线选模；BraTS19 的 val 选模指标与独立 test 指标分开报告。

```bash
../.conda/envs/SSL29/bin/python tests/verify_core.py
../.conda/envs/SSL29/bin/python scripts/summarize.py
```

恢复训练使用相同配置的 `code/train.py --resume .../last_checkpoint.pth`，不会兼容父版本的
归一化训练断点。断点保存 EMA、优化器与随机状态；DataLoader 恢复不保证逐批位级复现。
