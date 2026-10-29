# A1 实现与验证

实现日期：2026-10-26。条件名：`A1_boundary_supervision_v1`。

网络结构、四尺度 CE/Dice 和 Mean Teacher MSE 未改变。新增损失只使用标注病例，位置为半分辨率和最终分割输出。当前阶段不增加边界预测分支、通道路由或跨流交互。

## 边界目标

在原始完整标注体积内计算欧氏距离，按体素中心计量：前景到背景距离 <= 2 voxel 的位置为内侧，背景到前景距离 <= 2 voxel 为外侧。无前景或无背景时没有可用边界。未向完整体积额外填充背景；这仍是体素中心距离近似，并非精确连续表面距离。

仅为 train.list 前 8/6 个标注病例生成 uint8 缓存：0=非窄带，1=内侧，2=外侧。缓存记录源文件大小/mtime、标签 SHA-256、形状和两侧计数。训练时使用与图像/标签相同的旋转、翻转、padding 和 crop；新 padding 的边界编码为 0。无标签样本返回全零编码，不从其真值构造边界目标。

完整标注的裁剪平面不会被重新检测为边界。原始体积视野之外的未知结构无法恢复，若器官已在原始视野截断，该限制仍然存在。

## 目标公式

对样本 i、输出尺度 s、内外侧 k，用面积池化将原分辨率二值窄带投影到该尺度，得到权重 w(i,s,k,x)。内侧固定目标为前景，外侧固定目标为背景：

`CE(i,s,k) = sum_x w(i,s,k,x) * [-log P_s(class_k,x)] / sum_x w(i,s,k,x)`。

每个样本内平均有效侧，每个尺度平均有效样本，再平均半分辨率和全分辨率两个目标。无效侧/样本不进入分母；两个尺度均无可用边界时，损失是可微的 0。

一个粗网格可以同时接收内外侧监督，两侧保持各自的目标和权重。此实现等价于把粗网格预测复制回其细网格覆盖区域后，在细网格窄带上计算对应侧 CE，避免最近邻下采样丢失薄边界。

`L = L_sup + lambda_mt * L_mt + lambda_b * L_boundary`。

`lambda_mt` 延续原方案：前 2000 次更新为 0，之后为 0.3。`lambda_b` 前 2000 次更新为 0，第 2001 到 4000 次更新线性升到 0.1，第 4000 次及以后为 0.1。不使用 `L/L.detach()`。

## 观测项

training_metrics.csv 记录每尺度边界 CE、有效样本数、内外侧面积池化质量、标注 crop 无前景比例及窄带体素占比。

每 200 次更新、以及首次更新前，在已有计算图上使用 autograd.grad 观测 L_sup 和 L_boundary 对以下参数的梯度：

- `block_seven_up.conv[0].weight`：半分辨率上采样参数，输出通道轴为 1（ConvTranspose3d）。这只覆盖该特征的上采样路径，不覆盖 skip 的全部贡献。
- `block_nine.conv[0].weight`：最终卷积，输出通道轴为 0。

记录两目标的梯度范数、余弦、按当前 lambda_b 加权的范数比、逐输出通道范数/余弦和有效通道标记，写入 boundary_gradient_metrics.jsonl。这个范围是明确指定的局部参数子空间，不是全模型梯度范数。余弦是同一个有标签样本的区域/边界目标关系，不是 CHAP 式有无标签对齐评分。

观测不增加模型前向、不调用 backward、不更新 BN/EMA、不读写随机状态或参数 .grad；不参与路由或优化。全部正式实验保存配置、代码快照、代码哈希、病例划分及边界缓存元数据。

## 验证与启动

已通过 `tests/verify_boundary.py` 的距离场、混合粗网格 CE 数值、无边界、系数调度、两数据集变换/随机轨迹、无标签边界隔离、探针零系数 SGD/BN/Dropout 完全等价、非零系数梯度可达检查。

原 `tests/verify_core.py` 也已通过：原始四尺度输出/参数键一致、Teacher 梯度隔离、各头梯度、EMA 与滑窗拼接。

成对 GPU smoke 使用正式 patch、8 次随机初始化训练更新，将 MT warmup 和边界 start/ramp 缩到 2 步，以覆盖真实非零新增损失。它仅用于训练流程验证，不用于性能结论。两个数据集都成功完成、梯度有限非零、病例划分等同 Baseline 且保存断点后，启动器才允许 full。

正式训练继承原两组 Baseline 的 seed=42、15000 步、优化器/学习率、batch、patch、增强、BN/EMA 与历史选模协议；原 Results 不覆盖。后台监督器每 30 秒刷新两组状态及参考结果。

两组真实 patch 的 8 步验证均已通过，峰值显存 LA=5762.48 MiB、Pancreas=5113.07 MiB，均覆盖 3 次非零边界权重梯度探针。两组首步监督 CE/Dice、一致性损失和总梯度范数与对应历史 Baseline 记录完全一致；两组第 8 步断点均已成功重新读取，Student/Teacher 张量有限且优化器状态存在。记录位于 `verification_runs/A1_boundary_v1/{unit_results.json,smoke_results.json,checkpoint_and_initial_step_checks.json}`。

正式训练于 2026-10-26 23:27:45（Asia/Shanghai）启动，LA 使用 GPU 2，Pancreas 使用 GPU 1；实际进度见 `Results/A1_boundary_supervision_v1/summary.md`。正式运行目前处于执行阶段，没有性能提升结论。

如果更改 code/scripts/tests 下的 Python 源码，验证哈希失效，正式启动前需要重新验证。已有正式运行使用启动时保存的代码快照进行追溯，训练中不修改算法。
