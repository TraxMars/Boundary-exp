**独立边界头配套实验：实现与启动说明（2026-10-27）**

用户已明确要求将独立边界头方案也启动。该授权覆盖此前等待0.03权重对照完成的排程建议。本条件使用相同的1/2＋full尺度，区别于原分割logits窄带CE；不在本轮加入通道交互、Memory Bank或编码器skip路由。

已于2026-10-27 18:53:49（Asia/Shanghai）启动两组正式训练：LA 10%使用GPU2、PID1227265；Pancreas 10%使用GPU3、PID1227267；后台监督器PID1227209。已分别观察到40步/10步真实更新，当前为预热阶段，边界有效系数0是预期行为。静态PID仅作启动记录，实时状态以运行目录为准。

**实现与训练目标**

- [独立模型](../code/networks/vnet_boundary_head.py)继承原VNetDS；边界头读取原`out_conv_2`及`out_conv`的相同输入特征。half为32通道融合特征，full为16通道、含原有Dropout的最终特征。
- 每个头：`Conv3d(C,min(C,32),3,padding=1) → ReLU → Conv3d(min(C,32),1,1)`，不新增BN或Dropout。共新增34658个参数。
- 新头初始化使用独立CPU随机数上下文，seed为42+1000003；原骨干初始化、后续数据加载和Dropout随机数流保持原设置。
- 原四尺度分割CE＋Dice、Mean Teacher四尺度概率MSE、Student/Teacher原有BN与参数EMA策略保留。Teacher和无标签Student不执行边界头前向；有标签Student执行边界头，边界损失梯度同时更新头与共享骨干。
- 完整GT先生成输入尺度2 voxel的内外带，使用与A1相同的标注缓存和同步增强。此处将两侧合并为边界正类；非边界区域为负类。通过面积平均池化生成各尺度的软覆盖目标。
- [独立边界损失](../code/boundary_head.py)在每个尺度计算正类加权BCE＋soft Dice。正类权重由有效样本该尺度的负/正目标质量计算，detach并截断至[1,100]；Dice逐有效样本平均。完整目标边界质量为0的样本跳过；整批无有效边界则不把该项加入优化目标。
- 两个尺度的边界损失取平均，不因增加一个头翻倍总系数。

公式：

\[
B=\mathbf1[\text{encoded band}>0],\qquad B_s=\operatorname{AvgPool}_s(B),
\]

\[
L_b=\tfrac12\sum_{s\in\{1/2,1\}}
\left[\operatorname{BCE}_{\beta_s}(z_s,B_s)
+\operatorname{mean}_i\left(1-\frac{2\sum p_{s,i}B_{s,i}+\epsilon}
{\sum p_{s,i}+\sum B_{s,i}+\epsilon}\right)\right],
\]

\[
L=L_{sup}^{four\ scales}+\lambda_{MT}L_{MSE}^{four\ scales}+\lambda_bL_b.
\]

最大边界系数固定0.03作为未验证的工程起点；前2000次更新系数0，2001–4000线性增加到0.03，之后固定。它与原窄带CE的0.03不代表相同的损失或梯度强度，不宣称已经匹配两者的优化贡献。

**运行与复现**

- 新入口：[train_boundary_head.py](../code/train_boundary_head.py)。原`train.py`、`vnet_ds.py`、`boundary.py`、数据、目标及评估源码未修改。
- 新启动器：[run_boundary_head_pair.py](../scripts/run_boundary_head_pair.py)，启动前强制检查原源码哈希、病例划分以及新实现的单元和配套GPU短跑结果，拒绝覆盖已有目录。
- LA 10%（8标注）与Pancreas 10%（6标注）均随机初始化，seed42，15000步；沿用patch=(112,112,80)/(96,96,96)，SGD、LR=.01、EMA=.99、batch4（2+2）、原增强和验证协议；断点每100步保存。
- 正式目录：`Results/A1_independent_boundary_head_w003_v1/seed_42/{LA_8l,Pancreas_6l}`。
- [动态汇总](../Results/A1_independent_boundary_head_w003_v1/summary.md)、[完整启动清单](../Results/A1_independent_boundary_head_w003_v1/full_launch_manifest.json)。最终PID与启动时间以该清单为准。
- 推理仍仅使用原全分辨率分割输出；默认forward不执行边界头，也不需要GT或供体。加载本条件state_dict应使用新增模型类，因为保存状态包含辅助头参数。

**验证证据**

[数值与梯度检查](../tests/verify_boundary_head.py)已通过：

- 解析可验证的小例子与BCE＋Dice一致，内外带都对应边界正类，非边界位置梯度推动负类。
- 半尺度保留边界覆盖质量，空样本跳过且空批loss有限。
- 新增头不改变原骨干初始参数或训练随机数状态。
- 零权重时四尺度分割logits、BN、Dropout、SGD及Teacher EMA与原实现完全一致；梯度探针不写入参数梯度。
- 激活边界loss后两个辅助头与编码器均有有限、非零梯度；Teacher无参数梯度。

[配套GPU短跑](../verification_runs/independent_boundary_head_v1/smoke_results.json)均完成8步：使用正式patch，压缩warmup/ramp以实际执行非零边界项；LA峰值6221.7MiB，Pancreas5521.1MiB。两组第8步断点均已严格加载，包含Student/Teacher/优化器/RNG；两个辅助头的优化器动量非零。单元与短跑报告记录当前源码哈希；短跑不作为性能有效性的证据。

原有每200步参数梯度诊断继续记录，只监测半尺度上采样和最终卷积两个权重张量的边界/区域梯度范数、方向及系数比例；这不是全模型占比，也不是已经实现的编码器通道归因。

**解释边界**

本条件比较的是“独立边界预测头＋边界分类任务”这一整套方式，改变了辅助头参数、任务定义及loss，不能仅凭其与A1的差值断言某个loss或某层位置单独产生作用。半/full位置、完整标注边界宽度及主要训练协议已保持一致。两条件都有共享骨干上的优化取舍；独立头不保证更好的分割或更可靠的通道归因。

开发评估继续使用历史test在线选模，距离为voxel、ASD为单向定义；不把本轮结果当成独立测试结论。
