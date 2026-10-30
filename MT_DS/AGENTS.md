# 用户已确定的实验约束

- 后续实验的实现、配置与输出均放在本项目 `/aita_data/home/aita_duanxuelong/Boundary_exp/MT_DS`；原 `/aita_data/home/aita_duanxuelong/MT_DS` 作为保留的 Baseline。
- 每个实验条件、每组超参数和每个随机种子，都必须配套在 **LA 10%（8 个标注病例）与 Pancreas 10%（6 个标注病例）** 上训练。真实数据的训练验证短跑也按两数据集配套执行；两组不必同时占用 GPU。不要自行先只跑一个数据集、增加 20% 或加入 BraTS。
- 新实验条件从随机初始化训练；不加载已有 Baseline 权重微调，不以已有权重的事后测试代替训练实验。同一条件中断后的原配置续训可以使用该次实验自己的断点。
- 用户已授权按第一轮方案实现并启动 A1：LA 10% 与 Pancreas 10% 各从头训练 15000 步。后续交互实验按第一轮结果再推进，不能把方案文件或配置准备描述为已启动。
- 保留现有 Results、源码快照与日志，新实验使用独立输出目录。实验条件间保持可比；更改评估协议、病例划分或训练设置时，明确旧结果的可比性限制。
- 已完成的 seed=42、15000 步 LA_8l/Pancreas_6l 结果已核对，见 `docs/next_training_plan.md`。同协议的原 MT_DS 不必因增加新实验重复训练；新增种子或新协议需要对应的 Baseline。
- 现有 LA/Pancreas 协议使用 test 集在线选模，结果仅作为该历史协议下的开发参考，不作为独立测试证据。独立确认协议须统一用于 Baseline 与方法。
- 实际数据根目录为 `/aita_data/home/aita_duanxuelong/Dataset`；复制项目后旧脚本的默认相邻目录不成立，不要直接使用旧六组启动器。

详细方案与文献边界见 `docs/boundary_channel_research_2026-10-26.md` 和 `docs/next_training_plan.md`。

A1 两组正式训练已于 2026-10-26 23:27:45 启动。先读取 `Results/A1_boundary_supervision_v1/{summary.json,full_launch_manifest.json}` 与各运行状态再采取行动，勿重复启动。运行中的算法代码保持不变；新条件使用明确版本和独立目录。实现公式与验证记录见 `docs/a1_implementation.md`。

2026-10-27 用户明确要求先续上中断的 Pancreas。已从其第 3000 步断点恢复（原日志到 3760），GPU 0，训练 PID 437241，监督器 PID 436780；仅将保存间隔改为 100 步。LA 训练更新到 15000 步，最终验证/测试未收尾，本次不重启。先查实时状态，勿重复续训；细节见 `docs/resume_A1_2026-10-27.md`。

2026-10-27 15:31 用户已明确要求立即启动0.03的两个实验，覆盖此前等待0.1胰腺完成的排程建议。已启动 `Results/A1_boundary_supervision_w003_v1/seed_42/{LA_8l,Pancreas_6l}`：GPU 1/5，PID 2646884/2646893，后台监督器2646821；两组随机初始化、seed42、15000步。训练算法源码保持原A1，仅最大边界系数0.03；保存间隔统一100步。原0.1胰腺GPU0继续运行。先检查该条件实时清单和状态，勿重复启动；详见 `docs/A1_w003_launch_2026-10-27.md`。

2026-10-27 18:10 用户要求重新启动0.03条件的LA。已从本次实验第1300步完整断点恢复到GPU4，训练PID659169，监督器PID657943；已确认推进至1320步且loss/梯度有限。旧LA日志记录到1400步，检查点后的记录已先归档至 `Results/A1_boundary_supervision_w003_v1/seed_42/LA_8l/resume_history/A1_w003_resume_LA_20261027_180959/` 再从实时指标中裁去，TensorBoard旧事件也已归档。原监督器2646821已仅按PID退出，由 `scripts/resume_a1_w003.py` 接管两组状态汇总；Pancreas原训练PID2646893继续运行于GPU5。所有训练算法与超参数保持不变，原DataLoader采样器/预取状态不支持逐批精确恢复。先查实时清单和进程，勿重复启动；恢复记录见 `launch_logs/A1_w003_resume_LA_20261027_180959/resume_record.json`。

2026-10-27 18:53 用户明确要求把独立边界头实验也启动，覆盖此前等待0.03条件完成的排程建议。已启动 `Results/A1_independent_boundary_head_w003_v1/seed_42/{LA_8l,Pancreas_6l}`，GPU2/3、PID1227265/1227267、监督器1227209；两组随机初始化、seed42、15000步。新增独立入口 `code/train_boundary_head.py`，不修改现有运行中的原训练源码；half＋full二分类边界头、BCE＋Dice最大权重0.03、2000步预热＋2000步ramp、每100步保存。数值/零权重等价检查及两数据集完整patch GPU短跑均通过。此条件仅边界监督，无Bank、skip路由或通道交互。后续先查实时状态，勿重复启动；详见 `docs/independent_boundary_head_v1_implementation.md`。

2026-10-28已按用户要求完成本轮结果分析，未启动新训练。0.03窄带CE和独立边界头四组均completed、15000步及best复测完成；原0.1 Pancreas也已完成（其旧summary过期），原0.1 LA仍缺最后验证/复测，仅引用截至14800步best。分析见 `docs/boundary_results_review_2026-10-28.md` 及 `analysis/boundary_results_review_2026-10-28/`。注意：续训未显式保存采样器状态不等于本次采样已改变；LA1300步/Pancreas3000步恰为4步/3步epoch边界，预热及各1501个日志点的目标质量/占比保持一致，不可把续训乱序当作已知退化原因。后续归因、Bank、交互等仍需明确的新实验条件。

2026-10-28 22:42:52用户明确要求E4与D3两方案同时运行，已启动自动监督器PID1155497，正式清单 `Results/E4_D3_comparison_v1/full_launch_manifest.json`。范围为 `E4_legacy_boundary_w003_v1`（当前MT_DS的E4接旧辅助分支，0.03从第1步）和 `D3_boundary_skip_w0015_v1`（当前D3单头联合监督，最大0.015，2000预热+2000ramp，在线E2 skip归因）。每条件LA10%/Pancreas10%配套随机初始化、seed42、15000步，共4组；未额外启动只读出器M0、交换或Bank。初始E4/LA GPU6 PID1155510、D3/LA GPU4 PID1155511，均已过100步并保存断点；两个Pancreas因显存不足登记queued，由同一监督器自动等待资源。核对实时汇总后再行动，勿重复启动或把queued说成training。四组完整patch 8步GPU验证、损失/状态隔离检查通过，原训练源码未改。详见 `docs/E4_D3_launch_2026-10-28.md`。

2026-10-29用户要求分析四组结果，已核对E4/D3的LA/Pancreas均completed、15000步、72次主验证及best复测，监督器已完成，不要按历史排队记录重复启动。报告 `docs/e4_d3_results_review_2026-10-29.md`，可复现脚本与图表 `analysis/e4_d3_review_2026-10-29/`。E4胰腺有持续收益，LA退化；D3未有整体性能优势，但56次成熟期诊断支持其10% skip弱化的评分及部分主边界CE对应。注意E4/LA中期11次表面指标只有19例（同病例空预测），四组best/最终均20例；低分通道10%弱化有利不代表完全置零有利。没有通道交互、没有M0对照；分析时未加载权重追加推理或启动新训练。下一轮仅为建议，仍需明确新条件。

2026-10-29 11:36:38用户明确要求补M0并启动固定D3目标/0.015时间表/MT设置的最小L→U验证，已实现并启动正式后台监督器PID1870896，清单 `Results/D3_stage2_comparison_v1/full_launch_manifest.json`。本轮三条件为 `M0_D3_readout_v1`（边界头训练输入detach、主干保持原MT_DS）、`D3_L2U_random_v1`、`D3_L2U_boundary_v1`；每条件LA10%/Pancreas10%、seed42、随机初始化15000步，共6组。初始M0/LA GPU2 PID1870925、M0/Pancreas GPU4 PID1870927已实际开始，其余4组因剩余显存排队，资源满足自动启动；先查实时汇总，勿重复启动或将queued称为training。边界最大0.015、前2000步关闭再2000步ramp不变；交互仅Student E2→D3、4/32通道、L边界内外原型、Teacher伪边界软置信度约束，4000步后用1000步ramp至残差范数上限α=0.1。随机/边界选择有相同结构、候选池、有效供体过滤及一致性预算；无Bank、U→L、U边界训练项。新增独立训练入口，原训练源码未修改；CPU等价/梯度/BN/RNG检查及全部6组完整patch 8步GPU短跑通过。正式运行期间算法代码保持不变；说明与验证入口见 `docs/D3_stage2_launch_2026-10-29.md`。

2026-10-29 20:12 用户要求恢复 `D3_L2U_random_v1/seed_42/Pancreas_6l`。原进程在4000步后因CUDA显存不足退出，已核验本次4000步完整断点、最佳模型对应关系、源码和划分SHA，从该断点按原配置恢复到GPU6，训练PID403430、监督器PID403066；确认推进到4010步，交互实际触发（2个样本），loss和梯度有限。旧记录保存在该运行的 `resume_history/D3_stage2_resume_random_Pancreas_20261029_201157/`；原DataLoader采样器/预取状态未保存，不宣称逐批精确重放。原监督器1870896已仅按PID退出，由 `scripts/resume_d3_stage2.py` 接管六组动态清单；其他三个训练原进程保持运行，已完成的M0/LA和random/LA未重启。实际状态以实时清单和进程为准，勿重复启动。完整恢复命令和检查点SHA见 `launch_logs/D3_stage2_resume_random_Pancreas_20261029_201157/resume_record.json`。

2026-10-30用户要求分析已完成实验，已核对M0/随机L→U/边界L→U共6组均completed、15000步、72次完整主验证与best复测，未新训练或加载权重追加推理。报告 `docs/d3_stage2_results_review_2026-10-30.md`，复现脚本、CSV和图 `analysis/d3_stage2_review_2026-10-30/`。M0两数据集全部72次四指标与Baseline完全一致；D3联合监督辅助读出稍好但无整体分割收益。两种L→U未建立选择优势，实际残差约配置参考范数的0.04%～0.05%、投影梯度约1e-7、一致性改变量极小，不能把非零注入等同于有效学习信号。胰腺random恢复后1100个日志采样点的L/U配对均与M0不同，boundary与LA则保持匹配；该采样混杂已由ID和目标质量记录确认，不仅是“未保存采样器”的推测。另只读分析IDE中的独立LossNorm E3六组：其none为自身对照，末尾状态构造错误已有完成核验，不重训；胰腺random在13800步Dice24.8758%、7例空预测、表面指标13例，随后恢复，不能忽略该不稳定性。下一优先建议实际注入幅度/方向校准，尚未授权或启动新条件，勿重复提交已完成任务。
