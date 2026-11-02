# π0.5 SFT / LIBERO-10 Task3 流形验证执行计划

按“本地 2 卡调试 → Volc 单节点 8 卡正式 eval → 视频标注 → 离线流形统计”完成 task3 的 50 条 rollout，交付分类视频、冻结标签、逐轨迹指标、分组统计与图。

更新时间：2026-11-02（Asia/Shanghai）。状态：计划已确定，尚未执行本批正式 eval。本次更新只落计划文件。

**用户已明确：**本地使用 2 卡调试，调试完成后提交 Volc 侧 8 卡正式 eval；工作区与实验产物统一放在 `/vepfs-C区/jiajunlu/rlinf-pirl-official-30e0236`。复用已完成的 SFT、环境与模型加载验证，不重复评测 SFT SR 是否正常，不增加不必要的 smoke 或验收链。

## 1. 实验配置

本轮限定为 task3，每个初始状态采一条轨迹。模型保持原始 SFT 权重，仅推理和记录数据。

| 项目 | 固定设置 |
|---|---|
| 工作区 | `/vepfs-C区/jiajunlu/rlinf-pirl-official-30e0236` |
| Checkpoint | `/vepfs-C区/jiajunlu/ckpt/RLinf-Pi05-LIBERO-SFT` |
| Benchmark | `libero_10`，`task_order_index=0`，task3 |
| 任务 | `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` |
| 初始状态 | 官方 task3 的 trial0–49；episode ID 为 150–199 |
| 正式样本 | 50 条，每个初始状态 1 条；不扩为 group8 |
| 采样 | 评测 ODE；4 次去噪；一次规划执行最多 10 个动作 |
| 动作预算 | 最多 480 个 policy actions；首次环境成功时结束，保留成功动作 |
| 初始化 | 沿用 πRL 原生 reset/settling；记录实际配置，settling 不计入 policy action 数 |
| 模型输入和数值设置 | 沿用现有 πRL 原生图像、proprio、预处理、精度与 compile 设置 |
| 单卡推理 | 每卡一个模型进程，模型 batch size=1 |
| 随机性 | 固定 episode 对应的 policy noise seed：`7 + episode_id`；固定并记录环境 seed 和初始化过程，分片不改变种子 |
| 特征 | 第 4 次、即最后一次去噪 forward 中，`action_out_proj` 之前的 action-expert hidden |
| 点云粒度 | 一个有效 action 对应一个 1024 维点；完整轨迹为 `[T,1024]` |
| 主统计 | 前 200 个有效 actions，`X=[200,1024]`，k=10 |
| 补充统计 | 完整有效轨迹 `X=[T,1024]`，k=10，作为长度诊断 |
| 指标 | `D`、`V`、`D/V`、`G=V*exp(-0.1*D)` |

πRL 的 LIBERO-10 recipe 使用 4 次去噪、执行 10 个动作和 480 步上限；不沿用 Spatial 的 3/5/240 设置。旧 SimpleVLA 离线实验的 520 步预算也不带入本批。

## 2. 工作区和产物目录

采集、标注、统计与提交入口在当前工作区内实现；旧 SimpleVLA 文档只作为标注和指标定义的参照。已有运行及源码修改保持原样。

计划路径：`/vepfs-C区/jiajunlu/rlinf-pirl-official-30e0236/docs/manifold/pi_rl/manifold_verify_plan.md`。

实验根目录统一为：

```text
/vepfs-C区/jiajunlu/rlinf-pirl-official-30e0236/experiments/pi_rl/
  pi05_sft_libero10_task3_mani_ode480_<run_id>/
    config/                 # 本批配置、来源与种子表
    source_snapshot/        # 实际执行源码，包含新增或未提交的采集改动
    local_debug/            # 本地 2 卡短段调试，不进入正式统计
    volc/                   # 8 卡正式 eval 配置、提交回执、日志
    rollouts/               # 正式 50 条轨迹及逐动作特征/状态
    annotations/            # 标注、复核记录、冻结标签
    videos/                 # 原始与分类的主视角/腕部视频及索引
    geometry/prefix200_k10/ # 主统计
    geometry/full_k10/      # 完整轨迹及长度诊断
    figures/
    reports/
```

`<run_id>` 在启动本批时生成并冻结。复用既有 checkpoint、环境与资产记录；仅记录本批实际使用的路径、配置、版本和新增源码。快照需要包含本次采集改动，不能只归档干净官方 HEAD 而漏掉新增代码。

## 3. 本地 2 卡调试

本地阶段只确认新增的 hidden、视频和谓词采集能贯通。先让两张卡各运行一条短段，例如 trial0、trial1 各 32 个 policy actions；该长度足以检查 k10 计算与不足一个完整执行 chunk 的尾部处理。

同一次短段内完成以下必要检查：

1. 取到最后一次去噪的 action-expert hidden，最后一维为 1024，并正确对应实际执行的 action 位置。
2. 动作、hidden、chunk/slot 索引及终止 mask 对齐；`T` 个动作对应 `T` 个 hidden 和 `T+1` 个状态记录。
3. 主视角、腕部视频可读，逐状态 `In`/`Close` 谓词和原始 success 能保存。
4. 已采集点云能完成一次 k10 的 D/V/G 计算并写出结果；数值异常有明确记录。

短段通过即结束调试，不要求先跑完 480 步、不要求调试样本出现成功，也不以 SR 达到历史区间作为提交门槛。若有明确失败，只重跑能定位该失败的最小片段；通过且未被后续改动影响的检查不重复。

**验证范围遵循用户本次明确要求：**不另跑 SFT 基线 SR、不重复完整 checkpoint/资产扫描、不做采集开关的全量双跑、跨 GPU 逐位一致性、动作回放全像素比对、train/val/infer 生命周期或单独的 Volc 8 卡 preflight。该任务是新增推理数据采集，既有训练或复现流程的历史验收门禁不机械套用到本批。只有出现新错误或改动影响相关功能时，才增加针对性检查。

## 4. Volc 单节点 8 卡正式 eval

本地调试完成后，使用本批固定配置和实际采集源码提交一个 Volc 8 卡正式 eval 作业。复用当前工作区已有的提交、镜像、模型和资产接入方式，正式采集不调用训练入口。

每卡驻留一个模型，batch size=1，按 `trial_ids = list(range(50))[rank::8]` 分配任务：rank0、1 各 7 条，其余 rank 各 6 条，总计 50 条。随机种子只由 episode ID 决定，不取决于 rank 或运行顺序。本地调试产物不混入正式结果。

每条轨迹在首次成功时停止，否则执行至 480 个 policy actions。若在 10-action chunk 中途成功，保留该动作及其后状态，余下槽位不纳入数据。行为失败原样保留，技术错误独立记录；重试保留 attempt，汇总时使用预先约定的首个技术完整 attempt，不按成功或流形分数挑选。

Volc 正式作业同时产生完整轨迹、视频和标注所需状态证据，后续标注和几何计算直接消费这些文件，不为制作视频或计算指标再次运行 policy。保存提交回执与任务 ID；提交状态不明确时先只读查询，避免重复提交。任何停止既有 Volc 实验的操作仍须针对具体任务另行征得用户确认。

## 5. 同次 rollout 保存的数据

保存完整有效轨迹，保证后续改变统计前缀时可以离线重算。本轮主要特征只取最后一次去噪，不把去噪阶段拼接或把一个 chunk 展平成一个流形点。

| 数据 | 记录要求 |
|---|---|
| 身份与初始条件 | task/trial/episode ID、初始状态及来源、policy seed、环境 seed、checkpoint 路径 |
| 动作与特征 | 每个有效动作的 7 维环境动作、模型动作记录、最后一次去噪 hidden `[T,1024]` |
| 索引 | 连续 action/state/frame 索引、chunk ID、slot、执行 mask、去噪步索引 |
| 环境结果 | 每步 reward、success、done、终止原因、首次成功时间 |
| 目标谓词 | 同次 rollout 的每个 `s0...sT` 上的官方 `In`、`Close`，以及终态摘要 |
| 图像和视频 | 主视角和腕部的原始无损帧、两路 MP4；帧从 settling 后的 `s0` 开始 |
| 运行来源 | 实际配置、源码版本/快照、日志、错误与 attempt 记录 |

模型预测 horizon 中未执行的位置不参与流形。hidden 保持原生输出投影前的数值，离线转 FP64；不额外进行 BF16 截断、PCA、标准化、去重或 jitter。特征采集使用原 forward，不额外运行模型。

## 6. Task3 标注和视频归类

按旧交接文档的终态成果原则分类，并使用同次 rollout 的官方谓词与双视角证据。参照文档：`/vepfs-C区/jiajunlu/SimpleVLA-RL/docs/manifold/rollout_mani_v1/SFT_TRAJ1_TASKS_1_2_3_9_实验标注流形接手.md`。本次核对的文档 SHA256 为 `8f38474971e73c85686be635254d0f14f6a7f9fe0f3165e4b31aa53dc5ff1177`。

目标谓词为 `In(akita_black_bowl_1,white_cabinet_1_bottom_region)` 和 `Close(white_cabinet_1_bottom_region)`。

| 分类 | 规则 |
|---|---|
| level0 | 终态未保留碗入抽屉成果；包括关空抽屉、曾放入但最后掉出 |
| level1 | 终态碗在抽屉内，但抽屉未关好 |
| level2 | 原始环境完整成功，且对应碗在抽屉内、抽屉关闭 |
| unknown/disputed/invalid | 证据不足、证据冲突或数据异常；独立保留，不默认填为 level0 |

先由官方谓词和原始 success 生成候选标签，再查看全部 50 条双视角终态；对遮挡、掉出、关空抽屉或冲突案例补看过程。记录复核依据与帧位置，如实注明模型/人工复核来源，不宣称未进行的人类复核。标注时隐藏 D/V/G；标签冻结后才连接分组几何结果。

曾完成某个子目标仅作为过程事件记录，不覆盖终态标签。另保留第 200 步谓词和首次入抽屉、成果丢失、首次关闭等事件，便于解释前缀与终态的关系。

分类目录保存 `level_0/`、`level_1/`、`level_2/` 等类别的视频副本和索引，每条包括主视角与腕部视频，共 100 个正式视频。原始数据保留。归类后做一次文件齐全及视频帧数检查，不增加重复全量解码、独立 inode 审计或源视频删除流程。

## 7. 流形统计和分析

主分析沿用前 200 个有效 actions，固定点数以控制原始 V 的直接长度影响。完整轨迹结果单列，用于观察长度相关性；两种口径都只使用 action 粒度、最后一次去噪特征和 k10。

令 `X=[N,1024]`，逐轨迹计算：

```text
D = mean(local TLE dimensions), k=10, epsilon=1e-4
Xc = X - mean(X, axis=0)
B = Xc @ Xc.T
V = 0.5 * sum(log1p((1024/N) * eigvalsh(B)))
R = D / V
G = V * exp(-0.1 * D)
```

使用 `scikit-dimension==0.3.7` 的 TLE 定义、FP64 欧氏近邻和中心化 Gram 谱；Gram 不再额外除以 N。负谱仅在原实现数值容差内截零；保留 warning 与最终有效状态，不因中间 warning 自动删除轨迹。

| 口径 | 纳入规则 | 用途 |
|---|---|---|
| `prefix200_k10` | `T>=200`，精确取前 200 个有效动作，不向 chunk 边界取整 | 主要完成度分组分析 |
| `full_k10` | 取完整有效轨迹；点数需满足 k10 要求 | 完整轨迹指标、V 与长度关系诊断 |
| 短轨迹 | `T<200` 单列，仍计入全部 50 条的 SR、标注与视频 | 透明报告主分析的样本覆盖情况 |

不补点、不拼接其他轨迹，不因指标无效改变原始 SR 分母，也不临时降低 k。主表按终态 level0/1/2 分组，另报告成功/失败；标签是终态成果，前缀指标不直接解释为第 200 步的动作价值。

每类报告有效 n、mean、样本 std（ddof=1）、median 和四分位数。D/V 和 G 先逐轨迹计算再汇总。空类为 null，单例 std 为 null。对有足够样本的比较报告效应量/AUC；若报告显著性，预先列明比较集合并校正多重比较，不按结果挑选 K 或类别。

图包括四指标分组分布、D–V 散点、V/G 与长度关系、前 200 步样本覆盖情况。不同模型的 hidden 维度和尺度不同，不直接比较 π0.5 与 OpenVLA 的原始 V/G 绝对值来判断模型优劣。

## 8. 交付与一次汇总检查

正式采集完成后，基于已保存数据做一次汇总检查：trial0–49 覆盖且无重复；每条动作/hidden/状态/视频索引一致；50 条标签和原始结果可连接；主统计的纳入与排除原因明确。该步骤不再次生成轨迹、不复测基线 SR。

| 交付物 | 内容 |
|---|---|
| `reports/rollout_summary.json` | 50 条采集覆盖、成功数、长度、技术异常 |
| `annotations/annotations_frozen.json`、CSV | 类别、官方谓词、视觉依据、冻结记录 |
| `videos/video_index.csv` | 50 条轨迹的双视角分类视频路径，共 100 个视频 |
| `geometry/*/trajectory_metrics.csv` | 每条 D、V、D/V、G、T、纳入状态、类别 |
| `geometry/*/group_statistics.csv` | level0/1/2 和成功/失败的统计 |
| `figures/` | 分组图、散点和长度诊断 |
| `reports/manifold_report.md` | 配置、SR、类别分布、指标分析、覆盖率和解释边界 |
| `volc/`、`config/`、`source_snapshot/` | 正式提交与实际执行来源 |

执行顺序固定为：本地新增采集功能与 2 卡短段调试 → Volc 单节点 8 卡正式 50 条 eval → 基于原始视频和同次谓词标注并冻结 → k10 离线统计与图 → 一次汇总交付。有新故障时仅补充对应检查；已有通过结论不循环复验。
