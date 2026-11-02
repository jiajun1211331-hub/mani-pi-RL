# mani_v1 迁移到 πRL：按子需求拆分的 Git Commit 计划

按这份计划，用 10 次提交（C00–C09，含本计划）完成 mani_v1 的迁移、诊断和验证。每次提交只解决一个可验收的子需求，相关测试随功能提交；最后一次负责真实 GPU 生命周期验证和交接证据。

当前状态：2026-11-02，C00 已暂存，C01–C08 已实现并通过对应组件验证，尚未提交。Git 作者身份等待用户确认；C09 正在执行。disabled 的真实双 A100 完整生命周期已通过；G 混合已通过真实 rollout/评分/更新/val/save，正在验证 strict reload/infer，G-only 与 V 混合尚待验证，不能视为整体验收完成。

## 1. 固定迁移边界

先按已经确认的选择完成首版，再通过独立配置选择实验模式。迁移验收回答“实现是否忠实、可运行”，训练曲线回答“方法是否有效”，两类结论分别记录。

| 项目 | 约定 | 状态 |
| --- | --- | --- |
| Advantage 模式 | 同时支持 task_plus_geometry、geometry_only，并保留 disabled 基线路径 | 用户确认两种方法模式；disabled 用于兼容 |
| PPO 单位 | 保留 πRL 原生 chunk-level ratio，10 个 action 合成一个 ratio | 用户确认 |
| 几何前缀 | 用精确前 K 个 action 计算几何；仅完整落在 K 内的 chunk 接收几何 advantage | 用户确认 |
| 采样 | 64 个并行环境 × 8 个 rollout epoch，512 条原始轨迹，group_size=8，共 64 个原始组 | 用户确认 |
| 组筛选 | 固定采样后做 mask；不补采到 64 个 mixed 组，不追加全成功组配额 | 用户确认 |
| 组类型接口 | 支持 mixed_only、any_success；后者接收 mixed 与有效且非常数的全成功组 | 拟采用接口；any_success 为推荐默认值 |
| 前缀长度 | K 为该组成功轨迹中，首次成功前有效 action 数的最小值，包含成功 action | 迁移 mani_v1 口径 |
| 特征 | 实际训练 rollout 最后一个 denoise step 的 action_out_proj 输入，每个 action 一点，形状 [T,1024] | 首版方案 |
| 几何参数 | TLE k=10、epsilon=1e-4；评分可选 G=V exp(-0.1D) 或 V；选定评分组内标准化后权重 0.2 | 保留 G 默认值；新增用户要求的 A_GRPO + 0.2 × A_V 对照 |
| 后端 | 同步、colocated、pipeline_stage_num=1；组完整落在一个评分分片中 | 首版支持范围 |
| 基础模型 | π0.5 LIBERO SFT；沿用原生 flow_sde 训练采样、4 次 denoise、action expert 训练范围 | 首版方案 |

64×8 中的 8 是 rollout epoch 数；group_size=8 是每组轨迹数，二者含义不同。原生 actor.global_batch_size=2048 计量的是训练用 chunk 样本，不是 2048 条完整轨迹。筛选后的有效组可能少于 64，日志须分别记录原始数量、有效数量与实际 optimizer 更新次数。

## 2. 源码依据与实现入口

所有实现从当前真实 HEAD 出发，避免把目录名当作版本。2026-11-02 核对的 HEAD 为 051b23348ead20c06c336bd743c590a9b6c4ba17；目录名中的 30e0236 是历史命名。

- 源方法仓库：/vepfs-C区/jiajunlu/SimpleVLA-RL。
- 目标仓库：/vepfs-C区/jiajunlu/rlinf-pirl-official-30e0236。
- 现有 checkpoint：/vepfs-C区/jiajunlu/ckpt/RLinf-Pi05-LIBERO-SFT。
- 本计划目录：/vepfs-C区/jiajunlu/rlinf-pirl-official-30e0236/docs/manifold/pi_rl/mani_v1/。

下表中的相对路径均相对于目标仓库，且已经核对存在。每次实施前应重新检查对应符号和工作区变化。

| 已有路径 | 迁移用途 |
| --- | --- |
| scripts/pi_rl/manifold_geometry.py | 已有离线 FP64 几何实现，作为数值对齐依据 |
| scripts/pi_rl/collect_manifold.py | 已有离线 hidden 采集，作为特征位置依据 |
| rlinf/models/embodiment/openpi/tasks/rl.py | Pi0RL._predict_train 的真实采样链 |
| rlinf/models/embodiment/openpi/pi0.py | run_suffix、velocity_from_suffix 与 FP32 输出投影输入 |
| rlinf/workers/rollout/hf/huggingface_worker.py | _build_policy_output 的显式字段转发 |
| rlinf/data/schema/embodied_types.py | PolicyOutput、EnvPart、TrajectoryStep、Trajectory 的拆分、合并和堆叠 |
| rlinf/workers/env/env_worker.py | 环境结果、动作有效性、轨迹来源信息的转发 |
| rlinf/envs/sim/libero/libero_env.py | task/trial/reset-state 身份、终止与 chunk 内逐 action 结果 |
| rlinf/workers/actor/embodied_fsdp_actor_worker.py | 原生筛选、advantage 计算、minibatch 打乱前的接入点 |
| rlinf/algorithms/advantages.py | 原生 GRPO 组内标准化 |
| rlinf/algorithms/utils.py | chunk-level reward/logprob 预处理 |
| rlinf/config.py | 配置解析和组合约束校验 |
| examples/embodiment/config/libero_10_grpo_openpi_pi05.yaml | 首版配置基础 |
| scripts/pi_rl/prepare_training.py | 现有官方源码归档入口，存在版本限制 |
| tests/unit_tests/、tests/e2e_tests/embodied/ | 单元与端到端验收入口 |

现有离线采集采用 ODE 推理，在线训练采用带随机性的原生采样链。共享几何算法，并不意味着两种 hidden 分布等价。首版在线特征必须来自生成本批 action 的那次 forward，不能追加 ODE 推理替代。

## 3. 统一接口与数学口径

先固定从 action hidden 到 advantage 的数据契约，后续提交各自实现一段。新增模块建议放在 rlinf/algorithms/manifold/，其中 metrics.py、groups.py、advantages.py 分别负责数值、分组前缀和优势组合；这些路径目前是规划。

### 3.1 特征、分组与前缀

模型可选返回 manifold_hidden，形状为 [B,C,1024]，C=10，FP32、detach。它来自最后一次实际 denoise 的 suffix hidden，仅截取即将执行的前 C 个动作位置。它是该次速度预测所用表征，不是对最后采样 action 再编码得到的表征。

数据沿 PolicyOutput → TrajectoryStep → Trajectory → actor 传递。环境身份通过 EnvPart 合入轨迹。新增字段不进入 forward_inputs，避免混入 PPO 重算输入；高维 hidden 在 CPU 完成组评分后释放，不进入 actor GPU minibatch。

组身份至少包含 behavior-policy version、collector step、rollout epoch、环境 source rank/stage、group slot。每组必须验证 8 个唯一 member ID，并验证 task、trial、reset-state 一致。不能仅依赖 reshape(-1,8) 假定邻接样本属于同组。首版不支持跨评分分片拼组；遇到不满足的 placement 立即给出明确错误。

每条轨迹按整个 episode 的首次 done 截断，包含触发 done 的 action；成功标签来自环境明确的成功语义，时间截断不自动算成功。即使模拟器在后续 chunk 继续执行，所有首次终止后的 hidden 都无效。初始化的 settling steps、模型未执行的预测 horizon、padding 均不计入 T。

对至少有一条成功轨迹的组，K=min(T_i | success_i=1)。所有成员都必须至少有 K 个有效点；若某条早终止失败轨迹短于 K，整组几何无效，不能补点或悄悄改变 K 规则。

### 3.2 D、V、G 与几何 advantage：支持 G 或 V 评分

对每条轨迹的 X∈R^(K×1024)，使用已有离线算法的数值口径：

- X_c=X−mean_rows(X)，λ_j 为 X_c X_c^T 的特征值。
- V=(1/2) Σ_j log(1+(1024/K) max(λ_j,0))。
- D 为 scikit-dimension 0.3.7 中 TLE 的逐点 dimension_pw_ 均值；k=10，epsilon=1e-4。
- G=V exp(−0.1D)。评分对象 S 由 geometry.score 选择：volume_exp_neg_dim 对应 G，volume 对应 V。D/V 始终只作诊断，不作为评分对象。
- 计算使用 FP64，不额外标准化 hidden，不做 PCA、随机投影或 jitter。
- 沿用谱负值容差 1e-12+1e-10×max(λ_max,0)；两种评分共用 shape、有效前缀与 K>k 门槛。超容差负谱、非有限输入或非正/非有限 V 使 V 与 G 无效；TLE 局部维度无效或 D 非正/非有限使 D、D/V 与 G 无效，单独保留 V 的有效状态。

整组所选评分 S 有效后才做组内标准化：z_i^S=(S_i−mean_g(S))/(std_g(S,correction=1)+1e-6)。std≤1e-6 的组按无几何信号处理；不删除无效成员后用剩下几条重新标准化，也不使用 nanmean 掩盖失败。常数组与有效性均按所选 S 判断，不能用 G 的 std 或 valid 决定 V 分支是否有信号。

令 c 从 0 开始，M^mani_i,c 表示该 chunk 的全部动作有效且 (c+1)C≤K。两种模式为：

- task_plus_geometry：A_i,c=A^task_i,c+0.2 M^mani_i,c z_i^S。
- geometry_only：A_i,c=0.2 M^mani_i,c z_i^S。

A^task 沿用原生 GRPO 计算，几何项为 detach 的奖励信号，最终 A 不再二次标准化。geometry_only 仍然通过成功标签确定组资格和 K，不能描述成“不使用成功信息”。

新增的 V 混合方案使用同一轨迹前缀的原始 V 作为评分对象，明确写为：

```text
z_i^V = (V_i - mean_g(V)) / (std_g(V, correction=1) + 1e-6)
A_V,i,c = M^mani_i,c * z_i^V
A_mani,i,c = A_V,i,c
A_i,c = A_GRPO,i,c + 0.2 * A_V,i,c
```

这里 A_GRPO 即原生 A^task；0.2 乘的是带完整 chunk 前缀 mask 的组内标准化 V advantage，不是原始 V。V 分支不使用 G 或 exp(−αD) 修正，不对 V 再取 log、标准化 hidden 或更改归一化分母；alpha 对它不生效。D/G 可作为可选诊断，TLE 缺失或 D/G 无效不能屏蔽有效的 V advantage；若开启诊断，独立记录诊断状态。V-only 由 geometry_only 与 volume 组合表达，仍受相同成功标签、K 和组资格约束；本次新增实验对照为混合方案 A_GRPO + 0.2 × A_V。

例如 K=173、C=10 时，用 173 个点计算所选评分（G 分支计算 D/V/G，V 分支计算 V），只给前 17 个完整 chunk 加几何项。第 18 个 chunk 不接收几何项；混合模式中，它是否保留任务优势由原生 task loss mask 决定。

### 3.3 筛选、无效分支与 loss 契约

组类型选择、advantage 模式与评分对象是三个独立接口。下表中的 S 指配置所选 G 或 V。mixed_only 仅允许 mixed 组；any_success 同时允许 mixed 和全成功组。两者都使用本轮固定原始采样，不改变采样配额。

| 组状态 | task_plus_geometry | geometry_only |
| --- | --- | --- |
| mixed，S 有效且有变化 | task + 几何前缀 | 仅几何前缀 |
| mixed，S 无效或常数 | 保留原生 task；几何项为 0 | 整组无更新，禁止回退到 task |
| 全成功，any_success，S 有效且有变化 | task 自然为 0，仅几何前缀 | 仅几何前缀 |
| 全成功但 S 无效或常数 | 无更新 | 无更新 |
| 全成功且 mixed_only | 筛掉 | 筛掉 |
| 全失败 | K 未定义，筛掉 | K 未定义，筛掉 |
| 身份、版本、shape 不一致 | 明确报错 | 明确报错 |

enabled 模式由新逻辑负责组筛选，不能再叠加原生 rewards_upper_bound=0.9 把全成功组删掉。disabled 完整走旧路径，包括原生筛选，不采集 hidden、不要求安装几何可选依赖。

保留原生 chunk PPO ratio、clip 与 loss 归一化方式。几何前缀外设 A_geo=0，不根据 A!=0 重新推导 loss_mask 或缩小归一化分母。组筛选、原生动作有效性、几何更新范围分别保存，防止三种 mask 混用。

保留固定张量布局和各 rank 一致的训练循环，不按本地有效组数物理删样本。每个 optimizer 更新边界都检查全 rank 是否有信号：局部为空但其他 rank 有信号时正常参与同步；全局无信号时一致跳过 optimizer 更新，并明确 scheduler/step 计数语义，避免 Adam 动量或 weight decay 在“无更新”时仍改参数。

## 4. 提交总览

按以下顺序实施，每个提交都附带对应组件测试及本计划状态更新。C01–C04 在默认关闭的内部路径上完成组件，C05 才接通在线训练；每个中间提交都必须保持原有配置可用。

| 编号 | 建议 Commit Subject | 单一子需求 | 依赖 |
| --- | --- | --- | --- |
| C00 | docs(manifold): record the mani v1 migration plan | 冻结迁移口径与提交计划 | 无 |
| C01 | feat(manifold): add shared action manifold metrics | 建立在线离线共用的几何计算核心 | C00 |
| C02 | feat(openpi): expose rollout action expert features | 从真实训练 forward 可选导出 action hidden | C00 |
| C03 | feat(rollout): carry manifold features and episode identities | 跨 worker 保真传递 hidden 和组身份 | C02 |
| C04 | feat(manifold): score groups with matched action prefixes | 完整组校验、有效前缀与可选 G/V 评分 | C01、C03 |
| C05 | feat(actor): support both manifold advantage modes | 将双模式、G/V 评分选择与筛选接入 PPO | C04 |
| C06 | feat(manifold): report group and advantage diagnostics | 让有效信号、失败原因与成本可观测 | C05 |
| C07 | feat(pi-rl): snapshot the modified manifold training source | 确保运行实际迁移后的代码 | C05 |
| C08 | feat(config): add mani v1 training recipes | 提供三模式及 G/V 评分组合配置 | C05–C07 |
| C09 | test(manifold): verify the training and inference lifecycle | 真实 train/val/infer 验收并形成交接证据 | C08 |

C01 和 C02 在设计上互不依赖，但实际提交仍可线性完成。C09 的集成验证不能替代前面每次提交的局部测试。

### C00：固定计划与已有决策

将本文件单独提交，作为后续实现和审查依据。只暂存本计划，不把目录中其他实验结果、图片或用户已有修改带入提交。

验收：确认两种模式、G/V 评分选择及 A_GRPO + 0.2 × A_V 公式、完整 chunk 边界、固定 64×8 采样、建议默认值与未定训练资源均写清。提交前重新记录 HEAD、git status 和已有未提交文件；后续每个 Cxx 填写实际 SHA、测试命令及结果。

### C01：统一离线与在线几何计算

把已有 metrics 算法抽成可在训练中调用的纯计算组件，避免训练与离线脚本各自维护一份公式。

改动范围：新增 rlinf/algorithms/manifold/__init__.py、metrics.py；修改 scripts/pi_rl/manifold_geometry.py 使其调用共享核心；按实际依赖组织补充可选 scikit-dimension 版本要求。接口接受 [N,d] 数组及参数，返回 D、V、G、D/V、N、各指标独立的 valid/reason 和数值诊断；π0.5 适配层验证 d=1024。

验收：用已有保存的 π0.5 hidden 对齐新旧算法的 D/V/G 和 invalid 状态；用小型已知矩阵检查 V；覆盖常数、重复点、N≤k、NaN/Inf 等退化输入。数值差异报告容差与最大误差。缺少 scikit-dimension 时，基线和纯 V 评分仍可运行；启用 G 评分时提示依赖问题。覆盖 V 有效但 TLE/D/G 无效的情况，确保 volume 不误用全指标共用的 invalid 标志；原离线报告继续保留其 D/V/G 定义与状态。

边界：不在此提交接入训练，不改变现有离线实验定义，也不引入优化过的替代指标。

### C02：从真实 rollout forward 导出 hidden

只增加一个可关闭的模型输出能力，验证特征采集不会改变生成本身。

改动范围：rlinf/models/embodiment/openpi/tasks/rl.py；必要时扩展已有策略调用的可选参数。直接保留 Pi0RL._predict_train 最后一次 run_suffix 的结果，截取前 C 个 action 位置，返回 detached FP32 manifold_hidden。默认关闭，eval 默认关闭。

验收：固定模型、输入与完整随机状态，对比采集开关前后的 actions、chains、选中的 denoise index、prev_logprobs，以及随机状态消耗；同时验证 hidden 与本次 action_out_proj 的实际输入一致。测试包含原生随机采样路径和现用 torch_compile 设置；不能只测离线 ODE。

边界：不加额外 denoise pass，不在 actor 的 PPO 重算过程中重新评分，不改变 logprob 定义、噪声参数或可训练层。

### C03：贯通 hidden 与 episode 身份

把模型输出可靠地送到 actor，同时提供可验证的组身份，保证评分时用的是同一 behavior-policy 下的真实同组轨迹。

改动范围：huggingface_worker.py、embodied_types.py、env_worker.py；必要时在 libero_env.py 暴露已有 task/trial/reset-state 元数据。显式扩展 PolicyOutput、EnvPart、TrajectoryStep、Trajectory 的可选字段，以及 split/merge/to_batch/CPU detach 路径。保留原始 per-action done 信息，不用已折叠的 chunk mask 反推精确动作长度。

验收：用可辨认的 member/feature 序列经过真实 schema split、merge、stack、epoch 重排后，检查点序和身份无丢失；可选字段缺失时兼容旧数据；错误混组能被检测。分别检查 hidden 不进入 forward_inputs、不被当成训练模型输入、没有持有 autograd graph。

边界：首版验证组在本地评分分片中完整存在；跨 rank 分组不满足约束时报错，不悄悄拼邻近样本。

### C04：完成组前缀裁剪与评分

接收完整原始组，生成精确 K 的几何评分，以及独立的完整 chunk 前缀 mask。

改动范围：新增 rlinf/algorithms/manifold/groups.py，建立显式 GroupBatch/GroupScore 类或等价类型契约；actor 在 minibatch 打乱前调用。输出每组 K、成员长度与成功标签、每条 D/V/G 及各自状态、所选 score 的均值/std 与组级 valid/reason、几何 chunk mask。

验收：覆盖 K=173、C=10；首次成功在 chunk 中间；成功后模拟器继续执行；全失败；全成功；成员有效长度短于 K；成员重复/缺失；跨 epoch 或版本混组；所选评分常数或单成员计算失败；V 与 G 的常数/有效状态不同。身份错误必须报错，数值无效必须给出可统计的原因。

边界：此提交产出分数和 mask，不修改 PPO advantage。未使用的模型 horizon、padding 与 settling steps 均不能贡献几何点。

### C05：接入两种 advantage 模式

在 actor 已完成原生 task advantage、尚未打乱训练样本的位置组合几何项，并统一管理 enabled 模式的组筛选。

改动范围：新增 rlinf/algorithms/manifold/advantages.py；修改 embodied_fsdp_actor_worker.py、rlinf/config.py；按实际配置传递链启用 C02/C03 的特征能力。优先复用原生 GRPO，不复制整个训练循环或改写 chunk PPO loss。

接口为 algorithm.manifold.mode=disabled|task_plus_geometry|geometry_only，eligible_groups=mixed_only|any_success，geometry.score=volume_exp_neg_dim|volume；其他字段见第 5 节。配置缺省按 disabled；未知值、非支持后端、auto_reset/ignore_terminations 等不兼容组合在启动阶段报错，不静默改写用户配置。

验收：手算小组对齐两种 mode 在 G/V 两种评分下的 A，明确覆盖 A_GRPO + 0.2 × A_V；构造 V 相同但 D 不同、V 不同但 G 相同的组，验证常数判定跟随所选评分，并验证仅改变 D/alpha 不改变 V advantage；逐行覆盖第 3.3 节分支；确认 additive 的非几何前缀保留 task，geometry_only 的非前缀严格为零；确认原生 0.9 上界不会二次删除全成功组；确认总 A 不再标准化。disabled 同输入同 seed 的 advantage、mask、loss 与原生路径一致。

分布式验收覆盖本地全空但其他 rank 有效，以及全局全空的更新边界。同步跳过必须覆盖参数、Adam 状态、scheduler 与更新计数；不能只把 loss 乘零后仍执行 optimizer.step。

边界：不根据非零优势比例改变原生 loss 归一化，不做按任务 α、自适应 λ、残差化等方法升级。

### C06：记录信号与计算成本

让训练日志可以解释“采到了多少、哪些进入更新、几何项多大、哪里失败”，便于判断后期 SR 停滞时是信号问题还是实现问题。

改动范围：actor 诊断聚合，以及 docs/manifold/pi_rl/mani_v1/ 下的字段说明。保留原生 task reward、环境 SR 和任务计数；新增指标使用独立 manifold 命名空间。

至少记录：原始/eligible/mixed/全成功/全失败组数，invalid 和 constant 原因计数，K 的分布，几何覆盖的完整 chunk 比例，D/V/G 各指标状态、所选 score 及其组内均值/std，task/geometry/final advantage 范数及同向比例，实际更新与跳过次数，几何耗时、CPU hidden 字节数。

保存轻量组级与轨迹级标量审计记录，携带 group/member/task/reset-state/version、长度、K、D/V/G 及各自状态、score、所选 score 的 valid/reason、配置和代码版本。默认不持久化所有高维 hidden；提供有数量上限的调试抽样以便复算。所有字段在产生时记录，不能训练后用新模型重新计算来代替。

验收：计数守恒、分布式聚合权重正确、无效值不混入正常均值；重放一批标量可还原组筛选与几何 advantage。日志中的 enabled/disabled、mode、score、eligible_groups 和真实有效组数清晰可见。

实施记录：已新增独立诊断聚合和带源码/配置指纹的标量审计；默认不保存 hidden，有上限调试保存跨重启计数。196 项数学/配置检查、9 项 actor/依赖检查与双 A100 更新/空批审计检查通过；无 TLE 环境 101 passed、43 skipped。字段见 [DIAGNOSTICS.md](DIAGNOSTICS.md)，命令、硬件和结果见 [C06_VALIDATION.md](C06_VALIDATION.md)。尚未创建提交。

### C07：保证启动器使用改动后的源码

建立 mani_v1 专用的准备与启动路径，把实际迁移代码冻结进运行目录，避免在工作区改好了代码却仍跑旧版官方包。

已确认现有 scripts/pi_rl/prepare_training.py 强制校验官方 30e0236，并从 state 中的 rlinf_repo 归档源码。该规则服务于原有官方复现，保留其语义。新入口建议为 scripts/pi_rl/mani_v1/prepare_training.py 和 launch_training.py；优先复用适用的通用功能，新增路径名以实现时审查为准。

准备阶段记录实际 source repo、HEAD、提交序列、必要工作区 diff/文件清单与 hash、最终 resolved config、依赖版本和模型身份。正式实验优先使用已提交的干净源码；如允许开发态快照，必须明确包含被追踪改动与所需新增文件，不能仅 git archive HEAD 后遗漏修改。

运行时记录已加载关键模块的文件位置及 hash，核对它们来自本次快照。prepare 只生成文件，不提交云任务；launch 的实际入口与资源选择分开。使用独立输出目录，保留已有官方复现记录。

验收：离线准备后验证快照包含几何核心与 actor 接入代码；篡改快照能被发现；导入来源能被验证；原官方 prepare 路径行为保持不变。此提交不以实际长训练作为验收条件。

实施记录：已新增独立 mani_v1 准备/启动入口；默认要求干净提交，显式开发态包含运行所需 tracked 改动、新增与删除文件。34 项启动器检查、30 项审计回归通过；现有真实 π0.5 checkpoint 的离线准备与 check-only 通过，关键模块导入报告与快照 hash 对齐。方法配置与真实 worker 生命周期仍待 C08/C09。操作见 [SOURCE_SNAPSHOT.md](SOURCE_SNAPSHOT.md)，证据见 [C07_VALIDATION.md](C07_VALIDATION.md)。尚未创建提交。

### C08：提供可切换的训练配置

在已有 LIBERO-10 π0.5 配置上增加独立 mani_v1 recipe，让 G 混合、G-only、新增 V 混合方案与基线使用明确可比的模型、采样和优化参数；评分对象与 mode 独立选择。

改动范围：examples/embodiment/config/ 下新增 mani_v1 配置及所需小型公共配置；配套启动说明。三种 mode 共用静态配置结构，避免复制三份容易漂移的超参数。正式 recipe 保持 64 env×8 epoch、group_size=8、C=10、480 action horizon、4 denoise，以及原生 optimizer/update_epoch 等设置。

验收：实际 compose 并保存 disabled，以及 task_plus_geometry/geometry_only × volume_exp_neg_dim/volume 的 resolved YAML，重点标明 V 混合配置对应 A_GRPO + 0.2 × A_V；逐字段比较与基础配置的差异，明确哪些是方法必要变更。mixed_only/any_success 可独立选择，错误字段或不支持组合能在启动前被拒绝。

小规模 GPU smoke 使用单独覆盖配置，允许缩小环境数与 epoch 但保持完整 8 人组，并满足 global/micro batch 整除要求。smoke 覆盖与正式 64×8 配置分开存档，不能把 smoke 的采样预算误用为正式默认值。

实施记录：已提供一份共享正式配置和独立两 GPU smoke 配置，五种 mode/score × 两种组选择器的静态检查通过。保存五份正式、五份 smoke resolved YAML 与逐字段差异；66 项配置检查通过，无 TLE 环境 10 passed、4 skipped。见 [RECIPES.md](RECIPES.md)、[C08_VALIDATION.md](C08_VALIDATION.md)。尚未创建提交；不把配置检查当作真实训练验收。

### C09：验证完整模型生命周期并交接

用有界的真实 GPU 检查完成 train、val、checkpoint 保存重载和 infer；结果记录为可复查证据，再判断实现是否可以用于正式实验。

改动范围：tests/e2e_tests/embodied/ 的适用配置或测试入口，以及 docs/manifold/pi_rl/mani_v1/ 下的 VALIDATION_REPORT.md、RUNBOOK.md。测试与说明引用 C07 的真实源码快照，记录具体硬件、命令、随机种子、数据/模型版本、耗时、日志路径、通过/失败项。

建议使用 2 张 GPU，先验证 disabled 的有限基线一致性，再分别对 G 的 task_plus_geometry、geometry_only，以及新增 V 的 task_plus_geometry 做有限更新；报告明确列出实际覆盖的 mode/score 组合，不把 V 混合验证扩称为 V-only 全生命周期已验证。确认确实出现过非零几何信号及参数更新；若自然采样全被筛掉，只能说明空批处理通过，不能声称几何训练路径已覆盖。必要的人工构造组分支在单元测试覆盖，真实环境检查仍使用真实 rollout。

验收包括：训练完成且 loss/梯度有限；val 可运行且原生 SR 正常记录；checkpoint 可保存重载；重载后执行独立 infer；未启用采集时推理不携带 hidden 开销；训练结束后高维 hidden 不残留在 checkpoint 或 minibatch。测量附加内存和几何耗时，不能把理论估算写成实测结果。

实施进展：已新增真实生命周期观测与 strict 重载/推理测试入口，disabled 双 A100 完整链通过（每 rank 24 次更新、477 个冻结张量不变、190 个可训练张量变化）；35 项启动器组件回归通过。G 混合正在运行，其他目标组合及内存对照尚未完成。当前证据与未完成项见 [VALIDATION_REPORT.md](VALIDATION_REPORT.md)，操作见 [RUNBOOK.md](RUNBOOK.md)。

父目录 AGENTS.md 要求实质代码变更以 GPU train/val/infer smoke 验证。若硬件或环境不可用，报告具体阻塞和已完成部分，状态保持“实现完成、GPU 验证未完成”。本计划不启动或停止现有训练；任何清理必须限于本次创建的资源。

## 5. 配置草案

以下 YAML 表达目标接口，C05/C08 完成前不能直接用于现有训练。数值属于初始迁移配置；将来调整应产生新实验配置，保留旧结果的可追溯性。

```yaml
algorithm:
  adv_type: grpo
  reward_type: chunk_level
  logprob_type: chunk_level
  group_size: 8
  manifold:
    mode: task_plus_geometry  # disabled | task_plus_geometry | geometry_only
    eligible_groups: any_success  # mixed_only | any_success
    weight: 0.2
    feature:
      source: action_expert_pre_output
      denoise_step: last
      storage_dtype: float32
    geometry:
      granularity: action
      tle_k: 10
      tle_epsilon: 0.0001
      compute_dtype: float64
      score: volume_exp_neg_dim  # volume_exp_neg_dim (G) | volume (V)
      alpha: 0.1  # 仅 G 评分使用
    prefix:
      rule: min_success_actions
      update_boundary: full_chunk
    normalization:
      std_correction: 1
      epsilon: 0.000001
      min_group_std: 0.000001

env:
  train:
    total_num_envs: 64
    rollout_epoch: 8
    group_size: 8
    auto_reset: false
    ignore_terminations: false
    max_episode_steps: 480
    max_steps_per_rollout_epoch: 480

rollout:
  pipeline_stage_num: 1

actor:
  model:
    num_steps: 4
    num_action_chunks: 10
    openpi:
      task: rl
      noise_method: flow_sde
```

新增 A_GRPO + 0.2 × A_V 方案只需选择以下组合，其余设置沿用上面的 recipe：

```yaml
algorithm:
  manifold:
    mode: task_plus_geometry
    weight: 0.2
    geometry:
      score: volume
```

此片段是评分选择覆盖项，不是独立完整配置。G 混合、V 混合使用相同的采样、组类型、K、完整 chunk mask、标准化与 loss 契约；区别仅为 A_mani 的评分对象。

其余模型、placement、optimizer 和 loss 参数继承已核对的基础 recipe。公开字段只接受首版已经实现的枚举；字段可扩展不代表目前支持其他 denoise 位置、粒度或精度。

## 6. 每次提交的完成标准

实施时按提交目标检查 diff，避免把独立方法改动或实验产物混入同一提交。每个功能提交随附会因该功能破坏而失败的必要测试，不添加只复述实现细节的测试。

1. 实施功能前按仓库要求查询 RLinf/RLinf 的相关 open PR/issue，并记录是否重叠；2026-11-02 已通过 GitHub Search API 检索 manifold：开放 PR 为 0；唯一 issue #784 是 ABot-M0 支持请求，与本迁移不重叠。
2. 检查适用 AGENTS.md、当前 HEAD 与工作区；保留已有用户改动，使用明确文件清单暂存。
3. 每个提交包含该子需求的实现、必要组件测试和最小文档更新；新模块含 __init__.py、Apache header，检查 CI 路径覆盖。
4. 运行与本次改动匹配的 lint、组件测试；涉及真实模型行为的核对记录硬件条件。C09 汇总全链路 GPU 证据，不伪称已经跑过。
5. 使用 Conventional Commits；按仓库规则用真实用户 Git 身份和 Signed-off-by。若没有有效用户身份，记录并处理身份问题，不虚构作者。
6. 每提交一次，把下表填入实际 SHA、证据路径与限制；默认不 push、不创建 PR、不提交长期训练任务。

| 编号 | 状态 | 实际 SHA | 验收记录 |
| --- | --- | --- | --- |
| C00 | 已暂存，待作者身份确认后提交 | — | 本文；服务器回读与路径检查 |
| C01 | 已实现并通过组件验证，待提交 | — | [C01 验收记录](C01_VALIDATION.md)；不代表在线/GPU 生命周期已通过 |
| C02 | 已实现，真实 A100 采样一致性通过，待提交 | — | [C02 验收记录](C02_VALIDATION.md)；torch_compile 关/开均通过 |
| C03 | 已实现，schema 与真实 LIBERO 元数据检查通过，待提交 | — | [C03 验收记录](C03_VALIDATION.md)；全数据套件 142 通过，1 项既有 Transformers 兼容失败已在原测试复现 |
| C04 | 已实现，56 项组件测试通过，待提交 | — | [C04 验收记录](C04_VALIDATION.md)；无 TLE 环境 50 通过、6 跳过 |
| C05 | 已接入，优势/配置/actor 与真实双 GPU 更新边界检查通过，待提交 | — | [C05 验收记录](C05_VALIDATION.md)；110 项数值/优势、45 项配置、8 项 actor、1 项双 A100、1 项真实 π0.5 worker 通过 |
| C06 | 待实施 | — | — |
| C07 | 待实施 | — | — |
| C08 | 待实施 | — | — |
| C09 | 待实施 | — | — |

当前已有未提交的 pi05_tasks1239_manifold_20261102.md 及其 assets 位于相邻目录，属于既有分析材料；本次计划不覆盖、不暂存这些文件。本文是 docs/manifold 下的研究实施计划，不在 docs/source-en、docs/source-zh 的 Sphinx 构建树中；本次验收核对 Markdown、路径与源码口径。以后若增加公开使用文档，按仓库规范同步两种语言并执行对应构建。

## 7. 首版交付边界与正式实验选择

C09 通过后，交付的是可复现的双模式与 G/V 评分选择实现、对应配置、必要诊断，以及明确标注已测试组合的生命周期证据。效果检验再按明确的训练预算执行，不把短 smoke 的 SR 当作方法收益。

正式运行前需要确定首跑 mode、geometry.score、eligible_groups、训练步数、保存/评估频率和计算资源。这些选择不会阻塞当前接口设计；已确认的完整 chunk 边界、固定 64×8 和双模式支持不再重复询问。

首版保留 G=V exp(-0.1D) 作为默认评分，并加入用户要求的 V 混合对照 A_GRPO + 0.2 × A_V，以比较 A_mani 使用 G 或 V 时的训练表现。固定 K 消除了同组点数差异，但不能保证跨组、跨任务、跨 policy 阶段的 D/V/G 已校准；1024 维 π0.5 表征与旧 OpenVLA 28672 维表征也不能直接比较绝对数值。后续若调整 α、加入停滞校正或改变 action 选择方式，应另立方法实验与提交，避免把迁移正确性和指标升级混为同一结论。
