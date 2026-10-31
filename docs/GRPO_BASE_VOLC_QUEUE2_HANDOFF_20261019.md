# SimpleVLA 纯 GRPO baseline：Volc 队列 2 接手文档

更新日期：2026-10-19（UTC）。当前状态：修复后的 r2 任务 `t-20261019210705-b7xbl` 已在队列 2 正常进入 rollout，达到本次提交验收目标；云端任务继续运行。本文记录本次新运行，不复用或续训历史 RL 权重。

## 最先查看的路径

**本中文接手文档：**

```text
/vepfs-C区/jiajunlu/SimpleVLA-RL/docs/GRPO_BASE_VOLC_QUEUE2_HANDOFF_20261019.md
```

**中文逐 RL step 训练日志：**

```text
/vepfs-C区/jiajunlu/SimpleVLA-RL/experiments/train/grpo_base/libero10-grpo-base-seed7-q2-20261019T130046Z-r2/step_metrics_zh.txt
```

该日志由 `TrainingStepAudit` 在完成 actor update 后写入；刚进入 rollout 时尚无完成步，因此文件可以尚不存在。不要为生成日志重复启动训练。对应结构化主记录为同目录的 `rl_step_metrics.jsonl`。

**实验根目录（下文记为 RUN_ROOT）：**

```text
/vepfs-C区/jiajunlu/SimpleVLA-RL/experiments/train/grpo_base/libero10-grpo-base-seed7-q2-20261019T130046Z-r2
```

完整训练输出为 `RUN_ROOT/train.log`；提交回执、平台状态、远端预检及 rollout 验收证据均保留在该实验目录。所有路径位于共享 VEPFS，退出当前连接节点不影响云端任务。

## 已确认的训练配置

| 参数 | 本次值 |
|---|---|
| 算法 | `algorithm.adv_estimator=grpo` |
| 流形方法 | `algorithm.rollout_mani_v1=False`，`algorithm.manifold.advantage_weight=0.0`，不采集流形 hidden、不计算几何辅助项 |
| Reward model / critic / reference policy | RM 关闭且系数 0；GRPO 不创建 critic；KL 系数 0，不创建 reference policy |
| 任务奖励 | 仅 LIBERO 完成标志，成功 × 5，失败 0；无额外 shaping |
| 数据 | `libero_10`，task 0–9，每任务 trial 0–49，共 500 个初始状态 |
| 分组 | `data.train_batch_size=64`，`data.n_samples=8`，每次 RL update 使用 512 条有效轨迹 |
| 筛选 | `filter_accuracy=True`，成功率区间 `[0.1,0.9]`，每组保留成功数 1–7；全成功、全失败组剔除 |
| 其他采样 | `oversample_factor=1`，`filter_truncated=False` |
| 跨 epoch 保留 | **`data.accumulate_valid_groups_across_epochs=True`**，保留未凑满更新批次的完整组，再 shuffle 继续采集 |
| LR / schedule | `5e-6`，constant，无 warmup |
| 优化器 | AdamW，betas `(0.9,0.999)`，weight decay `0.01` |
| PPO mini-batch / epochs | `128 / 1`，每个完成 RL step 包含 4 次 optimizer update |
| PPO micro-batch | 全局 8，每卡 1 |
| log-prob micro-batch | 全局 32，每卡 4 |
| 轨迹内部切片 | `traj_mini_batch_size=16` |
| PPO clip | low `0.20`，high `0.28`，对应 ratio 区间 `[0.80,1.28]` |
| Gradient clip / KL / entropy | `1.0 / 0 / 0` |
| 训练动作采样 | `do_sample=True`，temperature `1.6` |
| 动作 | 7 维，chunk length 8，LIBERO rollout 上限 512 个策略动作 |
| 初始化 | 等待 10 个 dummy action，不计入策略 horizon |
| 观测 | 单图，center crop=True，use_proprio=False；unnorm_key 按模型统计量解析 |
| Prompt / response 配置 | data `256 / 128`；rollout.max_prompt_length `512`，不将 response_length 当作环境 action horizon |
| 模型训练 | 全参数，LoRA rank 0，BF16；gradient checkpointing=False；remove padding=False |
| FSDP offload | param=False，grad=True，optimizer=True |
| 并行 | 单机 8 卡，tensor model parallel=1，HF rollout |
| 随机种子 | `trainer.seed=7`；worker 使用 rank 派生种子；LIBERO 环境仍按原实现 env.seed(0) |
| 更新预算 | `trainer.total_training_steps=300`，`trainer.total_epochs=300` |
| 保存 | `trainer.save_freq=40`，`trainer.save_at_end=True` |
| 内置验证 | `val_before_train=False`，`test_freq=-1`，`val_only=False`；val_batch_size=496 仅作为 loader 配置 |
| 日志 | console + SwanLab online + 中文逐步日志 + JSONL 审计 |

跨 epoch 开启后，DataLoader 在内层重新遍历，不消耗外层 `total_epochs` 计数；因此 **300 个完成 RL update 是本次实际停止条件**，不能把 300 epochs 宣称为真实数据遍历上限。采集始终使用同一 policy version，收齐 64×8 才更新。超额通过筛选的组仍沿用原实现截断，不跨 policy version 保存旧轨迹。

若完成 300 个 RL update，accepted budget 为 153,600 条轨迹、1,200 次 optimizer update。筛选前物理采样量不固定。训练池与标准评测初始状态池重合，不将其表述为未见状态泛化评测。本次目标是确认云端正常进入 rollout，不要求等待完整训练或独立评测。

## Checkpoint 位置与编号

```text
RUN_ROOT/actor/global_step_39/    # 完成第40次RL更新
RUN_ROOT/actor/global_step_79/    # 完成第80次RL更新
RUN_ROOT/actor/global_step_119/
RUN_ROOT/actor/global_step_159/
RUN_ROOT/actor/global_step_199/
RUN_ROOT/actor/global_step_239/
RUN_ROOT/actor/global_step_279/
RUN_ROOT/actor/global_step_299/   # 达到300步上限时额外保存
```

目录索引从 0 开始。checkpoint 是模型/推理资产，现有实现不提供完整 optimizer、scheduler、RNG、DataLoader 进度恢复；不要将其当作等价断点续训。

## 模型、代码和依赖路径

| 用途 | 路径 |
|---|---|
| 当前仓库 | `/vepfs-C区/jiajunlu/SimpleVLA-RL` |
| 实际远端代码 | `RUN_ROOT/source_snapshot/`，包含当前未提交修改，SHA256 见 `launch_source_manifest.json` |
| 初始 SFT | `/vepfs-C区/wam/zouyangang/openvla_rl/model/Openvla-oft-SFT-libero10-traj1` |
| 独立 checkpoint view | `RUN_ROOT/checkpoint_view/`；权重只读使用链接，Python/config/tokenizer 为真实副本 |
| SFT 权重与代码审计 | `RUN_ROOT/checkpoint_view_manifest.json`，完整四分片 SHA256 |
| 共享 Python | `/vepfs-C区/jiajunlu/shared/envs/simplevla/bin/python` |
| OpenVLA-OFT | `/vepfs-C区/jiajunlu/openvla-oft` |
| LIBERO Python 包 | `/vepfs-C区/jiajunlu/LIBERO` |
| LIBERO 仿真资产 | `/vepfs-C区/jiajunlu/LIBERO/libero/libero/{assets,bddl_files,init_files}` |
| 本次 LIBERO 配置 | `RUN_ROOT/libero_config/config.yaml`；不用全局 `~/.libero` 配置 |
| 10任务资产审计 | `RUN_ROOT/libero_asset_manifest.json` |
| Ray 环境 | `RUN_ROOT/runtime/ray_runtime_env.json` |
| OSMesa 依赖兼容目录 | `RUN_ROOT/runtime/osmesa/`，库来源与 SHA256 见 `runtime/osmesa_manifest.json` |
| Ray/临时/编译缓存 | `/vgrpo` 是云端指向 `RUN_ROOT/runtime/work` 的短 ASCII 链接，避免 Ray 非 ASCII 与 Unix socket 长度限制 |
| SwanLab | 使用现有登录凭据，经运行私有目录提供给节点；文档、YAML、普通日志不记录密钥 |

本地核对的共享 Python 版本：Python 3.10，torch `2.4.0+cu124`，transformers `4.40.1`，ray `2.10.0`，tensorflow `2.15.0`，mujoco `2.3.7`，robosuite `1.4.1`，libero `0.1.0`，swanlab `0.4.12`，flash-attn `2.5.5`，timm `0.9.10`，hydra-core `1.3.5`，omegaconf `2.3.1`，tensordict `0.5.0`。远端实际版本与 GPU 型号以 `RUN_ROOT/remote_preflight.json` 为准。

渲染采用 OSMesa + fork + persistent LIBERO workers，每 rank 8 个环境子进程。线程数为 1，TensorFlow 使用按需 GPU 内存增长；不升级共享环境。OSMesa 及其非 glibc 动态依赖单独复制到运行目录，避免镜像缺系统库。内置 val 关闭不影响训练 rollout 的 SR 日志。

## 提交与实验文件

| 文件 | 用途 |
|---|---|
| `examples/run_openvla_oft_rl_grpo_base.sh` | 纯 GRPO 配置入口，固定本次已确认超参 |
| `examples/run_openvla_oft_rl_libero10_overfit50.sh` | 复用原有训练启动与运行时资源适配，由新入口覆盖为500状态/64×8 |
| `scripts/run_grpo_base_volc.sh` | 云端预检、环境变量和训练日志入口 |
| `scripts/grpo_base_volc_queue2_20261019.yaml` | 本次 Volc 提交文件 |
| `RUN_ROOT/submission.yaml` | 随实验冻结的提交文件副本 |
| `RUN_ROOT/launch_source_manifest.json` | 全部262个冻结文件的 SHA256 与 Git HEAD |
| `RUN_ROOT/source_manifest.json` | 训练审计器实际记录的12个关键源码文件 SHA256，与完整清单逐项一致 |
| `RUN_ROOT/git_status.txt`、`worktree.patch` | 提交时工作树背景；不是完整实验代码的替代品 |
| `RUN_ROOT/resolved_config.yaml` | 启动参数解析预览；远端实际生效配置为同目录 `resolved_config.json`，已逐项核验 |

队列 2：`q-20260403140809-n4flj`；单 worker，`ml.hpcpni2l.28xlarge`，8×A800 80GB。镜像为 `vemlp-cn-beijing.cr.volces.com/preset-images/pytorch:2.10.0-cu12.9.1-py3.13-ubuntu24.04`；实际训练使用共享 Python 3.10，而非镜像默认 Python。挂载 `vepfs-cnbjba6dad11eac7` 到 `/vepfs-C区`。

```bash
cd /vepfs-C区/jiajunlu/SimpleVLA-RL
/root/.volc/bin/volc ml_task submit -c scripts/grpo_base_volc_queue2_20261019.yaml --output json
```

这条命令会创建新云任务；接手后先查现有任务，避免重复提交。本次任务 ID：`t-20261019210705-b7xbl`。查询命令：

```bash
/root/.volc/bin/volc ml_task get -i t-20261019210705-b7xbl --output json
/root/.volc/bin/volc ml_task instance list -i t-20261019210705-b7xbl --output json
/root/.volc/bin/volc ml_task logs -t t-20261019210705-b7xbl -i worker-0 -l 200
```

## 验证与运行证据

已完成配置解析检查：`RUN_ROOT/config_validation.json`；262 个快照文件 SHA256 校验、Python AST 与启动脚本语法核对、4 个 SFT 权重分片完整 SHA256、10 个任务各至少50个初始状态均通过。

本地执行 `python -m pytest tests/test_rollout_mani_v1.py -q -k actual_fit_collect_filter_update_validation_save_and_log`（指定 source_snapshot 的 PYTHONPATH），结果 **8 passed**；覆盖方法开/关、跨 epoch 累积开/关、validation 开/关及保存/日志控制流，使用 mock worker，不是 GPU 模型生命周期测试。首个尝试输出在其 `preflight_tests.log`；r2 直接从修复后源码快照导入后的测试与渲染记录为 `RUN_ROOT/local_import_renderer_tests.log`。

提交回执：`RUN_ROOT/submission_result.json`，成功返回任务 `t-20261019210705-b7xbl`。r2 的本地快照导入、OSMesa context、LIBERO task0 reset 与10个 dummy action 以及8项mock fit测试通过，原始输出见 `RUN_ROOT/local_import_renderer_tests.log`。远端预检已通过：`RUN_ROOT/remote_preflight.json` 确认 8×NVIDIA A800-SXM4-80GB、共享 Python 与 source_snapshot 中的 verl、OSMesa context、LIBERO task0 reset 和10个 dummy action 均正常；依赖版本与上述列表一致。远端 FSDP 初始化已完成，全部8个rank已持续推进实际rollout，验收通过。本次新增的是运行入口和低频进度输出，不修改 GRPO 目标、奖励或模型更新数学；未将配置检查描述为完整 GPU train/val/infer 验证。


### 首个尝试与修复

首个任务 `t-20261019210400-krrqc` 于 2026-10-19 13:05:00 UTC 被平台标记 Failed，原因是源码快照漏带 `verl/version/version`，在 bootstrap 导入阶段退出；未加载模型，未采集训练 rollout。原尝试目录 `/vepfs-C区/jiajunlu/SimpleVLA-RL/experiments/train/grpo_base/libero10-grpo-base-seed7-q2-20261019T130046Z` 完整保留，包含 `train.log` 和 `terminal_platform_status.log`。修复时同时补齐 `verl/single_controller/version/version`，在独立 `-r2` 目录重新验证、重新提交，未覆盖或拼接旧训练记录。


### 只读监控

本次 `RUN_ROOT/monitor_once.py` 会查询固定任务 ID、保存 `platform_latest.json` 与 `platform_observations.jsonl`，并读取共享 `train.log` 的 rollout 进度，不提交或重启任务。

```bash
python /vepfs-C区/jiajunlu/SimpleVLA-RL/experiments/train/grpo_base/libero10-grpo-base-seed7-q2-20261019T130046Z-r2/monitor_once.py
```

本次启用 `LIBERO_ROLLOUT_PROGRESS_INTERVAL_ACTIONS=64`，仅在首个chunk及每64个策略动作后打印 `LIBERO rollout progress`，不改变采样或奖励。其 `policy_actions` 是当前 group 的策略动作推进位置，不是完成的RL更新次数。预检 Python 退出时出现过一次 MuJoCo `GLContext.__del__` 的 ignored 清理异常；预检返回码为0且主训练随后正常启动，不应将该单条 traceback 误读成任务终态。


## 最终 rollout 验收

验收时间（UTC）：**2026-10-19T13:12:43.411499+00:00**。当前任务：**`t-20261019210705-b7xbl`**；平台状态 **Running**；队列 **`q-20260403140809-n4flj`**；8×A800。

8个rank均有 `policy_version=0`、`policy_actions=8/512` 以及后续至少 `64/512` 的实际执行记录。日志在模型生成动作、环境执行并返回观测后打印，证明本次主训练正常进入并持续推进真实rollout。各rank最新已记录的动作位置：`{"0": 384, "1": 384, "2": 384, "3": 384, "4": 320, "5": 384, "6": 384, "7": 320}`。

证据文件：`RUN_ROOT/rollout_acceptance.json`、`RUN_ROOT/rollout_progress_evidence.log`、`RUN_ROOT/platform_observations.jsonl`、`RUN_ROOT/remote_preflight.json`、`RUN_ROOT/resolved_config.json`。远端生效配置逐项核实为纯GRPO、跨epoch累积True、64×8、LR5e-6、temperature1.6、300步上限、save_freq40、关闭内置val。

验收时已完成RL更新数：**0**。当前仍在第一个RL step的采集阶段，尚未产生第40步checkpoint；中文逐step日志将在首个完成更新后生成。本次不宣称300步训练完成、完整GPU train/val/infer生命周期通过或获得最终成功率。任务按既定配置继续运行，不因达到本次接手目标而停止。

最终中文逐step日志绝对路径：

```text
/vepfs-C区/jiajunlu/SimpleVLA-RL/experiments/train/grpo_base/libero10-grpo-base-seed7-q2-20261019T130046Z-r2/step_metrics_zh.txt
```

最终checkpoint根目录：

```text
/vepfs-C区/jiajunlu/SimpleVLA-RL/experiments/train/grpo_base/libero10-grpo-base-seed7-q2-20261019T130046Z-r2/actor
```
