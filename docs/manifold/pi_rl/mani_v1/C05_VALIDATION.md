# C05：双模式 advantage 与同步更新控制验收

本记录验证 manifold 接入 actor、G/V 两种评分的优势组合、启动配置约束，以及分布式空批的更新行为。C05 实现与对应组件验证已保存在工作区，提交等待真实作者身份确认。当前验证包括真实 π0.5 rollout worker 采样和双 A100 FSDP 小模型更新；完整 LIBERO train/val/checkpoint/infer 生命周期仍属于 C09。

## 接入与数学口径

`rlinf/algorithms/manifold/advantages.py` 使用显式组的 column 对应关系调用原生 GRPO，再恢复原始 batch 顺序。它保持原生任务分数、sample std 和 chunk 单位，只组合 detached 几何奖励信号：

```text
A_geo = 0.2 * full_chunk_prefix_mask * z(selected_score)
task_plus_geometry: A = A_GRPO + A_geo
geometry_only:     A = A_geo
```

所选评分独立选择 G 或 V。V 混合即 `A_GRPO + 0.2 * A_V`；最终 A 没有再次标准化。geometry_only 仍通过成功标签确定组资格和 K。

启用后，actor 跳过旧的 reward 上下界筛选，按新组规则设置 group mask：mixed 组保留；any_success 还保留所选评分有效且非常数的全成功组；全失败组移出更新。mixed 评分无效时，混合模式保留任务优势，geometry_only 的最终优势严格为零。原始动作 mask、组 mask 与完整 chunk 前缀 mask 分别保存在 CPU 结果中。前缀外零优势不会缩小原生 loss 分母，也不会物理删除样本。

评分在 shuffle 前完成，随后移除 rollout batch 的高维 hidden 和 episode context。保留的组级结果只含标量和低维 mask/advantage；high-dimensional hidden 不进入 GPU minibatch。disabled 使用原代码的 reward 筛选、advantage 与 loss 路径。

每个 optimizer 边界通过真实 all-reduce 判断全局是否存在非零的 masked advantage。局部为空但其他 rank 有信号时继续执行相同的 forward/backward；全局为空时所有 rank 跳过 optimizer 方法。原生 LR scheduler 每个 rollout iteration 至多推进一次；enabled 仅在本轮至少一次有限梯度的实际更新后推进。全空 iteration 不推进 scheduler 或 optimizer_steps。runner/collector 的 iteration 和版本编号保持原生采样计数语义。新增 `manifold/optimizer_updates` 和 `manifold/optimizer_skips` 分别记录实际更新及因全局无信号跳过的边界数。

`validate_manifold_cfg()` 检查公开字段、模式、评分对象、特征位置/精度、完整 chunk 前缀与归一化约束。enabled 要求同步 colocated π0.5 LIBERO-10、原生 chunk GRPO、C=10、4 denoise，并确保每个 scoring shard 有完整八人组。配置不会被该校验函数改写。async 入口明确拒绝 enabled 模式；disabled 不加载 TLE，V 不要求 TLE。

## CPU 数学、配置与 actor 检查

命令在仓库根目录执行，验证临时目录为 `experiments/pi_rl/mani_v1_checks`。使用现有 πRL Python 与已保存的 scikit-dimension 0.3.7 依赖：

```bash
TMPDIR="$PWD/experiments/pi_rl/mani_v1_checks" \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH="$PWD:$PWD/experiments/pi_rl/pi05_sft_libero10_tasks129_mani_ode480_20261102T073519Z/task2/runtime/geometry_packages" \
/vepfs-C区/jiajunlu/.envs/pirl-openpi-30e0236/bin/python -u -m pytest \
  tests/unit_tests/test_manifold.py -q \
  --basetemp=experiments/pi_rl/mani_v1_checks/pytest-c05-advantages-final \
  -p no:cacheprovider
```

结果：**110 passed，17.53 秒**，包含 C01/C04 既有组件测试。覆盖两种 mode × 两种 score × 两种 eligible_groups × 六类组状态的完整分支表；手算任务与几何项、精确分母、原生 chunk PPO loss、非前缀行为、显式乱序组的原生 GRPO、detach，以及真实双 CPU rank 的信号一致性均通过。输出为 `experiments/pi_rl/mani_v1_checks/pytest-c05-advantages-final.log`。

另用实际特征及 TLE 构造近似常数 G、变化 V：G 路径判定常数，V 混合仍产生几何优势；混合 G 回到原生 task，G-only 为零。与 C04 的常数 V/变化 G 反例一起覆盖所选评分独立性。

不安装 TLE 的系统 Python 执行相同文件：

```bash
TMPDIR="$PWD/experiments/pi_rl/mani_v1_checks" \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH="$PWD" \
python -m pytest tests/unit_tests/test_manifold.py -q \
  --basetemp=experiments/pi_rl/mani_v1_checks/pytest-c05-no-tle \
  -p no:cacheprovider
```

结果：**79 passed、31 skipped，5.08 秒**。跳过项需要真实 TLE；V 组合、V 无效诊断独立性和 CPU 双 rank 检查均通过。输出为 `pytest-c05-no-tle.log`。

```bash
TMPDIR="$PWD/experiments/pi_rl/mani_v1_checks" PYTHONPATH="$PWD" \
/vepfs-C区/jiajunlu/.envs/pirl-openpi-30e0236/bin/python -u -m pytest \
  tests/unit_tests/test_cluster_config.py -k manifold -q \
  --basetemp=experiments/pi_rl/mani_v1_checks/pytest-c05-config2 \
  -p no:cacheprovider
```

结果：**45 passed，10.38 秒**。检查值不被重写、未知字段/不支持组合、disabled 依赖隔离、G 依赖提示、async 拒绝，以及实际 Placement 数据的 colocated 顺序和完整组约束。45 条 warnings 均来自既有基础 recipe 缺少 defaults `_self_`。输出为 `pytest-c05-config2.log`。

```bash
TMPDIR="$PWD/experiments/pi_rl/mani_v1_checks" PYTHONPATH="$PWD" \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/vepfs-C区/jiajunlu/.envs/pirl-openpi-30e0236/bin/python -u -m pytest \
  tests/unit_tests/test_fsdp.py -k 'manifold_actor or manifold_disabled' -q \
  --basetemp=experiments/pi_rl/mani_v1_checks/pytest-c05-actor-final \
  -p no:cacheprovider
```

结果：**8 passed，26.46 秒**。真实 actor 方法检查 shuffle 前评分、高维 hidden 释放、三类组、geometry_only 边界、缺少 rollout 字段的明确错误，以及有效全成功组不再被 0.9 上界删除。disabled 的 advantage、loss mask、原分母、PPO loss 与 RNG 状态和 `051b23348ead20c06c336bd743c590a9b6c4ba17` 的原 actor 逐项一致。输出为 `pytest-c05-actor-final.log`。

## 两张真实 A100 的更新边界检查

使用当前主机两张 NVIDIA A100-SXM4-80GB，运行原生 FSDP NO_SHARD 小型可训练 logprob 模型、AdamW、grad scaler、LR scheduler 和 actor 的真实训练循环：

```bash
TMPDIR="$PWD/experiments/pi_rl/mani_v1_checks" PYTHONPATH="$PWD" \
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/vepfs-C区/jiajunlu/.envs/pirl-openpi-30e0236/bin/python -u -m pytest \
  tests/unit_tests/test_fsdp.py -k manifold_empty -q \
  --basetemp=experiments/pi_rl/mani_v1_checks/pytest-c05-fsdp2 \
  -p no:cacheprovider
```

结果：**1 passed，16.04 秒**，输出为 `pytest-c05-fsdp2.log`。其中目录名 fsdp2 是第二次检查的运行编号，实际 backend 为 FSDP NO_SHARD。

第一次 iteration 仅 rank 1 的一个 global batch 有信号，rank 0 的 loss mask 全空：两 rank 正常同步并各完成一次更新，同时跳过一个全局空边界。随后两 rank 全空；逐位比较参数、Adam state、scheduler state、grad scaler state、optimizer_steps 和 forward 次数，全部保持不变。切换 disabled 后，零优势仍执行原生 optimizer/scheduler 行为，已有 Adam 动量仍推动参数。

本项使用小模型验证同步与更新契约，不代表 π0.5 的完整优化训练已通过。

## 真实 π0.5 rollout worker 检查

同一主机 GPU 0 加载现有 `/vepfs-C区/jiajunlu/ckpt/RLinf-Pi05-LIBERO-SFT`，使用 bf16、flow_sde、4 denoise、C=10，关闭 torch_compile，通过真实 `MultiStepRolloutWorker.predict()` 与 `_build_policy_output()`：

```bash
TMPDIR="$PWD/experiments/pi_rl/mani_v1_checks" \
TORCHINDUCTOR_CACHE_DIR="$PWD/experiments/pi_rl/mani_v1_checks/c02_compile/inductor" \
TRITON_CACHE_DIR="$PWD/experiments/pi_rl/mani_v1_checks/c02_compile/triton" \
OPENPI_DATA_HOME="$PWD/experiments/pi_rl/handoff-verification-20261031-v2/runtime/model_assets/.cache/openpi" \
RLINF_PI05_TEST_CHECKPOINT=/vepfs-C区/jiajunlu/ckpt/RLinf-Pi05-LIBERO-SFT \
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
/vepfs-C区/jiajunlu/.envs/pirl-openpi-30e0236/bin/python -u -m pytest \
  tests/unit_tests/test_models.py -k manifold_worker -q \
  --basetemp=experiments/pi_rl/mani_v1_checks/pytest-c05-worker2 \
  -p no:cacheprovider
```

结果：**1 passed，53.08 秒**，输出为 `pytest-c05-worker2.log`。两个 enabled mode 的 train 输出经 PolicyOutput 保留 detached CPU FP32 `[1,10,1024]` hidden 和正确行为版本；disabled 的 train 与三个 mode 的 eval 都不携带 hidden。未启用资产路径的普通测试会跳过该 GPU 用例。

初轮局部 Worker fixture 未初始化计时器字段，actor/FSDP 与真实 worker 检查因此在进入相关方法时失败；补齐真实 Worker 所需的测试状态后，上述最终检查均通过。失败输出保留在 `pytest-c05-actor.log`、`pytest-c05-fsdp.log` 和 `pytest-c05-worker.log`。

本批相关 Python 文件的 Ruff lint/format 通过，`git diff --check` 通过。C06 组级诊断/审计、C07 源码快照、C08 recipe 与 C09 完整生命周期仍待完成；没有启动正式训练或提交云任务。
