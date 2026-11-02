# C06：诊断聚合与当次 rollout 审计验收

C06 已实现组、评分、优势和成本诊断，以及默认不保存 hidden 的标量审计。字段、配置和重放方法见 [DIAGNOSTICS.md](DIAGNOSTICS.md)。本批实现与验证保存在工作区，Git 提交仍等待真实作者身份。没有启动云任务或正式训练。

## 组件与无 TLE 检查

命令在仓库根执行，使用现有 πRL 环境与已保存的 scikit-dimension 0.3.7 / scikit-learn 1.6.1：

```bash
TMPDIR="$PWD/experiments/pi_rl/mani_v1_checks" \
PYTHONPATH="$PWD/experiments/pi_rl/pi05_sft_libero10_tasks129_mani_ode480_20261102T073519Z/task2/runtime/geometry_packages:$PWD" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/vepfs-C区/jiajunlu/.envs/pirl-openpi-30e0236/bin/python -m pytest \
  tests/unit_tests/test_manifold.py tests/unit_tests/test_cluster_config.py \
  -k 'manifold or volume_diagnostic' -q \
  --basetemp=experiments/pi_rl/mani_v1_checks/pytest-c06-final \
  -p no:cacheprovider
```

结果：**196 passed，36.60 秒**，记录在 `experiments/pi_rl/mani_v1_checks/pytest-c06-final.log`。52 条 warnings 均来自原有基础配置缺少 defaults `_self_`。覆盖 C01/C04/C05 既有数学与配置契约，并新增两种 mode × 两种 score × 六类组状态的标量重放、K/资格/std/z 篡改拒绝、乱序成员重放、默认审计位置，以及跨批次和重启的 hidden 保存数量上限。

两 CPU rank 使用不同本地样本量和不同失效原因，验证全局 23 个有效、1 个无效成员的真实加权均值、范数、耗时和字节数聚合。无效值不进入有效均值；count=0 时没有 mean。此测试同时验证动态原因键在 rank 间的并集与组计数守恒。

系统 Python 不安装 TLE，执行整个 `test_manifold.py`：**101 passed、43 skipped，8.62 秒**。跳过项依赖真实 TLE，V 评分、标量重放与分布式聚合正常；日志为 `pytest-c06-no-tle.log`。

原生 πRL Python 执行 `test_cluster_config.py test_fsdp.py -k 'volume_diagnostic or manifold_actor or manifold_disabled'`：**9 passed，54.87 秒**，日志为 `pytest-c06-integration-final.log`。检查 V 请求额外 D 诊断时依然不要求 TLE、真实 actor 的 hidden 释放和审计、实际组选择与 legacy 成功人数日志一致，以及 disabled 原路径等价。唯一 warning 来自原有基础配置缺少 defaults `_self_`。

## 双 A100 更新与审计关联

```bash
TMPDIR="$PWD/experiments/pi_rl/mani_v1_checks" PYTHONPATH="$PWD" \
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/vepfs-C区/jiajunlu/.envs/pirl-openpi-30e0236/bin/python -u -m pytest \
  tests/unit_tests/test_fsdp.py -k manifold_empty -q \
  --basetemp=experiments/pi_rl/mani_v1_checks/pytest-c06-gpu-final \
  -p no:cacheprovider
```

结果：**1 passed，17.84 秒**，日志为 `pytest-c06-gpu-final.log`。硬件为两张 NVIDIA A100-SXM4-80GB，运行真实 FSDP NO_SHARD 小型策略、AdamW、scaler、scheduler 和 actor 训练循环。

除 C05 的局部空/全局空边界外，本次增加真实 `compute_advantages_and_returns()` → NCCL 诊断归约 → 分 rank 审计 → train 的检查。两个原始组均有有效 V 信号，各 rank 完成六次更新，审计可逐项重放全部 mask 和 advantage；追加记录与实际累计 optimizer_steps=9 一致。

随后独立 all_failed 批次生成新审计文件，诊断为两个全失败组、selected_score/count=0；各 rank 六个边界全部跳过，参数、Adam 和 scheduler 状态逐项不变，累计 step 仍为 9。审计追加 updates=0、skips=6、scheduler_advanced=false，且关联的是当前批次。

本测试使用合成 hidden 与小型策略验证诊断、同步和更新契约，不代表真实 π0.5 LIBERO 完整生命周期已验收。C02/C05 的真实 π0.5 特征与采样证据继续有效；完整 train/val/checkpoint/infer 与附加峰值内存实测仍由 C09 完成。

相关 Python 文件的 Ruff lint 与 format 检查通过，`git diff --check` 通过。
