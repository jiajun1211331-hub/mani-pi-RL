# C03：特征、身份与逐 action 边界传递验收

本记录验证 manifold hidden 与显式 episode 身份在 rollout、环境和轨迹 schema 之间的传递。组件实现已保存在工作区，提交等待真实作者身份确认。在线 advantage、训练循环与完整 GPU 生命周期仍待 C05–C09 接入和验证。

## 数据契约

`PolicyOutput.manifold_hidden` 显式接收模型结果中的 FP32 `[B,C,1024]` 特征，进入 schema 时 detach 并移到 CPU。`EnvPart.manifold_context` 携带 task、trial、reset-state、group slot、member ID 和逐 action 的 done、success、valid。`TrajectoryStep.from_parts()` 将这些字段与同一 source 的 collector step、rollout epoch、env rank/stage、chunk ID 和行为 policy version 合并。特征和上下文分别堆叠为轨迹顶层字段，保留在 forward_inputs 之外。

LIBERO 的 `get_manifold_context()` 使用原始逐 action 终止信息，以明确的 LIBERO termination 语义记录成功。它要求训练环境关闭 auto_reset 和 ignore_terminations。reset-state 身份按当前 task/trial 重建，避免把未来 reset-state pool 当成当前 episode。首次终止后的整段有效性由 C04 在完整 episode 上处理；本层保留原始边界。

schema 会拒绝缺少对应环境上下文或 hidden、source 不匹配、非整数/不一致的行为版本，以及部分 step/shard 缺少可选字段。原数据完全缺少这些字段时保留旧路径。完整八人组与跨 chunk 身份一致性由 C04 检查。

## Schema 与兼容测试

使用现有 πRL Python 环境，在 CPU 上运行：

```bash
TMPDIR="$PWD/experiments/pi_rl/mani_v1_checks" PYTHONPATH="$PWD" \
/vepfs-C区/jiajunlu/.envs/pirl-openpi-30e0236/bin/python -u -m pytest \
  tests/unit_tests/test_data.py -q \
  --basetemp=experiments/pi_rl/mani_v1_checks/pytest-c03-full \
  -p no:cacheprovider
```

结果：142 passed、1 failed，69.88 秒。唯一失败为已有 `test_vlm_trend_batch_video_metadata_stays_nested_per_sample`：当前环境的 Transformers `VideoMetadata` 不接受 `frames_indices`。用 `git show HEAD:tests/unit_tests/test_data.py` 保存的原测试文件单独复跑，同样报错（1 failed、135 deselected，74.35 秒）；本批没有修改该测试及其 VLM 实现。两份日志分别为 `experiments/pi_rl/mani_v1_checks/pytest-c03-full.log` 和 `pytest-c03-baseline.log`。

最终代码再次运行定向检查：

```bash
TMPDIR="$PWD/experiments/pi_rl/mani_v1_checks" \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH="$PWD" \
/vepfs-C区/jiajunlu/.envs/pirl-openpi-30e0236/bin/python -u -m pytest \
  tests/unit_tests/test_data.py \
  -k 'manifold or trajectory_owns or trajectory_step_owns or policy_output or env_part' \
  -q --basetemp=experiments/pi_rl/mani_v1_checks/pytest-c03-confirm \
  -p no:cacheprovider
```

结果：**13 passed，130 deselected，54.52 秒**，输出为 `experiments/pi_rl/mani_v1_checks/pytest-c03-confirm.log`。包括直接构造 Trajectory 时的 CPU detach 边界检查。

新增的七个 manifold schema 测试均通过。随后强化了可辨认的每 action/feature 序列，并与 C04 联合定向执行：schema 七项仍全部通过。测试通过真实 PolicyOutput/EnvPart split、merge、complete、TrajectoryStep 构造、Trajectory stack/split/to_batch 和 epoch 重排，检查两个 epoch、两个八人组的 action 点顺序、来源身份、CPU detach 和旧数据兼容。还覆盖 rollout worker 的显式字段转发和 source/version/可选字段错误。

## 真实 LIBERO 检查

在 NVIDIA A100-SXM4-80GB 上使用 EGL，构造真实 LIBERO-10 的八个并行环境、group_size=8、C=10，执行两个 chunk。时间上限为 13 个 action，初始化 settling 为 15 步。

```bash
TMPDIR="$PWD/experiments/pi_rl/mani_v1_checks" \
LIBERO_CONFIG_PATH="$PWD/experiments/pi_rl/mani_v1_checks/libero-config" \
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl \
MUJOCO_EGL_DEVICE_ID=0 PYOPENGL_PLATFORM=egl LIBERO_TYPE=standard \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/vepfs-C区/jiajunlu/.envs/pirl-openpi-30e0236/bin/python -u \
  experiments/pi_rl/mani_v1_checks/check_env_context.py
```

结果：通过。八个成员的 task/trial/reset-state 均与实际 reset 一致，member ID 为 0–7；done 和 success 与实际逐 action 环境返回一致。第二个 chunk 每个成员恰好有三个有效 action，包含第 13 个时间截断 action，settling 不计入有效长度。实际数据见 [C03_ENV_CONTEXT.json](C03_ENV_CONTEXT.json)，执行输出保存在 `experiments/pi_rl/mani_v1_checks/c03_env_context_retry.log`。

首轮局部检查脚本缺少 multiprocessing main guard，启动时退出；补齐 guard 后以上真实检查通过。脚本按有界检查自然结束，并关闭其自身创建的环境。

本记录验证传递与环境元数据，未验证 actor 的 optimizer 更新、checkpoint 或完整 train/val/infer 生命周期。
