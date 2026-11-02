# C02：真实采样 action expert 特征验收

本记录验证 π0.5 在原生随机训练采样中导出最后一次 denoise hidden 的行为。C02 实现与组件测试已保存在工作区，本地提交等待真实作者身份确认；actor 接入和完整训练生命周期仍待 C03–C09 完成。

## 输出契约

`Pi0RL.predict_action_batch(..., collect_manifold_hidden=True)` 在 `mode="train"` 时增加 `manifold_hidden`，形状为 `[B,C,1024]`、FP32、detach、独立存储。它截取该次实际采样最后一次 `run_suffix` 输出的前 C 个 action 位置，直接对应 `action_out_proj` 的输入。默认关闭，eval 保持原有输出；字段位于结果顶层。

新增测试放在已有组件文件 `tests/unit_tests/test_models.py`，通过环境变量 `RLINF_PI05_TEST_CHECKPOINT` 显式启用真实模型测试。未提供 checkpoint 或没有加速器时跳过，普通 CPU 测试不会尝试加载大模型。

## 硬件、命令和结果

硬件为当前主机 NVIDIA A100-SXM4-80GB，测试仅使用可见设备 0。模型为 `/vepfs-C区/jiajunlu/ckpt/RLinf-Pi05-LIBERO-SFT` 的现有 checkpoint；使用仓库原生 model/pi0_5 配置，精度 `bf16`，flow_sde、noise_level=0.5、4 denoise、C=10、action expert 训练范围均沿用配置。观测使用固定图像、state 与 prompt，以比较模型输出；本项没有启动模拟器 rollout。

```bash
TMPDIR="$PWD/experiments/pi_rl/mani_v1_checks" \
TORCHINDUCTOR_CACHE_DIR="$PWD/experiments/pi_rl/mani_v1_checks/c02_compile/inductor" \
TRITON_CACHE_DIR="$PWD/experiments/pi_rl/mani_v1_checks/c02_compile/triton" \
OPENPI_DATA_HOME="$PWD/experiments/pi_rl/handoff-verification-20261031-v2/runtime/model_assets/.cache/openpi" \
RLINF_PI05_TEST_CHECKPOINT=/vepfs-C区/jiajunlu/ckpt/RLinf-Pi05-LIBERO-SFT \
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
/vepfs-C区/jiajunlu/.envs/pirl-openpi-30e0236/bin/python -u -m pytest \
  tests/unit_tests/test_models.py -k pi05_manifold_features -q \
  --basetemp=experiments/pi_rl/mani_v1_checks/pytest-c02-retry \
  -p no:cacheprovider
```

结果：**2 passed，64 deselected，113.61 秒**。两个参数化分支分别关闭/开启原生 `torch_compile`。实际输出在 `experiments/pi_rl/mani_v1_checks/pytest-c02-retry.log`；测试正常结束后，设备 0 的占用恢复到测试前的 717 MiB。

每个分支先预热，再固定 Python/NumPy/Torch CPU/加速器全设备及独立 Generator 的随机状态，比较采集关闭和开启的结果：

- actions、model_actions、prev_values、prev_logprobs、chains、denoise_inds 逐位一致（`rtol=0, atol=0`），logprob 有限。
- Python/NumPy/Torch CPU/加速器全设备/独立 Generator 的结束随机状态一致。
- 投影前 hook 恰好观测到 4 次调用；导出的 `[1,10,1024]` hidden 与第 4 次投影输入前 10 点逐位一致。
- hidden 为 FP32、没有 autograd graph，位于 forward_inputs 之外。
- eval 即使接收到采集参数也保持原有输出契约。

首次执行在模型加载前因测试 fixture 使用了不支持的精度字符串 `bfloat16` 失败；修正为仓库接受的 `bf16` 后，上述两项均通过。此前失败日志保存在 `experiments/pi_rl/mani_v1_checks/pytest-c02.log`。

```bash
python -m ruff check --preview \
  rlinf/models/embodiment/openpi/tasks/rl.py tests/unit_tests/test_models.py
python -m ruff format --check \
  rlinf/models/embodiment/openpi/tasks/rl.py tests/unit_tests/test_models.py
```

结果：lint 与两个文件的格式检查通过。

本项证明真实 checkpoint 的训练采样与 eval 模型调用可运行，采集开关保持采样行为。它没有执行反向传播、optimizer 更新、真实环境 val、checkpoint 保存重载或重载后独立 infer；这些完整生命周期证据仍属于 C09。附加内存和耗时尚未做对照测量。
