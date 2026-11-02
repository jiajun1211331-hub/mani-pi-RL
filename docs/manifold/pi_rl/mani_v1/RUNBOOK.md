# mani_v1 准备、有限验收与正式运行

本手册从可安装的 π0.5 LIBERO 环境和本地 SFT checkpoint 出发，先冻结源码并完成有限 GPU 生命周期检查，再准备正式 64×8 实验。配置含义见 [RECIPES.md](RECIPES.md)，源码与模型完整性约束见 [SOURCE_SNAPSHOT.md](SOURCE_SNAPSHOT.md)。实际通过范围以 [VALIDATION_REPORT.md](VALIDATION_REPORT.md) 为准。

## 环境与源码

使用支持当前 OpenPI π0.5 和标准 LIBERO 的目标环境。G 评分还需要安装固定 TLE 依赖：

```bash
python -m pip install -r requirements/manifold.txt
```

V 评分与 disabled 不依赖 TLE。不要为了有限检查修改正在被其他实验使用的环境；本次本地验证使用单独目录，依赖来源记录在实验的 `c09-runtime/dependency_sources.json`，每个 run 另存完整 package 版本。

检查 Git 状态后按 C00–C09 分批提交；使用真人作者和 Signed-off-by。正式准备默认要求干净提交，开发验证才显式使用 `--allow-dirty`。准备只生成文件，不申请云资源，不改变已有官方复现目录。

## 准备有限检查

在仓库根目录、所选 Python 环境中，为每个组合使用新的目录：

```bash
python -m scripts.pi_rl.mani_v1.prepare_training \
  --source-repo "$PWD" \
  --run "$PWD/experiments/<unique-smoke-run>" \
  --model /absolute/path/to/RLinf-Pi05-LIBERO-SFT \
  --config-name libero_10_grpo_openpi_pi05_mani_v1_smoke \
  --override algorithm.manifold.mode=task_plus_geometry \
  --override algorithm.manifold.geometry.score=volume_exp_neg_dim
```

依次检查 disabled、G 混合、G-only 和 V 混合。该 smoke 为两 GPU colocated、16 env×1 epoch、完整八人组、480 action、C=10、4 denoise，执行一轮训练、val 和 checkpoint。它不是正式预算，也不保证每批都有有效几何信号；没有非零几何信号时只能验收空批行为，需要继续有限真实 rollout 检查。

准备成功后核对 `manifests/source.json` 和 `configs/train.yaml`。模型、日志路径与实验名由 `--model`/`--run` 统一指定；数值覆盖或源码变化都应创建新 run，不修改已经准备的文件。

## 运行与验收

运行环境选择两张可用 GPU，设置标准 LIBERO 资产与 OpenPI tokenizer/cache 路径、EGL 后端和可写临时目录。Ray 的 socket 路径要足够短；本次使用 `/dev/shm/` 下独立目录，并将普通临时文件链接到 run 的 Vepfs 目录。不要使用 machine-wide `ray stop` 或清理其他运行。

从 run 目录先调用冻结启动器检查完整性：

```bash
PYTHONPATH="$PWD/source" python \
  source/scripts/pi_rl/mani_v1/launch_training.py \
  --run "$PWD" --check-only
```

有限验收通过测试入口添加观测子类。它仍调用 native runner、actor optimizer 和 rollout，只额外记录标量、内存和一个真实环境 observation：

```bash
PYTHONPATH="$PWD/source" python \
  /absolute/path/to/repo/tests/e2e_tests/embodied/run_mani_v1_lifecycle.py \
  --run "$PWD" > runtime/train.log 2>&1
```

记录该进程实际退出码；新版测试入口也写入 `runtime/training_exit.json`。native 训练完成且释放本次运行资源后，使用同一环境和一张可用 GPU 做独立重载验收：

```bash
PYTHONPATH="$PWD/source" python \
  /absolute/path/to/repo/tests/e2e_tests/embodied/verify_mani_v1_lifecycle.py \
  --run "$PWD" --training-returncode <actual-exit-code> \
  > runtime/verify.log 2>&1
```

验收脚本要求非零有限梯度、真实更新、原生 val/SR、完整 checkpoint、冻结参数不变及可训练参数实际变化、strict 全权重重载、独立 GPU infer 无 hidden。enabled 还要求真实 rollout 存在非零几何信号，并能用当次标量审计重放。只保存 checkpoint 或成功退出并不足以通过。

通过后 `runtime/lifecycle.json` 才会写入 passed=true。`runtime/observer/` 保存每个 actor/rollout 的 contemporaneous 事件；`runtime/source_reports/` 保存 driver/actor/rollout/env 的实际加载路径和 hash；`runtime/test_harness.json` 与 lifecycle 报告记录实际测试脚本 hash。训练 minibatch 和 DCP checkpoint 均检查不包含 manifold hidden/context。checkpoint 中 optimizer/scheduler 元数据会检查，但不把它扩称为已经验证 optimizer resume。

逐组比较 disabled 与 enabled 的实测 CPU RSS、高水位和设备分配/保留峰值，并查看评分耗时与实际 hidden 字节数。`hidden_bytes` 是张量大小，不能代替峰值内存测量；跨运行的峰值差异也包含 allocator 和轨迹差异，应保留原始观测范围。

## 正式预算

完成目标组合的有限验收并提交后，使用同样的准备命令，将配置名换为 `libero_10_grpo_openpi_pi05_mani_v1`，不携带 smoke 的环境数、epoch 或 batch 覆盖。默认是 G 混合/any_success；V 混合只改变 score 为 volume，仍是 `A_GRPO + 0.2 × A_V`。

正式启动使用冻结的 `launch_training.py`，去掉 `--check-only`。GPU 资源与云队列由外部选择，prepare/launch 不提交云任务。首版仅支持单节点、共享快照目录、同步 colocated 及每 rank 完整八人组；不支持跨 scoring rank 拼组。

检查独立 manifold 命名空间中的 raw/eligible/geometry/signal 组数、K、无效原因、优势范数与实际 optimizer 更新/跳过数。性能结论需要正式预算的训练曲线；有限 smoke 的 SR 只证明原生指标能正常产生。
