# C01：共享几何核心验收

本记录说明 C01 共享 FP64 几何组件的实现范围、组件测试和与迁移前算法的数值对齐。当前代码已保存到工作区，C00 和 C01 的本地提交等待用户确认真实作者身份；C02–C09 与 GPU train/val/infer 生命周期尚未完成。

## 实现与兼容范围

`rlinf/algorithms/manifold/metrics.py` 接收未缩放的 `[N,d]` 特征，返回 D、V、G、D/V 的独立值、有效状态和原因，以及谱与 TLE 诊断。两条评分路径共用 `N>k` 门槛；V 使用中心化 Gram 谱，G 使用 `V exp(-alpha D)`。`compute_dimension=False` 直接计算 V；TLE 依赖缺失或局部维度无效时，V 保留自身有效状态。G 调用方通过 `require_tle()` 在启动阶段检查固定版本。

`scripts/pi_rl/manifold_geometry.py` 保留离线 `metrics()` 接口、1024 维检查和原报告字段，并调用共享计算核心。可选依赖固定在 `requirements/manifold.txt`，CI 路径过滤已覆盖离线适配脚本。此提交的组件还没有接入 actor 或 rollout。

## 组件测试

在当前主机 CPU 上执行以下命令，均使用 `tests/unit_tests/test_manifold.py`。系统 `/tmp` 已满，因此显式将临时文件放入实验目录。验证过程中没有修改已有环境依赖或停止任何实验。

不带 TLE 的系统 Python：

```bash
TMPDIR="$PWD/experiments/pi_rl/mani_v1_checks" \
python -m pytest tests/unit_tests/test_manifold.py -q \
  --basetemp=experiments/pi_rl/mani_v1_checks/pytest-no-tle-final \
  -p no:cacheprovider
```

结果：22 passed，2 skipped；跳过的两项直接依赖 TLE。实际输出保存在 `experiments/pi_rl/mani_v1_checks/pytest-no-tle-final.log`。

使用现有 πRL 环境与已有离线实验中保存的 scikit-dimension 0.3.7 / scikit-learn 1.6.1：

```bash
TMPDIR="$PWD/experiments/pi_rl/mani_v1_checks" \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH="$PWD:$PWD/experiments/pi_rl/pi05_sft_libero10_tasks129_mani_ode480_20261102T073519Z/task2/runtime/geometry_packages" \
/vepfs-C区/jiajunlu/.envs/pirl-openpi-30e0236/bin/python \
  -m pytest tests/unit_tests/test_manifold.py -q \
  --basetemp=experiments/pi_rl/mani_v1_checks/pytest-tle2 \
  -p no:cacheprovider
```

结果：24 passed，耗时 5.16 秒。覆盖已知谱的 V、平移与旋转不变性、FP64 转换、常数与重复点、短前缀、NaN/Inf、Gram 溢出、负谱容差、依赖缺失/错误版本，以及 V 有效而 TLE/D/G 无效的分支。

```bash
python -m ruff check --preview rlinf/algorithms/manifold \
  scripts/pi_rl/manifold_geometry.py tests/unit_tests/test_manifold.py
python -m ruff format --check rlinf/algorithms/manifold \
  scripts/pi_rl/manifold_geometry.py tests/unit_tests/test_manifold.py
```

结果：lint 通过；4 个 Python 文件格式通过。

## 保存特征的数值对齐

从实施前 HEAD `051b23348ead20c06c336bd743c590a9b6c4ba17` 保留原始离线算法，逐项对比共享核心的离线适配结果。原算法 SHA256 为 `c3a79ae07f75948d24c4d9834de834f694610e1721cf13ca9e92822c8aad3644`。

输入是已有 `pi05_sft_libero10_tasks129_mani_ode480_20261102T073519Z` 实验 task1、task2、task9 中各两条已保存轨迹的前 250 个 hidden 点，每条形状 `[250,1024]`。对齐还包含常数、10 点短前缀和 NaN 输入。对比使用 `rtol=1e-12, atol=1e-12`；六条真实特征的 D/V/G/D:V 最大绝对误差均为 **0**，退化输入的原报告状态和原因一致。

逐输入路径、文件 hash、形状、误差及退化原因见 [C01_METRICS_ALIGNMENT.json](C01_METRICS_ALIGNMENT.json)。本机复算脚本和完整输出分别保存在 `experiments/pi_rl/mani_v1_checks/compare_metrics.py` 与 `metrics_alignment.log`；这些是局部验证产物，未加入提交清单。

以上 hidden 来自既有离线 ODE 采集，只用于验证共享数学实现。在线随机采样特征、GPU 行为一致性与完整训练生命周期仍分别需要 C02 和 C09 的证据。本文件属于研究实施记录，位于 Sphinx 构建树之外；本批没有修改公开 EN/ZH 页面。
