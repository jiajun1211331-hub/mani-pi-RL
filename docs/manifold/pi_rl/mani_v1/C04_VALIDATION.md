# C04：完整组、精确前缀与 G/V 评分验收

本记录验证八人完整组校验、首次 episode 终止裁剪、成功前缀 K，以及独立 G/V 评分。实现已保存到工作区，提交等待真实作者身份确认。C04 输出组级标量证据、完整 chunk 前缀 mask 和归一化分数；actor 的 advantage 组合与 optimizer 行为仍待 C05。

## 分组与评分契约

`rlinf/algorithms/manifold/groups.py` 定义 `GroupKey`、`MemberScore`、`GroupScore`、`GroupBatch` 与 `score_groups()`。输入是 CPU FP32 `[chunks,batch,actions,1024]` 特征和 C03 的显式上下文。组 key 包含 policy version、collector step、rollout epoch、env rank/stage、group slot；每组必须具有 0–7 八个唯一 member ID，以及一致的 task/trial/reset-state。成员可以在 batch 中非相邻，评分输出保留原 column 对应关系。缺失成员、重复成员、错组、跨 chunk 身份漂移和形状错误均明确报错。

每个成员先按整个 episode 的首次 done 截断，包含该 action，并使用该终止 action 的明确 success 标签。后续执行或成功不会改变首次失败结论。K 是成功成员有效长度的最小值；存在短于 K 的失败成员时，整组数值评分无效，K 保持原定义。全失败组 K 未定义。

所有成员的前 K 点交由共享 FP64 核心评分。V 分支可以不加载 TLE，也可以单独尝试 D/G 诊断；诊断失败保留有效 V。组内只使用所选评分计算 sample std 与 `(score-mean)/(std+1e-6)`；单成员评分失败或所选评分常数均让整组几何信号为零。前缀 mask 与所选评分有效状态分开存储。

## 组件测试命令与结果

使用现有 πRL 环境和保存的 scikit-dimension 0.3.7：

```bash
TMPDIR="$PWD/experiments/pi_rl/mani_v1_checks" \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH="$PWD:$PWD/experiments/pi_rl/pi05_sft_libero10_tasks129_mani_ode480_20261102T073519Z/task2/runtime/geometry_packages" \
/vepfs-C区/jiajunlu/.envs/pirl-openpi-30e0236/bin/python -u -m pytest \
  tests/unit_tests/test_manifold.py -q \
  --basetemp=experiments/pi_rl/mani_v1_checks/pytest-c04-confirm \
  -p no:cacheprovider
```

结果：**56 passed，47.72 秒**，包括 C01 的共享核心测试。完整输出为 `experiments/pi_rl/mani_v1_checks/pytest-c04-confirm.log`。

不安装 TLE 的系统 Python：

```bash
TMPDIR="$PWD/experiments/pi_rl/mani_v1_checks" \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m pytest tests/unit_tests/test_manifold.py -q \
  --basetemp=experiments/pi_rl/mani_v1_checks/pytest-c04-no-tle \
  -p no:cacheprovider
```

结果：**50 passed，6 skipped，2.23 秒**。六项跳过均需要真实 TLE；V 分支及缺失 TLE 的诊断独立性检查通过。输出为 `experiments/pi_rl/mani_v1_checks/pytest-c04-no-tle.log`。

覆盖的验收分支包括：

- G/V 两种评分分别使用完整 173 点计算，前 17 个完整 chunk 的 mask 为真，第 18 个为假；成功后 NaN hidden 不进入评分。
- 非相邻成员重排、首次成功在 chunk 中间、首次失败后再次成功、失败成员短于 K、全成功、全失败，以及两种评分的 K≤k 门槛。
- 重复/缺失成员、跨版本/epoch 混组、task/trial/reset-state 不一致、跨 chunk 身份漂移、错误特征维度/精度、无效成功标签与不连续有效点。
- 单成员 NaN 使整组失效；常数 V 分支无几何信号；V 有效而 TLE/D/G 无效时 V 分支仍然有效。
- 通过保持 Gram 谱的行空间旋转构造近似相同 V、不同 D 的组，V 判为常数，G 保持有效且有变化，验证常数判断跟随所选评分。
- 所有 z 分数使用 correction=1 的组内 std 加 1e-6；改变 V 路径的 alpha 或缺失 D/G 诊断不改变 V 分数。

扩展 K=173 测试到 G 时，首轮参考计算仍关闭 D，造成一项测试期望失败；补齐参考计算的 G 分支后，上述最终 56 项全部通过。此前联合执行日志为 `experiments/pi_rl/mani_v1_checks/pytest-c04-final.log`，其中 C03 强化后的七个 schema 测试全部通过。

```bash
python -m ruff check --preview rlinf/algorithms/manifold \
  rlinf/data/schema/embodied_types.py rlinf/workers/env/env_worker.py \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  rlinf/envs/sim/libero/libero_env.py \
  tests/unit_tests/test_manifold.py tests/unit_tests/test_data.py
python -m ruff format --check rlinf/algorithms/manifold \
  rlinf/data/schema/embodied_types.py rlinf/workers/env/env_worker.py \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  rlinf/envs/sim/libero/libero_env.py \
  tests/unit_tests/test_manifold.py tests/unit_tests/test_data.py
```

结果：lint 和九个 Python 文件的格式检查通过。本层验证在 CPU 上执行；没有执行 PPO loss、分布式更新或完整 GPU 生命周期。
