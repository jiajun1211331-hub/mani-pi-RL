# C07：实际迁移源码快照验收

C07 已实现独立准备/启动路径、开发态完整运行源码快照、配置/模型/环境指纹，以及 driver 与 worker 的加载来源检查。使用方法和范围见 [SOURCE_SNAPSHOT.md](SOURCE_SNAPSHOT.md)。代码与证据保存在工作区，提交仍等待真实作者身份。没有启动训练或云任务。

## 组件验证

```bash
TMPDIR="$PWD/experiments/pi_rl/mani_v1_checks" PYTHONPATH="$PWD" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/vepfs-C区/jiajunlu/.envs/pirl-openpi-30e0236/bin/python -m pytest \
  tests/test_pi_rl_training.py -q \
  --basetemp=experiments/pi_rl/mani_v1_checks/pytest-c07-final2 \
  -p no:cacheprovider
```

结果：**34 passed，68.07 秒**，日志为 `experiments/pi_rl/mani_v1_checks/pytest-c07-final2.log`。包括原官方启动器既有契约，以及新路径的干净/开发态源文件、tracked 修改、新增文件、删除文件、外部 symlink 拒绝、独立输出路径和冲突覆盖拒绝；真实快照的内容修改、新增、缺失、symlink 和配置篡改均被发现。子进程验证冻结模块的实际导入路径，并拒绝从原工作区导入；`--check-only` 没有创建训练 attempt。

测试在临时 Git fixture 中创建测试提交；没有替用户设置仓库身份或签署 DCO。模型身份用小型文件 fixture 检查完整哈希与外部权重变更拒绝，不把它当成可推理模型。

修改审计 provenance 后，使用保存的 scikit-dimension 0.3.7 依赖运行 `test_manifold.py -k 'scalar_audit or scalar_replay or hidden_dump or audit_default'`：**30 passed，13.20 秒**，日志为 `pytest-c07-audit.log`。C06 标量重放与默认/有上限保存契约继续通过。

首轮 launcher 检查发现 `get_logger()` 不接受 name 参数，已按仓库 API 修正。原始失败保存在 `pytest-c07-source.log` 和 `c07-initial-launch-error.log`；修正后的最终命令如上。相关 Ruff lint/format 和 `git diff --check` 通过。CPU CI 已接入该启动器组件测试，新增脚本和测试都有路径过滤覆盖。原官方两个入口的 diff 为空。

## 现有 π0.5 checkpoint 的离线检查

实际准备命令：

```bash
TMPDIR="$PWD/experiments/pi_rl/mani_v1_checks" PYTHONPATH="$PWD" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/vepfs-C区/jiajunlu/.envs/pirl-openpi-30e0236/bin/python \
  -m scripts.pi_rl.mani_v1.prepare_training \
  --source-repo "$PWD" \
  --run "$PWD/experiments/pi_rl/mani_v1_checks/c07-real-prepared" \
  --model /vepfs-C区/jiajunlu/ckpt/RLinf-Pi05-LIBERO-SFT \
  --config-name libero_10_grpo_openpi_pi05 --allow-dirty
```

本项使用既有 disabled recipe 核对准备功能；C08 才提供正式方法 recipe。准备通过，日志为 `c07-real-prepare.log`，记录 1464 个运行源码文件。原 HEAD 为 `051b23348ead20c06c336bd743c590a9b6c4ba17`，快照明确标记 development。实际 tree SHA-256：

```text
736400017be5a54dd117d9e4cd3a10deac588e610023710fafb1d38a8671959e
```

`model.safetensors` 为 7,473,091,464 字节，完整 SHA-256：

```text
4616e4c1966a5fb063b442ea5d69857f45e873222d839b794634d87d252ab9e7
```

从该 run 目录，以 `PYTHONPATH=<run>/source` 调用冻结 `launch_training.py --run <run> --check-only` 通过，日志为 `c07-real-launch-check.log`，记录 106 个已加载模块。

另一个子进程从同一快照实际导入 config、几何、actor、HF rollout、env worker、LIBERO、π0.5 tasks/rl 和 pi0 模块，记录 203 个 RLinf 模块。`c07-real-loaded-modules.json` 中上述关键模块全部位于本快照，hash 匹配；审计 provenance 的 snapshot tree hash 与 manifest 一致。导入日志为 `c07-real-imports.log`，其中 Gym 和 robosuite 提示属于现有依赖。该检查没有构造模型或环境，也没有执行 GPU forward。

本批未声称 worker 初始化或完整 train/val/checkpoint/infer 已通过。三类 worker 的运行时检查已接入初始化末尾，实际进程证据仍由 C09 收集；正式方法配置属于 C08。
