# mani_v1 生命周期验证记录

当前 C09 正在执行，尚未完成整体验收，也尚未创建任何 C00–C09 提交。C01–C08 的组件证据见各批次 VALIDATION 文档；本页只记录实际本地 train/val/checkpoint/infer 的进展，不把静态配置或小模型测试当作真实 π0.5 生命周期。

## 已执行与待验证范围

| 组合 | 真实 train/val/save | strict 重载与独立 infer | 几何信号 |
| --- | --- | --- | --- |
| disabled | 通过：每 rank 24 次更新，梯度/loss 有限，step 1 保存 | 通过：477 个冻结张量不变，190 个可训练张量变化，GPU infer 有限且无 hidden | 不适用，train/eval 均不采集 hidden |
| task_plus_geometry / G | 通过：每 rank 24 次有限更新，val 与 step 1 保存完成 | 正在验证 | 两组有效 mixed，K=370/208，几何 L2≈3.995 |
| geometry_only / G | 已准备，尚未运行 | 尚未运行 | 尚未验证 |
| task_plus_geometry / V | 已准备，尚未运行 | 尚未运行 | 尚未验证 |
| geometry_only / V | C08 静态组合已通过 | 不在本轮生命周期覆盖范围 | 不作生命周期结论 |

disabled 的原生 advantage/mask/loss/RNG 与 HEAD 基线一致性由 [C05_VALIDATION.md](C05_VALIDATION.md) 的对照证明；当前真实运行补充原生训练、val、保存与重载证据，不宣称训练曲线等价。

## 硬件与版本

本地主机两张 NVIDIA A100-SXM4-80GB，Python 3.11.14、torch 2.7.1+cu126、numpy 1.26.4、scikit-dimension 0.3.7、scikit-learn 1.6.1。隔离验证环境为 `experiments/pi_rl/mani_v1_checks/c09-runtime/venv`，原 πRL 环境没有修改。每个 run 的 `manifests/environment.json` 保存完整版本与解释器身份。

初始模型为 `/vepfs-C区/jiajunlu/ckpt/RLinf-Pi05-LIBERO-SFT`，权重身份见 [C07_VALIDATION.md](C07_VALIDATION.md)。actor seed=42，env seed=0；所有成功/失败标签来自真实 LIBERO rollout。使用 [RECIPES.md](RECIPES.md) 的独立两 GPU smoke，不改变正式 64×8 配置。

当前实际源码 HEAD 为 `051b23348ead20c06c336bd743c590a9b6c4ba17`，开发态快照明确包含未提交迁移代码。disabled-02 与三份已准备方法组合的实际 tree SHA-256 均为：

```text
d67ca1ba0107112486025ebded2e25695e7b0115f918500e2b44fa577e541ef6
```

这不是新提交 SHA。真人作者身份仍待用户提供；按批次独立保存的源码副本与 SHA-256 清单位于 `experiments/pi_rl/mani_v1_checks/pending_commits/`。

## disabled 的真实运行

运行目录：`experiments/pi_rl/mani_v1_checks/c09-disabled-02`。准备使用 C07 入口、smoke recipe、mode=disabled、现有 checkpoint 与 `--allow-dirty`。测试入口和环境参数见 [RUNBOOK.md](RUNBOOK.md)；实际 harness hash 保存在 `runtime/test_harness.json`。

`runtime/train.log` 记录 native train、val 和 step 1 checkpoint，进程返回 0。两个 actor 各完成 24 次更新，所有训练指标和梯度有限，首次 grad norm 为 14.219902038574219。真实 rollout 包含一个全失败组与一个 2/8 成功 mixed 组；native 训练汇总 SR=0.1875。val 的两个环境产生原生 success 指标，SR=0.0；小样本 SR 不用于评估方法效果。

driver、两个 actor、两个 rollout、两个 env 的实际导入来源报告已保存。`runtime/observer/` 保存 contemporaneous 事件与实际 train/eval hidden 开关；训练 minibatch 的观测断言通过。完整权重文件约 6.5 GiB，位于本 run 日志下的 `checkpoints/global_step_1/actor/model_state_dict/full_weights.pt`。

首个 disabled-01 在 Ray 启动前失败：Hydra 未将 JSON 的 Unicode escape 转回中文路径。启动参数已改为保留 literal Unicode，并添加解析测试。该项及全部启动器组件回归为 **35 passed，86.30 秒**，日志为 `pytest-c09-launcher.log`。

disabled-02 的重载脚本初轮发现通用 model builder 默认将模型放在 GPU，无法直接与 CPU mmap checkpoint 比较；核对阶段已显式将模型移至 CPU，保留失败日志 `runtime/verify.log`，重试日志为 `runtime/verify2.log`。训练本身不受该核对脚本问题影响。

最终重载与独立推理通过，`runtime/lifecycle.json` 记录 passed=true。477 个冻结张量逐项相等，190 个可训练张量实际变化，strict reload 没有 missing/unexpected key。DCP 包含 optimizer/scheduler 元数据且不含 manifold collection 字段；optimizer resume 未测试。重载后使用保存的真实环境 observation，在独立进程/GPU 执行 eval，输出形状 [8,10,7]、所有动作有限、无 manifold_hidden。核对耗时 83.37 秒，checkpoint SHA-256 为 `d4d86f2a6f5eb58847893b6b181610213e385876028cedca211b40e63d351e60`。

## G 混合的真实运行

运行目录为 `experiments/pi_rl/mani_v1_checks/c09-g-mixed-01`，与 disabled-02 使用相同源码 tree hash。训练进程自然退出，`runtime/training_exit.json` 记录 returncode=0。两个 actor 各完成 24 次实际更新、0 次跳过，训练指标与梯度有限，原生 val 和 step 1 checkpoint 保存完成。

当次实际 hidden 对两组均产生有效 D/V/G，无 invalid 组；两组为 mixed，K 分别为 370 和 208，组 G 均值/std 分别约为 36.272312/2.281670 与 40.045219/1.546754。第一组首成员 D=19.786557、V=277.873164、G=38.417344，第二组首成员 D=18.376725、V=262.224836、G=41.742918；原始精度值及独立状态保存在 contemporaneous 标量审计中。

几何 advantage 的全局 L2=3.9949949736623407，评分最慢 rank 耗时 1.5783165879547596 秒，实际 CPU hidden 张量合计 31,457,280 字节（30 MiB）。后者是实际张量字节数，不是 CPU 峰值内存。strict 重载、标量重放及独立 infer 验收正在运行，日志为该目录的 `runtime/verify.log`。

## 尚未完成的验收

必须完成所列 G 混合、G-only、V 混合的真实运行，并确认非零几何信号、实际参数变化及各自完整权重 strict 重载与独立 infer。CPU/GPU 附加内存和评分耗时正在采集，当前不把 hidden 理论大小写成峰值实测。没有启动正式训练或云任务，没有停止已有实验。
