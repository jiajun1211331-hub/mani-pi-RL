# C08：模式/评分 recipe 与采样预算验收

C08 已提供共享正式 recipe 与独立两 GPU smoke recipe。用法见 [RECIPES.md](RECIPES.md)，完整 resolved 模板见 [resolved_configs/](resolved_configs/)，逐字段差异与文件哈希见 [C08_CONFIG_DIFF.json](C08_CONFIG_DIFF.json)。尚未创建提交；本记录只证明配置可 compose、口径匹配和静态校验通过，GPU 生命周期仍属 C09。

```bash
TMPDIR="$PWD/experiments/pi_rl/mani_v1_checks" \
PYTHONPATH="$PWD/experiments/pi_rl/pi05_sft_libero10_tasks129_mani_ode480_20261102T073519Z/task2/runtime/geometry_packages:$PWD" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/vepfs-C区/jiajunlu/.envs/pirl-openpi-30e0236/bin/python -m pytest \
  tests/unit_tests/test_cluster_config.py -k 'manifold or volume_diagnostic' -q \
  --basetemp=experiments/pi_rl/mani_v1_checks/pytest-c08-recipes2 \
  -p no:cacheprovider
```

结果：**66 passed，22.48 秒**，日志为 `experiments/pi_rl/mani_v1_checks/pytest-c08-recipes2.log`。包括既有 52 项 manifold 配置检查和新增 14 项 recipe 检查。五种 mode/score × 两种 eligible_groups 均实际 compose，逐字段排除新增方法/诊断/实验名后与原生训练配置完全相等。smoke 另外核对每 rank 八人组、chunk 总样本数、global/micro/world-size 整除，以及独立的一轮 train/val/save 预算。不兼容覆盖在静态校验阶段报错。

62 条 warnings 来自原生基础 recipe 缺少 defaults `_self_`；新的正式和 smoke recipe 都显式声明 `_self_`。初版尝试直接继承原生主配置时触发 Hydra 的 `hydra.searchpath` 限制，已改为一份由所有方法共用的静态 recipe；初轮失败日志保留在 `pytest-c08-recipes.log`。没有修改原生 recipe。

使用无 TLE 的系统 Python 执行 `test_cluster_config.py -k 'manifold_recipe or manifold_smoke'`：**10 passed、4 skipped，6.79 秒**，日志为 `pytest-c08-no-tle.log`。四个启用 G 的组合因缺少真实 TLE 跳过，disabled 和两种 V 模式仍通过。Hydra 兼容模式提示属于测试环境现有版本。

此外在原生 πRL Python 和固定 scikit-dimension 0.3.7 依赖下独立 compose 并校验五份正式、五份 smoke YAML，保存 resolved 内容与 SHA-256。正式变体全部保持 64×8、group_size=8、C=10、480 action、4 denoise、update_epoch=4、原生 optimizer 与 batch 参数；V 混合明确对应 `A_GRPO + 0.2 × A_V`。smoke 与正式的每个覆盖差异单独列出。

Ruff lint/format 与 `git diff --check` 通过；新 recipe 已加入 embodied CI 路径过滤。研究 Markdown/YAML 不属于 Sphinx EN/ZH 页面，本批核对文件路径、字段、公式与 compose 产物，没有改动公开 EN/ZH 文档。
