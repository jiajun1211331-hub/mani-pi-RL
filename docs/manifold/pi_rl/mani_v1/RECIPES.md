# mani_v1 正式配置与 smoke 配置

正式配置使用 `libero_10_grpo_openpi_pi05_mani_v1`，在同一份静态 recipe 上选择 mode、score 和组类型。它沿用原生 LIBERO-10 π0.5 的模型、采样、chunk PPO 与优化参数；先完成独立 smoke，再用干净提交准备正式源码快照。

## 模式与评分

通过 C07 的准备入口传入下表的两个覆盖项，不复制训练配置：

| 对照 | `algorithm.manifold.mode` | `algorithm.manifold.geometry.score` | 优势定义 |
| --- | --- | --- | --- |
| 原生基线 | disabled | volume_exp_neg_dim | 原生 GRPO，不采集 hidden 或加载 TLE |
| G 混合（默认） | task_plus_geometry | volume_exp_neg_dim | A_GRPO + 0.2 × M × z(G) |
| G-only | geometry_only | volume_exp_neg_dim | 0.2 × M × z(G) |
| V 混合 | task_plus_geometry | volume | A_GRPO + 0.2 × A_V，其中 A_V=M×z(V) |
| V-only | geometry_only | volume | 0.2 × M × z(V) |

M 只覆盖完整落在精确 K 内的 chunk，z 使用完整八人组的 sample std 与 epsilon=1e-6。V 是原始 FP64 centered-Gram volume；V 混合中的 0.2 乘组内标准化并带 mask 的 A_V。V-only 支持配置组合，但完整生命周期验证的范围以 C09 报告为准。

`algorithm.manifold.eligible_groups=mixed_only|any_success` 独立于 mode/score，默认 any_success，允许有效且非常数的全成功组。geometry_only 仍由成功标签决定 K 和组资格。disabled 完整保留旧筛选路径；enabled 不再叠加旧 0.9 上界。

G 需要目标环境安装 `requirements/manifold.txt`。V 不要求 TLE；默认不计算额外 D/G/R，设置 `algorithm.manifold.diagnostics.dimension_metrics=true` 可尝试独立诊断。缺失或无效诊断不会屏蔽有效 V。

## 采样和优化预算

正式 recipe 保持 64 个并行环境 × 8 个 rollout epoch，即每轮 512 条原始轨迹、64 个八人组，不补采被筛掉的组。保留 480 action horizon、C=10、4 denoise、flow_sde、action-expert-only、update_epoch=4、micro_batch_size=128、global_batch_size=2048，以及原生 optimizer/loss 参数。

placement 为同步 colocated、pipeline_stage_num=1；每个 scoring rank 必须保留完整八人组。64 env 对八个 scoring rank 分配时每 rank 正好一组；其他资源布局也必须满足该约束及原生训练 batch 整除条件。正式配置不绑定云队列或具体 GPU 型号。

新增运行字段仅为 manifold 配置、组成功人数诊断开关与实验名。native 配置和新 recipe 使用同样的 env/model/FSDP/weight-sync defaults；Hydra 不允许继承含 `hydra.searchpath` 的另一个主配置，因此 mani_v1 自身持有一份共享静态 recipe。逐字段回归测试约束它与原生 recipe 的训练数值一致，不复制五套优化参数。

两张 GPU 的独立 `libero_10_grpo_openpi_pi05_mani_v1_smoke` 保持 horizon/C/denoise/八人组，使用 16 env × 1 epoch、update_epoch=1、micro_batch_size=2、global_batch_size=32、一轮训练和一次 val/checkpoint，eval 为两个环境。smoke 关闭 train/eval 视频与 torch compile，避免有限检查的大量编译和视频开销。这些覆盖不进入正式默认值，也不构成训练曲线对照。

## 准备示例与已解析配置

在所选 Python 环境中准备 V 混合 smoke：

```bash
python -m scripts.pi_rl.mani_v1.prepare_training \
  --source-repo "$PWD" \
  --run "$PWD/experiments/<unique-v-mixed-smoke>" \
  --model /absolute/path/to/RLinf-Pi05-LIBERO-SFT \
  --config-name libero_10_grpo_openpi_pi05_mani_v1_smoke \
  --override algorithm.manifold.mode=task_plus_geometry \
  --override algorithm.manifold.geometry.score=volume
```

开发态需显式加 `--allow-dirty`。改用正式 recipe 名即可准备正式预算，实际启动和资源环境见 [SOURCE_SNAPSHOT.md](SOURCE_SNAPSHOT.md)。模型路径、日志目录和实验名由准备参数确定，不能再用数值覆盖项重复指定。

[resolved_configs/](resolved_configs/) 保存五种组合各自的正式与 smoke YAML。它们是 compose/resolve 后的可核对模板，模型仍是 `/path/to/model/...` 占位路径；真实运行必须用准备入口生成包含实际模型与独立日志目录的配置。[C08_CONFIG_DIFF.json](C08_CONFIG_DIFF.json) 保存逐字段差异和每份文件的 SHA-256，便于区分方法配置与 smoke 预算。错误枚举、未知字段或 auto_reset 等不支持组合会在启动训练前被拒绝。
