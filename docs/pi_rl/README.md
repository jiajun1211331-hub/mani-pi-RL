# π0.5 + GRPO 复现资料

从[中文接手文档](../PI05_GRPO_REPRODUCTION_HANDOFF.md)开始，在新服务器安装环境、接入 LIBERO、评测 SFT、训练 GRPO 并执行配对评测。英文版为 [Reproduction Handoff](REPRODUCTION_HANDOFF_EN.md)。

| 资料 | 用途 |
|---|---|
| [迁移清单](COPY_MANIFEST.json) | 源工作树文件、原始 SHA256、迁移后 SHA256 与适配说明 |
| [完整运行包重放](requirements-runtime-replay.txt) | 安装后按历史精确版本重放依赖，最后恢复 Transformers fork/替换文件 |
| [环境约束](environment-constraints.txt) | 固定 NVIDIA / Python 3.11 安装的关键版本 |
| [原始脚本](provenance/original_scripts) | 当前 SimpleVLA 工作树中 25 个脚本的原样副本 |
| [SFT 基线证据](provenance/baseline) | 两协议各 500 条原始记录、验收、环境和资产 hash |
| [200 步训练证据](provenance/200step) | 云端训练退出、8 个 actor 的更新计数、checkpoint 严格重载及推理验收 |
| [历史计划](../manifold/pi_rl/PI05_SPATIAL_GRPO_REPRODUCTION_PLAN_20261025.md) | 2026-10-25 至 29 日的设计、诊断与预算变更背景 |
| [本次迁移验证](VERIFICATION.md) | 当前代码与文档的实际检查范围和结果 |

`provenance` 中的绝对路径和任务状态属于记录产生时的历史证据。新服务器必须生成自己的 run manifest、GPU 探针、SFT 验收和训练生命周期报告；不要复制历史报告来通过新服务器启动门槛。
