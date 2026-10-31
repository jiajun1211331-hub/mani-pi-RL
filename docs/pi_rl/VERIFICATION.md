# π0.5 + GRPO 迁移验证

本文记录 2026-10-31 在目标仓库执行的检查，区分迁移验证、历史结果与新服务器仍需执行的完整流程。主入口为[新服务器接手文档](../PI05_GRPO_REPRODUCTION_HANDOFF.md)。

## 复制与版本

已核对源工作树的 25 个 Python 脚本、4 个测试文件及相关文档/提交配套文件，共 33 个输入文件。25 个原始脚本在 `provenance/original_scripts/` 逐字节保留，当前适配及源/目标 SHA256 见 [COPY_MANIFEST.json](COPY_MANIFEST.json)。适配限定为显式资产路径、tokenizer 缓存、Volc 资源字段、初始化和实际末步预算，官方 `rlinf/`、`examples/`、`evaluations/`、`requirements/`、`pyproject.toml` 与固定基线一致。

封装仓库可独立提交。数值源码由干净检出 `/vepfs-C区/jiajunlu/rlinf-pirl-official-source-30e0236` 提供，HEAD 为 `30e0236981c6a1dc19202de64116834c12673828`。

## CPU、路径和文档

以下检查已完成：

- `python -m pytest tests/test_pi_rl_*.py -q`：57 passed；包括预算门槛、源码冻结、协议覆盖、配对和 200/1000 步末 checkpoint 选择。
- 将叠加包解压到另一份全新 `30e0236` 检出，使用包内脚本实际完成新 run 初始化、配置审计和源码冻结；没有依赖 SimpleVLA 仓库中的可执行代码。
- 真实执行 `initialize_run.py`，使用独立官方检出、不同的资产根目录及测试用 Volc 资源 JSON；配置审计和完整源码冻结通过，没有提交任务。
- `prepare_run.py link-assets`：585 个资产文件已记录 hash，20/20 Spatial BDDL/init 与安装包一致。
- `prepare_run.py verify-sft`：4 个文件大小/hash、32 维 norm stats 检查通过。
- 26 个当前 Python 文件通过语法解析；中英文 22 个代码块完全相同，shell 与内嵌 Python 语法通过。
- `docs-check/build_docs.py`：EN、ZH 各 0 warnings。首次因长 TMPDIR 触发 Unix socket 路径限制，改用独立短 `/dev/shm/pirl-docs-*` 后通过；失败日志保留。
- `check_rst_markup.py`：No inline-markup problems found。
- `check_doc_symbols.py`：上游现有 Sphinx 页面有 26 个待识别名称；本次没有修改这些 EN/ZH 页面。新增 Markdown 的脚本路径、CLI 参数、模型/环境配置和相对链接另行核对。

环境重放的离线 dry-run 揭示了仓库默认 Torch/torchvision overrides 冲突；文档已要求官方安装器在干净源码目录运行，包重放使用 `uv --no-config pip install --no-deps`。禁用项目配置后的离线检查因 FlashAttention wheel 未缓存而无法完成解析；未在当前共享训练 venv 执行重装，也未将这个检查记为完整新机安装成功。

## 本次 GPU 生命周期

独立验证运行目录：`/vepfs-C区/jiajunlu/rlinf-pirl-official-30e0236/experiments/pi_rl/handoff-verification-20261031-v2`。硬件为双 NVIDIA A100-SXM4-80GB，使用既有隔离 Python 3.11.14 / Torch 2.7.1+cu126 环境和同一已验证 SFT/资产。

实际执行入口：

```bash
export CHECK_RUN=/vepfs-C区/jiajunlu/rlinf-pirl-official-30e0236/experiments/pi_rl/handoff-verification-20261031-v2
export CHECK_PY=/vepfs-C区/jiajunlu/.envs/pirl-openpi-30e0236/bin/python
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 "$CHECK_PY" scripts/pi_rl/probe_runtime.py \
  --run "$CHECK_RUN" --output "$CHECK_RUN/reports/runtime_gpu0.json"
CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 "$CHECK_PY" scripts/pi_rl/probe_runtime.py \
  --run "$CHECK_RUN" --output "$CHECK_RUN/reports/runtime_gpu1.json"
"$CHECK_PY" scripts/pi_rl/audit_config.py --run "$CHECK_RUN"
"$CHECK_PY" scripts/pi_rl/audit_actor_model.py --run "$CHECK_RUN"
"$CHECK_PY" scripts/pi_rl/prepare_training.py --run "$CHECK_RUN"
CUDA_VISIBLE_DEVICES=0,1 "$CHECK_PY" "$CHECK_RUN/source_snapshot/harness/launch_training.py" \
  --run "$CHECK_RUN" --phase local-smoke
```

双 GPU EGL/reset/step/原生推理和 CPU actor 参数审计已通过。有限训练验证正在运行，最终结果将在完成后更新；当前不将其记为完整 train/val/save/reload/infer 通过。

本次 local-smoke 使用历史 SFT manifest 和已完成基线 summary 作为输入，未重新测量两个 500-episode SFT 基线。这些复用只用于当前同环境、同权重的有限生命周期检查。新服务器必须按主文档重新生成自己的 SFT 数据、readiness 和验收，不能复制这些报告通过启动门槛。

## 历史证据与验证范围

随附历史 SFT 原始 JSONL 已重新汇总：recipe-240 为 428/500，standalone-220 为 421/500，覆盖完整且原文件 SHA256 与验收一致。历史 200 步训练的 8 个 actor 均为 200 外层步/2400 optimizer update；最终 checkpoint 的实际文件 SHA256 已重新计算并与 lifecycle 一致。

本次没有提交或停止 Volc 任务，没有更改历史实验和源 SimpleVLA 工作树。未在全新物理服务器从零安装并实跑整个 200 步实验；未执行最终 RL 两协议各 500 条评测，也未声称已得到 GRPO 最终 SR。完整新机验收应按主文档执行。
