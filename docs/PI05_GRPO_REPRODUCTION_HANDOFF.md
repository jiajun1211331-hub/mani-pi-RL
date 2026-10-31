# π0.5 + GRPO：新服务器复现接手文档

按本文在一台新的 NVIDIA 服务器上完成 LIBERO Spatial 的 π0.5 SFT 评测、GRPO 训练、checkpoint 验证及训练前后配对评测。先迁移代码和资产，再安装隔离环境，最后逐阶段运行与验收。命令以仓库固定版本 `30e0236` 和当前 200 步、每 40 步保存的预算为主线；官方默认 1000 步作为显式选项。英文版见 [Reproduction Handoff](pi_rl/REPRODUCTION_HANDOFF_EN.md)。

更新：2026-10-31 UTC。本文的运行步骤以当前代码和产物为依据；[旧计划](manifold/pi_rl/PI05_SPATIAL_GRPO_REPRODUCTION_PLAN_20261025.md)保留设计与失败诊断背景。

## 1. 已完成的复现与接手边界

先区分历史实测结果与新服务器需要重新生成的证据，避免把旧报告当成新环境已经通过的验收。

| 项目 | 已核验的历史结果 | 证据 |
|---|---|---|
| 官方代码 | `30e0236981c6a1dc19202de64116834c12673828` | 仓库 Git HEAD；`prepare_run.py` 的版本门槛 |
| SFT | `RLinf/RLinf-Pi05-LIBERO-SFT`，revision `45ccfcc4e28634f1576ebf78cab0fbe2fd82432d` | [Hub 文件清单](pi_rl/provenance/baseline/manifests/hf_model_info.json)、[本地文件 hash](pi_rl/provenance/baseline/manifests/sft_files.json) |
| SFT 主协议 | recipe-240：428/500，85.6% | [验收](pi_rl/provenance/baseline/reports/sft_acceptance.md)；[原始逐 episode 数据](pi_rl/provenance/baseline/eval/sft/full/recipe-240) |
| SFT 补充协议 | standalone-220：421/500，84.2% | [原始逐 episode 数据](pi_rl/provenance/baseline/eval/sft/full/standalone-220) |
| 本地生命周期 | 双 A100 80GB；真实训练、validation、保存、严格重载、4 条推理通过 | [历史 lifecycle](pi_rl/provenance/baseline/train/local-smoke/reports/lifecycle.json) |
| 200 步云端训练 | 2026-10-30 14:26:23 UTC 训练子进程正常退出；8 个 actor 各 200 次外层更新、2400 次 optimizer update | [launch](pi_rl/provenance/200step/train/formal/reports/launch.json)、[lifecycle](pi_rl/provenance/200step/train/formal/reports/lifecycle.json) |
| 200 步 checkpoint | `global_step_200`，667 tensors，冻结 VLM 保持不变，严格重载与 4 条推理通过 | [checkpoint](pi_rl/provenance/200step/train/formal/reports/checkpoint.json) |
| 最终 SR | 两协议各 500 条 RL 后评测尚未执行；4 条重载推理不代表最终 SR | 历史运行的 `eval/rl/` 为空；后续按第 9 节执行 |

200 步任务名称为 `pi05-spatial-grpo-200step-save40-30e0236-20261029t083915z-formal-r2`，ID 为 `t-20261029164440-j846b`。上表证明训练子进程和生命周期完成，平台当前状态应另行只读查询；提交回执中的 `Staging` 与旧进度文件中的 `Running` 是当时的观测。

历史实验根目录为 `/vepfs-C区/jiajunlu/SimpleVLA-RL/experiments/pi_rl/`，本次迁移没有移动或修改这些运行。200 步运行位于 `pi05-spatial-grpo-200step-save40-30e0236-20261029T083915Z`，最终模型为该目录下 `train/logs/<run_id>/checkpoints/global_step_200/actor/model_state_dict/full_weights.pt`，SHA256 为 `dd58e08f9e98174d7770c6839cbc3b898fd1dc0c0fcfe1a93297339019136211`。

> 停止任何 Volc 实验前，必须列明本次具体实验名、任务 ID、停止原因，重新请求用户确认，并等待当次明确确认。不得通过删除任务、kill、释放资源、清理共享 Ray 等方式间接停止。本文的有限阶段按创建时确定的结束条件自然完成；接手、修改方案或重提任务不构成停止既有任务的授权。

## 2. 准备服务器与固定版本

准备 Linux x86_64、支持 CUDA 12.6 的 NVIDIA 驱动和离屏 EGL。历史本地环境为 glibc 2.35，训练容器为 Ubuntu 24.04；优先使用 Ubuntu 22.04/24.04 并在实际训练容器内验收驱动、NCCL、FlashAttention 和 EGL。

| 资源 | 本流程要求 |
|---|---|
| 本地评测 / 有限训练 | 两张可独占的 80GB GPU；历史为 2×A100-SXM4-80GB |
| 正式训练 | 单节点 8×80GB NVIDIA GPU；历史为 A800，论文参考硬件为 H100 |
| CPU / RAM | 历史本地 20 逻辑 CPU、约 460GiB RAM；训练节点以 8 卡规格分配。较小 RAM 配置需重新测量 |
| 共享内存 | launcher 检查 `/dev/shm` 至少空闲 8GiB；容器启动时配置足够的 `--shm-size` |
| 存储 | launcher 检查运行盘至少空闲 200GiB；按 checkpoint 实测大小和保留数量规划，建议预留 0.5–1.5TiB |
| 网络 / 工具 | Git、curl、uv、HF Hub；可访问固定 Git commit、Python wheel 和模型资产来源 |

先定义所有位置。以下 `/data/pirl` 是新服务器示例路径，按实际数据盘修改；不要将原服务器的 Python venv 原样复制后当作可移植环境。

```bash
export PIRL_BASE=/data/pirl
export PIRL_REPO="$PIRL_BASE/rlinf-pirl-official-30e0236"
export PIRL_ENV="$PIRL_BASE/.envs/pirl-openpi-30e0236"
export PIRL_ASSETS="$PIRL_BASE/LIBERO/libero/libero"
export PIRL_SFT="$PIRL_BASE/ckpt/RLinf-Pi05-LIBERO-SFT"
export PIRL_RUN="$PIRL_BASE/experiments/pi_rl/pi05-spatial-grpo-200step-$(date -u +%Y%m%dT%H%M%SZ)"
export TMPDIR="$PIRL_BASE/tmp"
export UV_CACHE_DIR="$PIRL_BASE/cache/uv"
export HF_HOME="$PIRL_BASE/cache/huggingface"
export UV_LINK_MODE=copy
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$HF_HOME"
nvidia-smi
df -h "$PIRL_BASE" /dev/shm
```

这些变量把下载、解压、编译和实验产物放在数据盘。先检查 GPU 占用和盘余量；原服务器曾因根分区 `/tmp` 满而安装失败。

新服务器可直接复制本次完整仓库（保留 `.git`），或先检出官方版本，再叠加本次迁移文件。下面在原服务器打包叠加层：

```bash
cd /vepfs-C区/jiajunlu/rlinf-pirl-official-30e0236
tar -czf /vepfs-C区/jiajunlu/pi_rl-handoff-overlay.tar.gz \
  scripts/pi_rl scripts/volc_empty_code \
  scripts/grpo_base_volc_queue2_20261019.yaml \
  tests/test_pi_rl_*.py docs/pi_rl \
  docs/PI05_GRPO_REPRODUCTION_HANDOFF.md \
  docs/manifold/pi_rl docs/GRPO_BASE_VOLC_QUEUE2_HANDOFF_20261019.md
```

将压缩包传到新服务器，执行：

```bash
git clone https://github.com/RLinf/RLinf.git "$PIRL_REPO"
git -C "$PIRL_REPO" checkout --detach 30e0236981c6a1dc19202de64116834c12673828
tar -xzf /path/to/pi_rl-handoff-overlay.tar.gz -C "$PIRL_REPO"
cd "$PIRL_REPO"
git rev-parse HEAD
git diff --exit-code HEAD
```

`HEAD` 必须为上述完整 commit，官方受跟踪文件的 diff 必须为空。叠加的 `scripts/pi_rl`、测试和 Markdown 保持为未跟踪文件，这是本实验封装在固定官方基线上的布局。`prepare_training.py` 使用 `git archive HEAD` 冻结官方源码，另外复制 harness；若先提交叠加层导致 HEAD 改变，会触发版本门槛。需要将封装另行版本管理时，保留一份该官方 commit 的干净检出，使用 `--repo` 指向它。

## 3. 接入 LIBERO 仿真器和资产

准备 LIBERO 的 XML、mesh、纹理、Spatial 的 BDDL 和初始状态；在线 RL 使用环境奖励，当前流程不依赖 SFT 演示数据集。Python 包在训练 venv 中安装 `rlinf-libero==0.1.3`，资产单独存放并通过每次运行的私有配置引用。

优先从原服务器复制这三个目录，跟随软链接复制实际内容：

```bash
# 在新服务器执行；替换 SOURCE_HOST，并确认目标目录属于本次迁移。
mkdir -p "$PIRL_ASSETS"
rsync -aL SOURCE_HOST:/vepfs-C区/jiajunlu/LIBERO/libero/libero/assets/ "$PIRL_ASSETS/assets/"
rsync -aL SOURCE_HOST:/vepfs-C区/jiajunlu/LIBERO/libero/libero/bddl_files/ "$PIRL_ASSETS/bddl_files/"
rsync -aL SOURCE_HOST:/vepfs-C区/jiajunlu/LIBERO/libero/libero/init_files/ "$PIRL_ASSETS/init_files/"
```

资产验收以内容为准。[历史资产清单](pi_rl/provenance/baseline/manifests/libero_assets.json)包含 585 个资产文件的 SHA256；[Spatial 对照](pi_rl/provenance/baseline/manifests/packaged_spatial_asset_comparison.json)记录 10 个 BDDL 和 10 个初始状态文件与官方包一致。

完全没有资产时，用单独的小环境下载，避免为了获取资产先安装整套训练依赖：

```bash
uv venv "$PIRL_BASE/.envs/libero-assets" --python 3.11.14
uv pip install --python "$PIRL_BASE/.envs/libero-assets/bin/python" \
  --no-deps rlinf-libero==0.1.3
uv pip install --python "$PIRL_BASE/.envs/libero-assets/bin/python" \
  pyyaml filelock tqdm termcolor huggingface-hub
export LIBERO_CONFIG_PATH="$PIRL_BASE/bootstrap-libero-config"
mkdir -p "$LIBERO_CONFIG_PATH" "$PIRL_ASSETS"
"$PIRL_BASE/.envs/libero-assets/bin/libero-download-assets" \
  --assets-dir "$PIRL_ASSETS/assets" --no-cache --skip-existing
"$PIRL_BASE/.envs/libero-assets/bin/python" - <<'PY'
import os, shutil
from pathlib import Path
from libero import libero
source = Path(libero.__file__).resolve().parent
target = Path(os.environ['PIRL_ASSETS'])
for name in ['bddl_files', 'init_files']:
    shutil.copytree(source / name, target / name)
PY
```

下载器来自固定 Python 包。Hub 的资产库可能更新，因此下载完成后与随附历史清单逐文件核验，不能只凭目录存在放行：

```bash
python3 - <<'PY'
import hashlib, json, os
from pathlib import Path
repo, assets = Path(os.environ['PIRL_REPO']), Path(os.environ['PIRL_ASSETS'])
reference = json.loads((repo / 'docs/pi_rl/provenance/baseline/manifests/libero_assets.json').read_text())
for entry in reference['files']:
    file = assets / entry['path']
    assert file.is_file(), file
    assert hashlib.sha256(file.read_bytes()).hexdigest() == entry['sha256'], file
print('585/585 reference simulator asset hashes match')
PY
```

如有差异，优先取回上述原服务器资产；保留差异并排查，不能把另一套资产的结果写为同一基线。

## 4. 初始化运行、安装隔离环境

为新服务器生成自己的目录、路径 manifest 和冻结评测脚本。新入口不复制历史通过状态，也不自动提交任务。

```bash
cd "$PIRL_REPO"
python3 scripts/pi_rl/initialize_run.py \
  --run "$PIRL_RUN" --repo "$PIRL_REPO" \
  --environment "$PIRL_ENV" --sft "$PIRL_SFT" \
  --asset-root "$PIRL_ASSETS" --outer-steps 200 --save-interval 40
source "$PIRL_RUN/runtime/run.env"
export PYTHONDONTWRITEBYTECODE=1
```

该命令要求 run 目录尚不存在，生成 `manifests/run.json`、`scripts/eval_snapshot/` 及其 SHA256、安装适配 diff、官方配置副本和 `runtime/run.env`。需要严格采用官方默认预算时，在初始化时指定 `--outer-steps 1000 --save-interval 40`。预算一经冻结，后续不原地改写。

训练节点需要系统 EGL/GLVND、GLFW、构建工具等依赖。对新服务器按固定仓库的系统依赖脚本安装一次，再使用已有库安装 venv：

```bash
bash requirements/sys_deps.sh nvidia
export UV_CONSTRAINT="$PIRL_REPO/docs/pi_rl/environment-constraints.txt"
bash "$PIRL_RUN/runtime/install_reuse_assets.sh" embodied \
  --model openpi --env libero --venv "$PIRL_ENV" \
  --python 3.11.14 --torch 2.7.1 --no-root \
  > "$PIRL_RUN/reports/environment-install.log" 2>&1
```

系统脚本可能使用 sudo 并安装系统包；由服务器管理员在新机器/镜像中执行。系统库已经齐全的机器直接执行 venv 安装命令。适配脚本保留官方依赖安装，仅将 LIBERO 下载/全局配置重设替换为资产校验与私有配置，并保留共享全局 Git 镜像规则。原脚本与适配 diff 位于 `manifests/official_source.json`、`manifests/install_asset_reuse.patch`。

关键版本如下，完整历史依赖见 [pip_freeze.txt](pi_rl/provenance/baseline/manifests/pip_freeze.txt)。它是环境证据，两个 Transformers 发行包共用模块目录，不能把这份 freeze 当成可直接联合求解的锁文件。

| 组件 | 历史实际值 |
|---|---|
| Python / PyTorch / torchvision | 3.11.14 / `2.7.1+cu126` / 0.22.1 |
| OpenPI / LIBERO / robosuite | `rlinf-openpi 0.1.1` / `rlinf-libero 0.1.3` / 1.4.1 |
| MuJoCo / dm-control | 3.3.7 / 1.0.34 |
| Ray / NumPy | 2.59.0 / 1.26.4 |
| JAX / jaxlib / orbax-checkpoint | 0.5.3 / 0.5.3 / 0.11.13 |
| FlashAttention / Triton | 2.7.4.post1 / 3.3.1 |
| 实际 `import transformers` | 4.53.2，来自 `rlinf-transformer-openpi` 及官方替换文件 |
| 发行包元数据 / tokenizers | stock transformers 4.57.6、fork 4.53.2 / 0.21.4 |

安装器必须完成最后的 Transformers 文件替换与 tokenizers 约束。历史顺序安装曾将 MuJoCo 升到 3.8.1，违反 LIBERO 的 `<3.4` 要求；本次约束文件在安装全过程固定 3.3.7 和 dm-control 1.0.34。若安装中断，先保留日志、完成或排查当前环境，不能仅看到 `pip freeze` 就进入训练。

检查实际依赖和 CUDA：

```bash
"$PIRL_ENV/bin/python" - <<'PY'
import torch, transformers, mujoco, ray, tokenizers, flash_attn
assert torch.cuda.is_available()
assert torch.__version__ == '2.7.1+cu126', torch.__version__
assert transformers.__version__ == '4.53.2'
assert mujoco.__version__ == '3.3.7'
assert ray.__version__ == '2.59.0'
assert tokenizers.__version__ == '0.21.4'
print(torch.__version__, torch.version.cuda, transformers.__file__)
PY
uv pip freeze --python "$PIRL_ENV/bin/python" > "$PIRL_RUN/manifests/pip_freeze.txt"
```

如果驱动检测选择了不同 CUDA wheel，应显式设置受驱动支持的 `UV_TORCH_BACKEND=cu126` 后重建独立 venv，或记录为另一次环境适配并重新验证。不要静默将 CPU wheel 或其他模型栈当成该已验证版本。`pip check` 可能报告 stock Transformers 对 tokenizers 的元数据冲突；最终以实际 fork 导入和 5 个替换文件的 hash 验收为准，GPU 探针会核对这些文件。

## 5. 下载 SFT 与 tokenizer

保留完整 SFT 仓库结构；模型权重与归一化文件共同定义起点，不能只复制 `model.safetensors`。

```bash
"$PIRL_ENV/bin/hf" download RLinf/RLinf-Pi05-LIBERO-SFT \
  --revision 45ccfcc4e28634f1576ebf78cab0fbe2fd82432d --local-dir "$PIRL_SFT"
"$PIRL_ENV/bin/python" scripts/pi_rl/prepare_run.py verify-sft --run "$PIRL_RUN"
"$PIRL_ENV/bin/python" scripts/pi_rl/prepare_run.py link-assets --run "$PIRL_RUN"
```

`verify-sft` 核对 4 个文件的大小和已发布 LFS hash，并记录全部 SHA256、32 维 state/actions 的 mean/std/q01/q99。模型文件为 7,473,091,464 字节，SHA256 为 `4616e4c1966a5fb063b442ea5d69857f45e873222d839b794634d87d252ab9e7`。`link-assets` 核对 20 个 Spatial 文件与安装包一致，写出私有 `runtime/libero_config/config.yaml` 和资产 manifest。

同时固定 tokenizer。官方安装器缓存的位置与 OpenPI 的 `gs://big_vision/…` 解析位置不同，所以把相同文件放到这两个入口，并设置运行私有 `OPENPI_DATA_HOME`：

```bash
"$PIRL_ENV/bin/hf" download RLinf/openpi_tokenizer paligemma_tokenizer.model \
  --local-dir "$PIRL_RUN/runtime/model_assets/.cache/openpi"
mkdir -p "$OPENPI_DATA_HOME/big_vision"
cp "$PIRL_RUN/runtime/model_assets/.cache/openpi/paligemma_tokenizer.model" \
  "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
"$PIRL_ENV/bin/python" - <<'PY'
import hashlib, os
from pathlib import Path
file = Path(os.environ['OPENPI_DATA_HOME']) / 'big_vision/paligemma_tokenizer.model'
assert hashlib.sha256(file.read_bytes()).hexdigest() == '8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6'
print('tokenizer SHA256 passed')
PY
```

也可从原服务器复制该 hash 的文件。缓存要对所有训练 rank 可见，否则原流程会在云端每个 rank 分别访问 Google Storage。迁移后的 `runtime.py` 和 launcher 将同一运行缓存传给模型及最终评测 view。

## 6. 验证推理与评测协议

先验证两张卡各自能加载模型、执行 LIBERO reset/step 和 EGL 渲染，再判断分片封装是否保持既定协议。以下 `EVAL` 指向刚刚冻结的评测脚本；正式 SFT 与 RL 都用它，不在评测中途替换文件。

```bash
export EVAL="$PIRL_RUN/scripts/eval_snapshot"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 "$PIRL_ENV/bin/python" \
  "$EVAL/probe_runtime.py" --run "$PIRL_RUN" --output "$PIRL_RUN/reports/runtime_gpu0.json"
CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 "$PIRL_ENV/bin/python" \
  "$EVAL/probe_runtime.py" --run "$PIRL_RUN" --output "$PIRL_RUN/reports/runtime_gpu1.json"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 "$PIRL_ENV/bin/python" \
  "$EVAL/verify_serial_schedule.py" --run "$PIRL_RUN" --output "$PIRL_RUN/reports/serial_schedule_parity.json"
```

GPU 探针输出 `.fixture.pt`、模型加载审计、观测/动作、仿真结果和环境包/替换文件 hash。要求 667/667 tensors 正确加载、missing/unexpected 为空、动作形状 `(1,5,7)`、数值有限、固定观测与噪声重复推理一致。串行检查要求每个 episode 的 noise seed 唯一，batch 1 与两张卡的固定 fixture 动作差满足 `1e-5`。

还需记录原生模型与旧外部 OpenPI、batch 2 与 batch 1 的差异。下面两项在历史生产精度下返回码为 1，脚本仍会写诊断 JSON；应保留结果并按 readiness 的独立条件判断，不能修改 `passed`：

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 "$PIRL_ENV/bin/python" \
  "$EVAL/calibrate_openpi.py" --run "$PIRL_RUN" \
  --fixture "$PIRL_RUN/reports/runtime_gpu0.fixture.pt" \
  --output "$PIRL_RUN/reports/native_external_calibration.json"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 "$PIRL_ENV/bin/python" \
  "$EVAL/verify_sharding.py" --run "$PIRL_RUN" \
  --output "$PIRL_RUN/reports/sharding_parity_diagnostics.json"
```

历史原生/外部生产动作最大差为 0.010104，batch 2/1 也不逐位相同。主流程固定同一原生 batch-1 模型，不用这些差异来改变正式模型精度。若 shell 使用 `set -e`，逐条执行上述诊断并检查返回码；导入、文件或 CUDA 错误导致报告缺失时必须修复。

下面的 FP32/highest/eager 联合诊断需通过；它不用于训练或正式评测：

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 "$PIRL_ENV/bin/python" \
  "$EVAL/calibrate_openpi.py" --run "$PIRL_RUN" --fp32-diagnostic \
  --fixture "$PIRL_RUN/reports/runtime_gpu0.fixture.pt" \
  --output "$PIRL_RUN/reports/native_external_fp32_highest_diagnostic.json"
"$PIRL_ENV/bin/python" "$EVAL/assess_eval_readiness.py" --run "$PIRL_RUN"
```

readiness 综合严格权重加载、batch-1 调用一致性、仿真 state/reward/termination、图像量化差异和 FP32 诊断。图像差异允许极稀疏的 uint8 最低位变化（最大差 1、变化比例 ≤0.001），不会把两条生产路径宣称为完全等价。门槛不通过时先诊断，不复制历史 readiness 放行。

## 7. 完成两个 SFT 基线评测

每张卡驻留一个模型，GPU 0 执行 task 0–4、GPU 1 执行 task 5–9。两协议按顺序运行，避免同时占用同一对 GPU。

| 项目 | recipe-240：主结果 | standalone-220：补充结果 |
|---|---|---|
| 任务与初始状态 | 10×50，500 唯一 episode | 相同 task/init ID |
| 动作预算 | 240 个 policy action；记录 success_once | 最多 220 个 policy action，成功时结束 |
| settling | 15 步 | 10 步 |
| 协议 seed | 0 | 7 |
| 模型与动作 | 原生 π0.5，3 次去噪、执行 5 步、7 维环境动作 | 同一原生模型与 3/5 设置 |
| 推理 | batch 1、precision=null、torch_compile=true | 相同 |

每 episode 的 policy noise seed 为 `seed*1000003 + task_id*50 + init_state_id`。recipe 的环境 seed 为 `(seed + (init_state_id*10+task_id)%2)*250 + (init_state_id*10+task_id)//2`，完整覆盖 0–499；standalone 的环境 seed 为 7。固定 seed 映射与 wrapper hash 可用于前后配对，两个协议的成功数不能合并。

```bash
"$PIRL_ENV/bin/python" "$EVAL/run_evaluation.py" --run "$PIRL_RUN" --stage smoke
"$PIRL_ENV/bin/python" "$EVAL/run_evaluation.py" --run "$PIRL_RUN" --stage full
"$PIRL_ENV/bin/python" scripts/pi_rl/accept_sft.py --run "$PIRL_RUN"
```

smoke 为每协议 10 条，full 为每协议 500 条。输出在 `eval/sft/{smoke,full}/<protocol>/`：两份 `episodes_gpu*.jsonl`、GPU 日志和 `summary.json`。验收重新读取原始 episode，检查覆盖、重复、系统错误、协议字段及权重/封装/资产 hash，生成 `reports/sft_acceptance.json/.md`。偏离参考 SFT 84.6% 超过 3 个百分点时触发排查，这不是统计等价容差。

整个流程关闭视频，但保留模型输入必需的离屏渲染。不要直接在两卡服务器上运行默认 500-env 的官方统一评测命令。已有输出时脚本拒绝覆盖；失败后先核对进程和日志，保留原 attempt，在明确终止状态后用新的独立 run 重跑，不拼接或静默丢弃失败样本。

## 8. 审计配置、冻结源码并训练

完成 SFT 后，冻结训练配置与完整生命周期的来源。算法基准为 [Spatial 官方 YAML](../examples/embodiment/config/libero_spatial_grpo_openpi_pi05.yaml)，不是旧 SimpleVLA/VERL GRPO 配置。

| 配置 | 必须保留的值 |
|---|---|
| GRPO | group 8、advantage normalize、reward/logprob chunk_level、clip 0.2/0.2、KL/entropy 0 |
| rollout | 64 个总环境（已含 group 8）、8 个 rollout epochs、240 步、3 次去噪/5 步执行 |
| group mask | reward 均值 [0.1,0.9]；不在同一外层步内补采到固定有效轨迹数 |
| Flow-SDE | task=rl、noise_method=flow_sde、noise_level=0.5 |
| 训练范围 | 冻结 VLM，仅 action expert；无 LoRA、无 value head |
| batch / 优化器 | micro 128、global 2048、update_epoch 1、lr 5e-6、Adam β=(0.9,0.95) |
| 执行 | co-located、FSDP no_shard、无 offload/gradient checkpointing、precision=null、torch_compile=true |
| 预算 | 当前 200 外层步/每 40 步保存；官方默认 1000；论文表为 500；正式 val_check_interval=-1 |
| 视频 | train/eval save_video=false |

一个外层 RL step 包含 `64×8=512` 个原始轨迹配额、`512×(240/5)=24576` 个 chunk 训练位置，按 global batch 2048 推导每个 actor 12 次 optimizer update。过滤通过 mask 控制有效 loss，512 不是筛选后固定有效轨迹数。历史训练的可训练参数精确值为 430,098,464。

```bash
"$PIRL_ENV/bin/python" scripts/pi_rl/audit_config.py --run "$PIRL_RUN"
"$PIRL_ENV/bin/python" scripts/pi_rl/audit_actor_model.py --run "$PIRL_RUN"
"$PIRL_ENV/bin/python" scripts/pi_rl/prepare_training.py --run "$PIRL_RUN"
export SNAPSHOT="$("$PIRL_ENV/bin/python" -c \
  'import json,os; from pathlib import Path; print(json.loads((Path(os.environ["PIRL_RUN"])/"manifests/training_source_snapshot.json").read_text())["directory"])')"
export HARNESS="$SNAPSHOT/harness"
CUDA_VISIBLE_DEVICES=0,1 "$PIRL_ENV/bin/python" "$HARNESS/launch_training.py" \
  --run "$PIRL_RUN" --phase local-smoke
```

审计只允许路径、实验名、视频关闭和显式预算差异。冻结产物包括 `source_snapshot/{official,harness,evaluation}`、三份 phase YAML 和审计、Volc YAML 及 SHA256。local-smoke 使用 16 train env、1 外层步、2 eval env，每卡应有 3 次 optimizer update，包含真实 train/val/save/严格重载及两协议共 4 条推理。结果须为 `train/local-smoke/reports/lifecycle.json` 的 `passed=true`。

新服务器默认 `local_lifecycle=same_snapshot`、`cloud_preflight=separate`；重新完成自己的测试。历史 200 步运行的 `reuse_baseline/inline` 是当时特定授权与相同数值源码下的路径，不能直接沿用。

已有独占 8 卡服务器时，有限 2 步检查和正式训练都可以直接本地启动：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PIRL_ENV/bin/python" "$HARNESS/launch_training.py" \
  --run "$PIRL_RUN" --phase cloud-preflight
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PIRL_ENV/bin/python" "$HARNESS/launch_training.py" \
  --run "$PIRL_RUN" --phase formal
```

`cloud-preflight` 是 phase 名；这里仍在本地执行，不提交云任务。它使用正式 64 env、2 外层步、16 eval env，先完成 NCCL/FlashAttention/EGL 与真实生命周期。正式训练从原始 SFT 开始，不从 smoke checkpoint 续训。正式每卡预期 `200×12=2400` 次 optimizer update。

launcher 使用独立 Ray token、`RAY_ADDRESS=local`、短 ASCII `/dev/shm/pirl-*` 路径，并将 Ray socket 临时路径链接到本次运行的盘目录。不要执行机器级 `ray stop`。重新运行已产生 `launch.json` 的 phase 会被拒绝；先检查原进程是否还在运行，不因日志暂时不更新就重启。

### 8.1 通过 Volc 提交

Volc 需要共享盘把代码、run、Python 环境、模型和资产以相同绝对路径挂到容器。迁移到另一个账号时，在第 4 节初始化前创建本账号资源 JSON，并给 `initialize_run.py` 增加 `--volc-config /path/to/volc.json`：

```json
{
  "queue_id": "YOUR_QUEUE_ID",
  "flavor": "YOUR_SINGLE_NODE_8_GPU_FLAVOR",
  "image_url": "YOUR_EGL_READY_IMAGE",
  "mount_path": "/data",
  "vepfs_id": "YOUR_VEPFS_ID"
}
```

五个字段全部填写实际值。旧默认队列为 `q-20260403140809-n4flj`、规格 `ml.hpcpni2l.28xlarge`、挂载 `/vepfs-C区`、VEPFS `vepfs-cnbjba6dad11eac7`，仅用于原账号。生成的 `configs/volc-*.yaml` 中 `UserCodePath` 指向随仓库迁移的 `scripts/volc_empty_code/`；真正代码从共享盘的不可变快照加载。

原预设镜像 `vemlp-cn-beijing.cr.volces.com/preset-images/pytorch:2.10.0-cu12.9.1-py3.13-ubuntu24.04` 使用共享 Python 3.11 venv；镜像默认 Python 3.13 不作为训练解释器。该镜像曾缺少 EGL loader，导致第一个正式尝试在 collective probe 失败；历史 r2 使用私有 GLVND loader 和 NVIDIA vendor JSON 适配，见 [cloud_bootstrap](pi_rl/provenance/200step/manifests/cloud_bootstrap.json)。新服务器优先构建包含系统依赖的兼容镜像，例如：

```dockerfile
FROM vemlp-cn-beijing.cr.volces.com/preset-images/pytorch:2.10.0-cu12.9.1-py3.13-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 libopengl0 libglvnd0 libgl1 libglfw3 \
    && rm -rf /var/lib/apt/lists/*
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
```

把已构建镜像的不可变 digest 填入 `image_url`，由 GPU runtime 注入当前宿主的 NVIDIA 驱动和 vendor 配置。不要拷贝旧服务器的 NVIDIA 驱动二进制来代替注入。8 卡 probe 必须在实际分配的容器内通过；共享 venv 可见不证明 glibc/EGL/NCCL 一致。

在本账号安装并配置 Volc CLI，先检查生成 YAML，再按阶段提交：

```bash
export VOLC=/root/.volc/bin/volc
"$PIRL_ENV/bin/python" "$HARNESS/submit_training.py" --run "$PIRL_RUN" \
  --phase cloud-preflight --volc "$VOLC" --check-only
"$PIRL_ENV/bin/python" "$HARNESS/submit_training.py" --run "$PIRL_RUN" \
  --phase cloud-preflight --volc "$VOLC"
# 只在 cloud-preflight/reports/lifecycle.json 的 passed=true 后继续：
"$PIRL_ENV/bin/python" "$HARNESS/submit_training.py" --run "$PIRL_RUN" \
  --phase formal --volc "$VOLC" --check-only
"$PIRL_ENV/bin/python" "$HARNESS/submit_training.py" --run "$PIRL_RUN" \
  --phase formal --volc "$VOLC"
```

提交会创建 GPU 任务。工具先按完整任务名查询所有状态，保存 submission intent，再提交；返回不确定时保留 intent 并只读核对，不能直接再提交。YAML 在冻结后不可编辑；资源变更应在新 run 冻结前填写。凭据由 CLI 私有配置管理，不写进 YAML、文档或日志。

只读监控命令如下，用回执里的实际任务 ID 替换占位符：

```bash
"$VOLC" ml_task get -i TASK_ID --output json
"$VOLC" ml_task instance list -i TASK_ID --output json
"$VOLC" ml_task logs -t TASK_ID -i worker-0 -l 200
```

结合 `train/<phase>/reports/launch.json`、`collective_gpu*.json`、`training.log`、`observer/actor_rank*.jsonl` 和 TensorBoard 判断进度。只有 rollout 正在推进可称启动通过，只有末步 checkpoint、训练正常退出和 lifecycle 通过才称训练完成。

## 9. 验证末步 checkpoint 并配对评测

等待配置确定的自然终点，检查 `train/formal/reports/lifecycle.json`。它核对所有 rank 的更新计数、有限 loss/grad、冻结 VLM 未变化、action expert 有变化、完整权重和 DCP optimizer/scheduler 元数据，并严格重载后运行 4 条真实推理。DCP 元数据存在不代表 optimizer resume 已经过测试；当前报告明确为 false。

正式训练的保存目录为 `train/logs/<run_id>/checkpoints/global_step_{40,80,120,160,200}/actor/`。主评测使用末步 200 的完整 `model_state_dict/full_weights.pt`；不能把 `global_step_199` 当作第 200 步，也不能把 `.pt` 直接当成外部 OpenPI `.safetensors`。

```bash
"$PIRL_ENV/bin/python" "$HARNESS/run_final_evaluation.py" --run "$PIRL_RUN" --check-only
CUDA_VISIBLE_DEVICES=0,1 "$PIRL_ENV/bin/python" "$HARNESS/run_final_evaluation.py" --run "$PIRL_RUN"
```

迁移后的入口从冻结 formal 配置读取最终预算，支持 200 和 1000 步；原始副本曾硬编码只接受 1000 步。它重新核对源码/配置/SFT/输入 hash，创建指向同一官方源码快照的私有 view，用同一冻结评测脚本先跑两协议各 10 条 smoke，再跑各 500 条完整评测，最后自动运行 `compare_results.py`。

检查 `reports/final_evaluation_launch.json` 的状态和 `eval/rl/full/<protocol>/summary.json`，最终报告为 `reports/final_comparison.json/.md`，包含总体/逐任务 SR、Wilson 95% 区间、相对本地 SFT 提升、新增成功和退化 episode，以及实际训练预算。报告参考为 SFT 84.6%、GRPO/Flow-SDE 97.4%；当前 200 步不等于官方 1000 步或论文表中 500 epochs 的预算，不保证得到同一数值。当前范围为 Spatial，Object/Goal/Long 与四套平均值需另行实验。

## 10. 常见故障与交付检查

按最先失败的阶段排查，保留原始失败证据；不要直接缩小算法参数让配置审计通过。

| 现象 | 检查与处理 |
|---|---|
| EGL 无法加载 / `NoneType.eglQueryString` | 检查当前容器的 libEGL/libOpenGL/GLVND、NVIDIA vendor JSON、driver graphics 注入；见 8.1，修复后新 run 验证 |
| 多 rank 重复下载 tokenizer | 检查 `OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model` 的实际存在、hash 和容器内挂载；只设置 DOWNLOAD_DIR 不够 |
| Ray Unix socket 非 ASCII / 路径过长 | 使用 launcher 创建的短 ASCII runtime，不手改共享 Ray 或 `/vgrpo` |
| Ray 观测 JSONL 缺失 | 检查 `RLINF_EXT_MODULE=training_observer`、冻结 harness PYTHONPATH；driver 中注册不会自动传播到 worker |
| TensorBoard validation 找不到 | 指标在 `runner.logger.log_path/tensorboard`，checkpoint 在 `log_path/experiment_name/checkpoints` |
| MuJoCo 版本超出范围 | 在安装前应用随附 constraints；保留联合解析/安装日志，不在共享环境升级依赖 |
| Transformers 导入 tokenizers 错误 | 核对 fork 4.53.2、tokenizers 0.21.4、官方替换文件 hash；完成安装器末尾步骤 |
| `--run` 被 torchrun 误解析 | 使用封装自带的 `--` 分隔方式，不改写 collective 命令 |
| Volc list 返回 400 | limit 必须 ≤100；提交封装已设为 100。保留查询结果，不因此重复 submit |
| GPU 数不匹配 / OOM | 使用独占 2 卡评测、2 卡 local-smoke 或 8 卡正式训练；保存诊断。改 batch/offload/precision 属于新的资源适配实验 |
| phase 已有 launch / episode 输出已存在 | 核对原进程或任务的权威状态；原运行仍活跃时保持运行。确认终态后另建独立 attempt，不覆盖旧日志 |

向下一位接手者交付以下产物及实际路径。新服务器上的每个阶段必须有自己的证据。

- 固定 Git HEAD、完整封装/原始副本、[迁移 SHA256 清单](pi_rl/COPY_MANIFEST.json)和实际环境 freeze/import/hash。
- 私有 LIBERO 配置、585 个资产 hash、20 个 Spatial 文件对照、tokenizer 与 SFT 全文件校验。
- 双 GPU 推理探针、fixture、串行一致性、生产差异诊断、FP32 校准与 readiness。
- SFT 两协议各 500 条完整无重复 episode、原始 JSONL、逐任务统计与验收。
- 三阶段 resolved YAML/白名单审计、不可变源码/配置 manifest、CPU 测试、本地 train/val/save/reload/infer。
- 实际 8 卡探针、有限 preflight lifecycle、正式训练所有 rank 的更新/保存证据、最终完整 checkpoint 与 hash。
- RL 两协议各 500 条配对评测及最终对比、实际硬件/预算/耗时/存储、未完成项。

本次代码迁移的验证范围与命令见 [VERIFICATION.md](pi_rl/VERIFICATION.md)。该交付文档覆盖完整流程；本次迁移不会自动重跑历史 200 步训练或提交新 Volc 任务。
