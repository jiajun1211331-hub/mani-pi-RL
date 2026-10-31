# π0.5 + GRPO: Reproduction Handoff

Use this guide to prepare a new NVIDIA server, evaluate the π0.5 SFT checkpoint on LIBERO Spatial, train GRPO, verify checkpoints, and compare matched SFT/RL episodes. The main workflow selects 200 outer steps and saves every 40 steps. Select 1000 steps explicitly for the fixed upstream default. Commands match the [Chinese guide](../PI05_GRPO_REPRODUCTION_HANDOFF.md).

Updated 2026-10-31 UTC. The [historical plan](../manifold/pi_rl/PI05_SPATIAL_GRPO_REPRODUCTION_PLAN_20261025.md) preserves earlier decisions and failed diagnostics.

## 1. Established Results and Handoff Scope

Separate historical measurements from evidence you must generate on the new server. Never copy historical passed reports into a new run to satisfy admission gates.

| Item | Verified historical result | Evidence |
|---|---|---|
| RLinf | `30e0236981c6a1dc19202de64116834c12673828` | Git HEAD and preparation gate |
| SFT | `RLinf/RLinf-Pi05-LIBERO-SFT`, revision `45ccfcc4e28634f1576ebf78cab0fbe2fd82432d` | [Metadata](provenance/baseline/manifests/hf_model_info.json), [hashes](provenance/baseline/manifests/sft_files.json) |
| SFT recipe-240 | 428/500 = 85.6% | [Acceptance](provenance/baseline/reports/sft_acceptance.md), [raw data](provenance/baseline/eval/sft/full/recipe-240) |
| SFT standalone-220 | 421/500 = 84.2% | [Raw data](provenance/baseline/eval/sft/full/standalone-220) |
| Local lifecycle | Two A100 80GB GPUs; train/val/save/strict reload/four inference episodes passed | [Lifecycle](provenance/baseline/train/local-smoke/reports/lifecycle.json) |
| 200-step cloud training | Training subprocess exited normally at 2026-10-30 14:26:23 UTC; eight actors each completed 200 outer steps and 2400 optimizer updates | [Launch](provenance/200step/train/formal/reports/launch.json), [lifecycle](provenance/200step/train/formal/reports/lifecycle.json) |
| Final checkpoint | Step 200; 667 tensors; frozen VLM unchanged; strict reload and four inference episodes passed | [Checkpoint](provenance/200step/train/formal/reports/checkpoint.json) |
| Final RL SR | Full 500-episode evaluation under each protocol remains unmeasured | Historical `eval/rl/` is empty; follow section 9 |

The historical task is `pi05-spatial-grpo-200step-save40-30e0236-20261029t083915z-formal-r2`, ID `t-20261029164440-j846b`. The table proves training-subprocess and lifecycle completion. Query the platform separately for its current status; receipt/progress-file states are historical observations.

Original runs remain under `/vepfs-C区/jiajunlu/SimpleVLA-RL/experiments/pi_rl/`. The 200-step run is `pi05-spatial-grpo-200step-save40-30e0236-20261029T083915Z`. Final weights are `train/logs/<run_id>/checkpoints/global_step_200/actor/model_state_dict/full_weights.pt`, SHA256 `dd58e08f9e98174d7770c6839cbc3b898fd1dc0c0fcfe1a93297339019136211`.

> Before stopping any Volc experiment, identify its exact name, task ID, and reason, request fresh user confirmation, and wait for explicit approval for that operation. Deletion, process termination, resource release, and shared Ray cleanup also count as stopping. Finite phases finish naturally at their configured endpoint. A handoff, revised plan, or replacement submission does not authorize stopping an existing task.

## 2. Prepare the Server and Fixed Source

Use Linux x86_64, a driver supporting CUDA 12.6, and offscreen EGL. Historical local glibc was 2.35; the cloud image used Ubuntu 24.04. Prefer Ubuntu 22.04/24.04, then validate the actual container's driver, NCCL, FlashAttention, and EGL.

| Resource | Requirement |
|---|---|
| Evaluation/local smoke | Two exclusively available 80GB GPUs; historical 2×A100 |
| Formal training | One node, 8×80GB NVIDIA GPUs; historical A800, paper reference H100 |
| CPU/RAM | Historical local host: 20 logical CPUs, about 460GiB RAM; re-measure smaller hosts |
| Shared memory | Launcher requires at least 8GiB free in `/dev/shm`; configure container shared memory |
| Disk | Launcher requires 200GiB free; plan 0.5–1.5TiB, then revise from actual checkpoint sizes |
| Tools/network | Git, curl, rsync, Python 3/venv; install uv below. Access to pinned source, wheels, and assets |

On minimal Ubuntu, have the administrator install git, curl, rsync, python3-venv, build-essential (through sudo apt-get install -y), and the current NVIDIA driver.

Set paths on your data volume. Rebuild the venv rather than treating its old absolute paths as portable:

```bash
export PIRL_BASE=/data/pirl
export PIRL_REPO="$PIRL_BASE/rlinf-pirl-official-30e0236"
export PIRL_HARNESS_REPO="$PIRL_REPO"
export PIRL_OFFICIAL="$PIRL_BASE/rlinf-official-source-30e0236"
export PIRL_ENV="$PIRL_BASE/.envs/pirl-openpi-30e0236"
export PIRL_ASSETS="$PIRL_BASE/LIBERO/libero/libero"
export PIRL_SFT="$PIRL_BASE/ckpt/RLinf-Pi05-LIBERO-SFT"
export PIRL_RUN="$PIRL_BASE/experiments/pi_rl/pi05-spatial-grpo-200step-$(date -u +%Y%m%dT%H%M%SZ)"
export TMPDIR="$PIRL_BASE/tmp"
export UV_CACHE_DIR="$PIRL_BASE/cache/uv"
export UV_PYTHON_INSTALL_DIR="$PIRL_BASE/uv-python"
export HF_HOME="$PIRL_BASE/cache/huggingface"
export UV_LINK_MODE=copy
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HOME"
nvidia-smi
df -h "$PIRL_BASE" /dev/shm
python3 -m venv "$PIRL_BASE/.envs/bootstrap-tools"
"$PIRL_BASE/.envs/bootstrap-tools/bin/python" -m pip install uv==0.11.14
export PATH="$PIRL_BASE/.envs/bootstrap-tools/bin:$PATH"
uv --version
```

Caches, extraction, compilation, and outputs now use the data volume. UV_PYTHON_INSTALL_DIR also uses the shared volume so the venv interpreter symlink resolves inside Volc; sharing only the venv while leaving managed Python in the login host's /root/.local fails at startup. Inspect GPU ownership and disk space first; a full root `/tmp` caused an earlier installation failure. Package the overlay on the original server:

```bash
cd /vepfs-C区/jiajunlu/rlinf-pirl-official-30e0236
tar --exclude=__pycache__ --exclude='*.pyc' -czf /vepfs-C区/jiajunlu/pi_rl-handoff-overlay.tar.gz \
  scripts/pi_rl scripts/volc_empty_code \
  scripts/grpo_base_volc_queue2_20261019.yaml \
  tests/test_pi_rl_*.py docs/pi_rl \
  docs/PI05_GRPO_REPRODUCTION_HANDOFF.md \
  docs/manifold/pi_rl docs/GRPO_BASE_VOLC_QUEUE2_HANDOFF_20261019.md
```

Transfer the archive and check out the exact official source before extracting it:

```bash
git clone https://github.com/RLinf/RLinf.git "$PIRL_REPO"
git -C "$PIRL_REPO" checkout --detach 30e0236981c6a1dc19202de64116834c12673828
tar -xzf /path/to/pi_rl-handoff-overlay.tar.gz -C "$PIRL_REPO"
git -C "$PIRL_REPO" worktree add --detach "$PIRL_OFFICIAL" \
  30e0236981c6a1dc19202de64116834c12673828
cd "$PIRL_HARNESS_REPO"
git -C "$PIRL_OFFICIAL" rev-parse HEAD
git -C "$PIRL_OFFICIAL" diff --exit-code HEAD
```

Require the exact commit and an empty tracked-file diff in PIRL_OFFICIAL. Keep editable, versioned harness/docs in PIRL_HARNESS_REPO and a separate clean worktree for numerical source. Preparation archives the clean source and copies the harness separately, so committing documentation does not move the numerical baseline. The current host has this worktree at `/vepfs-C区/jiajunlu/rlinf-pirl-official-source-30e0236`.

## 3. Prepare LIBERO and Simulator Assets

Provide XML, meshes, textures, Spatial BDDL, and initial states. Online RL uses simulator rewards; this workflow does not require demonstration datasets. Install `rlinf-libero==0.1.3` in the training venv and reference separately stored assets through each run's private config.

Prefer copying the three directories from the original host, following symlinks to actual contents:

```bash
# 在新服务器执行；替换 SOURCE_HOST，并确认目标目录属于本次迁移。
mkdir -p "$PIRL_ASSETS"
rsync -aL SOURCE_HOST:/vepfs-C区/jiajunlu/LIBERO/libero/libero/assets/ "$PIRL_ASSETS/assets/"
rsync -aL SOURCE_HOST:/vepfs-C区/jiajunlu/LIBERO/libero/libero/bddl_files/ "$PIRL_ASSETS/bddl_files/"
rsync -aL SOURCE_HOST:/vepfs-C区/jiajunlu/LIBERO/libero/libero/init_files/ "$PIRL_ASSETS/init_files/"
```

The [reference manifest](provenance/baseline/manifests/libero_assets.json) contains 585 asset hashes. The [Spatial comparison](provenance/baseline/manifests/packaged_spatial_asset_comparison.json) checks ten BDDL and ten initial-state files against the official package. Without existing assets, download through a small separate environment:

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

The downloader comes from the pinned package, but its Hub assets can change. Verify every reference file:

```bash
python3 - <<'PY'
import hashlib, json, os
from pathlib import Path
repo, assets = Path(os.environ['PIRL_HARNESS_REPO']), Path(os.environ['PIRL_ASSETS'])
reference = json.loads((repo / 'docs/pi_rl/provenance/baseline/manifests/libero_assets.json').read_text())
for entry in reference['files']:
    file = assets / entry['path']
    assert file.is_file(), file
    assert hashlib.sha256(file.read_bytes()).hexdigest() == entry['sha256'], file
print('585/585 reference simulator asset hashes match')
PY
```

On a mismatch, retrieve the reference tree from the original host or preserve and diagnose differences before claiming the same baseline.

## 4. Initialize a Run and Install Its Environment

Create new-server manifests and freeze evaluation scripts. Initialization imports no historical passed state and submits no tasks:

```bash
cd "$PIRL_HARNESS_REPO"
python3 scripts/pi_rl/initialize_run.py \
  --run "$PIRL_RUN" --repo "$PIRL_OFFICIAL" \
  --environment "$PIRL_ENV" --sft "$PIRL_SFT" \
  --asset-root "$PIRL_ASSETS" --outer-steps 200 --save-interval 40
source "$PIRL_RUN/runtime/run.env"
export PYTHONDONTWRITEBYTECODE=1
```

The run directory must not exist. After sourcing run.env, PIRL_REPO points to clean official source; PIRL_HARNESS_REPO and the shell working directory retain the harness repository. This creates `manifests/run.json`, frozen scripts/hashes, the installer diff, official config copies, and `runtime/run.env`. For the upstream default select `--outer-steps 1000 --save-interval 40` here; do not edit a frozen budget.

Install system dependencies once on the new server/image, then create the venv:

```bash
bash "$PIRL_OFFICIAL/requirements/sys_deps.sh" nvidia
export UV_CONSTRAINT="$PIRL_HARNESS_REPO/docs/pi_rl/environment-constraints.txt"
(
  cd "$PIRL_OFFICIAL"
  bash "$PIRL_RUN/runtime/install_reuse_assets.sh" embodied \
    --model openpi --env libero --venv "$PIRL_ENV" \
    --python 3.11.14 --torch 2.7.1 --no-root \
    > "$PIRL_RUN/reports/environment-install.log" 2>&1
)
```

Run the installer from PIRL_OFFICIAL so its temporary --torch patch and uv project discovery use the same pyproject. The system script may invoke sudo; have the administrator run it on the new host/image. If its libraries already exist, run the venv command directly. The adaptation replaces LIBERO download/global-path reset with verified reuse/private config and preserves shared global Git mirror rules. It retains upstream dependency/model installation. Audit `manifests/install_asset_reuse.patch` and `official_source.json`.

| Component | Historical runtime |
|---|---|
| Python/Torch/torchvision | 3.11.14 / `2.7.1+cu126` / 0.22.1 |
| OpenPI/LIBERO/robosuite | 0.1.1 / 0.1.3 / 1.4.1 |
| MuJoCo/dm-control | 3.3.7 / 1.0.34 |
| Ray/NumPy | 2.59.0 / 1.26.4 |
| JAX/jaxlib/orbax-checkpoint | 0.5.3 / 0.5.3 / 0.11.13 |
| FlashAttention/Triton | 2.7.4.post1 / 3.3.1 |
| Imported Transformers | Fork 4.53.2 plus official replacement files |
| Distribution metadata/tokenizers | Stock Transformers 4.57.6 and fork 4.53.2 / 0.21.4 |

The [full historical freeze](provenance/baseline/manifests/pip_freeze.txt) is evidence, not a directly resolvable lock: two distributions own the same Transformers module directory. Finish the installer's replacement-copy/tokenizer steps. An earlier sequential install selected MuJoCo 3.8.1, outside LIBERO's `<3.4` bound; supplied constraints pin 3.3.7 and dm-control 1.0.34 throughout installation.

After installation, pin all remaining transitive packages with the [runtime replay list](requirements-runtime-replay.txt). Use --no-deps to replay recorded packages without jointly resolving the incompatible metadata of the two Transformers distributions. Also use --no-config for replay, otherwise the harness pyproject can override pinned Torch/torchvision with 2.11/0.26. Restore the fork, tokenizers, and official replacement files last, in that order. Apply this only to the isolated venv, then check actual imports/CUDA and save its freeze:

```bash
uv --no-config pip install --python "$PIRL_ENV/bin/python" --no-deps \
  -r "$PIRL_HARNESS_REPO/docs/pi_rl/requirements-runtime-replay.txt"
uv --no-config pip install --python "$PIRL_ENV/bin/python" --no-deps --reinstall \
  rlinf-transformer-openpi==4.53.2 tokenizers==0.21.4
"$PIRL_ENV/bin/python" - <<'PY'
from pathlib import Path
import shutil, openpi, transformers
source = Path(openpi.__file__).parent / 'models_pytorch/transformers_replace'
target = Path(transformers.__file__).parent
for file in source.rglob('*.py'):
    shutil.copy2(file, target / file.relative_to(source))
PY
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
readlink -f "$PIRL_ENV/bin/python"
uv pip freeze --python "$PIRL_ENV/bin/python" > "$PIRL_RUN/manifests/pip_freeze.txt"
```

If detection selects another CUDA build, explicitly select driver-compatible `UV_TORCH_BACKEND=cu126` when rebuilding a separate venv, or record and validate a separate adaptation. A CPU wheel does not satisfy this environment. `pip check` can report stock Transformers/tokenizers metadata conflicts; runtime gates are fork imports and five replacement-file hashes, checked by GPU probes.

## 5. Download SFT and Tokenizer Assets

Keep the complete SFT directory, including normalization statistics:

```bash
"$PIRL_ENV/bin/hf" download RLinf/RLinf-Pi05-LIBERO-SFT \
  --revision 45ccfcc4e28634f1576ebf78cab0fbe2fd82432d --local-dir "$PIRL_SFT"
"$PIRL_ENV/bin/python" scripts/pi_rl/prepare_run.py verify-sft --run "$PIRL_RUN"
"$PIRL_ENV/bin/python" scripts/pi_rl/prepare_run.py link-assets --run "$PIRL_RUN"
```

Verification checks four sizes, published LFS hashes, all local hashes, and finite 32-dimensional state/actions mean/std/q01/q99. `model.safetensors` is 7,473,091,464 bytes, SHA256 `4616e4c1966a5fb063b442ea5d69857f45e873222d839b794634d87d252ab9e7`. Asset linking checks all 20 Spatial files and writes a private config/manifest.

The installer and OpenPI's `gs://big_vision/…` lookup use different cache locations. Stage identical tokenizer bytes for both:

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

You may copy the same-hash tokenizer from the original host. All ranks must see this private cache; otherwise they may each download from Google Storage. Runtime and launcher propagate `OPENPI_DATA_HOME` into evaluation views.

## 6. Validate Inference and Evaluation Protocols

Check loading, reset/step, EGL, and inference on each GPU. Keep the frozen evaluation scripts identical for SFT and RL:

```bash
export EVAL="$PIRL_RUN/scripts/eval_snapshot"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 "$PIRL_ENV/bin/python" \
  "$EVAL/probe_runtime.py" --run "$PIRL_RUN" --output "$PIRL_RUN/reports/runtime_gpu0.json"
CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 "$PIRL_ENV/bin/python" \
  "$EVAL/probe_runtime.py" --run "$PIRL_RUN" --output "$PIRL_RUN/reports/runtime_gpu1.json"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 "$PIRL_ENV/bin/python" \
  "$EVAL/verify_serial_schedule.py" --run "$PIRL_RUN" --output "$PIRL_RUN/reports/serial_schedule_parity.json"
```

Probes write fixtures, loading audits, observations/actions, simulator results, imported-package information, and replacement hashes. Require 667/667 tensors, empty missing/unexpected keys, finite `(1,5,7)` actions, and reproducible fixed-input/noise inference. Serial checks verify unique episode seeds and batch-one fixture equality within `1e-5`.

Retain production diagnostics against external OpenPI and batch-two execution. Historically both commands below exited 1 after writing reports; do not edit their passed fields:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 "$PIRL_ENV/bin/python" \
  "$EVAL/calibrate_openpi.py" --run "$PIRL_RUN" \
  --fixture "$PIRL_RUN/reports/runtime_gpu0.fixture.pt" \
  --output "$PIRL_RUN/reports/native_external_calibration.json"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 "$PIRL_ENV/bin/python" \
  "$EVAL/verify_sharding.py" --run "$PIRL_RUN" \
  --output "$PIRL_RUN/reports/sharding_parity_diagnostics.json"
```

Historical native/external production maximum action difference was 0.010104; batch-two and batch-one execution were also not bitwise identical. Main measurements use the same native batch-one model for SFT/RL. With `set -e`, execute these individually and inspect status. Missing reports caused by imports, file errors, or CUDA failures require repair.

The combined FP32/highest/eager diagnostic must pass; it does not change production precision:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 "$PIRL_ENV/bin/python" \
  "$EVAL/calibrate_openpi.py" --run "$PIRL_RUN" --fp32-diagnostic \
  --fixture "$PIRL_RUN/reports/runtime_gpu0.fixture.pt" \
  --output "$PIRL_RUN/reports/native_external_fp32_highest_diagnostic.json"
"$PIRL_ENV/bin/python" "$EVAL/assess_eval_readiness.py" --run "$PIRL_RUN"
```

Readiness combines strict loads, batch-one consistency, simulator state/reward/termination, raster changes, and FP32 calibration. Raster changes must be sparse uint8 least-significant bits: maximum difference 1, changed fraction ≤0.001. This does not establish legacy production equivalence. Diagnose failures instead of copying old readiness reports.

## 7. Measure Both SFT Baselines

GPU 0 handles tasks 0–4 and GPU 1 tasks 5–9, each with a resident model. Protocols run sequentially.

| Setting | recipe-240, primary | standalone-220, supplementary |
|---|---|---|
| Coverage | 10 tasks × 50 states = 500 | Same task/init IDs |
| Budget | 240 policy actions, success_once | Up to 220, stop on success |
| Settling/seed | 15 / 0 | 10 / 7 |
| Model/actions | Native π0.5, 3 denoise steps, 5 executed actions, environment dimension 7 | Same |
| Inference | Batch 1, precision=null, torch_compile=true | Same |

Policy noise seed is `seed*1000003 + task_id*50 + init_state_id`. Recipe environment seed is `(seed + (init_state_id*10+task_id)%2)*250 + (init_state_id*10+task_id)//2`, covering 0–499; standalone seed is 7. Preserve mappings and wrapper hashes for paired comparison. Do not pool the protocols' successes.

```bash
"$PIRL_ENV/bin/python" "$EVAL/run_evaluation.py" --run "$PIRL_RUN" --stage smoke
"$PIRL_ENV/bin/python" "$EVAL/run_evaluation.py" --run "$PIRL_RUN" --stage full
"$PIRL_ENV/bin/python" scripts/pi_rl/accept_sft.py --run "$PIRL_RUN"
```

Smoke produces ten episodes per protocol; full produces 500. Each output directory has two JSONL shards, GPU logs, and a summary. Acceptance re-reads episodes to check coverage, duplicates, errors, protocol fields, and input hashes, then writes `reports/sft_acceptance.json/.md`. A difference over three percentage points from reference SFT 84.6% triggers diagnosis; it is not an equivalence tolerance.

Video remains disabled; offscreen rendering is required. Avoid the upstream 500-concurrent-env evaluator on two GPUs. Existing outputs are protected. Inspect actual processes/logs after failure, preserve the attempt, and use a new independent run after confirming termination. Do not silently discard failures or splice attempts.

## 8. Audit, Freeze, and Train

Use the fixed [Spatial recipe](../../examples/embodiment/config/libero_spatial_grpo_openpi_pi05.yaml) and inherited defaults, rather than the old SimpleVLA/VERL configuration.

| Setting | Required value |
|---|---|
| GRPO | Group 8, normalized advantages, chunk-level reward/logprob, clip 0.2/0.2, zero KL/entropy |
| Rollout | 64 total envs including group 8, eight rollout epochs, 240 actions, 3 denoise/5 executed actions |
| Filtering | Group reward mean [0.1,0.9], masking without in-step valid-group refill |
| Flow-SDE | task=rl, flow_sde, noise_level=0.5 |
| Trainable scope | Action expert only, frozen VLM, no LoRA/value head |
| Optimization | Micro 128, global 2048, update_epoch 1, lr 5e-6, Adam β=(0.9,0.95) |
| Execution | Co-located, FSDP no_shard, no offload/checkpointing, precision=null, torch_compile=true |
| Budget/video | Selected 200/save40; upstream 1000; paper table 500; formal val interval -1; video disabled |

An outer step contains `64×8=512` raw trajectory slots and `512×(240/5)=24576` chunk positions: 12 optimizer updates per actor at global batch 2048. Masks determine valid loss; 512 is not a fixed valid-trajectory count. Exact historical trainable numel is 430,098,464.

```bash
"$PIRL_ENV/bin/python" -m pytest tests/test_pi_rl_*.py -q
"$PIRL_ENV/bin/python" scripts/pi_rl/audit_config.py --run "$PIRL_RUN"
"$PIRL_ENV/bin/python" scripts/pi_rl/audit_actor_model.py --run "$PIRL_RUN"
"$PIRL_ENV/bin/python" scripts/pi_rl/prepare_training.py --run "$PIRL_RUN"
export SNAPSHOT="$("$PIRL_ENV/bin/python" -c \
  'import json,os; from pathlib import Path; print(json.loads((Path(os.environ["PIRL_RUN"])/"manifests/training_source_snapshot.json").read_text())["directory"])')"
export HARNESS="$SNAPSHOT/harness"
CUDA_VISIBLE_DEVICES=0,1 "$PIRL_ENV/bin/python" "$HARNESS/launch_training.py" \
  --run "$PIRL_RUN" --phase local-smoke
```

The audit permits only paths, experiment names, video settings, and explicit budget. Freeze writes official source/harness/evaluation, three audited configs, Volc specs, and hashes. Local smoke uses 16 train envs, one outer step, and two validation envs. Require three optimizer updates per actor, train/val/save/strict reload/four inference episodes and `passed=true` in local lifecycle.

Fresh runs use same_snapshot/separate lifecycle gates. Historical reuse_baseline/inline relied on specific authorization and identical numerical sources; do not inherit it on a new server. On an exclusive eight-GPU host, run preflight and formal directly:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PIRL_ENV/bin/python" "$HARNESS/launch_training.py" \
  --run "$PIRL_RUN" --phase cloud-preflight
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PIRL_ENV/bin/python" "$HARNESS/launch_training.py" \
  --run "$PIRL_RUN" --phase formal
```

Here cloud-preflight is a phase name; commands run locally. It uses formal 64 envs, two outer steps, and 16 validation envs, with NCCL/FlashAttention/EGL plus real lifecycle gates. Formal starts from original SFT and expects 2400 optimizer updates per actor for 200 steps.

Launcher uses private Ray authentication, RAY_ADDRESS=local, and short ASCII `/dev/shm/pirl-*` paths linked to the run volume. Avoid machine-wide Ray cleanup. Existing phase launch records prevent restarts; inspect authoritative process state when logs stall.

### 8.1 Submit Through Volc

Mount source, outputs, venv, SFT, and assets at identical absolute paths in the container. Before initialization, provide your account's resource JSON via `--volc-config /path/to/volc.json`:

```json
{
  "queue_id": "YOUR_QUEUE_ID",
  "flavor": "YOUR_SINGLE_NODE_8_GPU_FLAVOR",
  "image_url": "YOUR_EGL_READY_IMAGE",
  "mount_path": "/data",
  "vepfs_id": "YOUR_VEPFS_ID"
}
```

Replace all five placeholders. Historical queue `q-20260403140809-n4flj`, flavor `ml.hpcpni2l.28xlarge`, mount `/vepfs-C区`, and VEPFS `vepfs-cnbjba6dad11eac7` belong to the original account. UserCodePath uses supplied `scripts/volc_empty_code/`; real code comes from the frozen shared snapshot.

The preset image uses shared Python 3.11, not default Python 3.13. A missing EGL loader failed the first formal attempt; r2 used private GLVND/vendor registration recorded in [cloud_bootstrap](provenance/200step/manifests/cloud_bootstrap.json). Prefer an image with required libraries:

```dockerfile
FROM vemlp-cn-beijing.cr.volces.com/preset-images/pytorch:2.10.0-cu12.9.1-py3.13-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 libopengl0 libglvnd0 libgl1 libglfw3 \
    && rm -rf /var/lib/apt/lists/*
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
```

Set image_url to its immutable digest. GPU runtime must inject the current host NVIDIA driver/vendor configuration; do not replace it with old driver binaries. Validate eight ranks inside the actual allocated image; shared venv visibility does not prove glibc/EGL/NCCL compatibility.

Configure the private Volc CLI, inspect generated specs, then submit in phase order:

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

Submission creates GPU tasks. The tool queries the exact name across all states, saves intent, then submits. Reconcile uncertain responses read-only before any further submission. Configure placement before freezing; do not edit frozen specs. Keep credentials in private CLI configuration.

Monitor with the receipt's actual task ID:

```bash
"$VOLC" ml_task get -i TASK_ID --output json
"$VOLC" ml_task instance list -i TASK_ID --output json
"$VOLC" ml_task logs -t TASK_ID -i worker-0 -l 200
```

Read phase launch JSON, collective reports, training logs, actor observer JSONL, and TensorBoard. Advancing rollout proves startup; completed training requires successful exit, final checkpoint, and passed lifecycle.

## 9. Verify the Final Checkpoint and Paired Evaluation

Wait for the configured natural endpoint. Formal lifecycle checks all actor counts, finite loss/gradients, unchanged frozen VLM, changed action expert, complete weights and DCP optimizer/scheduler metadata, strict reload, and four real inference episodes. Optimizer resume remains untested and explicitly false.

Formal saves steps 40/80/120/160/200 under `train/logs/<run_id>/checkpoints/`. Choose final step 200, not 199, and its `actor/model_state_dict/full_weights.pt`; this is not an external OpenPI safetensors file.

```bash
"$PIRL_ENV/bin/python" "$HARNESS/run_final_evaluation.py" --run "$PIRL_RUN" --check-only
CUDA_VISIBLE_DEVICES=0,1 "$PIRL_ENV/bin/python" "$HARNESS/run_final_evaluation.py" --run "$PIRL_RUN"
```

The migrated final evaluator reads the frozen budget and supports 200/1000; its original archived version required 1000. It re-checks source/config/SFT/input hashes, creates a private view of the same official snapshot, runs ten-episode smoke then 500 full episodes per protocol with the same frozen scripts, and automatically compares results.

Inspect final launch state, RL summaries, and `reports/final_comparison.json/.md`. They contain overall/per-task SR, Wilson 95% intervals, improvement, gained/regressed episodes, and actual budget. Paper references are SFT 84.6% and GRPO/Flow-SDE 97.4%. Current 200 steps differ from upstream 1000 and paper-table 500 epochs and do not guarantee that score. Object/Goal/Long and four-suite averages require separate experiments.

## 10. Diagnose Failures and Deliver Evidence

Preserve failures and diagnose the first failing stage; do not silently shrink algorithm settings.

| Symptom | Check |
|---|---|
| EGL loader / NoneType.eglQueryString | Container GLVND/libEGL/libOpenGL, vendor JSON, graphics-driver injection; section 8.1 |
| Repeated tokenizer downloads | Shared OPENPI_DATA_HOME/big_vision path/hash; DOWNLOAD_DIR alone is insufficient |
| Ray socket/path errors | Launcher's private short ASCII runtime; avoid shared cleanup or /vgrpo changes |
| Missing observer JSONL | RLINF_EXT_MODULE=training_observer and frozen PYTHONPATH; driver registration does not propagate automatically |
| Missing validation metrics | log_path/tensorboard; checkpoints use log_path/experiment_name/checkpoints |
| Replay torchvision 0.26 / 0.22 conflict | Use uv --no-config pip install; run the upstream installer from PIRL_OFFICIAL |
| MuJoCo mismatch | Constraints throughout install and preserved resolver logs; isolated venv |
| Transformers import | Fork 4.53.2, tokenizers 0.21.4, replacements, completed installer |
| torchrun consumes --run | Retain the wrapper's `--` separator |
| Volc query HTTP 400 | Limit ≤100; wrapper uses 100; avoid duplicate submit |
| OOM/GPU count mismatch | Exclusive two/eight-GPU allocation; batch/offload/precision changes are separate adaptations |
| Existing launch/output | Check authoritative process/task status, preserve active runs, use a new attempt only after terminal confirmation |

Deliver each new-server stage's own evidence:

- Fixed HEAD, harness/original copies, [migration manifest](COPY_MANIFEST.json), actual imports/freeze/hashes.
- Private LIBERO config, 585 asset hashes, 20 Spatial comparisons, tokenizer and complete SFT checks.
- Both GPU probes/fixtures, serial parity, production diagnostics, FP32 calibration, readiness.
- Two complete 500-episode SFT datasets, summaries, task statistics, and acceptance.
- Three audited configs, immutable source manifest, CPU tests, local train/val/save/reload/infer.
- Actual eight-GPU preflight and formal actor update/save evidence, final checkpoint/hash.
- Two 500-episode RL datasets and paired comparison; hardware, budget, timing/storage, and limitations.

See [migration verification](VERIFICATION.md) for the checks performed on this delivery. This migration does not automatically submit another Volc task or repeat historical 200-step training.
