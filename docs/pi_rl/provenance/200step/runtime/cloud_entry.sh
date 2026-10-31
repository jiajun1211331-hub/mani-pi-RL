#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH='/vepfs-C区/jiajunlu/SimpleVLA-RL/experiments/pi_rl/pi05-spatial-grpo-200step-save40-30e0236-20261029T083915Z/runtime/egl':"${LD_LIBRARY_PATH:-}"
export __EGL_VENDOR_LIBRARY_DIRS='/vepfs-C区/jiajunlu/SimpleVLA-RL/experiments/pi_rl/pi05-spatial-grpo-200step-save40-30e0236-20261029T083915Z/runtime/egl/egl_vendor.d'
exec '/vepfs-C区/jiajunlu/.envs/pirl-openpi-30e0236/bin/python' -u '/vepfs-C区/jiajunlu/SimpleVLA-RL/experiments/pi_rl/pi05-spatial-grpo-200step-save40-30e0236-20261029T083915Z/source_snapshot/harness/launch_training.py' --run '/vepfs-C区/jiajunlu/SimpleVLA-RL/experiments/pi_rl/pi05-spatial-grpo-200step-save40-30e0236-20261029T083915Z' --phase formal
