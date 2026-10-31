# π0.5 / πRL：LIBERO Spatial GRPO 复现实验计划

> **2026-10-29 当前执行目标：** 用户最终指定为 **200 个外层 RL step、8 卡、`save_interval=40`，直接提交 Volc，不等待新的本地双卡适配**。观察到实际 rollout 正常推进即完成当前 goal；不把提交成功或仅模型初始化成功作为验收。本次独立运行目录为 `experiments/pi_rl/pi05-spatial-grpo-200step-save40-30e0236-20261029T083915Z`，配置审计显式允许预算覆盖，其余算法参数沿用固定官方版本。复用原始基线已通过的真实 train/val/save/reload/infer，启动前重算源文件与 SFT 证据；在云端训练任务内执行 8 卡 NCCL/FlashAttention/EGL 检查，再进入 200 步训练。下文 1000 步记录保留为旧预算，当前任务不要求跑满该上限。当前 goal 的启动验收不代表完整训练及最终 SFT/RL 对比已经完成。

日期：2026-10-25。目标：完成“官方代码检出 → 环境与资产复用 → 官方 SFT 权重下载 → 本地双卡 SFT 评测与论文对照 → Volc 队列运行官方 Spatial GRPO → 训练后评测”的闭环。

本文是执行计划，现已开始落地。截至 2026-10-25 19:10 UTC，固定官方 commit 的独立检出、SFT 下载/hash 校验、隔离环境安装及双卡 EGL/原生模型推理探针已完成。正式 SFT 两个协议各 500 条已通过验收：recipe-240 为 428/500（85.6%），standalone-220 为 421/500（84.2%），无缺失、重复或系统错误。配置、覆盖率、冻结权重、提交去重、配对结果与最终 checkpoint 选择的 47 项 CPU 测试在根环境和隔离环境中通过。GPU 冒烟第 3 次已完成真实 GRPO 更新、官方验证、保存与独立 CPU 严格重载，但 Ray 观测记录缺失使完整生命周期验收失败；该次结果与失败证据保留。修复经过真实 CPU Ray worker 验证后，第 4 次 GPU 冒烟完成每卡 3 次更新、验证 2/2、保存及严格重载；验收另发现 TensorBoard 路径误判，修正后独立检查通过，重载后的两协议共 4 条 GPU 推理全部完成。该轮原始失败日志和修正检查器 hash 均保留。新 r8 快照只改变验收脚本，官方源码及三份配置 hash 不变；第 5 次同快照有限 GPU 验收已完整通过，每卡 3 次更新、验证 2/2、严格重载、重载后推理 4/4 均通过。独立的 8 卡、2 步 cloud preflight 已提交，任务 ID `t-20261026030926-4vrmg`，2026-10-25 19:10 UTC 状态为 Queue；正式 1000 步任务尚未提交。当前执行目录为 `experiments/pi_rl/pi05-spatial-grpo-official-30e0236-20261025T144123Z/`，具体阶段状态与证据见该目录的 `manifests/run.json`。文中的后续命令仍需按阶段验收门槛执行。

2026-10-25 用户补充：暂不设置算力/费用/存储预算上限；本轮只完成 Spatial；策略推理不保存视频。本计划据此关闭本地评测、云端 rollout 与 preflight 的视频录制/保存，保留模型输入必需的离屏渲染和逐 episode 数值记录。预算不设上限不影响共享资源隔离、空间检查和 Volc 停止前逐次确认规则。

## 1. 必须满足的约束

1. **LIBERO、仿真资产、已有环境和下载缓存优先复用。** 复用以版本和行为兼容为前提；保留现有仓库、环境、LIBERO 修改及其他实验，不在共享环境中直接升级冲突依赖。
2. **先在本地 2 张 GPU 上评测官方 SFT checkpoint。** 本次主线至少完成 `libero_spatial` 的 10 个任务 × 50 个初始状态，共 500 个 episode；与论文 SFT Spatial **84.6%** 对照。未完成配置、资产、推理和指标验收前，不进入正式 GRPO。
3. **在 `libero_spatial` 上运行 GRPO，算法、模型、训练配置对齐固定版本的官方配置，通过 Volc 队列提交。** 首轮不修改 advantage、reward、Flow-SDE、冻结范围、batch、学习率等设置，不加入轨迹监督或其他方法。
4. **训练前后使用相同评测协议。** 论文对照结果、当前官方配置结果、吞吐冒烟结果分别记录；不能混用不同 episode 长度、去噪步数、动作执行长度或权重。
5. 现有 [SimpleVLA-RL/AGENTS.md](../../../AGENTS.md) 要求：停止任何 Volc 实验前，必须列出实验名、任务 ID、原因，并取得用户当次明确确认。执行本计划不得自动停止、删除或释放其他任务，也不得通过清理共享 Ray 进程间接中止任务。
6. **本轮限定 Spatial，不保存推理视频。** `env.train.video_cfg.save_video=false`、`env.eval.video_cfg.save_video=false` 列入用户明确要求的配置差异白名单；standalone 和自写分片入口同样关闭视频。不能以“官方默认录像”为由保存视频。

## 2. 已核查的服务器情况与选型结论

### 2.1 代码是否需要重新拉取

**结论：需要另建一份固定官方版本的干净检出，保留现有目录。** 可以复用旧仓库的 Git 对象来减少下载，不需要把所有已有资产重新下载。

| 项目 | 2026-10-25 实际检查结果 | 本次处理 |
|---|---|---|
| 现有 RLinf | `/vepfs-C区/jiajunlu/rlinf` | 保留，不原地 pull/reset |
| 当前提交 | `b9f9c7f7c56c6bbaa75151101df17c9feec2a9f1`，2026-06-08，`Add DreamZero LIBERO eval adaptations` | 属于现有实验代码，不当作当前官方基线 |
| 远端 | 用户 Gitee 镜像；前一提交为 2026-05-26 的 `1061a89` | 本次基线从 `RLinf/RLinf` 获取 |
| 工作区 | 本次检查 `git status --short` 为空 | 干净不等于与当前官方版本相同 |
| 本次官方版本 | `30e0236981c6a1dc19202de64116834c12673828` | 固定 commit，不随执行时的 main 漂移 |

已确认的差异包括：FSDP 配置从 `training_backend/fsdp` 调整为 `hybrid_engines/fsdp`；rollout 次数字段从 `algorithm` 移到 `env.train/env.eval`；OpenPI 模型配置新增明确的任务工厂和模型结构字段；新代码包含原生 OpenPI checkpoint 兼容加载逻辑。因此不能把新版 YAML 直接放进旧仓库，或把旧版模型环境直接视为新版可用。

本次复现定义为“**固定当前官方实现与官方 Spatial GRPO recipe**”。它与“恢复论文发表时的所有软件版本”不是同一个保证；论文数值是对照目标，版本差异必须在最终报告保留。

### 2.2 可复用的资产与算力

| 资源 | 已确认路径 / 状态 | 复用方式 |
|---|---|---|
| LIBERO 代码 | `/vepfs-C区/jiajunlu/LIBERO`，commit `8f1084e3132a39270c3a13ebe37270a43ece2a01` | 保留；新环境所用包按官方依赖安装，优先复用其数据资产 |
| LIBERO 资产 | `/vepfs-C区/jiajunlu/LIBERO/libero/libero/assets` | 校验后通过私有配置或软链接复用 |
| BDDL | `/vepfs-C区/jiajunlu/LIBERO/libero/libero/bddl_files` | 核对 Spatial 的 10 个任务及文件 hash |
| 初始状态 | `/vepfs-C区/jiajunlu/LIBERO/libero/libero/init_files` | 核对每任务 50 个状态，保留固定 ID 清单 |
| 本地 GPU | 2 × NVIDIA A100-SXM4-80GB | SFT 与训练后评测；启动前重新检查占用 |
| 本地 CPU / 内存 | 20 个逻辑 CPU，约 460 GiB RAM；检查时可用约 411 GiB | 评测优先有限并发，避免直接启动 500 个仿真实例 |
| 共享盘 | `/vepfs-C区`，检查时剩余约 7.8 TiB | 这是共享盘余量，不代表本实验独占配额 |
| 已有模型目录 | `/vepfs-C区/jiajunlu/ckpt/RL` | 本次限定范围扫描未发现目标 π0.5 SFT；下载前再按模型 ID 检查缓存 |

LIBERO 当前唯一已发现的工作区改动，是 `benchmark/__init__.py` 中给可信本地初始状态加载显式设置 `torch.load(..., weights_only=False)`，用于兼容新版 PyTorch。保存 diff，不撤销，不把这处改动误判为任务资产变化。

### 2.3 环境尽量复用的具体边界

已检查常用环境的包元数据：

| 环境 | Python / Torch / Transformers | 判断 |
|---|---|---|
| `shared/envs/simplevla` | 3.10.20 / 2.4.0+cu124 / 4.40.1 | 无 OpenPI，不能直接作为本次官方环境 |
| `.envs/simplevla_robo` | 3.10.20 / 2.4.0+cu124 / 4.40.1 | 无 OpenPI，保留其现有用途 |
| `.envs/openwam` | 3.10.21 / 2.7.1+cu128 / 5.17.0 | 无 OpenPI、LIBERO、Ray，Transformers 不符合本次依赖范围 |
| `.envs/openwam-libero` | 3.10.21 / 2.7.1+cpu / 4.21.1 | 适合作为已有资产位置参考，不能直接做 GPU 模型环境 |
| `.envs/openwam-robotwin` | 3.10.20 / 2.4.1 / 未安装 | 非本任务完整环境 |

旧 RLinf 目录中未发现 `.venv` 或 `.venv-openpi`。上述扫描不代表服务器绝对不存在其他环境；执行阶段先补查明确登记的 OpenPI/RLinf 环境。

复用顺序：**已有兼容环境 → 在隔离环境中复用缓存和兼容依赖 → 按固定官方安装脚本新建独立环境**。当前证据倾向最后一种；不为了复用目录而强行保留冲突版本，也不原地升级 SimpleVLA/OpenWAM 环境。

固定版本安装脚本为 LIBERO 安装 `rlinf-libero`，为 OpenPI 安装 `rlinf-openpi==0.1.1`，还会处理 Transformers 替换文件、tokenizer 约束和辅助资产。只看 `pip freeze` 或只安装一个 `openpi` 包，不能证明环境一致。

## 3. 工作目录、版本与产物

建议使用以下独立路径；不存在的目录在执行阶段创建：

```text
/vepfs-C区/jiajunlu/rlinf-pirl-official-30e0236/      # 新官方代码检出
/vepfs-C区/jiajunlu/.envs/pirl-openpi-30e0236/       # 仅在无兼容环境时新建
/vepfs-C区/jiajunlu/ckpt/RLinf-Pi05-LIBERO-SFT/     # 若已有相同内容则复用原位置
/vepfs-C区/jiajunlu/SimpleVLA-RL/experiments/pi_rl/<run_id>/
  manifests/                                     # 版本、资产、权重、环境、episode 清单
  configs/                                       # 官方原配置、解析配置、差异报告
  runtime/libero_config/                          # 每次运行私有 LIBERO 配置
  scripts/                                       # 待实现的评测分片、提交、汇总脚本
  source_snapshot/                               # 云端使用的冻结源码与启动脚本
  eval/sft/                                      # SFT 的 episode JSONL / 汇总；不保存视频
  train/                                         # 日志、checkpoint 与提交回执
  eval/rl/                                       # 训练后同协议评测
  reports/                                       # SFT 验收、资源测量、最终对比
```

至少固定：RLinf commit、OpenPI/LIBERO 实际安装版本及代码位置、Torch/Transformers/CUDA/MuJoCo/robosuite/Ray 版本、安装后被替换模块的 hash、镜像 digest、SFT revision 和文件 SHA256、资产 hash、完整 resolved config、GPU 型号、随机种子。

## 4. 阶段 A：检出官方代码，复用资产并配置环境

### A1. 创建独立检出

以下是计划中的命令，执行前确认目标目录不存在；如已存在，先核验而不是覆盖：

```bash
PIRL_REPO=/vepfs-C区/jiajunlu/rlinf-pirl-official-30e0236
PIRL_COMMIT=30e0236981c6a1dc19202de64116834c12673828
git clone --reference-if-able /vepfs-C区/jiajunlu/rlinf --dissociate \
  https://github.com/RLinf/RLinf.git "$PIRL_REPO"
git -C "$PIRL_REPO" checkout --detach "$PIRL_COMMIT"
git -C "$PIRL_REPO" rev-parse HEAD
git -C "$PIRL_REPO" status --short
```

归档当前官方训练配置、模型配置、Spatial 环境配置和所有 Hydra defaults，记录与旧仓库的相关差异。新目录内执行操作前，读取其适用的 AGENTS.md。

### A2. 环境安装与资产复用

1. 对候选环境检查实际 import 路径、CUDA 可用性、OpenPI 模型工厂、LIBERO reset/step/offscreen rendering，而不只检查包名。只有完整通过且无需破坏其他项目依赖时才直接使用。
2. 未发现兼容环境时，以固定 commit 的安装脚本为依赖基准，创建独立 venv，复用服务器既有 pip/uv/HF 缓存。安装入口为：

   ```bash
   cd "$PIRL_REPO"
   bash requirements/install.sh embodied --model openpi --env libero \
     --venv /vepfs-C区/jiajunlu/.envs/pirl-openpi-30e0236
   ```

   **先完成下面的资产接入与安装脚本检查，再运行此入口。** 这不是允许直接执行默认资产下载的无条件命令。

   执行核查发现 `rlinf-openpi==0.1.1` 精确依赖 `torch==2.7.1`，而当前项目基础依赖默认 Torch 2.11。实际安装使用官方支持的 `--torch 2.7.1 --no-root`：匹配 OpenPI 自带 pin、避免先安装另一套 Torch，并复用已经检查存在的 EGL/GLFW/Mesa 系统库；不改变训练 YAML。安装适配另外禁用上游脚本对全局 Git 镜像规则的清理，保留其他实验的配置。原脚本、适配 diff、失败尝试和安装决策分别留档。

   顺序安装还曾选中 MuJoCo 3.8.1，违反 `rlinf-libero==0.1.3` 声明的 `mujoco>=3.0,<3.4`；随后联合解析 `rlinf-libero==0.1.3` 与 `rlinf-openpi==0.1.1`，收敛到 MuJoCo 3.3.7、dm-control 1.0.34。预览与实际变更日志均留档。系统根分区 `/tmp` 已满，编译临时目录与缓存使用本次 run 的私有目录。官方 Transformers fork 实际导入版本 4.53.2、替换文件 hash 和 tokenizers 0.21.4 已在 GPU 探针中验证；stock Transformers 的发行包元数据仍报告 tokenizers 范围冲突，不将此环境描述为 `pip check` 完全无告警。

3. 固定版本的 `install_libero_env` 会调用 `libero-download-assets --skip-existing`，并重设 LIBERO 配置。旧目录存在资产并不代表新包路径会自动识别。先核验 `LIBERO_CONFIG_PATH` 是否被当前包和重设函数完整遵循；使用本次运行私有配置，不能改写其他实验使用的全局配置。
4. 新安装包实际解析到的 `assets / bddl_files / init_files` 优先指向已有经过校验的资产。若官方安装流程不能直接复用，制作留档的安装适配脚本，仅把资产下载/配置重设阶段替换为复用步骤；依赖版本、模型实现、训练代码不改。禁止覆盖非空目录，禁止向共享缓存文件写入定制修改。
5. 对新包期望的资产清单做缺失与 hash 检查。只下载缺失或确认不兼容的文件；本任务使用已发布 SFT 做在线 RL，通常不需要重新下载完整 SFT 演示数据集。
6. 优先使用官方 EGL 渲染。若需要 OSMesa，先记录原因并做图像与动作一致性检查，再决定是否接受该环境偏差，不能直接沿用旧 SimpleVLA 的渲染设置。
7. 若 Ray 对共享盘中文或长路径不兼容，为本次 run 创建唯一短 ASCII 运行路径，使用独立临时目录和端口；不得覆盖旧 `/vgrpo`，不得执行清理所有 Ray 进程的命令。

**A 阶段验收：** 双卡可见；模型与仿真库 import 正常；每张卡完成一次离屏渲染与策略推理；资产来自预期位置；环境快照、安装适配 diff 与资产清单已落盘。

## 5. 阶段 B：下载并验证官方 SFT checkpoint

模型 ID：[`RLinf/RLinf-Pi05-LIBERO-SFT`](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT)。本次核对的 HF revision：`45ccfcc4e28634f1576ebf78cab0fbe2fd82432d`。

```bash
PIRL_SFT=/vepfs-C区/jiajunlu/ckpt/RLinf-Pi05-LIBERO-SFT
hf download RLinf/RLinf-Pi05-LIBERO-SFT \
  --revision 45ccfcc4e28634f1576ebf78cab0fbe2fd82432d \
  --local-dir "$PIRL_SFT"
```

下载前检查现有同 revision 的完整缓存，能够复用就不再下载。保留完整仓库结构，尤其是 `model.safetensors` 和 `physical-intelligence/libero/norm_stats.json`，不能只保留权重文件。禁止把 π0、官方全量 SFT 模型或 RL 后权重当作本次 few-shot SFT 起点。

加载后记录：关键权重缺失/多余键、实际模型配置、归一化文件路径、总参数与可训练参数数量。新 RLinf 原生 OpenPI 加载器支持旧 OpenPI 布局的兼容转换，必须核验关键主体权重全部加载；不得用 `strict=False` 掩盖关键参数未加载。

**B 阶段验收：** 文件完整且 hash 固定；norm stats 可读取；固定观测与固定初始噪声的前向结果为有限数；动作形状、反归一化、夹爪方向符合 LIBERO。

## 6. 阶段 C：本地双卡 SFT 评测与论文对照

### C1. 对照数值与主验收范围

论文 Table 9 的 π0.5 对照如下。**GRPO Spatial 的参考值是 97.4%，不是 PPO 的 99.6%。** [论文 Table 9](https://arxiv.org/html/2510.25889v3)

| 模型 / 方法 | Spatial | Object | Goal | Long | 四套平均 |
|---|---:|---:|---:|---:|---:|
| π0.5 few-shot SFT | **84.6%** | 95.4% | 84.6% | 43.9% | 77.1% |
| π0.5 + GRPO / Flow-SDE | **97.4%** | 99.8% | 91.2% | 77.6% | 91.5% |

本轮只完成 Spatial SFT 验收与 Spatial GRPO 闭环，Object、Goal、Long 不纳入本轮。没有测齐四套就不能声明复现了 77.1% 平均值。Spatial-only RL 训练后，也不能预期直接得到论文分别训练四套任务的 GRPO 平均值。

### C2. 两种公开评测口径必须分开

| 项目 | 当前 RLinf Spatial 评测 recipe | 官方 standalone 评测脚本 |
|---|---|---|
| 来源 | `evaluations/libero/libero_spatial_openpi_pi05_eval.yaml` | `toolkits/standalone_eval_scripts/openpi/libero_eval.py` |
| Spatial 时间上限 | `max_episode_steps=240`、rollout budget 240 | 220 个策略动作步，另加默认 10 个 settling 步 |
| 模型步数 / 执行动作 | π0.5 Spatial 为 3 次去噪 / 5 步执行 | 必须显式指定 3 / 5，不能使用脚本默认 5 次去噪 |
| 随机种子 | 环境配置继承 seed 0 | 脚本默认 seed 7 |
| settling | 原生 `LiberoEnv.reset()` 默认 15 步 | 默认 10 步 |
| 模型加载 | 当前 RLinf 原生 OpenPI 模型 | 外部 OpenPI `PI0Pytorch`，原脚本读取 safetensors |

不能仅凭现有代码断言论文 84.6% 的所有运行细节等同于今天的任一入口。执行时归档论文超参表、官方说明及这两条路径的差异，核验 settling、终止条件、初始状态映射和噪声种子。

安排两份 Spatial 结果：

- **主结果：当前官方 recipe，240 步。** SFT 与 RL 后权重均采用同一协议、3 次去噪、5 步执行，用于判断本次训练收益。
- **论文对照补充结果：官方 standalone 协议，220 步。** 显式记录 seed 7、settling 10 步和 3/5 参数；用于排查公开入口差异。若论文原始评测细节仍未完全确认，报告为“与论文参考值比较”，不声称已逐项恢复论文协议。

220/240 的上限、seed 0/7 和模型加载路径不一起随意切换后归因；诊断时每次只改变一项。不得把两个协议下的成功数合并成一个 SR。

### C3. 本地两张卡如何执行

1. 优先让 GPU 0、GPU 1 各驻留一个策略实例；每个实例复用已加载模型，串行或低并发执行自己的 episode。
2. 固定 `task_id × init_state_id` 清单。每个协议覆盖 10 × 50 = 500 个唯一组合；例如 GPU 0 负责 task 0–4，GPU 1 负责 task 5–9，各 250 条。记录 task 名称到 ID 的映射。
3. 不把“500 个 episode”误解成“必须同时创建 500 个 env”。若使用分片执行，保证评测样本、观测预处理、策略、reset/step 和统计语义与对应官方入口一致，改变的只是执行调度。
4. 分片工具已实现，正式运行使用冻结的 `scripts/eval_snapshot/`；它通过自写 task-range 封装使用官方模型和环境接口。SFT 和 RL 的最终对比使用同一原生模型封装；官方外部 OpenPI standalone 入口用于 SFT 的参考校准，检查固定观测、固定噪声下的动作一致性，不让两种加载实现成为未控制的前后差异。噪声按预先固定的 episode 种子或输入张量控制，避免分片顺序改变结果却未记录。已通过与未通过的数值比较分别见本节执行记录。
5. 先运行每任务 1 条，共 10 条冒烟；检查相机翻转/缩放、双视角、文字指令、proprio 输入处理、夹爪方向、norm stats、动作执行长度、reset 和 success 统计。通过后再执行正式 500 条。
6. 保存每个 episode 的任务、初始状态、seed、协议、checkpoint hash、成功与否、首次成功步、长度、异常。环境异常单独列出并按预先规则补跑原 episode ID，不得静默剔除后缩小分母。
7. 不录制或保存视频；使用固定观测/噪声的动作一致性、图像与动作张量统计以及逐 episode 日志诊断，并记录成功/失败样本的 episode ID（若该类样本存在）。自动 reset 后的新 episode 不混入原 500 条统计；success 定义与官方一致，避免终止后重复计数。

执行记录：两协议的 10 条冒烟分别为 recipe-240 **8/10**、standalone-220 **9/10**，无系统错误；这不是正式 SFT 验收。 正式 recipe-240 已完成 **428/500 = 85.6%**，95% Wilson 区间 **82.25%–88.41%**，相对论文 84.6% 为 **+1.0 pp**；覆盖完整，无系统错误。standalone-220 的正式 500 条仍在运行，整体验收尚未完成。正式两协议各 500 条按同一冻结入口执行，评测脚本冻结于当前 run 的 `scripts/eval_snapshot/`。主线统一使用原生模型、batch 1、`precision=null`、`torch_compile=true`。相同固定输入/噪声的 batch-1 分片 API 在两张卡的 fixture 上与官方原生调用动作差为 0；不同 EGL 上下文存在少量 RGB 最低位变化，batch 2 与 batch 1 不声称逐位相同。当前原生与旧外部 OpenPI 的生产精度动作最大差为 0.010104，未通过最初设置的 `atol=rtol=1e-3` 测试；FP32/highest/eager 联合诊断最大差为 0.000338。失败记录全部保留，联合诊断不用于单因素归因，也不改变正式评测精度。`reports/native_eval_readiness.json` 只放行当前官方原生链路的统计测量，不代表旧实现生产动作完全等价、论文软件版本完全复原或正式 SFT SR 验收通过。

当前统一评测入口的参考命令如下。它保留了默认的 500-env 配置，**不是要求在本地直接以 500 并发执行**；本地正式运行用验收过的分片方案：

```bash
cd "$PIRL_REPO"
CUDA_VISIBLE_DEVICES=0,1 bash evaluations/run_eval.sh \
  libero libero_spatial_openpi_pi05_eval \
  rollout.model.model_path="$PIRL_SFT" \
  env.eval.video_cfg.save_video=false
```

### C4. SFT 验收标准

- **硬性条件：** 正确 checkpoint 和 norm stats；500 条完整且无重复/遗漏；关键权重全部加载；图像与动作处理正确；代码、环境、协议和随机种子均可追溯；没有未解释的系统错误。
- **数值条件：** 报告总体与逐任务 SR、95% Wilson 区间、相对 84.6% 的百分点差。将绝对偏差超过 **3 个百分点**作为排查触发值，而不是官方容差或统计等价证明。N=500、SR≈84.6% 时，简单二项近似标准误约 1.6 个百分点；任务间差异和训练种子不确定性仍需另外考虑。
- **偏差处理：** 先核验资产、协议、模型加载与预处理，再按预先登记的额外 seed 复测并完整报告。不得“重复到达标”或只保留最高的一次。
- **进入正式训练：** 硬性条件全部通过，且不存在未解释的显著指标偏差；输出 `reports/sft_acceptance.md`。单次 SR 接近 84.6% 不能独自证明实现无误。若仍不能解释偏差，继续修复评测链路，不提交正式 GRPO。

## 7. 阶段 D：冻结官方 Spatial GRPO 配置

唯一训练配置基准是固定 commit 下的 [`libero_spatial_grpo_openpi_pi05.yaml`](https://github.com/RLinf/RLinf/blob/30e0236981c6a1dc19202de64116834c12673828/examples/embodiment/config/libero_spatial_grpo_openpi_pi05.yaml)，连同其继承的模型、环境、FSDP 与权重同步配置。

| 配置组 | 正式运行必须保留的值 |
|---|---|
| 算法 | `adv_type=grpo`，`loss_type=actor`，`normalize_advantages=true`，`group_size=8` |
| 更新 | `update_epoch=1`，`loss_agg_func=token-mean` |
| PPO-style ratio clip | `clip_ratio_low=0.2`，`clip_ratio_high=0.2`；其余官方 loss 字段原样保留 |
| 正则 | `kl_beta=0`，`entropy_bonus=0` |
| reward / logprob | `chunk_level`；`reward_coef=1.0` |
| group 过滤 | `filter_rewards=true`，阈值 `[0.1, 0.9]` |
| 训练环境 | `libero_spatial`，总 env 64，group 8，`rollout_epoch=8` |
| episode / rollout budget | 240 / 240 |
| env 默认行为 | seed 0；固定 reset-state ID 池；`auto_reset=false`；`ignore_terminations=false`；`use_rel_reward=true`；无 step penalty；reset 打开夹爪 |
| 模型 | `pi05_libero`，`pi05=true`，`num_steps=3`，`num_action_chunks=5`，env action dim 7 |
| Flow-SDE | `openpi.task=rl`，`noise_method=flow_sde`，`noise_level=0.5` |
| 冻结策略 | `train_expert_only=true`，`is_lora=false`，`add_value_head=false` |
| 输入与结构 | 两张图像；模型 action dim 32；保留官方 proprio/transforms 和 token 配置，不人工替换为旧 OpenVLA 输入处理 |
| batch | `micro_batch_size=128`，`global_batch_size=2048` |
| optimizer | lr `5e-6`；Adam β=(0.9,0.95)，eps `1e-8`，weight decay `0.01`，clip grad `1.0` |
| actor seed | 42 |
| 运行方式 | 单节点，actor/env/rollout co-located，placement `all` |
| FSDP | `strategy=fsdp`，`sharding_strategy=no_shard`，`gradient_checkpointing=false` |
| offload | actor 和 rollout 均 false |
| precision | 维持官方 `null` 和模型内部 dtype 处理，记录最终各模块 dtype，不擅自全局改成 bf16/fp16 |
| 编译 | 保留模型配置 `torch_compile=true` |
| 长度与保存 | `max_epochs=1000`，`max_steps=-1`，`save_interval=40`，`val_check_interval=-1` |
| 用户要求的视频设置 | 覆盖官方默认录像：`env.train.video_cfg.save_video=false`、`env.eval.video_cfg.save_video=false` |
| 其他 | `recompute_logprobs=false`；`unnorm_key=libero_10` 原样保留；无 reward model、无 critic |

说明：

- Spatial 是 **3 次去噪、执行 5 步、对本轮采样 batch 遍历 1 轮（`update_epoch=1`）**；后者不表示只做一次 optimizer update，具体计数见 D1。不能沿用 Long 的参数。网络预测 horizon 与每次实际执行长度不是同一字段，不能为“5 步执行”擅自修改模型结构。
- `unnorm_key=libero_10` 虽然名称带 10，也是该官方 Spatial 配置的值；核验实际使用的 norm stats/transforms，不凭名称改成 spatial。
- 配置中出现 value/GAE/dual-clip 等字段，不代表它们在 GRPO 的实际调用链中必然生效。以原值归档，并记录真实执行路径，不能把存在的 YAML 字段当作已启用功能。
- 优化器只接收可训练 action expert 相关参数；运行时输出总参数量、可训练参数量及名称清单，不把论文“300M expert”的命名直接当作精确 numel。冻结 VLM 在短更新后应保持不变。
- 按 64 env × 8 rollout 轮计，每个外层训练轮有 512 个 env-rollout 配额；240/5=48 个 chunk 位置。它不等于 512 条一定完整有效的成功/失败轨迹，过滤和终止后的实际样本数必须从日志记录。具体分组、筛选与更新计数见下节。

只允许改动：模型本地路径、日志/输出路径、实验名、集群实际路径映射、用户明确要求的训练/评测视频关闭项，以及已经验证不改变数值行为的运行适配。与官方 resolved config 做逐字段 diff，允许项列白名单；训练参数不得因显存或速度压力静默缩小。

如果原配置在目标资源上 OOM 或不能运行，停止推进该次启动并保存诊断；后续调整应另列为“资源适配实验”，不能继续标成严格官方配置复现。涉及停止已经运行的 Volc 任务时遵守第 1 节的当次确认规则。

### D1. RL step、64×8 与 optimizer update 的实际定义

以下来自固定 commit 的 runner、env worker 与 actor 调用链，不把外层 RL step、采样轮和优化器更新混为一谈：

1. [`EmbodiedRunner.set_max_steps`](https://github.com/RLinf/RLinf/blob/30e0236981c6a1dc19202de64116834c12673828/rlinf/runners/embodied_runner.py#L669) 设置 `num_steps_per_epoch=1`。`run()` 每完成一次采样、advantage 计算和 actor 训练，才执行 `global_step += 1`。因此本配置的 `max_epochs=1000`、`max_steps=-1` 对应 **1000 个外层 RL step**，不是 1000 次单独的 optimizer update。
2. **64 是总并发环境数，已经包含 group size 8。** 每轮是 8 个 group × 8 条轨迹；[`EnvWorker`](https://github.com/RLinf/RLinf/blob/30e0236981c6a1dc19202de64116834c12673828/rlinf/workers/env/env_worker.py#L1113) 重复 8 个 rollout epochs，合计 64 个 group 采样 × 8 条轨迹 = **512 个原始轨迹配额/RL step**。组内共享 task/init-state，见 [`LiberoEnv.update_reset_state_ids`](https://github.com/RLinf/RLinf/blob/30e0236981c6a1dc19202de64116834c12673828/rlinf/envs/sim/libero/libero_env.py#L512)。不同 group 的 reset-state 允许重复，不能称作 64 个唯一初始状态。
3. [`_process_received_rollout_batch`](https://github.com/RLinf/RLinf/blob/30e0236981c6a1dc19202de64116834c12673828/rlinf/workers/actor/embodied_fsdp_actor_worker.py#L252) 用 reward group 均值 `[0.1,0.9]` 构造 `loss_mask`；在本次二元成功奖励语义下，全成功与全失败组不参与有效 loss。**当前实现没有在同一 RL step 内继续补采、直到凑满 64 个有效 group 的逻辑。** 这与已有 SimpleVLA 跨 epoch 累积有效组的方案不同；512 是原始采样配额，不能写成筛选后固定 512 条有效轨迹。
4. [`run_training`](https://github.com/RLinf/RLinf/blob/30e0236981c6a1dc19202de64116834c12673828/rlinf/workers/actor/embodied_fsdp_actor_worker.py#L579) 将时间与轨迹维展开，再按 global batch 切分。按当前固定形状，`512 × (240/5) = 24,576` 个 chunk 训练位置，`global_batch_size=2048`、`update_epoch=1`，对应 **12 次 optimizer update/RL step**。这是代码与配置推导值；终止与 reward 筛选通过 mask 控制有效 loss，运行时仍需核对实际计数、有效位置数和异常情况。8 卡、每卡 micro batch 128 时，每次 optimizer update 有 2 次梯度累积。

### D2. 论文预算与当前官方配置的差异

重新核对计划引用的 [论文 v3 附录 J、Table 11](https://arxiv.org/html/2510.25889v3)：π0.5 Spatial 一栏列出 **Train epochs=500**、global batch 2048、update epochs 1、actor lr `5e-6`、interaction steps 240、parallel envs 64、rollout epochs 8、prediction horizon 10、replan horizon 5、denoise steps 3、Flow-SDE noise 0.5。这里的 `64×8` 同样是并行环境数 × 采样轮数。

固定 commit 的当前 Spatial GRPO YAML 则为 **max_epochs=1000**。原计划已经选择“固定当前官方实现和 recipe”，因此本轮仍以 1000 个外层 RL step 为配置基准，并明确记录它是论文表中 500 epochs 的两倍外层预算；不能称为完全相同的论文训练预算。如后续改为论文预算 500，必须在提交前单独冻结配置与命名，不能在运行到 500 时擅自停止现有 Volc 任务。

论文 B.2 明确使用 RLinf、8×H100 80GB、env/rollout/actor co-located，并在 RL 阶段冻结 VLM、仅训练 action expert。附录 F.1/Table 9 中，π0.5 Spatial few-shot SFT 为 84.6%，GRPO/Flow-SDE 为 97.4%，PPO 为 99.6%。Table 11 是 LIBERO 的通用超参表，包含 critic/GAE 等项目；论文未单列完整的 GRPO 专用 group size、过滤/补采与 checkpoint 选择清单，不能把当前代码的每项实现细节倒推成论文原始运行已逐项确认的设置。

## 8. 阶段 E：Volc 队列提交与训练

### E1. 复用已验证过的提交设施

参考服务器已有 [队列交接文档](../../GRPO_BASE_VOLC_QUEUE2_HANDOFF_20261019.md) 和 [队列 YAML](../../../scripts/grpo_base_volc_queue2_20261019.yaml)。这些资料只用于复用 Volc 队列、镜像、挂载和日志查询方式，不继承其 SimpleVLA/VERL 算法配置。

| 项目 | 现有资料记录的值 | 提交前检查 |
|---|---|---|
| CLI | `/root/.volc/bin/volc` | 命令可用、当前账号权限正常 |
| Queue 2 | `q-20260403140809-n4flj` | 队列仍有效、资源规格和排队状态 |
| 规格 | `ml.hpcpni2l.28xlarge`，1 worker，8 × A800 80GB | 实际实例的 GPU 数量/显存/驱动 |
| 文件系统 | `vepfs-cnbjba6dad11eac7` 挂载 `/vepfs-C区` | 代码、env、权重、资产和输出均可访问 |
| 既有镜像 | `vemlp-cn-beijing.cr.volces.com/preset-images/pytorch:2.10.0-cu12.9.1-py3.13-ubuntu24.04` | 仅作已用过的基础镜像候选；不能视为已兼容新版 OpenPI |

论文使用 8 × H100 80GB；本计划优先复用已有 Volc 队列的 8 × A800 80GB。报告必须注明硬件差异，不能用 H100 的运行时间承诺 A800 的耗时。[论文实现设置](https://arxiv.org/html/2510.25889v3)

### E2. 提交前准备

1. SFT 验收报告通过。
2. 对本次新写的封装做最小本地 GPU 冒烟：推理、评测结果汇总；若涉及训练/模型加载代码改动，按父目录 AGENTS 要求增加有限训练更新、保存、重载、validation、inference 生命周期测试。缩小规模的 smoke 必须单独命名，不能将其权重、修改过的配置或训练曲线当作正式结果。
3. 如果两卡无法完成必要训练冒烟，明确记录未验证项与原因，不能声称已通过；在正式训练前先安排独立且有限步数的云端 preflight，使用已写定的结束条件。
4. 冻结 `source_snapshot`、依赖 manifest、训练 resolved config、配置白名单 diff 和启动脚本。云端只从快照运行，避免共享工作区后续变动污染实验。
5. 检查队列容器内 Python/GLIBC/CUDA/EGL/NCCL/FlashAttention/编译缓存兼容性。不能因为共享路径可见，就假设本地 venv 可在该镜像直接运行。必要时用兼容镜像或在该镜像构建同版本环境，产出新的环境 manifest。
6. checkpoint 输出位置、写权限、空间和日志转存全部通过。正式运行从原始 SFT 初始化，`resume_dir=null`、`ckpt_path=null`，不从 smoke 权重接着训练。

当前准备了三份独立配置：`configs/local-smoke.yaml`（2 卡、16 env、1 外层步、验证 2 env）、`configs/cloud-preflight.yaml`（8 卡、正式 64 env、2 外层步、验证 16 env）和 `configs/formal.yaml`（正式 1000 外层步）。有限阶段保持 group 8、rollout 8、micro/global batch 128/2048、3/5、Flow-SDE 0.5 等算法设置，单独保存与每步验证；正式配置与最初审计的 recipe 完全一致。它们不是三个正式复现结果。最新不可变源码位置由 `manifests/training_source_snapshot.json` 指定，目前为 `source_snapshot-r8/`，tree SHA256 为 `84fb0459dae795191088d2c5a4d38220d2463991f0ab6d38c8d5cc0c2eb6ebd6`；前版快照和 manifest 保留。快照内含固定官方 git archive 及观测、验收、启动和配对评测封装，共 2263 个文件。正式配置 SHA256 仍为 `e50a61f931bb115cf98d873269a8676c169fcf2fc9d89d401556fe58e47b3360`。参数观测封装在 Ray 包装之前注册派生 actor，调用官方方法并记录实际返回值和参数集合，不改变 optimizer 调用或张量。

`manifests/actor_parameters.json` 记录实际总参数 3,353,433,872、可训练参数 430,098,464、冻结参数 2,923,335,408；第 4 次实际 GPU 运行两卡 post-FSDP optimizer 参数集合均与可训练集合完全一致。可训练参数包含 BF16 311,427,072 和 FP32 118,671,392；冻结参数包含 BF16 2,922,286,192 和 FP32 1,049,216，保留官方内部混合 dtype。官方配置校验会为 OpenPI expert-only 自动启用 `use_orig_params=True`，不将此派生行为误记为手工修改 YAML。`verify_training.py` 将核对真实 optimizer update 数、有限梯度、冻结 VLM 与原始 SFT 完全相同、严格 full-weight 重载、DCP optimizer/scheduler 元数据和重载后 GPU 推理；有限阶段另要求官方 runner 的验证指标。optimizer resume 暂不列为已测试能力。

官方训练入口的 `--cfg job --resolve` 已在 CPU 上成功解析；本地训练负向门槛测试正确拒绝占用仍在执行 SFT 的 GPU。Ray 2.59 的真实新实例、CPU worker、官方 RLinf Cluster/node-probe/manager 启动通过，私有认证 token 路径也已实测。默认长 TMPDIR 曾使 dashboard 子进程出现 AF_UNIX path too long；保留失败日志后，将 TMPDIR 映射到短 ASCII 符号链接、文件仍写入本次私有 VEPFS tmp，真实 socket 绑定及调度器复测均通过。这些不代表 GPU train/val 生命周期通过。`continue_after_sft.py` 可等待现有评测自然结束，执行正式 SFT 验收，只有通过才启动独立本地生命周期测试；它不提交云端任务。历史接续记录为 `reports/sft_followup_r3.json`，该进程已结束；后续本地有限尝试的原始错误与结果均保留，r8 完整同快照生命周期已通过。SFT 评测未因这些封装修正而重跑。正式验收还会重算 585 个资产 hash 与原始 episode 文件 hash，供最终配对报告核验。若正式 SR 偏差或训练门槛失败，保留诊断，不自动放宽阈值或换配置重试。

### E3. 正式启动命令及退出状态

已实现的 `launch_training.py` 调用固定快照内的官方 Python 入口，并直接保留 subprocess 返回码。启动前重算源码/config hash、SFT 权重与原始评测验收；正式阶段还要求本地与云端生命周期报告与同一源码 hash 对应。它设置私有 LIBERO、编译缓存、短 ASCII Ray 临时目录和 `RAY_ADDRESS=local`，避免连接其他实验的 Ray。下面命令只有满足前置门槛后才会运行：

```bash
PIRL_SNAPSHOT="$PIRL_RUN/source_snapshot-r8"
"$PIRL_ENV/bin/python" "$PIRL_SNAPSHOT/harness/launch_training.py" \
  --run "$PIRL_RUN" --phase local-smoke
# 云端 preflight 和正式任务由各自 YAML 调用：
"$PIRL_ENV/bin/python" "$PIRL_SNAPSHOT/harness/launch_training.py" \
  --run "$PIRL_RUN" --phase cloud-preflight
"$PIRL_ENV/bin/python" "$PIRL_SNAPSHOT/harness/launch_training.py" \
  --run "$PIRL_RUN" --phase formal
```

日志与运行状态写入 `train/<phase>/reports/`。同一 phase 已有 launch 记录时拒绝重复启动；失败需要保留该次证据并准备独立后续尝试，不能原地覆盖。

### E4. Volc 提交文件模板

具体提交文件已生成：`configs/volc-cloud-preflight.yaml` 和 `configs/volc-formal.yaml`，包含当前绝对路径和唯一任务名，hash 记录于训练快照 manifest。它们使用上表的队列、规格、候选镜像和挂载；preflight 在 2 个外层步后自然结束。候选镜像尚未通过本次实际云端兼容性测试，正式任务仍受此门槛约束。下列结构仅说明所用字段，实际提交以两份已冻结 YAML 为准。

```yaml
TaskName: __TASK_NAME__
Description: pi05 libero_spatial official GRPO Flow-SDE commit 30e0236
Entrypoint: 'bash -lc "exec __ENV__/bin/python __SNAPSHOT__/harness/launch_training.py --run __RUN_ROOT__ --phase formal"'
Framework: PyTorchDDP
ResourceQueueID: q-20260403140809-n4flj
UserCodePath: /vepfs-C区/jiajunlu/SimpleVLA-RL/scripts/volc_empty_code
Envs:
  - Name: NVIDIA_DRIVER_CAPABILITIES
    Value: compute,utility,graphics
    IsPrivate: false
TaskRoleSpecs:
  - RoleName: worker
    RoleReplicas: 1
    Flavor: ml.hpcpni2l.28xlarge
ImageUrl: vemlp-cn-beijing.cr.volces.com/preset-images/pytorch:2.10.0-cu12.9.1-py3.13-ubuntu24.04
Storages:
  - Type: Vepfs
    MountPath: /vepfs-C区
    ReadOnly: false
    SubPath: ''
    VepfsId: vepfs-cnbjba6dad11eac7
AccessType: Private
```

正式提交与只读检查命令：

```bash
"$PIRL_ENV/bin/python" "$PIRL_SNAPSHOT/harness/submit_training.py" \
  --run "$PIRL_RUN" --phase cloud-preflight \
  --volc "$PIRL_RUN/runtime/volc_cli_compatible"
"$PIRL_ENV/bin/python" "$PIRL_SNAPSHOT/harness/submit_training.py" \
  --run "$PIRL_RUN" --phase formal \
  --volc "$PIRL_RUN/runtime/volc_cli_compatible"
/root/.volc/bin/volc ml_task get -i "$PIRL_TASK_ID" --output json
/root/.volc/bin/volc ml_task instance list -i "$PIRL_TASK_ID" --output json
/root/.volc/bin/volc ml_task logs -t "$PIRL_TASK_ID" -i worker-0 -l 200
```

本次 Volc API 实测拒绝 `Limit=1000`，已用独立只读查询兼容适配器将该 list 参数转换为 100，其他参数与 submit 原样交给原 CLI；适配器路径与 SHA256 见 `reports/volc_query_compatibility.json`，固定训练快照未改动。仓库内后续提交工具也已修正默认查询上限。提交后列表曾短暂不可见，随后以任务 ID 与完整名称交叉确认，原始 intent/CLI 输出保留，没有重复提交。当前 preflight 名称为 `pi05-spatial-grpo-official-30e0236-20261025t144123z-cloud-preflight`，ID 为 `t-20261026030926-4vrmg`，回执位于 `train/cloud-preflight/submission/submission.json`。

提交封装先重新检查启动门槛，再按唯一名称查询包括终态在内的已有任务，并在实际 submit 前持久化 intent。保存 CLI 原始输出、任务 ID、任务名和实际镜像/实例信息；若返回不确定，保留 intent 并只读核对，不自动再次 submit。无需把账号密钥写入文档、启动脚本或提交 YAML。正式提交的负向门槛测试已确认：SFT 未验收时在调用 Volc 提交 API 前拒绝执行。

### E5. 训练期间记录

- 前几轮记录编译和初始化耗时，随后至少 10 个稳定轮记录 rollout、更新、同步、保存的耗时，以及 GPU 峰值显存、CPU/RAM、有效样本比例。
- 记录训练 reward/success、advantage 分布、ratio/clip fraction、loss、grad norm、过滤比例和实际 optimizer step 数。训练 rollout 的 success 不能替代独立评测 SR。
- 使用官方 1000 epochs 上限和每 40 轮保存。`val_check_interval=-1` 表示默认训练内评测关闭；后续评测独立进行，不为画曲线静默改变训练配置。
- 保留每个 checkpoint 的 epoch/global step、保存完整性和 hash。可以在本地两卡评测已经完整落盘的 checkpoint，但不得读半写入文件，也不能让共享盘 I/O 严重影响其他实验。

## 9. 阶段 F：训练后评测与结果报告

封装已实现为快照内 `harness/run_final_evaluation.py` 与 `harness/compare_results.py`。前者只接受固定官方 1000 步末尾 checkpoint，要求真实训练/保存/重载证据通过；在指向同一官方源码快照的私有评测 view 中，调用原封不动的冻结评测脚本，依次执行 10 条 smoke 和两协议各 500 条完整评测。后者检查 SFT 原始 JSONL 与验收时 SHA256 相同，按 task/init-state 配对，核对协议、seed、noise/env seed、3/5 与 settling，输出总体/逐任务 SR、Wilson 区间、新增成功与退化 episode。上述能力目前只有 CPU 契约测试；正式 RL checkpoint 与最终结果仍未生成。

```bash
"$PIRL_ENV/bin/python" "$PIRL_SNAPSHOT/harness/run_final_evaluation.py" \
  --run "$PIRL_RUN"
```

1. 预先指定主结果为官方预算结束时的最终完整 checkpoint；同时可报告按固定评测频率得到的 best checkpoint，但必须标明选择方式，不能只报告最好的一次。
   论文 C.1 明确每套使用 500 个初始状态评测，但正文与附录未明确披露 Table 9 的 checkpoint 评测频率及 best/final 选择规则；不能据此声称“每 40 轮评测选 best”是原论文做法。当前官方 YAML 的 `save_interval=40` 只是保存频率，`val_check_interval=-1` 关闭训练内评测。本轮以最终 checkpoint 为主结果，暂不追加全程 best 搜索；若后续增加，需预先登记评测频率与选择规则。
2. 在本地两卡用 C 阶段同一套 episode 清单、种子、3 次去噪、5 步执行、相同观测/动作处理完成 Spatial 500 条评测。分别报告 240-step 主协议和 220-step 补充协议。
3. 原生 RLinf checkpoint 通常在 `actor/model_state_dict/full_weights.pt`。保持 SFT 目录作为模型基础资产/norm stats 来源，再通过统一评测的 `runner.ckpt_path` 加载 RL 权重；必须核验实际加载的是目标 checkpoint。

   ```bash
   cd "$PIRL_REPO"
  CUDA_VISIBLE_DEVICES=0,1 bash evaluations/run_eval.sh \
    libero libero_spatial_openpi_pi05_eval \
    rollout.model.model_path="$PIRL_SFT" \
    runner.ckpt_path="$PIRL_RL_FULL_WEIGHTS" \
    env.eval.video_cfg.save_video=false
   ```

   上述命令用于说明官方加载入口；本地分片执行仍遵守 C3。新版原生 `full_weights.pt` 不能直接当成 standalone 脚本要求的外部 OpenPI `model.safetensors`。220-step 补充评测优先在已验收的原生模型评测封装中复用 standalone 环境协议；若必须转换权重，要单独验证 key 映射和固定输入输出一致性，不能只更改文件名。

4. 汇总 SFT → GRPO 的 SR 和百分点提升、逐任务 SR、配对 episode 成败变化、95% 区间、实际 GPU 小时和存储消耗。论文参考提升为 Spatial 84.6% → 97.4%，即 +12.8 个百分点；这是参考而非成功保证。
5. Object、Goal、Long 不属于本轮交付。若未来另行开展其他套件测试，明确这些是 Spatial-only RL 的迁移/保留能力测试，分别使用预先固定且适配该套件的评测配置；不能把 Spatial 的参数直接复制到所有套件，也不能把缺失的 Goal 评测配置名当作已有文件。

建议结果表：

| checkpoint | 协议 / seed | N | Spatial SR | 相对本地 SFT 提升 | 论文参考 | 代码 / 权重 hash |
|---|---|---:|---:|---:|---:|---|
| 官方 SFT | recipe-240 / 0 | 500 | **85.6% (428/500)** | — | 84.6% | 固定 commit / SFT SHA256 见运行 manifest |
| 最终 GRPO | recipe-240 / 0 | 500 | 待测 | 待测 | 97.4% | 待填 |
| 官方 SFT | standalone-220 / 7 | 500 | **84.2% (421/500)** | — | 84.6% | 同一固定 commit / SFT SHA256 |
| 最终 GRPO | standalone-220 / 7 | 500 | 待测 | 待测 | 97.4% | 待填 |

以上协议标签包含不同默认 seed，因此该表不用于单独估计 220→240 的因果影响。若需分析时间上限影响，另做相同 seed、相同模型路径的配对评测。

## 10. 算力、时间与存储预算

| 阶段 | 资源 | 时间预算方法 |
|---|---|---|
| 代码、依赖、权重准备 | CPU、网络、共享盘 | 缓存命中时较短；编译依赖/镜像适配可能成为主要耗时，执行后记录 |
| SFT 双卡冒烟 | 本地 2 × A100 80GB | 先测 10 条；不把首次 compile 计入稳态吞吐 |
| Spatial 正式 SFT 评测 | 本地 2 × A100 80GB | 两种协议各 500 条；以 20–50 条预跑吞吐外推，再留初始化/失败重试余量 |
| Spatial GRPO | Volc 单节点 8 × A800 80GB | 官方上限 1000 epochs；按云端稳定轮时间外推，排队时间另计 |
| 训练后评测 | 本地 2 × A100 80GB | 与 SFT 相同协议、相同样本数 |

本地 r8 的实测记录已落盘为 `reports/local_resource_measurement.json`：有限 1 步包含 16 env × 8 rollout，官方 TensorBoard 记录采样 425.73 秒、actor 更新 138.41 秒；从 collective 探针前到训练子进程退出共 741.10 秒。完整本地 checkpoint 的文件长度合计约 14.99 GiB，其中完整模型权重 6,946,523,365 字节。它们是双卡首次有限测试的观测，包含初始化/冷启动，不能直接作为 8 卡正式训练稳态吞吐或 DCP 存储承诺。云端仍需按实际启动时间、至少 10 个稳定轮和云端 checkpoint 重新计量，排队时间单列。

目前没有核验到“这个固定 commit + π0.5 Spatial GRPO + 8×A800”的官方完整 wall-clock 数字，不承诺某个小时数。现有 standalone README 提到单 H100 评测约 2 小时，但未把这里的全部模型/套件条件限定清楚，只可作为参考，不能直接换算本地双卡耗时。

执行阶段使用：

```text
T_eval ≈ 初始化耗时 + 目标 episode 数 / 实测双卡总吞吐
T_train ≈ 初始化编译耗时 + 1000 × 稳态每 epoch 耗时 + checkpoint 保存耗时
GPU-hours_train = 8 × 实际训练 wall-clock 小时
总日历时间 = 环境准备 + SFT 验收 + 排队 + 训练 + 最终评测
```

本轮暂不设置费用、GPU 小时或新增存储的预算上限，仍记录实测资源消耗与共享盘余量。预计保存约 25 个周期 checkpoint。环境、权重与 checkpoint 的初步规划空间为 **0.5–1.5 TiB**（不保存视频），最终按首个完整 checkpoint（含优化器/各 rank 状态）实测大小 × 保留数量重新预算；不能只按 SFT 权重体积计算。共享盘还有其他实验，不自动删除旧产物腾空间。

## 11. 分阶段交付与验收清单

- [x] A：官方独立检出和版本 manifest；保留旧 RLinf 与已有环境；资产复用通过。双卡 EGL/推理证据：当前 run 的 `reports/runtime_gpu{0,1}.json`。
- [x] B：官方 SFT 权重、norm stats、revision、SHA256 与加载检查完整。两张卡均加载 667/667 个 tensor，missing/unexpected 为空；固定观测/噪声重复动作差为 0。
- [x] C：本地双卡 Spatial SFT 500 条/协议；原始 episode 记录、逐任务统计、论文差值、协议差异与验收结论完整。见 `reports/sft_acceptance.json/.md`；两协议分别 85.6% / 84.2%。
- [x] D：官方配置继承展开、允许差异白名单、可训练参数清单和实际 dtype 完整。第 5 次 r8 真实两卡 post-FSDP 初始化记录位于 `train/local-smoke/reports/observer/`；完整本地生命周期通过，8 卡云端验收仍待完成。
- [ ] E-preflight：新增脚本与必要训练/保存/重载/评测生命周期测试完成，未验证项明确。
- [ ] E-submit：Volc 最终 YAML、源码快照、环境 manifest、提交回执与唯一任务 ID 完整。
- [ ] E-train：正式配置运行、checkpoint 可读取、实际资源和耗时记录完整。
- [ ] F：训练后同协议 Spatial 评测；SFT/RL/论文三方比较；失败任务和版本/硬件差异说明完整。

**完成定义：** 只有 SFT 复现验收、官方配置 GRPO、完整 checkpoint 和训练后同协议评测都完成，才算本计划执行完成。“任务提交成功”“训练 loss 正常”或“训练 rollout SR 很高”都不能替代最终验收。

## 12. 主要依据

- [RLinf 固定官方版本](https://github.com/RLinf/RLinf/tree/30e0236981c6a1dc19202de64116834c12673828)
- [Spatial π0.5 GRPO 配置](https://github.com/RLinf/RLinf/blob/30e0236981c6a1dc19202de64116834c12673828/examples/embodiment/config/libero_spatial_grpo_openpi_pi05.yaml)
- [π0.5 模型配置](https://github.com/RLinf/RLinf/blob/30e0236981c6a1dc19202de64116834c12673828/examples/embodiment/config/model/pi0_5.yaml)
- [Spatial 环境配置](https://github.com/RLinf/RLinf/blob/30e0236981c6a1dc19202de64116834c12673828/examples/embodiment/config/env/libero_spatial.yaml)
- [Spatial π0.5 统一评测配置](https://github.com/RLinf/RLinf/blob/30e0236981c6a1dc19202de64116834c12673828/evaluations/libero/libero_spatial_openpi_pi05_eval.yaml)
- [EmbodiedRunner：外层 RL step 与 epoch 定义](https://github.com/RLinf/RLinf/blob/30e0236981c6a1dc19202de64116834c12673828/rlinf/runners/embodied_runner.py)
- [EmbodiedFSDPActor：reward mask、batch 展开与 optimizer update](https://github.com/RLinf/RLinf/blob/30e0236981c6a1dc19202de64116834c12673828/rlinf/workers/actor/embodied_fsdp_actor_worker.py)
- [EnvWorker：rollout epoch 与 chunk 循环](https://github.com/RLinf/RLinf/blob/30e0236981c6a1dc19202de64116834c12673828/rlinf/workers/env/env_worker.py)
- [LIBERO：组内 reset-state 复用](https://github.com/RLinf/RLinf/blob/30e0236981c6a1dc19202de64116834c12673828/rlinf/envs/sim/libero/libero_env.py)
- [官方安装脚本](https://github.com/RLinf/RLinf/blob/30e0236981c6a1dc19202de64116834c12673828/requirements/install.sh)
- [官方 standalone LIBERO 评测](https://github.com/RLinf/RLinf/blob/30e0236981c6a1dc19202de64116834c12673828/toolkits/standalone_eval_scripts/openpi/libero_eval.py) 与 [说明](https://github.com/RLinf/RLinf/blob/30e0236981c6a1dc19202de64116834c12673828/toolkits/standalone_eval_scripts/openpi/README.md)
- [OpenPI checkpoint 兼容加载](https://github.com/RLinf/RLinf/blob/30e0236981c6a1dc19202de64116834c12673828/rlinf/models/embodiment/openpi/checkpoint.py)
- [官方 SFT 权重](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT/tree/45ccfcc4e28634f1576ebf78cab0fbe2fd82432d)
- [πRL 论文 v3，Table 9 与实现设置](https://arxiv.org/html/2510.25889v3)

服务器资产、环境和旧代码状态来自本次只读检查；Volc 队列配置来自 2026-10-19 的既有交接记录，执行时必须重新验证有效性。
