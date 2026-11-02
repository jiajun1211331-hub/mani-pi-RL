# mani_v1 源码准备与启动

mani_v1 使用独立入口冻结当前迁移代码。原有 `scripts/pi_rl/prepare_training.py` 和 `launch_training.py` 继续服务于官方版本复现，版本约束和准入规则保持不变。

## 离线准备

在目标 Python 环境和仓库根目录执行：

```bash
python -m scripts.pi_rl.mani_v1.prepare_training \
  --source-repo "$PWD" \
  --run "$PWD/experiments/<unique-run>" \
  --model /absolute/path/to/RLinf-Pi05-LIBERO-SFT \
  --config-name <recipe>
```

默认要求仓库已提交且干净；如需检查尚未提交的实现，显式添加 `--allow-dirty`。开发态快照会标记 `development_snapshot: true`，不能用 HEAD 冒充本批实际代码。准备阶段只复制文件、解析配置并计算指纹，不启动训练或提交云任务。

源码范围包括 `rlinf/`、`examples/`、`scripts/`、`requirements/`、`ray_utils/`、`toolkits/`、`evaluations/`，以及根目录的 pyproject、uv.lock、LICENSE 和 README。该范围内被跟踪的当前内容及非忽略新增文件都会复制；已删除文件保持删除。不复制 Git 元数据、模型权重、Python 字节码和研究实验输出。源码范围外的文件清单仍记录在 manifest；工作区 diff 包含 HEAD 之后的 tracked 改动。新增文件的实际内容由快照和逐文件 hash 保留。外部或绝对源码符号链接会被拒绝。

`--run` 必须是新目录，并放在源码范围外；仓库内推荐使用忽略的 `experiments/`。失败后保留证据，不覆盖旧目录。正式运行优先使用已提交、干净源码，再重新准备一个目录。

配置通过快照内的代码在独立进程中 compose 并 resolve，启用模式接受 manifold 静态契约校验。native driver 在实际启动时继续执行完整配置/placement 校验。使用重复 `--override '<key>=<value>'` 传入 Hydra 覆盖；模型路径由 `--model` 指定，日志目录和实验名由 `--run` 派生，冲突覆盖会明确报错。

准备产物：

| 路径 | 内容 |
| --- | --- |
| `source/` | 实际运行源码，包含几何核心和 actor 接入实现 |
| `configs/train.yaml` | 不含未解析引用的训练配置；日志位于本 run 的 `train/logs/` |
| `manifests/source.json` | 原仓库、HEAD、提交序列、status、diff hash、源码文件清单/hash、tree hash、开发态标记与配置/模型/环境清单 hash |
| `manifests/worktree.diff` | tracked 内容相对 HEAD 的 binary diff，不更改 index 或提交 |
| `manifests/model.json` | 外部模型的绝对路径、文件大小与 SHA-256，包括完整权重内容；排除模型目录的 `.cache` |
| `manifests/environment.json` | 实际 Python executable/prefix/version 与安装包版本 |

权重只做内容哈希，不复制到源码快照。准备和启动检查都要读取完整权重，耗时与存储吞吐有关。

## 检查与运行

使用准备时的 Python，并从运行目录选择冻结源码：

```bash
cd /absolute/path/to/run
PYTHONPATH="$PWD/source" python \
  source/scripts/pi_rl/mani_v1/launch_training.py \
  --run "$PWD" --check-only
```

检查会核对源码文件清单、内容/符号链接/执行权限、配置、环境和外部模型身份，并记录本进程实际加载的 RLinf 路径与 hash。文件新增、丢失或修改都会失败。`--check-only` 不创建训练 attempt，也不启动 Ray。只有冻结入口可以启动，工作区入口会被拒绝。

实际运行使用相同命令并去掉 `--check-only`。GPU 数量、`CUDA_VISIBLE_DEVICES`、LIBERO 资产、tokenizer/cache 和临时目录等运行环境由调用者设置；启动器不申请云资源。它显式启动本地 Ray，不连接已有集群，也不执行 machine-wide 清理。首版为共享源码目录上的单节点同步部署。

启动器固定使用准备好的配置，不接受启动时数值覆盖；需要修改配置时准备新 run。`runtime/launch.json` 防止同一 run 被意外重复启动。driver 和 actor/rollout/env 初始化完成后，各自检查已加载 RLinf 模块的真实路径及 hash，写入 `runtime/source_reports/`。若导入落到工作区或另一个安装位置，会明确失败。

manifold 标量审计的 provenance 同时包含源码快照 tree hash、原 HEAD 和开发态标记，可与这些进程报告关联。`--check-only` 和离线导入检查不能替代真实 worker 或完整训练生命周期验证；后者属于 C09。
