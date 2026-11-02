# mani_v1 诊断与标量审计

启用 manifold 后，actor 在 shuffle 前记录本批真实 rollout 的分组、评分和优势。日志使用独立的 `manifold/` 命名空间；runner 会再添加阶段前缀，例如 `rollout/manifold/groups/raw` 和 `train/manifold/optimizer_updates`。原生 task reward、环境 SR 和任务计数继续保留。

## 配置

以下字段放在 `algorithm.manifold` 下，可省略整个 `diagnostics`：

```yaml
diagnostics:
  audit_dir: null
  dimension_metrics: false
  debug_hidden_max_trajectories: 0
```

`audit_dir: null` 将审计保存在 `${runner.logger.log_path}/${runner.logger.experiment_name}/manifold/rank_00000/` 等 rank 子目录。自定义路径仍按 rank 分目录。`dimension_metrics` 控制 V 评分是否额外尝试 D/G/R；G 评分始终需要 D。V 评分中缺失 TLE 或 D 无效不会屏蔽有效 V。

默认不保存高维 hidden。`debug_hidden_max_trajectories` 为每个 rank 审计目录设置累计数量上限；重启同一目录仍计入已有文件。调试 NPZ 保存当次实际 hidden 的可用前缀、身份、点数和 hash，若轨迹不足 K，会明确标记不是完整评分前缀。不要用调试文件数量推断原始采样数量。

disabled 不写审计或采集 hidden；显式配置 manifold 时仍记录 mode/score/eligible_groups 标签、真实全局 raw/eligible 组数和零 hidden 字节数。

## 日志字段

以下路径均省略 `manifold/` 前缀。计数和范数先跨评分 rank 汇总充分统计量，再计算比率或均值；不会直接平均各 rank 的均值。

| 字段 | 含义 |
| --- | --- |
| `enabled`、`mode/<name>`、`score/<name>`、`eligible_groups/<name>` | 本轮声明的模式、评分与组选择器；枚举以数值 one-hot 标签记录 |
| `groups/raw`、`groups/eligible` | 原始组数与实际通过组选择 mask 的组数 |
| `groups/mixed`、`groups/all_success`、`groups/all_failed` | 按真实首次 episode 的成功标签分类 |
| `groups/valid`、`groups/constant`、`groups/invalid` | 所选评分的组级状态；全失败另计 |
| `groups/reason/<reason>` | 全失败、常数、短失败轨迹或数值失效等原因 |
| `groups/geometry` | 通过组选择、评分有效且非常数的组数 |
| `groups/signal` | 最终 masked advantage 至少有一个非零值的组数 |
| `groups/success_<n>_of_<G>` | 原始组成功人数直方图 |
| `K/*`、`episode_actions/*` | 组前缀长度与成员首次 episode 有效动作数的分布 |
| `K/bin_*` | 不重叠区间：0–10、11–50、51–100、101–200、201–480、>480 |
| `D/*`、`V/*`、`G/*`、`R/*` | 独立状态计数、原因和有效值分布；R 为 D/V |
| `selected_score/*` | 按配置选择 G 或 V 后，成员有效评分的分布 |
| `score_group_mean/*`、`score_group_std/*` | 完整组所选评分的均值与 sample std 的分布 |
| `chunks/native`、`chunks/selected` | 原生动作有效 mask 与叠加组选择后的 loss mask 中的 chunk 数 |
| `chunks/prefix` | 原始完整 chunk 前缀与动作有效 mask 的交集，不要求数值评分有效 |
| `chunks/geometry`、`geometry_chunk_fraction` | eligible 且评分有效的完整前缀 chunk 数，以及它占 native chunk 的比例 |
| `advantage/task_l2`、`geometry_l2`、`final_l2` | loss mask 上的全局 L2 范数；geometry 已包含配置权重 |
| `advantage/paired`、`same_sign`、`same_sign_fraction` | task/geometry 同时非零的 chunk 数、其中同号数及其比率；没有配对时比率为 0 |
| `geometry_seconds/*`、`audit_seconds/*` | 每 rank 评分、审计写入耗时分布；sum 是各 rank 工作时间之和，max 是最慢 rank 时间 |
| `hidden_bytes` | 评分时 CPU hidden 张量的实际字节数之和，不是 CPU 峰值内存 |
| `optimizer_updates`、`optimizer_skips` | 每 rank 一致的实际更新边界数与全局无信号跳过数，runner 按原生训练指标汇总 |

分布字段使用 `count/sum/mean/min/max/std`，其中 std 的 correction 为 1，单样本 std 为 0。没有有效样本时只记录 count=0，不写虚假的 mean。指标状态包括 `valid`、`invalid`、`disabled`、`unavailable`、`not_computed`。例如 V 模式未请求 D 时为 disabled；全失败没有 K 时为 not_computed。

成员级 D/V/G/R 与 selected_score 分布只接收各自有效数值，即使该成员所在组因另一成员失效而无几何更新。这仅用于诊断：组标准化仍要求全部成员有效，绝不删除失效成员后重算优势。`groups/eligible` 也不等于 `groups/geometry` 或 `groups/signal`；例如无效 mixed 组在混合模式仍保留 task 信号。

可检查两条守恒式：

```text
raw = mixed + all_success + all_failed
raw = valid + constant + invalid + all_failed
```

## 审计与重放

每次评分创建独立 `step_<iteration>_<uuid>.jsonl`。首条 `batch` 记录保存配置口径、布局、group/version/collector/epoch/source 身份、每个 member 的 task/trial/reset-state、长度、成功、K、D/V/G/R 独立状态、所选评分统计量、z-score、原生 task advantage 与 native 有效 chunk 数。训练完成后，同一文件追加 `updates` 记录，给出实际 optimizer 更新、跳过、累计 step 和 scheduler 是否推进。

`provenance_<sha256>.json` 保存完整 resolved 配置、实际源码根目录、HEAD（该源码根存在 `.git` 时）、关键文件 hash 和依赖版本。文件内容寻址且不可覆写；代码指纹覆盖 manifold 核心及 actor、rollout、env、schema、配置和 native loss/advantage 等关键实现。C07 负责完整源码快照与运行时导入来源校验。

可在相同实现版本下仅使用标量重放，无需模型或 hidden：

```python
import json
from pathlib import Path
from rlinf.algorithms.manifold.audit import replay_advantages

path = Path("<audit_dir>/rank_00000/step_<iteration>_<uuid>.jsonl")
record = json.loads(path.read_text().splitlines()[0])
advantages = replay_advantages(record)
```

重放会从成功标签和长度重建类别、K 和完整 chunk mask，从原始标量评分重建组标准化，并核对存储的统计量、资格和 z-score。返回 task、geometry、final、action/group/prefix/loss mask。它不能替代 hidden 到 D/V/G 的数值复算；后者需要有上限的调试 NPZ。
