# Codec V2 算法设计（易读版）

## 缩写与术语

首次阅读本文时，可先使用下表；正文在首次技术性使用时也遵循这些全称。

| 缩写 | 全称 | 本文含义 |
|---|---|---|
| MIDI | Musical Instrument Digital Interface（乐器数字接口） | 本文处理的音乐事件文件/协议。 |
| SMF | Standard MIDI File（标准 MIDI 文件） | MIDI 文件格式；其 raw track/tick 是目标 parser 的输入事实。 |
| QL | Quarter Length（四分音符长度） | 音乐时间单位；`1.0 QL` 等于一个四分音符。 |
| TS | Time Signature（拍号） | 决定 canonical bar 长度的 meta event。 |
| PPQN | Pulses Per Quarter Note（每四分音符 tick 数） | SMF 的 tick timing division。 |
| FIFO | First In, First Out（先进先出） | 重叠同 pitch Note On/Off 的配对顺序。 |
| SMPTE | Society of Motion Picture and Television Engineers | 另一种 MIDI timing division；首版不支持。 |
| JSON | JavaScript Object Notation | form metadata 的文件编码格式。 |
| NPZ | NumPy Zipped Archive | `voice_tensors.npz` 的归档格式。 |
| CC64 | MIDI Continuous Controller 64（延音踏板控制器） | 只采集/评估的 pedal 控制信息，不进入 V2 tensor。 |
| SHA-256 | Secure Hash Algorithm 256-bit | artifact/content hash 使用的散列算法。 |

## 先用一分钟理解它

Codec V2 可以理解为把 MIDI（Musical Instrument Digital Interface，乐器数字接口）变成一叠固定大小的“音乐表格”。

```text
一首 MIDI
  → 一个个已验证的小节（bar）
    → 每个小节最多 C 个短时间格（slot；C 由本次 encoding run 的配置冻结）
      → 每格 18 个语义位置（lane）
        → 每个位置用 6 个数描述
```

因此，一个小节的主张量 shape 为：

```text
[18, C, 6]
```

一个数据集有 `N` 个小节时：

```text
voice_tensors: float32 [N, 18, C, 6]
```

这里的 18 个位置不是 MIDI track。它们是稳定的音乐角色：

```text
lane 0       melody：当前最符合旋律语义的一个音
lane 1..16   harmony：当前和声集合中的最多 16 个音
lane 17      bass：当前最低音语义的一个音
```

Codec 的目标是让模型不必先理解每个 MIDI 的 track 排列，也能看到“旋律、和声、低音”
这些稳定角色；无法无损表示时明确失败，不静默丢音。

> `raw_smf_v1` 已是正式实现的 parser 权威。它从 PPQN SMF 的 raw tick 构建已验证的
> `CanonicalBarSpan`，再生成 `BarRecord`；production runtime 没有 music21
> Part/Measure 的回退路径，也不会猜测小节边界。

## 独立设计：可配置的 Slot Capacity（run 内冻结）

### 这项设计解决什么问题

`0.25 QL` 是 V2 的时间分辨率；`slot_capacity` 是一个 bar 最多可表达多少个这样的时间格。两者
不能混为一谈。此前的 `48` 只是一个未发布实验的容量值，等于最多 `12.0 QL`，不是 V2 语义本身。

真实数据已经证明容量不能继续固定在代码里：例如 `20/4 = 20.0 QL` 需要 80 个 slot，`41/8 =
20.5 QL` 需要 82 个 slot。它们是合法的 canonical bar，不可因容量不足被截断、拆成伪 bar 或改写
拍号。

本设计因此冻结以下边界：**时间量化仍固定为 `0.25 QL`；slot capacity 改为 encoding-run 配置；同
一个 run 和其训练输入中的全部 tensor 必须使用同一个 capacity。**

### 配置与当前 Beethoven profile

所有正式编码都必须显式声明配置，不能让 CLI 依赖代码中的默认 `48`：

```yaml
bar_tensor:
  schema_version: bar_tensor_schema.v2
  slot_grid:
    quantum_ql: 0.25
    capacity: 92
    epsilon_ql: 0.000001
```

这里 `92 × 0.25 = 23.0 QL`。`92` 是当前 Beethoven profile 的冻结选择，覆盖已发现最大的 82-slot
bar，并留出 10 个 slot 的余量。它不是“所有数据集永远必须用 92”：另一个 dataset 可以在**新 encoding
run、新 artifact、新训练 run**中选取不同的正整数 capacity；但不得把不同 capacity 的样本拼进同一
训练 dataset。

`capacity` 不要求是 16、32、48 的倍数。唯一数学要求是正整数，实际可表达最长 bar 为
`capacity × quantum_ql`。一个 bar 的 valid slot count 仍为：

```text
required_slot_count = max(1, ceil((bar_length_ql - epsilon_ql) / quantum_ql))
```

若 `required_slot_count > capacity`，codec 抛出 `CodecCapacityError` / `slot_capacity_exceeded`；绝不
自动截断、拆分、降低时间分辨率或重写拍号。异常必须至少记录：`song_id`、source file identity、file
path、tune index、canonical bar index、start/end tick、meter、bar length、quantum、required slot count 与
configured capacity。这是 **codec representability failure**，不是 parser failure：parser 正确识别了
该真实 bar。

### 一个 run 内为何不能“动态改变 shape”

同一个 run 中，所有输出必须是：

```text
voice_tensors:      float32 [N, 18, C, 6]
slot_valid_mask:    bool    [N, C]
slot_durations_ql:  float32 [N, C]
```

短 bar 仅使用前面的 valid slots；剩余位置是 padding，绝不是 rest。因而 4/4 在 C=92 时仍只有 slot
`0..15` 为真实音乐时间，`16..91` 都是 padding；20/4 使用 `0..79`，41/8 使用 `0..81`。

DVAE（Discrete Variational Autoencoder，离散变分自编码器）训练入口可以从 manifest 或第一批数据**读取
并验证** C，但不能在同一次训练中随 batch 改变 C：输入层、位置编码、DiT（Diffusion Transformer，扩散
Transformer）token layout、attention mask 和 loss mask 都在模型初始化时依赖这个 shape。训练侧必须
fail-fast 校验所有输入与 manifest 的 `quantum_ql`、`capacity`、lane/feature 定义及配置 hash 一致，并在
训练日志记录 `slot_capacity` 和 encoding manifest SHA-256。

### 实施规则：配置到 artifact、evaluation、训练的单一传播链

1. `SlotGrid.for_bar()` 必须接收 `quantum_ql`、`capacity`、`epsilon_ql`；模块内不得再有时间容量字面量
   `48`。track safety limit 的 `48` 与 slot capacity 无关，可以保留。
2. codec、empty-bar 初始化和 padding 均由 `SlotGrid.capacity` 分配，不能按 `48` 创建 array。
3. `encoding_manifest.json.slot_grid_policy` 必须写入这三个值；它们必须参与
   `configuration_sha256`。`voice_tensors.npz` descriptor、index 的 `voice_tensor_shape`、mask/duration
   shape 必须与 C 一致。
4. 所有 capture、exporter、evaluator 从 manifest 读取 C，校验 tensor axis 2、mask/duration axis 1、
   valid count 与 duration sum；不得假定 48。
5. 训练 dataset loader 在打开 archive 时读取 manifest，以 C 构造模型；首批和后续 batch 都校验
   `[18,C,6]`。不同 C、不同 quantum 或不同 feature/lane 定义的 artifact 一律拒绝混训。
6. 任何更改 C 的实验都使用新的 output/run id，完整重编码；旧的 `[18,48,6]` 未发布 artifact 不得与
   `[18,92,6]` artifact 混用。
7. 必须有配置传播、92-slot 20/4、92-slot 41/8、普通 4/4 padding、以及超出 92 的结构化 failure 回归
   测试；E2E 需验证 manifest、index、NPZ、evaluation 和训练 loader 读取到同一 C。

## Canonical Raw-SMF Parser：已实施的正式权威

### 阅读状态：哪些已经存在，哪些是待实施目标

本文有两个不同层次，必须分开阅读：

| 层次 | 状态 | 含义 |
|---|---|---|
| Codec V2 张量、`SlotGrid`、18 lanes、6D feature、可配置 slot capacity | **已实施** | lane/feature 语义为 `bar_tensor_schema.v2`；C 是本 run 显式冻结的配置。 |
| raw MIDI → `CanonicalBarSpan[]` 的 parser | **已实施：`raw_smf_v1`** | 是唯一 production bar authority。 |
| controls、form metadata、diagnostics、CLI acceptance | **边界已实施** | controls 只供 evaluation；legacy measure index form metadata fail-closed。 |

正式 parser 已直接读取 raw SMF event；标准 MIDI 并没有 Part 或 Measure 的物理约束，
因此 `canonical timeline` 是唯一 bar 坐标系，music21 Measure 不在 production codec
runtime 中。

```text
raw SMF（Standard MIDI File，标准 MIDI 文件）header + raw tracks
  → raw TS（Time Signature，拍号）/ note / control facts（tick 是时间权威）
  → resolved Time Signature chain
  → gap-free CanonicalBarSpan[]
  → absolute-time quantization、clip 到 bar
  → 不改变的 V2 slot / lane / feature codec
  → canonical artifact + evaluation-framework diagnostics
```

以下是已经冻结并实施的规则；production code 不得以 convenience behavior 替代。

### 1. 输入范围、时间单位和原始事实

#### 1.1 支持的 SMF 输入

Codec V2 production encoding 只接受 PPQN（Pulses Per Quarter Note，每四分音符 tick 数）
timing 的 SMF format 0 与 format 1：

```text
format 0: 一个 raw track；physical_track_index = 0
format 1: 每个 SMF track 保留其原始零基 index
format 2: smf_format_2_unsupported，fail-fast
SMPTE（Society of Motion Picture and Television Engineers）division: smpte_division_unsupported，fail-fast
```

format 2 的 tracks 是独立 pattern/tune，而不是同一首歌的并行 part；不能把它偷偷当成
format 1。当前 production parser 对每个 Type 0/1 source file 只产生一个 `SongRecord`，
并固定 `tune_index=0`。SMF Type 2 与 music21 `Opus` 不在支持范围；同一 source file 的
multi-tune identity 必须等独立 parser/API 设计完成后才能启用。

ABC、KRN、MusicXML 和其它 symbolic score format 不属于 V2 canonical parser 支持范围；
它们不是“解析失败的 MIDI”。`encoding_manifest.json` 必须记录
`supported_source_formats=["smf_ppqn_type_0", "smf_ppqn_type_1"]` 与
`canonical_parser_version="raw_smf_v1"`。

`physical_track_index` 永远是 SMF 的原始 track index。即使 Track 0 只有 tempo/TS、即使
某条 track 没有音符、即使某条 track 后来被 retention 排除，也不得重新编号。

Track 0 并非自动权威：TS 可以只写在某条 music track，也可以在多个 track 冗余出现。
48-track safety limit 只统计**成功配对出至少一个 note 的 physical track**；conductor-only
track 不占这个限额，但其 TS/tempo/key/CC64 事实仍必须读取并保留 provenance。

#### 1.2 精确时间坐标

raw tick 是 parser 进行 TS 合法性、排序和 note pairing 的唯一权威。对 PPQN `P`：

```text
raw_ql = Fraction(absolute_tick, P)
```

这里的 `Fraction` 表示精确有理数，不是 float。转换成 float QL 只用于最终 artifact、
显示和已有 API；`EPSILON_QL = 1e-6` 不能用于判断 TS 是否在 bar boundary。

每个 raw event 至少记录：

| Fact | 必须字段 | 说明 |
|---|---|---|
| `RawTimeSignatureFact` | `physical_track_index`, `event_ordinal`, `absolute_tick`, `numerator`, `denominator_exponent`, `denominator` | `denominator=2^denominator_exponent`；保留原始和解释后的值。 |
| `RawNoteEvent` | track、channel、event ordinal、tick、kind、pitch、velocity | 是未配对的 MIDI event，不是最终 note。 |
| `RawSourceNote` | track、channel、file-scoped `source_note_ordinal`、start/end tick、start/end QL、pitch、velocity | 是成功配对后的稳定 source note。 |
| `CanonicalBarSpan` | `canonical_bar_index`、start/end tick 与 QL、meter、TS provenance | 是唯一能进入 V2 codec 的 bar。 |

建议 TS 验证为：`1 <= numerator <= 255`、`0 <= denominator_exponent <= 7`，因此分母只可为
`1, 2, 4, ..., 128`。不合法值报 `time_signature_invalid`；合法但一个 bar 需要超过本 run 配置
的 C 个 `0.25-QL` slot 时，parser 仍可构建 span，但 codec 必须报 `slot_capacity_exceeded`，不能改拍号。

### 2. Time Signature chain：如何得到唯一全局小节线

#### 2.1 收集、合并和冲突

1. 按每条 raw track 自己的 delta ticks 累加，得到每个 TS meta event 的 `absolute_tick`。
2. 对所有 tracks 的 TS facts 按 `(absolute_tick, physical_track_index, event_ordinal)` 排序。
3. 对同一 tick：
   - 所有 `(numerator, denominator_exponent)` 相同：合并成一个 resolved TS；
     保留全部原始 fact 的 `(track, ordinal)` 列表，`duplicate_merge_count += fact_count - 1`。
   - 存在不同 meter：报 `time_signature_conflict`；诊断必须列出该 tick 的全部原始 facts，
     不区分“同一 track 冲突”或“跨 track 冲突”的严重性。
4. 正式训练默认策略为 `initial_time_signature_policy=error`：tick 0 必须恰好有一个
   resolved TS；若不存在，报 `time_signature_initial_missing`。
5. 显式兼容策略为 `smf_default_4_4`。仅当 tick 0 没有 TS 时（包括整个 SMF 完全没有
   TS event），在 tick 0 注入 synthetic 4/4。该注入绝不静默：provenance 必须写入
   `origin="smf_default"`、`injected_at_tick=0`、`meter="4/4"`，以及第一个真实
   TS 的 tick；若不存在真实 TS，则 `first_real_ts_tick=null`。parser-integrity 必须报告
   policy、origin count 和全部 injection source。兼容策略不改变正式训练默认边界。

因此，“第一个 TS 在 tick 960”在默认 `error` 下不是可编码输入；在显式兼容模式下，
`[0,960)` 使用带 provenance 的 synthetic 4/4，而非由实现者私下猜测。

#### 2.2 用精确 tick 生成 TS segment

某个 resolved TS 为 `(nn, dd)` 时，nominal bar length 是：

```text
bar_length_tick = Fraction(nn * 4 * PPQN, 2^dd)
bar_length_ql   = Fraction(nn * 4, 2^dd)
```

从 tick 0 开始，当前 segment 的 bar end 依次为：

```text
segment_start_tick + k * bar_length_tick,  k = 1, 2, ...
```

每个 resolved TS event 的精确 tick 都是下一段 canonical timeline 的起点，绝不因 float epsilon
被移动、snap 或拒绝。设当前 canonical cursor 为 `C`、当前 nominal old-meter bar length 为 `B`、
下一个 TS event 为 `T`；以精确 `Fraction` 重复生成完整 span：

```text
while C + B <= T:
    emit full CanonicalBarSpan [C, C + B), meter = old_meter
    C = C + B
```

随后分三种情况：

```text
C == T   → TS 恰在完整 bar boundary；不生成 partial bar。
C < T    → emit partial CanonicalBarSpan [C, T), meter = old_meter。
C > T    → 不可能；循环不得越过 T。
```

第二种叫 `canonical_partial_bar_before_ts_change`。它的 `is_partial=true`、
`partial_reason="time_signature_change"`、`nominal_meter=old_meter`，而新的 TS 从 `T` 精确开始
下一段 full-bar sequence。partial bar 的绝对 end 由 raw TS event 决定；它不是 guessed measure，
不是 pickup，也不是用 epsilon 吸附到相邻完整 bar。

例：当前 3/4 span 从 QL 1380 开始，raw 4/4 TS event 在 QL 1382：

```text
[1380, 1382)  → partial span，meter context = 3/4，实际长度 = 2.0 QL
[1382, 1386)  → 新的 4/4 full span
```

因此 `time_signature_mid_bar_change` 不再是 failure code。默认 `error` 策略下初始 TS 缺失、同 tick
冲突、非法 TS 数值仍 fail-fast；TS event 在 terminal bar 后仍记录 provenance，但不会单独 materialize trailing bar。

tempo 同 tick 的排序不影响本算法：tempo 用于秒级播放/诊断，TS 用于 QL bar grid。两者都
保留各自在 raw event stream 中的 ordinal，以便 provenance 审计。

所有 TS facts 均进入 diagnostics；即使 TS change 晚于 terminal bar，也记录其事实和链
验证结果，但不因它单独生成 trailing bar。

### 3. raw Note 配对与稳定身份

source note identity 必须在 quantization、排序、clip 和 role assignment 前产生：

```text
source_note_id = source_file_identity : physical_track_index : source_note_ordinal
```

对每条 physical track，按 `(absolute_tick, event_ordinal)` 处理 event stream，并为 key
`(track, channel, pitch)` 维护独立 FIFO（First In, First Out，先进先出）queue。
`source_note_ordinal` 在当前 one-file/one-song production boundary 内单调递增：先收集成功的
non-zero-velocity Note On，再按 `(absolute_tick, physical_track_index, event_ordinal)` 排序并编号
`0..K-1`。当前 `tune_index=0`。

具体配对步骤：

1. `NoteOn(pitch, velocity>0)`：为 key `(track, channel, pitch)` 的 FIFO queue 追加一条
   pending entry；其 file-scoped ordinal 由上面的全局排序决定。
2. `NoteOff`，或 `NoteOn(pitch, velocity=0)`：从同一 key 的 queue 取最早 pending entry，
   组成一条 `RawSourceNote`。
3. queue 为空时出现 off：按第 3.1 节的 `redundant_orphan_note_off` 规则处理；任何不满足
   该规则的将来 pairing 异常仍是 parser failure。
4. 文件结束仍有 pending on：`unterminated_note_on`，整首 song parser failure。
5. 配对后：
   - `end_tick > start_tick`：产生候选 `RawSourceNote`；
   - `end_tick == start_tick`：按第 3.1 节的 `same_tick_zero_duration_pair` 丢弃这对 event；
   - `end_tick < start_tick`：`note_nonpositive_duration`，整首 song parser failure。

FIFO 是冻结选择：同 pitch 可重叠时，最早开始的 note 先结束。同 tick chord 的稳定性来自
SMF track 内的 event ordinal；不同 track 的音本来就以不同 `physical_track_index` 区分。
velocity-zero Note On 永远当作 Note Off，不分配新的 ordinal。因此在当前每个 source file
只产生一个 `SongRecord` 的边界内，`source_file_identity:physical_track_index:source_note_ordinal`
保持唯一。

#### 3.1 Raw pairing normalization v1：只丢弃零时间 pair 与冗余 Off

这一步发生在 `RawSourceNote` 产生之前，不能与 0.25 QL 的量化投影混淆。它的依据是三次
只读 census：Mozart 有 331 个、Beethoven/mixed corpus 有 890 个 `note_nonpositive_duration`；
两者均为 `end_tick == start_tick`、`queue_depth_before == 1` 的同 tick FIFO pair。两次 census
都没有 `unterminated_note_on`；MAESTRO sample 有零个 pairing finding。该证据只支持以下两个
精确规则，不支持一般性 MIDI 容错。

```text
policy_version = "raw_pairing_normalization.v1"
key            = (physical_track_index, channel, pitch)
```

**规则 A：`same_tick_zero_duration_pair`**

当 Note Off（含 velocity=0 的 Note On）从同一 key 的 FIFO queue 取到最早 pending Note On，且：

```text
end_tick == start_tick
queue_depth_before == 1
```

则从 queue 移除这一 pending On，丢弃这对 On/Off，继续处理同 tick 及后续 events；不产生
`RawSourceNote`。它没有正的物理持续区间，因而既不是可训练音符，也不是 quantization
fragment。它不进入 `source_note_ordinal`、`source_note_id`、bar clipping、tensor 或
quantization audit。

**规则 B：`redundant_orphan_note_off`**

当 Note Off（含 velocity=0 的 Note On）到来、而同一 key 的 FIFO queue 为空时，丢弃该 Off，
queue 保持为空，然后继续。此 event 在本 parser 的确定性状态中没有可结束的活跃音，因此不改变
任一正时长 note 的 start/end；它同样不产生 source note。

每一次 A/B 都必须作为独立的 `raw_pairing_repair` diagnostic 发布，至少记录：

```text
repair_kind, source_file_identity, dataset_relative_posix_path, tune_index,
physical_track_index, channel, pitch,
on_tick/on_event_ordinal（规则 A）, off_tick/off_event_ordinal,
on_velocity（规则 A）, queue_depth_before, same_tick_events,
smf_format, ppqn
```

run-level manifest 必须绑定 repair policy version、repair artifact hash、总 repair 数、按 repair
kind 的数目与受影响文件数；`parser_integrity` 必须报告这些汇总。repair artifact 和现有
`raw_pairing_census` 都是诊断数据，不是训练样本。

**仍然严格 fail-fast 的边界：**

```text
unterminated_note_on
end_tick < start_tick
任何跨 physical track / channel / pitch 的补配
任何改变 FIFO 顺序的策略（例如 LIFO）
所有 TS、SMF format、SMPTE、track limit、slot capacity 等既有 failure
```

不能用“修复”把一个正时长音延长、缩短、移动或合并。若未来 census 出现 queue depth 大于 1、
不同 tick nonpositive、或 dangling On，必须新开设计决策，不能沿用本规则。

`source_note_ordinal` 的 raw-file scope 保持不变，但它现在只对 normalization 后的、成功形成
**正时长** `RawSourceNote` 的 Note On 编号。先收集这些 candidate On，再按
`(absolute_tick, physical_track_index, event_ordinal)` 排序为 `0..K-1`；被丢弃的 event pair
永不占 ordinal，因此不会制造 source identity 空洞语义或影响跨 bar continuity。

#### 3.2 Raw pairing census：持续监控，而非允许放宽边界

`raw_pairing_census` 仍必须直接读取 SMF bytes，使用与 parser 完全相同的 `(track, channel,
pitch)` FIFO state machine，但绝不生成 `SongRecord`、绝不写 tensor、绝不修改 event 顺序或文件。
它必须对未规范化前的每个异常输出可复查记录：

```text
source_file_identity, dataset-relative POSIX path, tune_index,
physical_track_index, channel, pitch, failure_code,
absolute_tick, event_ordinal, queue_depth_before,
matched_on_tick/event_ordinal（若有）, pending_on_queue snapshot,
same-tick events for this key, PPQN, SMF format
```

并输出按 `failure_code`、文件、track、channel、pitch 与同-tick/不同-tick 情形聚合的计数。
census 的 manifest 必须同时绑定输入文件清单、代码 revision、content hash，以及两个不可混淆的
版本字段：`census_pairing_policy_version="smf_fifo_track_channel_pitch_v1"`（观测使用的原始 FIFO）
和 `normalization_policy_version="raw_pairing_normalization.v1"`（encode 将采用的 repair 规则）。
它是诊断 artifact，不是训练数据；它可解释规则 A/B 为什么触发，但绝不能使本节列出的其它
fail-fast 边界失效。

### 4. 量化、clip 与 continuation：固定且可复算的顺序

全曲 absolute 0.25-QL quantization 与非整 0.25-QL bar length（例如 `5/32 = 0.625 QL`）
相位不兼容，因此 **不存在** `canonical_quantized_onset/end_ql` 这一全局 tensor 时间层。
它不得继续出现在代码、artifact 或 audit 中。

一条 `RawSourceNote` 先保留 raw global interval，再可生成零个或多个 fragment。每个
fragment 的 identity 是：

```text
(source_note_id, canonical_bar_index)
```

一个 fragment 有四组时间字段：

```text
raw_source_onset/end_ql                 # source 的全局精确事实；供 harmony tie-break/provenance
raw_local_start/end_ql                  # raw global interval clip 到本 span 后的 local 区间
quantized_local_start/end_ql            # 仅在本 bar 的 local grid 上量化的边界
bar_local_quantized_duration_ql         # quantized_local_end - quantized_local_start
```

对一个 canonical bar 的 local length `L`，先建立两个**不同**的可量化集合：

```text
S（onset 可取值） = 所有 valid slot 的 start
E（end 可取值）   = 所有 valid slot boundary，包含 bar end L
```

例如 `5/32 = 0.625 QL`：

```text
S = [0.00, 0.25, 0.50]
E = [0.00, 0.25, 0.50, 0.625]
```

`S` 不包含 `L`，因此一个 onset 不会被量化成“bar 结束以后才开始”的无效事件；`E` 包含 `L`，
因此 final partial slot 可以在真实 bar end 结束。

对每个 raw note 与每个相交 span `[bar_start,bar_end)`：

1. 先以 raw global 半开区间求交；空交集不产生 fragment。
2. 得到：
   ```text
   raw_local_start = max(0, raw_source_start - bar_start)
   raw_local_end   = min(L, raw_source_end - bar_start)
   ```
3. 独立量化：
   ```text
   quantized_local_start = nearest(raw_local_start, S)
   quantized_local_end   = nearest(raw_local_end, E)
   ```
   距离相同（half tie）时选择 away from zero 的较大 boundary。
4. 若 `quantized_local_end > quantized_local_start`，直接采用这对边界。
5. 若 `quantized_local_end <= quantized_local_start`，执行本设计冻结的
   **minimum-representable-slot projection（最小可表示 slot 投影）**，而不是失败或丢弃。
   它只在本条 fragment 已经坍缩时启用，绝不改变正常量化结果：

   ```text
   candidates = 当前 bar 的每一个 valid slot interval [S[i], E[i+1])
   overlap(i) = length([raw_local_start, raw_local_end) ∩ candidates[i])
   endpoint_error(i) = abs(S[i] - raw_local_start) + abs(E[i+1] - raw_local_end)

   选择 overlap 最大的 candidate；
   若并列，选择 endpoint_error 最小的 candidate；
   若仍并列，选择 slot index 较大的 candidate。

   quantized_local_start = S[i]
   quantized_local_end   = E[i+1]
   quantization_repair_kind = "minimum_representable_slot_projection"
   ```

   因为每个 candidate 是有效 slot，所以投影后的 duration 必然严格为正，且不会跨越
   canonical bar。例：`raw_local=[0.11,0.19)` 在 `0.25 QL` 网格中坍缩时，投影为
   `[0.00,0.25)`；若它恰好跨两个 slot 的边界且两个 overlap 相同，最后一条规则选择较后的
   slot，保证每次运行都得到同一答案。
6. 写入一个 bar-local quantized fragment；tensor 的 slot active/onset/hold 只读取它。

这不是“所有短音符一律拉长一个 slot”：只有独立量化两端已经给出零或负 duration 的
fragment 才投影。`quantization_audit` 必须记录原始两端、通常量化候选的两端、最终两端、
`quantization_repair_kind`、被选 slot index、overlap 与 endpoint error。run-level summary
还必须报告 projection 的 fragment 数、占全部 fragment 的比例，并可按 source file、meter 和
canonical bar 汇总。训练前的接受决策必须显式阅读这些计数；它们不得被算作“零误差”。

raw SMF tick→QL 是精确 Fraction，不先使用 float epsilon “吸附”。未来外部浮点 metadata
仅可在验证后于 `1e-6 QL` 内 snap 到已知 boundary；snap 不得改变上述 nearest/tie policy。

continuation 永远来自**未量化的 raw global source interval**：

```text
continues_from_previous_bar = raw_source_start < bar_start
continues_into_next_bar     = raw_source_end > bar_end
```

因此 `raw_local_start == 0` 或 `quantized_local_start == 0` 本身不表示 hold。只有
`raw_source_start < bar_start` 才写 hold；一个在当前 bar 内开始、但量化到 local 0 的 note
仍必须写 onset。

`quantization_audit` 的原子单位同样改为 `(source_note_id, canonical_bar_index)` fragment：

```text
onset_residual = abs(quantized_local_start - raw_local_start)
end_residual   = abs(quantized_local_end - raw_local_end)
meter          = fragment 所在 canonical span 的唯一 meter
```

跨 bar source note 因而有多个 audit sample；这反映它在多个 bar/meter 中被独立量化、独立
编码的事实。全局 `raw_source_onset_ql` 仍保存，但不再决定 audit meter，也不再是 tensor
onset 的坐标 authority。

### 5. Pickup、Terminal Bar、空白：完整时间轴语义

#### 5.1 Pickup：首版不推断，也不接受 metadata

MIDI 的首音位置不能证明弱起。为保证可复现，**当前已实施的 canonical raw-SMF parser 不
推断 pickup，也不接受 pickup metadata**：

```text
canonical grid 在 tick 0 / QL 0 开始
第一个 CanonicalBarSpan 是 initial TS 的完整 nominal bar
首音之前的时间是 valid rest，不是 pickup
is_pickup = false
pickup_policy = "not_inferred_from_midi"
pickup_status = "NOT_DECLARED"
```

例如初始 4/4，首音在 QL 1.5：bar 0 仍为 `[0.0,4.0)`；slot 0..5 是 rest，slot 6 起才
出现该音。不能因为首音迟到而把它移到 slot 0，也不能把后续 bar boundary 左移 1.5 QL。

未来若支持外部 pickup metadata，必须单独定义、版本化并绑定
`source_file_identity + tune_index`；其 `pickup_length_ql` 必须满足
`0 < length < initial_nominal_bar_length` 且可由 0.25 QL 表示。届时算法将是：首 span 为
`[0,pickup_length_ql)`，下一完整 bar 从 `pickup_length_ql` 开始，所有后续 TS boundary
均相对此新 phase 验证。该模式会改变全曲 phase，不能和首版模式混用。

#### 5.2 Terminal Bar：完整的 Rest 尾巴，而不是 note-off 截断

终止政策表达音乐时间轴的结束，而不是为了凑张量 shape。设 `last_end` 是所有 raw source
note 中最大的 `raw_source_end_ql`；它是物理音乐结束事实，不依赖每个 bar 的 local quantization：

1. 若 `last_end` 落在半开区间 `[bar.start,bar.end)`，该 span 就是 terminal bar；若 `last_end`
   **恰等于**一个 bar 的 `end`，terminal bar 是刚刚结束的前一个 bar，而不是下一 bar。
2. `terminal_bar_end = terminal_bar.end`。
3. materialize 从 canonical bar 0 到 terminal bar（含）之间的全部 spans。
4. terminal bar 内，note 覆盖的有效 grid 正常编码；最后 note-off 后、到该 bar end 的
   **有效** grid 全部是 Rest。
5. terminal bar 后不追加一整个 trailing empty bar，除非将来有独立的 source-duration
   contract。

例：4/4 terminal bar 为 `[40.0,44.0)`，最后 note 在 local QL 1.50 结束：

```text
local time:   0.00                 1.50                         4.00
              |---------------------|----------------------------|
valid slot:     0 .. 5                6 .. 15
meaning:      note onset/hold          Rest 尾巴

tensor slot:    0 .. 15 = valid musical time
                16 .. 47 = padding（全零，绝不是 Rest 尾巴）
```

这保留“音停后仍处于同一完整小节”的乐理语义。V2 技术上能以 mask/duration 表达短 bar，
但本政策不从 note-off 推导短 terminal bar。

#### 5.3 长静音和 Empty Bar

在 QL 0 到 `terminal_bar_end` 之间，TS chain 定义的每一个 span 都必须 materialize；绝不
使用“相邻音符间隔 >= 1 QL”之类规则补 bar。没有任何成功量化 fragment 的 span 是 Empty Bar：

```text
所有 valid lane-slot = Rest [0,1,0,0,0,0]
所有 padding lane-slot = 全零
base_pitch_valid = false
base_pitch storage value = 0
bar_context = 12 个零
```

Empty Bar 会清空 `SemanticCodecSequenceState` 的 melody continuity。后续再出现同一个
`source_note_id` 也不得穿过完整空 bar 获得 7-semitone preference。连续性只允许在
`canonical_bar_index` 连续、同一 song/tune/transpose variant、且中间没有 empty bar 时保留；
song、Opus tune、transpose variant、dataset split 的边界永远 reset。

若 input 有合法 TS 但没有任何成功配对的 note，报 `no_note_events` parser failure。不得输出
零行 encoding run，也不得凭空制造 all-rest bar；两者都会把“没有音乐 event”误表为可训练样本。

#### 5.4 Partial Slot 的来源边界

`SlotGrid` 作为通用 tensor contract 可以表达最后一个短于 `0.25 QL` 的 valid slot。raw-MIDI
parser 的 full span 仍使用 current meter 的 nominal length；但它现在允许**唯一一种** raw-derived
partial bar：第 2.2 节中由精确 TS event 截出的
`canonical_partial_bar_before_ts_change`。

这类 bar 的实际 `bar_length_ql = ts_event_ql - previous_canonical_boundary_ql`，仍以 old meter
作为 meter context；`SlotGrid.for_bar(bar_length_ql)` 产生有效 mask/duration。例：3/4 被 QL 2.0
处切断时，bar 有 8 个 `0.25 QL` valid slots；若实际长度不是 0.25 的整数倍，则最后一个 valid
slot 持有真实剩余 duration。它必须照常参与 raw clip、bar-local fragment quantization、chroma、
form full-coverage 判断、codec fidelity 与 quantization audit。

严格禁止其它 partial 来源：不得从 note-off、曲尾、长静音、music21 Measure heuristic 或猜测的
pickup 生成 partial bar。曲尾仍遵循完整 terminal Rest-tail bar；只有 exact raw TS event 才可产生
此类 partial canonical span。

### 6. Form metadata、diagnostics 和实施影响

#### 6.1 Form metadata 映射

当前实现的 parser 只接受**已绑定 canonical bar 的 form metadata**。其坐标系固定为
`canonical_bar_index.v1`；合法 payload 必须绑定 source identity、`raw_smf_v1` parser
version 与 canonical timeline hash：

```json
{
  "coordinate_system": "canonical_bar_index.v1",
  "source_file_identity": "...",
  "canonical_parser_version": "raw_smf_v1",
  "canonical_timeline_sha256": "sha256:...",
  "form": "binary",
  "sections": [{"name": "A", "canonical_start_bar_index": 0, "canonical_end_bar_index": 8}]
}
```

每个 section 必须是已有 canonical bar index 的非空、无重叠区间。只有整个 payload 都通过
验证后，parser 才设置全曲 `song.form`（如 `binary`）和局部
`bar.form`/`section_label`（如 `A`）。准确状态集合为：`absent`、`mapped`、
`unavailable_legacy_measure_index`、`unavailable_timeline_mismatch`、
`unavailable_empty_canonical_sections`。

当前 offline form generator 只能产生 `coordinate_system=legacy_measure_index`，尚不能生成
上面的 bound canonical metadata。因此 Codec V2 必须 fail-closed：不写入任何 form label，
`form_action_alignment` 为 `UNAVAILABLE`。canonical form classifier/adapter 是未来独立工作。
以 QL interval 表示 form section 也只是未来 proposal，当前 parser 不接受它。

`ActionLabeler` 可以没有 form metadata 独立运行；但 alignment 不得猜测标签。当前它仍是
legacy 16-bin diagnostic heuristic：每个 bar 按 `bar_length_ql / 16` 归一化 rhythm bins。
它不决定 V2 tensor，也不构成训练输入。在 canonical form classifier/adapter 启用前，必须
迁移至 `SlotGrid` valid-slot 语义，或继续使 `form_action_alignment` 为 `UNAVAILABLE`；
不能表述为已经使用 SlotGrid。

#### 6.2 parser-integrity 的发布状态

当前 `parser_integrity_raw_observation.v2` 只发布 runtime capture 实际 materialize 的字段：

| 当前已发布组 | 字段 |
|---|---|
| Measure map | `measure_map`（song/measure count、meter distribution、Opus-tune count、over-capacity count） |
| Initial TS | `initial_time_signature_policy`、`initial_time_signature_origin_counts`、`initial_time_signature_injections` |
| Partial span | `partial_span_count`、`partial_reason_counts`、`partial_spans` |
| Track retention | `track_retention` |
| Pairing normalization | `normalization_policy_version`、`repair_artifact`、`repair_count`、`repair_counts_by_kind`、`raw_pairing_repair_counts`、pairing-loss count/ratio、affected-file count |
| Failures | `parser_failures` |

当前 `quantization_audit` raw observation 发布 `audit_unit="source_note_fragment"`、
fragment/projection count、pairing-loss summary、grid policy 与按 file/meter 的 residual。每条
sample 的 key 是 `(source_note_id, canonical_bar_index)`，包含 meter、raw-local/quantized-local
start/end 与 onset/end residual；不发布或消费 global quantized onset/end 字段。

以下是**后续 parser-integrity 扩展**，不是当前 raw observation 字段：
`canonical_parser_version`、TS fact/duplicate/conflict count、`initial_ts_source`、
`canonical_span_count`、`empty_bar_count`、terminal/pickup policy/status、
note-pairing/quantization policy、`retained_note_track_count`。它们必须先补齐 schema 和
capture 实现，evaluation 才可消费。

parser 无法构造 canonical timeline 时，不发布“看起来 AVAILABLE 的 partial encoding”。该
source 是 parser failure；整个 encoding run 依既有 fail-fast policy 失败。已 materialize
run 的 capture 输入不完整、hash/index 不匹配时，evaluation raw 才是 `UNAVAILABLE`。

#### 6.3 必须同步修改的文件和顺序

这套实现不改动 `SemanticHarmonySetCodec` 的 V2 tensor 语义。已完成的迁移边界为：

1. `src/data/canonical_timeline.py` 产出 raw facts 与 `CanonicalBarSpan[]`；
2. `src/data/music_parser.py` 由 spans 产生 `SongRecord`/`BarRecord` 与
   `canonical_bar_index`；旧 `measure_map.py` 已退役；
3. controls、codec-fidelity source raw、pipeline、capture/export/evaluator 均消费
   canonical 坐标；
4. form loader 仅接受 identity、timeline hash 与 canonical coordinate system 均匹配的
   metadata；legacy measure index fail-closed。

`source_measure_index` 和 `source_bar_index` 是 `canonical_bar_index` 的 serialized
compatibility alias；若保留 `BarRecord.source_measure_index`，它必须严格等于
`canonical_bar_index` 的 serialized compatibility alias。任何 codec algorithm（包括 sequence
state）都不得读取它来解释 music21 Measure identity。术语 `bar-local source note` 统一改为
`bar-local clipped note`。

### 7. Canonical parser 的正式语料验收

固定 corpus run 只有在 published artifacts 证明以下结果后才可用于训练：

```text
parser failures == 0
archive / manifest / row index 全量 hash、shape、row 对齐
默认 retention 下 note 和 note-bearing track 零丢失
TS conflict、duplicate merge、empty bar、terminal/pickup policy 有完整报告
quantization audit 有总体、按文件、按 meter 的 residual 结果
若用于训练：measured harmony_lane_overflow_count == 0
```

这些是运行后测量值，不得在 parser 或文档中预设为真。

---

## 1. 一个小节如何变成固定大小的张量

以 4/4 小节为例：它长 `4.0 QL`，被切成 16 个 `0.25 QL` 的有效 slot。

```text
时间（QL）  0      .25    .50    .75    1.0                    4.0
            |-------|-------|-------|-------|------ ... --------|
slot 编号      0       1       2       3              ...       15
```

Codec 在**同一次 encoding run 内**预留配置的 `C` 个 slot，保证该 run 的模型输入 shape 固定。当前
Beethoven profile 使用 `C=92`：

```text
4/4：slot 0..15 是真实音乐时间，16..91 是 padding
3/4：slot 0..11 是真实音乐时间，12..91 是 padding
6/8：slot 0..11 是真实音乐时间，12..91 是 padding
12/8：slot 0..23 是真实音乐时间，24..91 是 padding
```

### 为什么从 16 个 slot 改为“0.25 QL + 可配置 capacity”

这次改变的是 **一个 bar 可容纳的最长音乐时间**，不是时间分辨率。

旧方案的 16 个 slot 在 `0.25 QL` 分辨率下最多表达：

```text
16 × 0.25 QL = 4.0 QL
```

它刚好覆盖普通的 4/4 或 2/2 小节，却无法原样表达更长的小节：

```text
12/8  = 6.0 QL  → 需要 24 个 slot
12/4  = 12.0 QL → 需要 48 个 slot
20/4  = 20.0 QL → 需要 80 个 slot
41/8  = 20.5 QL → 需要 82 个 slot
```

在 Mozart 语料的只读 census 中，发现有 104 个真实 measure 超过 16 个 slot，其中
包含大量 12/8 measure；也存在 `12/4` 的 48-slot measure。随后 Beethoven 语料又发现 80-slot
与 82-slot 的合法 bar。这说明“48”不是 V2 的乐理规则，而只是当时 corpus 观察到的临时上限。继续
使用 16-slot 只有三种
结果：

```text
1. 截断小节尾部的 note 或时间          → 数据丢失，不可接受
2. 把一个 source measure 硬拆成多个 bar → 改变 source measure 语义与跨 bar 状态
3. 降低时间分辨率以塞进 16 格           → 损失 0.25 QL timing 精度
```

因此 V2 固定 `0.25 QL` 的时间粒度，但将最长可表达时间交给 run 配置：

```text
C × 0.25 QL

当前 Beethoven profile：92 × 0.25 QL = 23.0 QL
```

这使一个 20/4 或 41/8 小节可以作为一个完整 bar 被表达；较短小节则通过
`slot_valid_mask` 和 padding 使用同一固定 tensor shape。模型不必处理同一 run 内的可变 shape，
也不会把 3/4、6/8 等短小节误认为 92 个真实时间格。

配置的 C 不是无限容量。验证后的实际 measure 若仍需要超过 C 个 slot，Codec 会抛出带完整 canonical
bar 上下文的 `CodecCapacityError(slot_capacity_exceeded)`；不会自动截断或隐式重新切分。

### Rest 不是 padding

这两个状态看起来都“没有音”，但意义不同：

```text
有效时间内没有音：这是 rest
小节本来没有那么长：这是 padding
```

它们的向量也不同：

```text
valid slot 中的 rest: [0, 1, 0, 0, 0, 0]
padding slot:          [0, 0, 0, 0, 0, 0]
```

两个辅助 array 明确告诉下游程序 slot 是否真实存在、真实持续多久：

```text
slot_valid_mask[row, slot]     # true：真实音乐时间；false：padding
slot_durations_ql[row, slot]   # 真实时长；padding 固定为 0
```

最后一个 valid slot 允许短于 `0.25 QL`。例如一个测量长度为 `6.375 QL` 的 bar 有
26 个有效 slot：前 25 个是 `0.25`，最后一个是 `0.125`。如果一个 bar 需要超过
C 个 slot，Codec 抛出 `slot_capacity_exceeded`。

---

## 2. 模型实际拿到的数据包

一个 encoding run 写出 `voice_tensors.npz`，其中包含：

| Array | Shape | 它回答的问题 |
|---|---|---|
| `voice_tensors` | `[N,18,C,6]` | 第 `row` 个小节，在某个角色、某个时间格中发生了什么？ |
| `slot_valid_mask` | `[N,C]` | 这个时间格是音乐，还是仅为了固定 shape 的 padding？ |
| `slot_durations_ql` | `[N,C]` | 这个时间格实际有多长？ |
| `bar_contexts` | `[N,12]` | 整个小节的相对和声背景是什么？ |
| `base_pitches` | `[N]` | 计算相对 pitch 的低音锚点是什么？ |
| `base_pitch_valid` | `[N]` | 当前小节真的有低音锚点吗？ |

`bar_tensor_index.json` 负责把一个 row 追溯回原始音乐：`raw_smf_v1` 写入
`canonical_bar_index`、`row`、`song_id`、transpose 和 `tensor_key`。同时写入的
`source_bar_index`/`source_measure_index` 必须严格等于 `canonical_bar_index`，仅作为
序列化兼容 alias，而不是任何 Measure identity。
`encoding_manifest.json` 记录 array shape/dtype、配置、数据集 identity、代码 revision、
`canonical_parser_version="raw_smf_v1"`、`supported_source_formats` 与 SHA-256 hash。也就是说：

```text
张量：音乐数值
index：数值来自哪个 bar
manifest：这些文件是否属于同一次可信编码
```

---

## 3. 每个 lane 的六个数字

`voice_tensors[row, lane, slot, :]` 是长度为 6 的向量：

| 下标 | 名称 | 直观含义 |
|---:|---|---|
| 0 | `relative_pitch` | 当前音相对当前 bar 最低音有多高。 |
| 1 | `is_rest` | 该语义位置此刻是否为空。 |
| 2 | `is_note_on` | 音是否在这个 slot 新开始。 |
| 3 | `is_hold` | 音是否从较早时间延续到这里。 |
| 4 | `normalized_velocity` | MIDI 力度归一化到 0 到 1。 |
| 5 | `velocity_ratio` | 当前音在同时发声语义音中的相对力度份额。 |

在有效时间中，状态只有三种：

```text
rest:  [0,              1, 0, 0, 0,                   0]
onset: [relative_pitch, 0, 1, 0, normalized_velocity, velocity_ratio]
hold:  [relative_pitch, 0, 0, 1, normalized_velocity, velocity_ratio]
```

`rest`、`onset`、`hold` 在 valid lane-slot 中有且仅有一个为 1。padding 不属于音乐
状态，六个值都为零。

---

## 4. 可手算复现的完整示例

下面用一个小型 4/4 bar 演示“音乐 → lane → 张量”。它使用的字段与 parser 提供给
codec 的字段相同；选择小例子是为了可以逐元素核对结果。

### 4.1 输入音乐

四个 note 都从小节开头开始，持续 `0.75 QL`：

| 音 | MIDI pitch | Velocity | Physical track | Bar-local 区间 |
|---|---:|---:|---:|---|
| C4 | 60 | 80 | 0 | `[0.00, 0.75)` |
| E4 | 64 | 70 | 1 | `[0.00, 0.75)` |
| G4 | 67 | 60 | 2 | `[0.00, 0.75)` |
| C5 | 72 | 100 | 0 | `[0.00, 0.75)` |

它们在前三个 slot 中同时发声：

```text
slot time       [0.00,.25)  [.25,.50)  [.50,.75)  [.75,1.00)
                 slot 0      slot 1      slot 2      slot 3
                 ------------------------------------------------
C5 (72)         active      active      active      ended
G4 (67)         active      active      active      ended
E4 (64)         active      active      active      ended
C4 (60)         active      active      active      ended
```

### 4.2 音如何分配到 lane

先找整个 bar 的最低音：

```text
base_pitch = C4 = 60
```

然后对每个 active slot 分配角色：

```text
最高音 C5 (72)                 → melody, lane 0
移除 melody 后的最低音 C4 (60) → bass, lane 17
剩余 E4、G4                    → harmony, lane 1、lane 2
```

结果：

```text
slot                  0          1          2          3
                       ─────────  ─────────  ─────────  ─────────
lane 0  melody         C5 onset   C5 hold    C5 hold    rest
lane 1  harmony_00     E4 onset   E4 hold    E4 hold    rest
lane 2  harmony_01     G4 onset   G4 hold    G4 hold    rest
lane 3..16             rest       rest       rest       rest
lane 17 bass           C4 onset   C4 hold    C4 hold    rest
```

slot 3 从 `0.75 QL` 开始；note 恰在这里结束。slot 使用左闭右开语义，因此它们不再
属于 slot 3，所有 lane 都是 rest。

### 4.3 slot 0 的实际张量值

同一 slot 中所有 assigned note 的 velocity denominator：

```text
80 + 70 + 60 + 100 = 310
```

例如 C5：

```text
relative_pitch       = (72 - 60) / 24 = 0.5
normalized_velocity  = 100 / 127      = 0.787402
velocity_ratio       = 100 / 310      = 0.322581
```

所以 slot 0 的非 rest 向量为：

| Lane | 音 | `[relative_pitch, rest, onset, hold, normalized_velocity, velocity_ratio]` |
|---:|---|---|
| 0 | C5 | `[0.500000, 0, 1, 0, 0.787402, 0.322581]` |
| 1 | E4 | `[0.166667, 0, 1, 0, 0.551181, 0.225806]` |
| 2 | G4 | `[0.291667, 0, 1, 0, 0.472441, 0.193548]` |
| 17 | C4 | `[0.000000, 0, 1, 0, 0.629921, 0.258065]` |

其余 valid lane 是 rest。slot 1、2 的 pitch/velocity 相同，但 `onset=0, hold=1`。
slot 16 以后则是 padding，全为零。

### 4.4 整小节的和声背景

除逐 slot 张量外，Codec 为每个 bar 保存一个 12D `bar_context`。它描述：

> 从当前 bar 的最低音出发，哪些 pitch class 占据了多少 duration/velocity 权重？

本例中：

```text
C4 (60) + C5 (72) → relative bin 0：80 + 100 = 180
E4 (64)           → relative bin 4：70
G4 (67)           → relative bin 7：60
总计                                      310
```

四个音的 duration 相同，duration 公因子会在 L1 normalization 中抵消：

```text
bin index:     0        1  2  3    4        5  6    7        8  9 10 11
bar_context: [0.580645, 0, 0, 0, 0.225806, 0, 0, 0.193548, 0, 0, 0, 0]
```

它不关心 E4、G4 分别处于哪个 harmony lane，只描述整小节的相对和声内容。

---

## 5. 三种角色的选择规则

### Melody：通常选最高音，但避免不必要跳变

先选 active notes 中最高的 pitch；同 pitch 时，velocity 更高者优先，最后以
`source_note_id` 保证确定性：

```text
pitch descending → velocity descending → source_note_id ascending
```

然后应用 7-semitone continuity rule。若上一 slot 的 melody 还在响，并且它只比当前
最高音低不超过 7 个半音，就继续把它当作 melody：

```text
previous_melody.pitch >= highest_active_pitch - 7
```

这避免一个短暂较高的装饰音导致 melody lane 不必要地交换。

### Bass：从剩余音中选最低音

移除 melody 后，若仍有 active note，选择最低 pitch；同 pitch 时 velocity 高者优先，
再以 `source_note_id` 打破平局。若只有 melody，bass lane 为 rest。

### Harmony：剩余 note 的确定性集合

移除 melody 与 bass 后，剩余 note 是 harmony。它们按下面的顺序排序，再依次写入
lane 1 至 lane 16：

```text
pitch ascending
physical_track_index ascending
bar-local clipped duration ascending
source_onset_ql ascending
velocity ascending
source_note_id ascending
```

这只是为了令同一输入总是产生同一 tensor；不意味着某条 harmony lane 是长期固定的
physical voice。

---

## 6. Active、onset 与 hold 怎么判定

一个 note 在 slot `[slot_start, slot_end)` 中 active，当且仅当：

```text
note.onset_ql < slot_end - 1e-6
且
note.onset_ql + note.duration_ql > slot_start + 1e-6
```

这表示 note 必须真正覆盖 slot 的一部分，而不是仅在边界接触它：

- note 恰在 slot 左边界开始：属于这个 slot；
- note 恰在 slot 左边界结束：不属于这个 slot；
- active 但不是本 slot 新开始的 note：写成 hold。

对于一个 assigned note，只有在以下两件事同时成立时写 `onset`：

```text
它不是从 previous bar 延续到当前 bar slot 0 的 note
且 note.onset_ql 与 slot_start 的差不超过 1e-6 QL
```

其他 active note 都是 `hold`。

---

## 7. 跨小节的音和 melody continuity

假设 C5 在 bar 0 开始并跨入 bar 1；bar 1 开头同时出现一个更高的 E5：

```text
bar 0: C5 开始，并持续超过 bar end
bar 1: C5 从上一个 bar 延续；E5 在当前 bar 开始
```

若 C5 是上一 bar 的 melody，且它只比 E5 低 4 个半音：

```text
4 <= 7
→ C5 保留 melody lane 0
```

而 C5 在 bar 1 的 slot 0 必然写为 hold：

```text
bar 0 lane 0 slot 0: C5 onset
bar 1 lane 0 slot 0: C5 hold
```

这类跨 bar 状态只能由 `encode_song(song)` 维护。`encode(bar)` 不传 state 时故意把
bar 当独立对象，不能在多小节循环中替代 `encode_song()`。

状态只有在 canonical bar 连续、且中间不是 Empty Bar 时才可跨 bar 传递：

```text
current.canonical_bar_index == previous.canonical_bar_index + 1
```

因此 tune、transpose variant、split boundary、canonical bar gap 或 Empty Bar 都会安全 reset。

---

## 8. 三个关键数值公式

### Relative pitch

```text
relative_pitch = (absolute_midi_pitch - base_pitch) / 24
```

`base_pitch` 是当前 bar 所有 source note 的最低 pitch。若最高和最低相差超过
96 semitones，Codec 报 `relative_pitch_range_overflow`，不输出越界数值。

### Velocity 与 vertical balance

```text
clamped_velocity    = clamp(source_velocity, 0, 127)
normalized_velocity = clamped_velocity / 127

velocity_ratio = clamped_velocity /
                 sum(clamped velocity of melody + active harmony + bass)
```

若分母为零，ratio 为零。它描述当前 slot 内的相对力度，不是整首曲子的 loudness。

### 12D bar context

对每个 bar-local source note：

```text
pitch-class bin = (pitch - (base_pitch mod 12)) mod 12
weight          = clipped_duration_ql * clamped_velocity / 127
```

按 bin 累加再做 L1 normalization。空 bar 或总 weight 为零时，context 是全零。

---

## 9. V2 的明确失败边界

| 条件 | 行为 | 为什么 |
|---|---|---|
| 一个 bar 需要超过配置的 C 个 slot | `slot_capacity_exceeded` | 不能静默截断时间。 |
| 一个 slot 有超过 16 个 harmony note | `harmony_lane_overflow` | 不能静默丢 note。 |
| pitch span 超过 96 semitones | `relative_pitch_range_overflow` | 不输出超出契约的 normalization。 |
| canonical TS chain 缺失、冲突或非法 TS 数值 | codec 前 parser failure | 不猜测 meter；非完整 boundary 的精确 TS event 会生成显式 partial span，而非失败。 |
| raw pairing 超出 normalization v1 | `unterminated_note_on` / `note_nonpositive_duration`（仅 `end_tick < start_tick`） | 只允许丢弃零时间 pair 与 queue 为空的冗余 Off；不得伪造 source note。 |
| physical track 超过 48，且未显式 truncate | `track_limit_exceeded` | track loss 必须显式。 |

正式 canonical training encoding 使用 error policy。显式 `truncate` 会记录
dropped-part count 与 dropped-note ratio，但不能被误认为无损结果。

`quantization_nonpositive_clipped_duration` 不再是训练编码的最终 failure：已成功配对且与
canonical span 有正 raw overlap 的 fragment 若在两端独立量化后坍缩，必须按照第 4 节的
最小可表示 slot 投影写入一个 slot，并在 quantization audit 中作为 repair 统计。它绝不能被
静默删除或伪装为无误差。

---

## 10. V2 保存与不保存的信息

### V2 直接保存

- bar-local note 的相对 pitch；
- 0.25-QL slot 上的 onset / hold / rest；
- note-local velocity 与当前 slot 的 vertical balance；
- melody、bass、harmony-set 的语义角色；
- 跨 bar source identity 与 continuation；
- 12-bin、duration/velocity-weighted chroma；
- actual slot duration、valid mask 与 partial final slot。

### V2 不直接保存

- 0.25 QL 以下的精确 onset/duration；
- note 在单个 slot 内的精确 overlap duration；
- harmony physical voice 在相邻 slot 的固定 lane identity；
- pedal/CC64、tempo、key 作为 tensor target；
- 独立的 bar-level 或 phrase-level dynamics-shape channel。

这些边界并不隐藏：quantization residual、performance controls、parser integrity
和 codec fidelity 均由已有 evaluation framework 采集和报告。

---

## 11. 实现与契约的对应关系

| 内容 | 文件 |
|---|---|
| Public artifact contract | `contracts/codec/bar_tensor_schema.v2.md` |
| Slot grid | `src/codec/slot_grid.py` |
| Melody/bass/harmony assignment 与 sequence state | `src/codec/semantic_harmony_assignment.py` |
| Tensor 写入 | `src/codec/semantic_harmony_set_codec.py` |
| Bar clipping 与 source identity | `src/data/music_parser.py` |
| NPZ、index 与 manifest 写入 | `src/pipeline/encoding_pipeline.py` |
| Codec fidelity evaluation | `src/diagnostics/codec_fidelity_raw_capture.py` 与 `evaluation/src/evaluation_framework/evaluation_codec_fidelity.py` |

若本文与 public schema contract 的 artifact compatibility 描述冲突，以 contract 为准；
若本文与当前 runtime 行为冲突，以实现和回归测试为准。
