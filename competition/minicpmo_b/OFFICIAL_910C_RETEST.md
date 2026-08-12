# 官方单卡 910C 最终复测流程

本流程只用于官方 `quay.io/ascend/vllm-omni:v0.25.0-a3`、单卡 910C。910B4
开发结果不能填入最终成绩表。所有基线/优化测试必须使用相同模型目录、数据、
请求顺序、两次 warmup 和空闲卡。

建议先用总编排器打印完整计划；dry-run 不启动服务，也不占用 NPU：

```bash
python competition/minicpmo_b/scripts/run_official_910c_retest.py \
  --image-digest quay.io/ascend/vllm-omni@sha256:REPLACE_WITH_REAL_DIGEST \
  --model-path /workspace/MiniCPM-o-4_5 \
  --daily-omni-root /tmp/minicpmo_b_daily_omni \
  --videomme-root /tmp/minicpmo_b_videomme \
  --seed-tts-root /tmp/minicpmo_b_seedtts
```

仅在官方空闲单卡 910C 主机确认计划后执行：

```bash
python competition/minicpmo_b/scripts/run_official_910c_retest.py \
  --image-digest quay.io/ascend/vllm-omni@sha256:REPLACE_WITH_REAL_DIGEST \
  --execute --confirm-single-910c
```

正式执行前必须先提交全部代码和配置，使
`git status --porcelain --untracked-files=all` 为空。编排器把 HEAD commit/tree
写入冻结计划和每个阶段 marker；未提交改动会被拒绝，避免官方成绩来自无法进入
源码制品的代码。

编排器按阶段写入 `orchestrator_state/*.json`，命令哈希不变且已通过的阶段会自动
跳过；可用重复的 `--phase NAME` 仅执行指定阶段。冻结计划一旦存在，同一
`--result-root` 只能用完全相同的输入、镜像 digest、Git commit 和 Git tree 恢复；
任何漂移都会立即停止，必须使用新的结果目录，禁止把不同轮次证据混在一起。它包含
910C/单可见卡/残留服务
preflight、模型全部 checkpoint 分片、Token2Wav ONNX/PT 资产、三套完整数据、
Whisper/WavLM/UTMOS、依赖版本、端口、磁盘空间与 S3Tokenizer/campplus CPU 加载探针，
以及环境冻结、基线/优化 c1/c4/c8、双工 RTF、多轮门禁、三项精度、性能
汇总及相对门禁。任何阶段失败都会立即停止。Demo 与录屏仍需按第 6 节人工完成。

执行模式还会先写入 `orchestrator_state/orchestrator_plan.json`，冻结完整输入参数、
26 阶段顺序、规范命令及 SHA256。最终证据审计会用当前提交中的
`build_phases()` 重新生成计划，并逐项核对计划与每个阶段 marker；即使有人同步修改
marker 中的命令与哈希，只要命令偏离规范计划也会失败。

使用 `--rerun-completed`，或恢复一个尚未通过/旧格式的阶段时，编排器会把该阶段及
其全部下游 marker 移入 `orchestrator_state/history/<UTC>/`，并保存
`invalidation.json`，随后只认可当前冻结计划重新生成的 marker。默认不删除历史
长测证据。若同时使用 `--phase NAME` 做局部重跑，必须继续重跑该阶段之后的全部下游
阶段（最稳妥是去掉 `--phase` 再恢复完整编排）；否则最终审计会因下游 marker 缺失
而失败。每个 marker 还绑定计划 SHA256、源码 commit/tree 和合法 UTC 起止时间。

总编排共 26 个阶段。性能矩阵、Realtime 双工和三项精度各自生成
`run_protocol.json`；基线与优化版必须具有完全相同的输入参数、请求顺序、warmup、
并发、元数据文件 SHA256 和媒体目录 inventory fingerprint。deploy YAML 及其 SHA256
记录在 `variant` 区域，允许基线与优化版不同。五个 `protocol-*` 阶段会在对应结果
门禁之前比较两侧 fingerprint；最终证据审计还会直接重算一次，不只信任已生成的
`protocol_gate_*.json`。

Whisper 与 WavLM 必须准备为实际展开的离线模型目录，不要直接指向 Hugging Face
cache 中含符号链接的 snapshot。协议 inventory 会拒绝 symlink，环境冻结还会对
`config.json` 和权重文件计算内容 SHA256，避免复测依赖容器外 cache 或悬空链接。
所有 benchmark 显式使用 `seed=0`；Daily-Omni 与 Seed-TTS 按该 seed 的确定性顺序
运行，Video-MME 固定 `disable_shuffle=true`。这些参数均写入 A/B protocol 并由最终
审计重新校验，不能依赖上游 CLI 默认值。

官方 26 阶段完成后，如评测时间允许，建议再执行一次反序 fresh-start 确认：

```bash
competition/minicpmo_b/scripts/run_paired_confirmation.sh
```

主轮顺序为 A1（基线）→B1（优化），确认轮反转为 B2→A2，并保持相同输入、两次
warmup、c1/c4/c8 与统计口径。辅助 gate 要求两个配对方向的 TTFT/TTFP 均至少改善
10%、SPEAK 生成 RTF 均至少改善 15%，且同一版本两次 fresh-start 的指标漂移不超过
10%。该结果用于排除编译热身、服务启动顺序和机器漂移造成的虚假收益，不替代主办方
官方脚本和 26 阶段主成绩；若目录存在，最终审计与归档会强制其所有 gate 通过。

## 1. 环境冻结

```bash
cd /vllm-workspace/vllm-omni
export MODEL_PATH=/workspace/MiniCPM-o-4_5
export NPU_DEVICE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CONTAINER_IMAGE_DIGEST=quay.io/ascend/vllm-omni@sha256:REPLACE_WITH_REAL_DIGEST

OUTPUT_DIR=competition/minicpmo_b/results/official_910c/environment \
HASH_MODEL_WEIGHTS=1 \
competition/minicpmo_b/scripts/collect_environment.sh
```

确认 `npu_smi.txt` 只显示一张 910C 被当前测试使用；保留镜像 digest、CANN、
torch/torch_npu、vLLM、vLLM-Ascend 和模型 SHA256。开始每组测试前确认没有残留
`vllm serve` 或 `StageEngineCoreProc`。

## 2. 官方基线性能矩阵

```bash
CASE_NAME=official_910c_baseline \
DEPLOY_CONFIG=/vllm-workspace/vllm-omni/vllm_omni/deploy/minicpmo_4_5.yaml \
REQUIRE_STAGE1_FULL_DECODE=0 \
C1_PROMPTS=32 C4_PROMPTS=64 C8_PROMPTS=128 \
competition/minicpmo_b/scripts/run_perf_matrix.sh
```

## 3. 优化版本性能矩阵

```bash
CASE_NAME=official_910c_optimized \
DEPLOY_CONFIG=/vllm-workspace/vllm-omni/competition/minicpmo_b/config/minicpmo_4_5_910c_low_latency.yaml \
REQUIRE_STAGE1_FULL_DECODE=1 \
C1_PROMPTS=32 C4_PROMPTS=64 C8_PROMPTS=128 \
competition/minicpmo_b/scripts/run_perf_matrix.sh
```

`REQUIRE_STAGE1_FULL_DECODE=1` 会强制检查 Stage 1 日志同时包含：

- `CUDAGraphMode.FULL_DECODE_ONLY`
- `enable_npugraph_ex=True`
- `Capturing CUDA graphs (decode, FULL)`

若 910C 环境把图模式静默降级为 PIECEWISE，脚本必须失败，不能把降级结果当作
优化版成绩。

汇总：

```bash
python competition/minicpmo_b/scripts/summarize_performance.py \
  baseline_c1=competition/minicpmo_b/results/official_910c_baseline/seed_tts_c1_n32.json \
  optimized_c1=competition/minicpmo_b/results/official_910c_optimized/seed_tts_c1_n32.json \
  baseline_c4=competition/minicpmo_b/results/official_910c_baseline/seed_tts_c4_n64.json \
  optimized_c4=competition/minicpmo_b/results/official_910c_optimized/seed_tts_c4_n64.json \
  baseline_c8=competition/minicpmo_b/results/official_910c_baseline/seed_tts_c8_n128.json \
  optimized_c8=competition/minicpmo_b/results/official_910c_optimized/seed_tts_c8_n128.json
```

上述 chat-completions 矩阵用于 TTFT、TTFP、并发吞吐和稳定性对照；其中
`audio_chunk_rtf` 是全部稳态 chunk 的辅助指标，不能冒充比赛主 RTF。

## 4. 官方 SPEAK 生成阶段 RTF

基线和优化版分别启动对应服务后，使用 Realtime 全双工端点执行相同数据、顺序、
warmup 与并发矩阵：

```bash
RESULT_DIR=competition/minicpmo_b/results/official_910c_baseline/duplex_rtf \
NUM_PROMPTS=32 MAX_CONCURRENCY=1 \
competition/minicpmo_b/scripts/benchmark_duplex_rtf.sh

RESULT_DIR=competition/minicpmo_b/results/official_910c_optimized/duplex_rtf \
NUM_PROMPTS=32 MAX_CONCURRENCY=1 \
competition/minicpmo_b/scripts/benchmark_duplex_rtf.sh

RESULT_DIR=competition/minicpmo_b/results/official_910c_optimized/duplex_rtf \
NUM_PROMPTS=64 MAX_CONCURRENCY=4 \
competition/minicpmo_b/scripts/benchmark_duplex_rtf.sh

RESULT_DIR=competition/minicpmo_b/results/official_910c_optimized/duplex_rtf \
NUM_PROMPTS=128 MAX_CONCURRENCY=8 \
competition/minicpmo_b/scripts/benchmark_duplex_rtf.sh
```

基线服务也必须执行相同 c1/c4/c8 三组命令；除 `RESULT_DIR` 外不得改变输入、
warmup、请求顺序或统计参数。

至少提取 mean/p50/p99 `audio_speak_generation_rtf`，并同时保留
`audio_speak_tail_rtf`、全部 `audio_chunk_rtf`、TTFT、TTFP、E2EL、吞吐、
continuity 和 underrun。`audio_speak_generation_rtf` 依据 Realtime audio delta
的模型阶段元数据分离 SPEAK 生成与尾部，是本地复测代理；最终排名必须使用
主办方脚本的阶段判定与统计口径。

每组优化结果必须执行自动门禁。当前默认方案包含 Stage 1 FULL decode、reference
cache、13/25 固定包和完整 runner 预热：

```bash
python competition/minicpmo_b/scripts/gate_duplex_candidate.py \
  competition/minicpmo_b/results/official_910c_baseline/duplex_rtf/native_duplex_c1_n32.json \
  competition/minicpmo_b/results/official_910c_optimized/duplex_rtf/native_duplex_c1_n32.json \
  --profile combined \
  --output competition/minicpmo_b/results/official_910c_optimized/duplex_rtf/gate_c1.json

python competition/minicpmo_b/scripts/validate_candidate_log.py \
  competition/minicpmo_b/results/official_910c_optimized/server.log \
  --require-stage0-mm-cache-disabled \
  --require-stage1-full-decode \
  --require-stage0-ref-cache \
  --require-stage2-prompt-cache \
  --require-stage2-runner-prewarm \
  --require-stage1-cpu-slot-mapping \
  --require-stage1-graph-sampler \
  --require-stage1-binary-argmax \
  --output competition/minicpmo_b/results/official_910c_optimized/activation_gate.json
```

若最终默认配置没有晋级某项实验开关，应删除对应的 `--require-*`，但不得保留开关
又跳过其 activation gate。c4/c8 使用相同方式分别生成 `gate_c4.json`、
`gate_c8.json`。

在任何性能成绩晋级前，必须额外通过同一 WebSocket session 的多轮边界门禁，防止
resumable segment 水位、TTS terminal control packet 或 playback 状态跨轮污染：

```bash
HOST=127.0.0.1 PORT=8091 \
NUM_PROMPTS=2 MAX_CONCURRENCY=1 TURNS_PER_SESSION=3 NUM_WARMUPS=0 \
RESULT_DIR=competition/minicpmo_b/results/official_910c_optimized/multiturn_s2_t3 \
RESULT_FILENAME=native_duplex_rtf.json \
competition/minicpmo_b/scripts/benchmark_duplex_rtf.sh
```

硬门槛：`audio_turns=6`、每个 run 的 `done_count=3`、`cancelled_count=0`、
`stale_audio_delta_count=0`、`truncate_count=0`、`lifecycle_counts_ok=true`、
`cross_turn_independent_ok=true`，并检查服务日志不存在下降水位丢包告警
`Enqueue save_async ... previous_chunks_sent=`。若任一项失败，不得用单轮 n=32/64/128
结果替代稳定性结论。

Stage 1 热路径开关已经进入当前默认配置；以下拆分命令可用于单独复核其 activation：

```bash
python competition/minicpmo_b/scripts/validate_candidate_log.py \
  competition/minicpmo_b/results/official_910c_optimized/server.log \
  --require-stage1-full-decode \
  --require-stage1-cpu-slot-mapping \
  --require-stage1-graph-sampler \
  --require-stage1-binary-argmax

python competition/minicpmo_b/scripts/gate_duplex_candidate.py \
  competition/minicpmo_b/results/official_910c_baseline/duplex_rtf/native_duplex_c1_n32.json \
  competition/minicpmo_b/results/official_910c_optimized/duplex_rtf/native_duplex_c1_n32.json \
  --profile stage1-hotpath
```

910C 日志若出现 CPU slot mapping 回退，或 graph sampler activation marker 缺失，
该轮成绩必须作废并恢复 local4 可靠配置复测。

## 5. 三项完整精度门禁

先在相同数据快照、请求数和并发下运行官方基线：

```bash
SUITE=daily-omni MAX_CONCURRENCY=1 REQUIRE_STAGE0_MM_CACHE_DISABLED=1 \
DEPLOY_CONFIG=competition/minicpmo_b/config/ablations/minicpmo_4_5_official_baseline_accuracy_cacheoff.yaml \
RESULT_DIR=competition/minicpmo_b/results/official_910c_baseline/accuracy/daily-omni \
competition/minicpmo_b/scripts/run_accuracy_case.sh

SUITE=videomme NUM_PROMPTS=2700 MAX_CONCURRENCY=4 REQUIRE_STAGE0_MM_CACHE_DISABLED=1 \
DEPLOY_CONFIG=competition/minicpmo_b/config/ablations/minicpmo_4_5_official_baseline_accuracy_cacheoff.yaml \
RESULT_DIR=competition/minicpmo_b/results/official_910c_baseline/accuracy/videomme \
competition/minicpmo_b/scripts/run_accuracy_case.sh

SEED_TTS_HF_WHISPER_MODEL=/workspace/whisper-large-v3 \
SEED_TTS_WAVLM_MODEL=/workspace/wavlm-base-plus \
SEED_TTS_UTMOS_JIT_FILE=/workspace/utmos/utmos.jit \
SEED_TTS_SIM_EVAL=1 SEED_TTS_UTMOS_EVAL=1 \
MIN_SEED_TTS_MEAN_SIM=-1 MIN_SEED_TTS_MEAN_UTMOS=0 \
SUITE=seed-tts NUM_PROMPTS=1000 MAX_CONCURRENCY=4 \
REQUIRE_STAGE0_MM_CACHE_DISABLED=1 \
DEPLOY_CONFIG=competition/minicpmo_b/config/ablations/minicpmo_4_5_official_baseline_accuracy_cacheoff.yaml \
RESULT_DIR=competition/minicpmo_b/results/official_910c_baseline/accuracy/seed-tts \
competition/minicpmo_b/scripts/run_accuracy_case.sh
```

基线精度 overlay 与上游 `vllm_omni/deploy/minicpmo_4_5.yaml` 的唯一差异是 Stage 0
`mm_processor_cache_gb: 0`，并由单测做完整配置等价比较。该设置只修复长稳缓存协议，
不改变模型输出；基线性能矩阵仍必须使用未经修改的上游 baseline YAML。

再运行优化版；除 deploy config、结果目录和优化版 activation gate 外，其余输入
必须与基线完全一致：

```bash
SUITE=daily-omni MAX_CONCURRENCY=1 REQUIRE_STAGE0_MM_CACHE_DISABLED=1 \
DEPLOY_CONFIG=competition/minicpmo_b/config/minicpmo_4_5_910c_low_latency.yaml \
RESULT_DIR=competition/minicpmo_b/results/official_910c_optimized/accuracy/daily-omni \
competition/minicpmo_b/scripts/run_accuracy_case.sh

SUITE=videomme NUM_PROMPTS=2700 MAX_CONCURRENCY=4 REQUIRE_STAGE0_MM_CACHE_DISABLED=1 \
DEPLOY_CONFIG=competition/minicpmo_b/config/minicpmo_4_5_910c_low_latency.yaml \
RESULT_DIR=competition/minicpmo_b/results/official_910c_optimized/accuracy/videomme \
competition/minicpmo_b/scripts/run_accuracy_case.sh

SEED_TTS_HF_WHISPER_MODEL=/workspace/whisper-large-v3 \
SEED_TTS_WAVLM_MODEL=/workspace/wavlm-base-plus \
SEED_TTS_UTMOS_JIT_FILE=/workspace/utmos/utmos.jit \
SEED_TTS_SIM_EVAL=1 SEED_TTS_UTMOS_EVAL=1 \
MIN_SEED_TTS_MEAN_SIM=-1 MIN_SEED_TTS_MEAN_UTMOS=0 \
REQUIRE_STAGE0_MM_CACHE_DISABLED=1 \
SUITE=seed-tts NUM_PROMPTS=1000 MAX_CONCURRENCY=4 \
DEPLOY_CONFIG=competition/minicpmo_b/config/minicpmo_4_5_910c_low_latency.yaml \
RESULT_DIR=competition/minicpmo_b/results/official_910c_optimized/accuracy/seed-tts \
competition/minicpmo_b/scripts/run_accuracy_case.sh
```

Seed-TTS 生成阶段会先把 WAV 与 `seed_tts_eval_manifest.jsonl` 保存到结果目录的
`generated_audio_checkpoint/`，然后运行 Whisper/WavLM/UTMOS。若质量评测中断，
先释放服务和 NPU，再用以下命令离线续评，无需重新生成 1000 条音频：

```bash
SEED_TTS_EVAL_DEVICE=cpu SEED_TTS_WHISPER_BATCH_SIZE=8 \
SEED_TTS_SIM_EVAL=1 SEED_TTS_UTMOS_EVAL=1 \
SEED_TTS_HF_WHISPER_MODEL=/workspace/whisper-large-v3 \
SEED_TTS_WAVLM_MODEL=/workspace/wavlm-base-plus \
SEED_TTS_UTMOS_JIT_FILE=/workspace/utmos/utmos.jit \
python competition/minicpmo_b/scripts/resume_seed_tts_quality.py \
  competition/minicpmo_b/results/official_910c_optimized/accuracy/seed-tts/generated_audio_checkpoint/seed_tts_eval_manifest.jsonl \
  --base-result competition/minicpmo_b/results/official_910c_optimized/accuracy/seed-tts/seed_tts_generation_performance_checkpoint.json \
  --output competition/minicpmo_b/results/official_910c_optimized/accuracy/seed-tts/seed_tts_quality_resumed.json
```

基线目录使用相同命令单独续评。manifest 必须为 1000 行，且最终质量 JSON 的
evaluated/failed 数量仍须满足下述完整性门槛；只有合并结果中的
`seed_tts_quality_complete=true` 才可进入精度比较。

Daily-Omni 不硬编码历史条数：脚本读取准备后的 `qa.json` 并评测全部行。当前本地
10-shard 镜像实测为 1196 条；若主办方最终数据版本为其他条数，以其 `qa.json`
完整行数为准，并在环境清单中记录数据文件 SHA256。

默认优化 YAML 在 Stage 0 显式设置 `mm_processor_cache_gb: 0`。这是长稳正确性
门禁：vLLM 的默认 LRU 需要 API sender 与 Engine receiver 保持完全相同的访问/
淘汰顺序，并发、重试或请求在入队前失败会造成两端状态分叉，最终出现
`AssertionError: Expected a cached item for mm_hash=...`。关闭该缓存不会跳过 VPM、
APM 或 LLM，也不改变权重、采样和输出；它只避免复用前端 HF processor 的结果。
精度与 Demo 服务日志必须通过 `validate_candidate_log.py`，上述断言即使只出现一次，
该轮完整结果也必须作废重跑。

硬门槛：Daily-Omni ≥ 0.78、Video-MME ≥ 0.68、Seed-TTS mean WER ≤ 0.05；
同时相对对应官方基线的精度降幅不得超过 2 个百分点。每个结果目录必须只保留
本轮唯一一个 `qwen_omni_acc_*.json`，再执行：

```bash
for suite in daily-omni videomme seed-tts; do
  python competition/minicpmo_b/scripts/compare_accuracy_results.py \
    "${suite}" \
    "competition/minicpmo_b/results/official_910c_baseline/accuracy/${suite}" \
    "competition/minicpmo_b/results/official_910c_optimized/accuracy/${suite}" \
    --output "competition/minicpmo_b/results/official_910c_optimized/accuracy/${suite}/baseline_gate.json"
done
```

`MIN_SEED_TTS_MEAN_SIM=-1` 与 `MIN_SEED_TTS_MEAN_UTMOS=0` 只用于强制完整产出
1000/1000 条 SIM/UTMOS，并非质量晋级阈值；真正晋级由随后基线比较的 2pp 门禁
决定。运行前必须把 Whisper、WavLM 和 `balacoon/utmos` 的 `utmos.jit` 放入离线
镜像/HF cache，禁止评测过程中临时联网。若主办方最终文档给出不同 TTS-Seed
指标或归一化口径，以官方口径替换该比较，不得用本地 WER 规避官方精度要求。

## 6. Demo 与稳定性

使用默认优化配置启动服务：

```bash
PORT=8091 competition/minicpmo_b/scripts/start_server.sh
```

先运行 `smoke_test.sh`，随后接入官方 vLLM-Omni Demo，依次录制文本、音频、
视频和 text+audio 流式输出。至少完成一轮连续交互稳定性测试，保存服务日志、
HTTP/Demo 错误、音频中断次数、空包、underrun 和 NPU 资源峰值。视频必须能看出
首段音频开始播放、后续 chunk 连续到达以及完整交互结束。

先复制运行元数据和逐请求场景模板。`DEMO_EVIDENCE_TEMPLATE.json` 是生成结果的
字段参考，不再手工填写总计数：

```bash
mkdir -p competition/minicpmo_b/results/official_910c/demo
DEMO_ROOT=competition/minicpmo_b/results/official_910c/demo
cp competition/minicpmo_b/DEMO_RUN_METADATA_TEMPLATE.json \
  "$DEMO_ROOT/demo_run_metadata.json"
for scenario in text audio video text_audio; do
  cp competition/minicpmo_b/DEMO_SCENARIO_EVIDENCE_TEMPLATE.json \
    "$DEMO_ROOT/scenario_${scenario}.json"
done
```

建议使用以下目录布局，所有清单路径都相对于 `demo/`，不得使用绝对路径、`..` 或
符号链接逃逸：

```text
demo/
├── demo_evidence.json
├── demo_run_metadata.json
├── server.log
├── demo.mp4
├── scenario_text.json
├── scenario_audio.json
├── scenario_video.json
└── scenario_text_audio.json
```

四份 `scenario_*.json` 必须分别由文本、音频、视频和 text+audio 的完整交互证明；
不要仅凭 HTTP 200 填写通过。每个请求记录唯一 request ID、带时区开始/结束时间、
`completed`、相对于 Demo 根目录的非空输出文件；音频、视频和 text+audio 请求还要
记录实际收到的正数 `audio_packet_count`。原始 Demo/浏览器事件导出优先于手写摘要。

`started_utc` 与 `finished_utc` 使用带时区的 ISO-8601，例如
`2026-08-12T12:00:00Z`；`continuous_run_minutes` 不得大于两者的真实时间跨度。
`audio_interruption_count`、`empty_audio_packet_count`、`audio_underrun_count` 和
`unexpected_error_count` 必须由实际采集结果填写且最终均为 0。服务必须自然退出，
并设置 `service_exit_clean=true`。

`server.log` 与 `demo.mp4` 必须是 Demo 目录内的非空文件。录屏仅接受可识别的
MP4/WebM 文件头；改扩展名的文本文件会被拒绝。录制与服务自然退出后运行收口工具：

```bash
python competition/minicpmo_b/scripts/finalize_demo_evidence.py \
  --demo-root competition/minicpmo_b/results/official_910c/demo
```

该工具从逐请求记录重算四场景请求数、完成数与音频包数，为元数据、四份场景 JSON、
每个输出文件、服务日志和录屏生成 SHA256，再原子写入 `demo_evidence.json` 并立即
运行 Demo-only 门禁。最终审计仍会独立重算全部计数和哈希，并扫描服务日志中的
`Traceback`、`ERROR`、
`ERR99999`、segmentation fault 和 engine core initialization failure。任何命中都
必须先定位并重新完成干净的 Demo 稳定性运行，不能只从日志中删除对应行。

结束后在服务所在终端发送一次 Ctrl-C，并等待 API Server 与全部 Stage 进程
自然完成清理；不要用 `kill/pkill` 跳过编排器的退出流程。

在构建最终包之前也可随时单独重跑 Demo 审计，不受其他尚未完成的 910C 证据影响：

```bash
python competition/minicpmo_b/scripts/validate_final_evidence.py \
  --demo-only \
  --demo-root competition/minicpmo_b/results/official_910c/demo
```

## 7. 提交审计

先执行静态门禁：

```bash
python competition/minicpmo_b/scripts/validate_submission_package.py \
  --require-tracked \
  --output competition/minicpmo_b/results/official_910c/submission_package_audit.json
```

确认工作树干净、所有必需文件已提交，并完成全部 910C/Demo 证据后生成源码快照：

```bash
python competition/minicpmo_b/scripts/build_submission_artifacts.py \
  --base-ref origin/minicpm-challenge \
  --output-dir competition/minicpmo_b/results/official_910c/final_submission/source
```

最终目录必须包含：冻结后的代码/patch、默认 YAML、启动与 benchmark 脚本、
三项精度原始 JSON/逐项输出、c1/c4/c8 性能 JSON、服务日志、环境与模型清单、
Demo 视频、性能报告和从全新官方镜像开始的复现命令。任何缺失或无法由日志证明
FULL decode 图真实生效的成绩均视为未完成。

根据原始 JSON 自动生成最终报告（仍需人工补充技术说明或异常说明时，可在生成后追加）：

```bash
python competition/minicpmo_b/scripts/render_final_report.py
```

报告保存为 `results/official_910c/final_submission/FINAL_REPORT.md`，且不得再含
“待填写”。最后执行全证据审计：

```bash
python competition/minicpmo_b/scripts/validate_final_evidence.py \
  --output competition/minicpmo_b/results/official_910c/final_evidence_audit.json
```

该门禁直接检查单卡 910C preflight、模型/数据/评测权重 hash、基线与优化版
c1/c4/c8 原始结果和服务日志、SPEAK RTF gate、多轮原始 JSON、三项完整精度与 2pp
gate、26 个阶段的命令/哈希/退出状态、Demo 四场景/视频/日志、干净源码制品及其 SHA256。
源码审计还会打开 `source_snapshot.tar.gz`，逐文件与覆盖完整 HEAD 的
`source_sha256.txt` 核对，并要求环境冻结时的 Git commit 与源码制品 commit 完全一致；
仅修改外层 `artifact_sha256.txt` 不能掩盖归档成员被替换或遗漏。只有审计结果
`passed=true` 才能声明提交包完成。

审计通过后构建确定性最终归档，并再次验证归档内部的逐文件 SHA256：

```bash
python competition/minicpmo_b/scripts/build_final_submission.py
python competition/minicpmo_b/scripts/validate_final_evidence.py --require-package
```

最终 `--require-package` 不要把 `--output` 写入 `results/official_910c/`；该目录本身已
属于归档权威证据，验证后改写其中任一文件会立即使刚验证的 tar 过期。若需要保存
最终验证输出，请使用结果树外路径，例如 `--output /tmp/final_package_audit.json`；
工具会主动拒绝写入结果树内部。

最终上传文件为
`results/official_910c/final_submission/package/minicpmo_b_official_910c.tar.gz`，外部
摘要保存在同目录 `archive_sha256.txt`。归档拒绝符号链接、重复/逃逸路径以及未被
清单覆盖的文件，并固定成员顺序、mtime、UID/GID 和 gzip 时间戳以便复现。构建器
先生成隐藏候选包，使用固定内存的流式 SHA256 验证全部成员、内嵌清单和 metadata，
再与当前结果目录的权威证据文件名及内容逐一对照；全部通过后才原子替换正式 tar，
不会因新包构建失败破坏上一份有效归档。`archive_verification.json` 与
`archive_sha256.txt` 同样以 fsync + 原子替换发布。最终 `--require-package` 会重新打开
tar 独立执行相同内部与权威文件集校验，不只信任已有 sidecar。
