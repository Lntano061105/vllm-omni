# 官方单卡 910C 最终复测流程

本流程只用于官方 `quay.io/ascend/vllm-omni:v0.25.0-a3`、单卡 910C。910B4
开发结果不能填入最终成绩表。所有基线/优化测试必须使用相同模型目录、数据、
请求顺序、两次 warmup 和空闲卡。

## 1. 环境冻结

```bash
cd /vllm-workspace/vllm-omni
export MODEL_PATH=/workspace/MiniCPM-o-4_5
export NPU_DEVICE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

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
```

至少提取 mean/p50/p99 `audio_speak_generation_rtf`，并同时保留
`audio_speak_tail_rtf`、全部 `audio_chunk_rtf`、TTFT、TTFP、E2EL、吞吐、
continuity 和 underrun。`audio_speak_generation_rtf` 依据 Realtime audio delta
的模型阶段元数据分离 SPEAK 生成与尾部，是本地复测代理；最终排名必须使用
主办方脚本的阶段判定与统计口径。

## 5. 三项完整精度门禁

```bash
SUITE=daily-omni NUM_PROMPTS=1197 MAX_CONCURRENCY=1 \
RESULT_DIR=competition/minicpmo_b/results/official_910c/accuracy/daily-omni \
competition/minicpmo_b/scripts/run_accuracy_case.sh

SUITE=videomme NUM_PROMPTS=2700 MAX_CONCURRENCY=4 \
RESULT_DIR=competition/minicpmo_b/results/official_910c/accuracy/videomme \
competition/minicpmo_b/scripts/run_accuracy_case.sh

SEED_TTS_HF_WHISPER_MODEL=/workspace/whisper-large-v3 \
SUITE=seed-tts NUM_PROMPTS=1000 MAX_CONCURRENCY=4 \
RESULT_DIR=competition/minicpmo_b/results/official_910c/accuracy/seed-tts \
competition/minicpmo_b/scripts/run_accuracy_case.sh
```

硬门槛：Daily-Omni ≥ 0.78、Video-MME ≥ 0.68、Seed-TTS mean WER ≤ 0.05；
同时相对主办方对应官方基线的精度降幅不得超过 2 个百分点。

## 6. Demo 与稳定性

使用默认优化配置启动服务：

```bash
PORT=8091 competition/minicpmo_b/scripts/start_server.sh \
  > competition/minicpmo_b/results/official_910c/demo_server.log 2>&1 &
demo_server_pid=$!
```

先运行 `smoke_test.sh`，随后接入官方 vLLM-Omni Demo，依次录制文本、音频、
视频和 text+audio 流式输出。至少完成一轮连续交互稳定性测试，保存服务日志、
HTTP/Demo 错误、音频中断次数、空包、underrun 和 NPU 资源峰值。视频必须能看出
首段音频开始播放、后续 chunk 连续到达以及完整交互结束。

结束后执行 `kill -INT "${demo_server_pid}"` 并等待服务清理完成，避免残留子进程
影响后续资源数据。

## 7. 提交审计

最终目录必须包含：冻结后的代码/patch、默认 YAML、启动与 benchmark 脚本、
三项精度原始 JSON/逐项输出、c1/c4/c8 性能 JSON、服务日志、环境与模型清单、
Demo 视频、性能报告和从全新官方镜像开始的复现命令。任何缺失或无法由日志证明
FULL decode 图真实生效的成绩均视为未完成。
