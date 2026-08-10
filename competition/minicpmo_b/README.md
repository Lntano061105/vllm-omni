# MiniCPM-o 4.5 昇腾挑战赛：vLLM-Omni 子赛道 B

本目录保存单卡 910C 方案的配置、启动脚本、Benchmark 命令、原始结果和复现说明。主线原则是：所有性能改动必须通过 Daily-Omni、Seed-TTS 和 Video-MME 精度门槛，并保持官方 Demo 的流式音频连续性。

## 快速开始

启动优化服务：

```bash
competition/minicpmo_b/scripts/start_server.sh
```

默认 YAML 已指向当前晋级方案：native duplex、Stage 1
`FULL_DECODE_ONLY`、Talker local decode ×2、conditional Euler Token2Wav，
以及只取消墙钟等待而保留静音 PCM 的 continuation 零等待路径。默认设备号为
逻辑卡 `0`；在多卡开发机上可通过 `NPU_DEVICE=5` 将该逻辑卡绑定到物理 NPU 5。

运行功能冒烟测试：

```bash
competition/minicpmo_b/scripts/smoke_test.sh
```

运行 Seed-TTS 性能测试：

```bash
NUM_PROMPTS=32 MAX_CONCURRENCY=1 competition/minicpmo_b/scripts/benchmark_seed_tts.sh
```

`benchmark_seed_tts.sh` 的 `mean_audio_rtf` 是请求级 RTF，
`mean_audio_chunk_rtf` 是全部稳态 audio chunk 的辅助均值；两者都不是比赛
主 RTF。官方排名口径只统计全双工 **SPEAK 生成**阶段，即 VPM/APM/LLM、
TTS 和 T2W 同时运行时的 `SPEAK→WAV` 完整链路。使用 Realtime 端点执行：

```bash
NUM_PROMPTS=32 MAX_CONCURRENCY=1 \
  competition/minicpmo_b/scripts/benchmark_duplex_rtf.sh
```

结果中的 `mean_audio_speak_generation_rtf` 才是本地比赛口径代理；
`mean_audio_speak_tail_rtf` 与 `mean_audio_chunk_rtf` 仅用于定位尾部和整体流式
表现。最终成绩仍以主办方官方脚本对 SPEAK 阶段的判定为准。

正式性能矩阵只启动一次服务，依次运行并发 1/4/8 的 32/64/128 请求：

```bash
CASE_NAME=optimized_910c competition/minicpmo_b/scripts/run_perf_matrix.sh
```

把多个原始 JSON 汇总成带相对变化的 Markdown 表：

```bash
python competition/minicpmo_b/scripts/summarize_performance.py \
  baseline=competition/minicpmo_b/results/baseline/seed_tts_c1_n4.json \
  optimized=competition/minicpmo_b/results/optimized_single_fastpath/seed_tts_c1_n4.json
```

将本地 parquet/ZIP 数据转换为官方 Benchmark 可直接使用的布局：

```bash
# 冒烟子集；--limit 0 表示准备完整数据
python competition/minicpmo_b/scripts/prepare_accuracy_data.py \
  daily-omni --limit 3
python competition/minicpmo_b/scripts/prepare_accuracy_data.py \
  videomme --limit 3
```

在同一进程命名空间内启动服务并执行单项准入精度测试：

```bash
SUITE=daily-omni NUM_PROMPTS=1197 MAX_CONCURRENCY=1 \
  competition/minicpmo_b/scripts/run_accuracy_case.sh

SUITE=videomme NUM_PROMPTS=2700 MAX_CONCURRENCY=4 \
  competition/minicpmo_b/scripts/run_accuracy_case.sh

SEED_TTS_HF_WHISPER_MODEL=/path/to/whisper-large-v3 \
SUITE=seed-tts NUM_PROMPTS=1000 MAX_CONCURRENCY=4 \
  competition/minicpmo_b/scripts/run_accuracy_case.sh
```

如果复用一个未设置 `--allowed-local-media-path` 的既有服务，可让 Video-MME
客户端内联本地视频，避免本地 `file://` 被 API 拒绝：

```bash
VIDEOMME_INLINE_LOCAL_VIDEO=1 SUITE=videomme PORT=8095 \
  competition/minicpmo_b/scripts/benchmark_accuracy.sh
```

一次启动完成 Daily-Omni 与 Video-MME 各 3 条端到端冒烟：

```bash
competition/minicpmo_b/scripts/run_accuracy_smoke.sh
```

冒烟只检查媒体加载、HTTP 成功率和答案解析，不用 3 条随机样本执行正式
精度门槛判定；完整评测仍由 `benchmark_accuracy.sh` 默认执行 0.78/0.68/0.05
三项官方门槛。需要自定义门槛时可设置
`MIN_DAILY_OMNI_ACCURACY`、`MIN_VIDEOMME_ACCURACY` 和
`MAX_SEED_TTS_MEAN_WER`。

Daily-Omni 本地镜像把媒体字节嵌在 parquet 中，准备脚本会流式生成
`qa.json + Videos/`；Video-MME 准备脚本会从 20 个 ZIP 中仅解出评测所需
视频。Seed-TTS WER 依照官方协议需要 `jiwer`、`zhon` 和
`openai/whisper-large-v3`，离线环境应提前提供 Whisper 本地路径。

在一个进程命名空间内自动启动服务并执行基线/优化 A/B：

```bash
NUM_PROMPTS=8 MAX_CONCURRENCY=1 competition/minicpmo_b/scripts/run_ab_smoke.sh
```

默认路径：

- 模型：`/workspace/MiniCPM-o-4_5`
- Seed-TTS：`/workspace/seed-tts`
- Daily-Omni：`/workspace/MTEB/Daily-Omni`
- Video-MME：`/workspace/Video-MME`
- 服务端口：`8091`

上述路径均可通过脚本同名环境变量覆盖。正式提交前应在官方 910C 镜像中重新执行完整的 32/64/128 请求性能测试和三项精度测试。

采集代码版本、依赖、NPU 信息、模型清单和复现 patch：

```bash
competition/minicpmo_b/scripts/collect_environment.sh
# 如需对全部大权重计算 SHA256（耗时较长）：
HASH_MODEL_WEIGHTS=1 competition/minicpmo_b/scripts/collect_environment.sh
```

## 结果目录

- `results/baseline/`：官方配置、未启用优化时的原始结果
- `results/optimized/`：优化配置结果
- `results/optimized_single_fastpath/`：加入 Stage 2 单请求缓存快路径后的结果
- `results/candidates/cond_rk4_chunk50_hift_wn_talker_sparse_retest_chunk_metric/`：
  O20 之前最佳配置的同口径逐 chunk RTF 复测
- `results/candidates/cond_rk4_chunk50_hift_wn_talker_sparse_full_decode/`：
  当前 910B4 最佳 FULL_DECODE_ONLY 性能与 Seed-TTS WER 门禁结果
- `results/candidates/cond_rk4_chunk50_hift_wn_talker_sparse_piecewise_same_period/`：
  当前最佳的同周期 PIECEWISE 因果对照
- 每次测试保留 JSON、服务日志、命令行、Git commit、镜像和 NPU 信息

详细路线见 [PLAN.md](PLAN.md)，报告模板见 [REPORT.md](REPORT.md)，官方单卡
910C 最终执行顺序见 [OFFICIAL_910C_RETEST.md](OFFICIAL_910C_RETEST.md)。
