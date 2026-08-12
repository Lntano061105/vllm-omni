# MiniCPM-o 4.5 昇腾挑战赛：vLLM-Omni 子赛道 B

本目录保存单卡 910C 方案的配置、启动脚本、Benchmark 命令、原始结果和复现说明。主线原则是：所有性能改动必须通过 Daily-Omni、Seed-TTS 和 Video-MME 精度门槛，并保持官方 Demo 的流式音频连续性。

## 快速开始

启动优化服务：

```bash
competition/minicpmo_b/scripts/start_server.sh
```

默认 YAML 已指向当前晋级方案：native duplex、Stage 1
`FULL_DECODE_ONLY`、Talker local decode ×4、conditional Euler Token2Wav，
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

候选与当前最佳使用相同输入和请求数测试后，用机器门禁判定，避免误用请求级或
全部 chunk RTF：

```bash
python competition/minicpmo_b/scripts/gate_duplex_candidate.py \
  competition/minicpmo_b/results/current/result.json \
  competition/minicpmo_b/results/candidate/result.json \
  --profile npugraph \
  --output competition/minicpmo_b/results/candidate/gate.json
```

`--profile` 可选 `npugraph`、`prompt-cache`、`combined`。工具只读取官方目标代理
`speak_generation_rtf`，并同时检查成功请求数、音频 turn、SPEAK chunk 数、TTFT、
TTFP、mean 和 p99，任一硬门槛不满足即以非零状态退出。

服务 A/B 完成后验证优化不是“配置存在但运行未命中”：

```bash
python competition/minicpmo_b/scripts/validate_candidate_log.py server.log \
  --require-stage0-mm-cache-disabled \
  --require-stage1-full-decode \
  --require-stage0-ref-cache \
  --require-stage2-prompt-cache \
  --require-stage2-npugraph \
  --min-npugraph-buckets 2
```

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
SUITE=daily-omni MAX_CONCURRENCY=1 \
  competition/minicpmo_b/scripts/run_accuracy_case.sh

SUITE=videomme NUM_PROMPTS=2700 MAX_CONCURRENCY=4 \
  competition/minicpmo_b/scripts/run_accuracy_case.sh

SEED_TTS_HF_WHISPER_MODEL=/path/to/whisper-large-v3 \
SEED_TTS_WAVLM_MODEL=/path/to/wavlm-base-plus \
SEED_TTS_UTMOS_JIT_FILE=/path/to/utmos.jit \
SEED_TTS_SIM_EVAL=1 SEED_TTS_UTMOS_EVAL=1 \
MIN_SEED_TTS_MEAN_SIM=-1 MIN_SEED_TTS_MEAN_UTMOS=0 \
SUITE=seed-tts NUM_PROMPTS=1000 MAX_CONCURRENCY=4 \
  competition/minicpmo_b/scripts/run_accuracy_case.sh
```

Daily-Omni 默认读取准备后的 `qa.json` 并跑完整数据，不依赖容易过期的硬编码条数。
当前本地 10 个 parquet 分片共 1196 条；官方最终镜像若不同，以实际完整行数为准。
比赛封装默认启用 `--require-complete-evaluation`，要求 `completed` 和对应
`evaluated_ok` 都等于请求数，且 HTTP、解析、ASR 等失败计数全部为零；不能用
“少量成功样本达到精度阈值”替代完整集准入。默认 YAML 同时关闭 Stage 0 的
双进程多模态 processor LRU，避免长稳并发/重试后 sender 与 receiver 淘汰顺序
分叉。

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
三项本地最低门槛。需要自定义门槛时可设置
`MIN_DAILY_OMNI_ACCURACY`、`MIN_VIDEOMME_ACCURACY` 和
`MAX_SEED_TTS_MEAN_WER`。TTS-Seed 正式门禁还应启用 SIM/UTMOS；一旦设置
`MIN_SEED_TTS_MEAN_SIM` / `MIN_SEED_TTS_MEAN_UTMOS`，脚本会强制每项质量指标
完整评测全部请求，缺失或失败一条即失败。

Daily-Omni 本地镜像把媒体字节嵌在 parquet 中，准备脚本会流式生成
`qa.json + Videos/`；Video-MME 准备脚本会从 20 个 ZIP 中仅解出评测所需
视频。Seed-TTS WER 依照官方协议需要 `jiwer`、`zhon` 和
`openai/whisper-large-v3`；SIM/UTMOS 还需要 WavLM 与 `utmos.jit`。离线环境应
提前提供并记录三个评测模型的 SHA256，不能在正式计时中临时下载。

Seed-TTS 默认会在 `${RESULT_DIR}/generated_audio_checkpoint/` 先保存全部 24 kHz
mono PCM16 WAV 与原子 `seed_tts_eval_manifest.jsonl`，再启动耗时较长的 Whisper
质量评测。若评测被中断，可在不启动服务、不占用 NPU 的情况下续跑：

```bash
SEED_TTS_EVAL_DEVICE=cpu SEED_TTS_WHISPER_BATCH_SIZE=8 \
python competition/minicpmo_b/scripts/resume_seed_tts_quality.py \
  "${RESULT_DIR}/generated_audio_checkpoint/seed_tts_eval_manifest.jsonl" \
  --base-result "${RESULT_DIR}/seed_tts_generation_performance_checkpoint.json" \
  --output "${RESULT_DIR}/seed_tts_quality_resumed.json"
```

使用 `--base-result` 后，续跑 JSON 同时包含请求 TTFT/TTFP/吞吐与质量字段。
未合并的性能检查点带有 `seed_tts_quality_complete=false`，不得作为正式精度结果。
正式提交时应同时保留 manifest、WAV、性能检查点和合并后的续跑 JSON。

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

官方 910C 完整流程可先无副作用预览：

```bash
python competition/minicpmo_b/scripts/run_official_910c_retest.py
```

只有在官方空闲单卡 910C 主机上才添加 `--execute --confirm-single-910c`。编排器会
先拒绝 910B、多张可见 NPU、残留 vLLM 进程、缺失/不兼容的模型及 Token2Wav
资产、非完整数据、评测模型、占用端口或磁盘不足，再对称执行基线/优化矩阵并生成
c1/c4/c8 双工性能门禁和三项精度相对门禁。详细参数、断点续跑和人工 Demo 步骤见
`competition/minicpmo_b/OFFICIAL_910C_RETEST.md`。

正式交付前还需填写 `DEMO_EVIDENCE_TEMPLATE.json` 并运行：

```bash
python competition/minicpmo_b/scripts/render_final_report.py
python competition/minicpmo_b/scripts/validate_final_evidence.py
python competition/minicpmo_b/scripts/build_final_submission.py
python competition/minicpmo_b/scripts/validate_final_evidence.py --require-package
```

官方 Demo 录屏结束后，先从 `DEMO_RUN_METADATA_TEMPLATE.json` 和
`DEMO_SCENARIO_EVIDENCE_TEMPLATE.json` 的逐请求记录生成哈希绑定的总清单：

```bash
python competition/minicpmo_b/scripts/finalize_demo_evidence.py \
  --demo-root competition/minicpmo_b/results/official_910c/demo
python competition/minicpmo_b/scripts/validate_final_evidence.py \
  --demo-only \
  --demo-root competition/minicpmo_b/results/official_910c/demo
```

该命令不会启动服务；它会拒绝缺失的 910C 原始结果、完整精度、Demo 视频/日志、
未填写报告或非干净源码制品。

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
- `results/candidates/fixed13_25_boundaryfix_hot_n4_v1/`：
  固定 13/25、完整启动预热与多轮边界修复后的 910B4 热态 n=4 三指标结果；
  `gate_vs_same_period_local4.json` 为同周期 combined 门禁，
  `gate_vs_local4.json` 为较早可靠 local4 基线的保守门禁
- `results/candidates/fixed13_25_multiturn_boundaryfix_s2_t3_v1/`：
  同一候选的 2 sessions × 3 turns 生命周期与跨轮稳定性门禁结果
- `results/environment_910b4_fixed13_25_boundaryfix_20260811/`：
  当前开发环境、仅 NPU 5 进程快照、代码 patch、模型元数据和数据版本清单
- 每次测试保留 JSON、服务日志、命令行、Git commit、镜像和 NPU 信息

详细路线见 [PLAN.md](PLAN.md)，报告模板见 [REPORT.md](REPORT.md)，官方单卡
910C 最终执行顺序见 [OFFICIAL_910C_RETEST.md](OFFICIAL_910C_RETEST.md)，逐项
完成度与证据缺口见 [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md)。

## Stage 1 profiler 热路径候选（仅物理 NPU 5）

当前 local4 可靠基线保持不变。获得明确硬件授权后，先分别验证二元控制 argmax
和 CPU slot mapping，再验证 graph sampler + CPU slot + argmax 的组合：

```bash
# 单变量：消除确定性 continue/stop 通用 sampler
DEPLOY_CONFIG=competition/minicpmo_b/config/ablations/minicpmo_4_5_duplex_euler_local4_binary_argmax_npu5.yaml \
PORT=8095 competition/minicpmo_b/scripts/start_server.sh

# 单变量：消除 scheduler-visible 与 runner-local 的单 token slot kernel
DEPLOY_CONFIG=competition/minicpmo_b/config/ablations/minicpmo_4_5_duplex_euler_local4_cpu_slot_npu5.yaml \
PORT=8095 competition/minicpmo_b/scripts/start_server.sh

# 组合：FULL graph 内 exact compact sampler + CPU slot mapping
DEPLOY_CONFIG=competition/minicpmo_b/config/ablations/minicpmo_4_5_duplex_euler_local4_graph_sampler_cpu_slot_npu5.yaml \
PORT=8095 competition/minicpmo_b/scripts/start_server.sh
```

二元控制单变量使用 `--profile host-control`，要求 SPEAK RTF mean 至少改善 3%；
CPU slot 单变量使用 `--profile slot-mapping`，要求至少改善 5%。

组合候选必须执行：

```bash
python competition/minicpmo_b/scripts/validate_candidate_log.py server.log \
  --require-stage1-full-decode \
  --require-stage1-cpu-slot-mapping \
  --require-stage1-graph-sampler \
  --require-stage1-binary-argmax

python competition/minicpmo_b/scripts/gate_duplex_candidate.py \
  baseline.json candidate.json --profile stage1-hotpath
```

也可以用一次服务启动完成 duplex 性能、activation、Seed-TTS 和音频质量门禁：

```bash
DEPLOY_CONFIG=competition/minicpmo_b/config/ablations/minicpmo_4_5_duplex_euler_local4_graph_sampler_cpu_slot_npu5.yaml \
NPU_DEVICE=5 PORT=8095 CASE_NAME=o40_o41_v1 \
REQUIRE_STAGE1_FULL_DECODE=1 \
REQUIRE_STAGE1_CPU_SLOT_MAPPING=1 \
REQUIRE_STAGE1_GRAPH_SAMPLER=1 \
REQUIRE_STAGE1_BINARY_ARGMAX=1 \
DUPLEX_GATE_PROFILE=stage1-hotpath \
BASELINE_DUPLEX_RESULT=competition/minicpmo_b/results/candidates/euler_local4_zero_wait_npu5_n4_v1/result.json \
competition/minicpmo_b/scripts/run_candidate_gate.sh
```

Profiler CSV 可用以下命令统一汇总，避免手工漏算 sampler 控制算子：

```bash
python competition/minicpmo_b/scripts/summarize_ascend_ops.py \
  path/to/ASCEND_PROFILER_OUTPUT/op_statistic.csv \
  --output stage1_ops.json
```

门禁只读取 SPEAK 生成阶段 RTF；目标为 mean 至少改善 12%、p99 不回退，且
TTFT/TTFP 回退不超过 2%。未通过前不得写入默认 910C 配置。

## 提交包静态审计

正式提交前执行不占用 NPU 的静态门禁：

```bash
python competition/minicpmo_b/scripts/validate_submission_package.py \
  --require-tracked \
  --output competition/minicpmo_b/results/submission_package_audit.json
```

它会验证核心文件与文档引用存在、脚本语法和执行权限、默认/基线 YAML 可解析、
默认 910C 配置只绑定可移植的逻辑设备 0，并确保全部必需文件已经纳入 Git。

静态门禁、官方 910C 结果和 Demo 材料全部通过并提交后，生成最终源码制品：

```bash
python competition/minicpmo_b/scripts/build_submission_artifacts.py \
  --base-ref origin/minicpm-challenge \
  --output-dir competition/minicpmo_b/results/final_submission/source
```

最终模式拒绝脏工作树，输出完整 Git archive、相对官方 challenge 分支的 binary
patch、HEAD/base SHA、Git tree、变更列表、源文件/制品 SHA256 和静态审计结果。
`--allow-dirty` 仅供开发检查，生成的 metadata 会标记 `final_candidate=false`。
