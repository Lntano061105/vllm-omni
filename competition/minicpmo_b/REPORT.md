# 性能与精度报告

## 1. 环境

- Git commit：待填写
- 镜像：`quay.io/ascend/vllm-omni:v0.25.0-a3`
- 开发验证 NPU：单卡 910B4-1 64GB；正式评测目标为单卡 910C
- 当前环境：CANN 9.0.0、torch/torch_npu 2.10.0、vLLM-Omni 0.25.0
- 模型路径与校验值：待填写

## 2. 改动摘要

| 编号 | 改动 | 预期影响 | 精度风险 | 状态 |
|---|---|---|---|---|
| O1 | 首块 codec frame 25 → 4，后续保持 25 | 降低 TTFP | 低 | 冒烟通过 |
| O2 | 共享内存轮询间隔 10 ms → 1 ms | 降低阶段交接等待 | 无 | 冒烟通过，待消融 |
| O3 | Token2Wav 初始 state 缓存 | 降低固定音色/重复参考音频 TTFP | 低 | 单元测试通过 |
| O4 | Stage 2 NPU 预热 4/25 帧 shape | 消除部分冷请求编译 | 低 | 冒烟通过 |
| O5 | 并发 1 Flow/HiFT cache 零复制快路径 + CFM 时间轴缓存 | 降低逐 chunk RTF | 低 | NPU A/B 通过 |
| O6 | speaker projection 缓存 + 默认初始 state 只读共享 | 降低首包前的重复投影和大 cache 拷贝 | 低 | 单元测试通过；Seed-TTS 无回归，主要收益在 Demo |
| O7 | Flow Matching Euler steps 10 → 8 | 直接减少 CFM 主干计算 | 中 | 性能与 3 条 WER 冒烟通过；扩大 WER/主观音质待测 |
| O8 | 缓存固定 timestep embedding；单请求 CFG latent 使用只读视图；预分配连续 estimator cache | 减少每个音频 chunk 的小算子、分配和张量复制 | 低 | 31 项定向单元测试与 NPU A/B 通过 |
| O9 | 已初始化的不同参考音频按 cache shape 跨 prompt 合批 | 提升并发 4/8 的 Stage 2 利用率 | 低 | 35 项定向单元测试通过；并发 NPU A/B 待测 |
| O10 | 条件单分支缩放近似 CFG + 6-step + 50 帧稳态 chunk | 大幅减少 DiT 分支/步数并摊薄逐块开销 | 高 | 淘汰：3 条 mean/median WER=1.0，且严重削波 |
| O11 | 不放大的 conditional-only + 6-step + 50 帧稳态 chunk | 保留单分支减算，避免 1.7× 幅度放大 | 高 | WER/波形评测中 |
| O12 | 完整 CFG + 6-step + 50 帧稳态 chunk | 保留 CFG 精度，仅减少积分步数和逐块开销 | 中 | 待性能与精度测试 |
| O13 | 精度门控 conditional-only RK4 + 50 帧稳态 chunk | 减少 Stage 2 分支计算并保持可接受音质 | 中 | 当前最佳候选；3/3 Seed-TTS WER=0 |
| O14 | HiFT weight norm 启动时物化 | 消除每次 vocoder forward 的 weight norm 重参数化 | 低 | 当前最佳候选保留 |
| O15 | Talker codec 稀疏传输与 latent 路由 | 仅在 4/50 帧边界执行 D2H/跨阶段传输 | 低 | 当前最佳候选保留 |
| O16 | Stage 1 FP16 / W8A8 | 降低计算精度或权重带宽 | 中 | 淘汰：单请求 TTFT、TTFP、RTF 均回退 |
| O17 | Stage 0 混合 W8A8（仅量化 Qwen3 2–33 层） | 降低权重内存、扩大 KV cache | 中 | 单请求首响淘汰；保留为 910C 并发消融候选 |
| O18 | 独立逐 chunk RTF 采集 | 按相邻音频包间隔 / 当前包音频时长统计 | 无 | 已实现并用于同周期 A/B |
| O19 | Talker compact Top-K sampling | 仅对候选集合执行 softmax/multinomial | 低 | 淘汰：微基准有收益，但端到端 Stage 1 TPOT 无改善 |
| O20 | Stage 1 `FULL_DECODE_ONLY` + Npugraph_ex/ACLGraph | 将 attention、KV 更新与 Talker decode 纳入整图，减少 Host/算子下发开销 | 低 | 晋级默认配置；chunk RTF -29.68%，3/3 WER=0 |
| O21 | Talker scatter repetition penalty | 用 `unique + gather/scatter` 避免完整词表 `bincount` | 低 | 淘汰：微基准加速，但端到端 TPOT +1.41%、chunk RTF +2.04% |
| O22 | Stage 1 同步调度 + `FULL_DECODE_ONLY` | 尝试减少异步 scheduler/图 replay 间的 Host 等待 | 低 | 淘汰：TPOT +1.36%、chunk RTF +0.82%，TTFT/TTFP 均回退 |
| O23 | Talker runner-local 两步直连 decode | 每次 scheduler/图 replay 连续生成 2 个 codec token，减少 Host 往返 | 低 | 910B4 c1 晋级候选：chunk RTF -11.09%、请求 RTF -9.85%、3/3 WER=0；c4/910C 待门禁 |
| O24 | Prompt shape bucket + 多 shape 启动预热 | 尝试消除不同参考音频的 Conformer 首次 shape 成本 | 低 | 淘汰：c4 TTFP 仍为 12.35 s，chunk RTF 恶化至 0.5272 |
| O25 | Ascend fused relative attention / CPU Conformer | 降低新 shape 编译成本或绕过 NPU 冷启动 | 中 | 淘汰：fused 稳态 RTF 约回退 9%；CPU BF16 encoder 约 6.66 s/chunk |
| O26 | 取消 native silence continuation 的墙钟 pacing | 保留送入模型的 1 秒静音 payload，仅取消每轮默认 1000 ms 人工等待 | 低 | 当前 910B4 最佳：TTFT/TTFP -38.82%，官方口径 SPEAK RTF -62.93% |

## 3. 性能结果

### 指标口径

历史 JSON 中的 `mean_audio_rtf` 是请求级指标；`mean_audio_chunk_rtf` 从第二个
音频包开始，以相邻包到达间隔除以当前包音频时长，汇总**全部**稳态 chunk。
根据官方说明，这两项都不能替代比赛主 RTF：排名只统计全双工 SPEAK 生成阶段，
即 VPM/APM/LLM、TTS 和 T2W 同时运行时的 `SPEAK→WAV` 完整链路。当前 Realtime
Benchmark 新增 `audio_speak_generation_rtf`，并把下游尾部单列为
`audio_speak_tail_rtf`；最终结论仍以主办方脚本的阶段判定为准。下文已有
`chunk RTF` 均为历史“全部稳态 chunk”辅助结果，除非明确标注 SPEAK 生成。

| 版本 | 并发 | 请求数 | TTFT mean/p50/p99 ms | TTFP mean/p50/p99 ms | audio RTF mean/p50/p99 | 吞吐 req/s |
|---|---:|---:|---:|---:|---:|---:|
| 官方基线 | 1 | 32 | 待填写 | 待填写 | 待填写 | 待填写 |
| 优化版 | 1 | 32 | 待填写 | 待填写 | 待填写 | 待填写 |
| 官方基线 | 4 | 64 | 待填写 | 待填写 | 待填写 | 待填写 |
| 优化版 | 4 | 64 | 待填写 | 待填写 | 待填写 | 待填写 |
| 官方基线 | 8 | 128 | 待填写 | 待填写 | 待填写 | 待填写 |
| 优化版 | 8 | 128 | 待填写 | 待填写 | 待填写 | 待填写 |

### 910B4 开发机小样本 A/B

以下结果使用同一代码、同一模型、同一 4 条 Seed-TTS 输入、并发 1、2 次预热。它只验证方向，不替代官方 910C 的 32 请求结果。

| 指标 | 官方 YAML | 低延迟 YAML | 相对变化 |
|---|---:|---:|---:|
| 成功请求 | 4/4 | 4/4 | 持平 |
| Mean TTFT | 376.10 ms | 373.53 ms | -0.68% |
| Mean TTFP | 1521.68 ms | 897.16 ms | -41.04% |
| Mean audio RTF | 0.7731 | 0.7571 | -2.07% |
| Mean E2EL | 2800.95 ms | 2752.81 ms | -1.72% |
| Throughput | 0.3569 req/s | 0.3632 req/s | +1.75% |
| Streaming continuity OK | 100% | 100% | 持平 |

首个预热请求由约 70.0 秒降至 37.0 秒。优化版启动时间由约 318 秒增加到 367 秒，说明一部分延迟被显式迁移到服务初始化阶段；官方性能统计执行两次预热，因此最终得分收益主要来自更早的 4 帧首音频块。

### 前一代最佳有效候选

配置：`config/ablations/minicpmo_4_5_cond_rk4_chunk50_hift_wn_talker_sparse.yaml`。
测试条件仍为单卡 910B4、相同 4 条 Seed-TTS、并发 1、两次预热；不能冒充
910C 正式成绩。

| 指标 | 最初官方 YAML | 当前最佳 | 相对变化 |
|---|---:|---:|---:|
| Mean TTFT | 376.10 ms | 367.44 ms | -2.30% |
| Mean TTFP | 1521.68 ms | 655.50 ms | -56.92% |
| 请求级 Mean audio RTF | 0.7731 | 0.5890 | -23.81% |
| Mean E2EL | 2800.95 ms | 2146.11 ms | -23.38% |
| Throughput | 0.3569 req/s | 0.4658 req/s | +30.52% |
| Streaming continuity | 100% | 100% | 持平 |
| Mean underrun | 0 s | 0 s | 持平 |

原始结果：
`results/candidates/cond_rk4_chunk50_hift_wn_talker_sparse/performance/seed_tts_c1_n4.json`。
Seed-TTS 小门禁为 3/3 成功、Mean/Median WER=0，无削波、空 PCM、ASR 或请求失败。

在加入逐 chunk 指标后的同时段复测中，当前最佳配置得到 TTFT 367.62 ms、
TTFP 658.22 ms、请求级 RTF 0.5936、稳态 chunk RTF 0.4239（median 0.4151、
p99 0.5559），连续率 100%、underrun 0。原始结果位于
`results/candidates/cond_rk4_chunk50_hift_wn_talker_sparse_retest_chunk_metric/performance/seed_tts_c1_n4.json`。

### 当前最佳：Stage 1 全 Decode 图

配置：
`config/ablations/minicpmo_4_5_cond_rk4_chunk50_hift_wn_talker_sparse_full_decode.yaml`，
并已提升到默认 `config/minicpmo_4_5_910c_low_latency.yaml`。该候选仅把 Stage 1
从 `PIECEWISE` 改为 `FULL_DECODE_ONLY`，抓图尺寸为 `[1, 2, 4]`；同周期对照
其余 YAML、4 条 Seed-TTS 输入、两次预热、并发和 NPU 均完全一致。两边生成
token 数均为 54、音频总时长均为 14.60 s，因此收益不是靠缩短输出获得。

| 指标 | 同周期 PIECEWISE | FULL_DECODE_ONLY | 相对变化 |
|---|---:|---:|---:|
| Mean TTFT | 374.28 ms | 376.70 ms | +0.65% |
| Mean TTFP | 665.73 ms | 646.28 ms | -2.92% |
| 请求级 Mean audio RTF | 0.59890 | 0.45273 | -24.41% |
| Mean steady chunk RTF | 0.42574 | 0.29937 | -29.68% |
| Median steady chunk RTF | 0.41108 | 0.26099 | -36.51% |
| P99 steady chunk RTF | 0.57348 | 0.56010 | -2.33% |
| Mean E2EL | 2174.64 ms | 1649.80 ms | -24.13% |
| Throughput | 0.45973 req/s | 0.60592 req/s | +31.80% |
| Streaming continuity / underrun | 100% / 0 s | 100% / 0 s | 持平 |

主测试请求的 Stage 1 平均 TPOT 从 16.993 ms 降到 10.864 ms，下降
36.07%。日志确认 `enable_npugraph_ex=True`、`splitting_ops=[]`，并成功完成
`Capturing CUDA graphs (decode, FULL)`，说明 attention/KV decode 路径实际进入
了全图，而不是配置被兼容性检查静默降级。Seed-TTS 小门禁 3/3 成功，
Mean/Median WER=0，请求失败、空 PCM 和 ASR/WER 失败均为 0。

性能原始结果：
`results/candidates/cond_rk4_chunk50_hift_wn_talker_sparse_full_decode/performance/seed_tts_c1_n4.json`；
同周期对照：
`results/candidates/cond_rk4_chunk50_hift_wn_talker_sparse_piecewise_same_period/performance/seed_tts_c1_n4.json`；
精度结果：
`results/candidates/cond_rk4_chunk50_hift_wn_talker_sparse_full_decode/accuracy/seed-tts/qwen_omni_acc_seed_tts_20260810-015804.json`。

### RTF 主瓶颈定量归因

在 O20 之前，50 个 codec frame 对应约 2 秒音频。服务复测中 Stage 1 的平均 TPOT 为
16.756 ms，因此收集一个稳态 chunk 的串行 Talker 时间约为：

`50 × 16.756 ms = 837.8 ms`，对应 RTF `837.8 / 2000 = 0.4189`。

当时实测端到端稳态 chunk RTF 为 0.4239，两者仅差约 0.005。另一方面，Stage 2
独立 NPU profile 在相同 conditional RK4、1 step、50 frame 条件下，Token2Wav
总耗时约 156.3 ms，独立 RTF 仅 0.0782，其中 CFM 120.6 ms、encoder 11.9 ms、
HiFT 22.6 ms。这证明当前 chunk RTF 几乎完全由 Stage 1 每个 codec token 的
串行 Talker 解码决定；继续优化 Stage 2 已不可能带来同量级 RTF 改善。
O20 随后直接验证了这一归因：整图消除大量 Host/launch 开销后，Stage 1 TPOT
下降 36.07%，steady chunk RTF 同步下降 29.68%。剩余 p99 改善较小，后续应
优先分析图 replay 间的 scheduler/connector 尾延迟，而不是继续压缩 Stage 2。

对 O20 的 Stage 1 FULL decode 路径做约 178 个 codec step 的 Ascend profiler
采样后，NPU 计算合计约 421.0 ms（约 2.365 ms/token），而 profiler 下的
device free 时间约 1949.5 ms（约 10.95 ms/token）。主要设备算子为 MatMulV2
152.36 ms、FusedInferAttentionScore 95.07 ms、ScatterElements 34.92 ms、
slot mapping 33.81 ms 和 Bincount 25.40 ms；Host 侧 `NOTIFY_WAIT` 851.0 ms，
`aten::copy_` 186.3 ms/5099 次，`aten::item` 154.5 ms/1203 次，
`aten::bincount` 139.0 ms。该证据说明剩余瓶颈仍以 Host 等待、小张量同步和
逐 token 调度为主，单纯压缩大矩阵计算的收益上限有限。原始 profiler 位于
`results/profiles/talker_full_decode/stage1_rank0/`。

按模型实际 `top_k=100` 复测，compact sampling 微基准把采样子路径由约
1.199 ms 降到 1.030 ms，但在完整
服务中 Stage 1 TPOT 从 16.756 ms 变为 16.863 ms，没有可测收益；同时 TTFT
和 TTFP 分别回退 4.47% 和 2.26%。其表面 chunk RTF 为 0.3993，但生成音频总
时长从 14.60 s 变为 13.72 s，且 Stage 1 TPOT 未下降，不能作因果归因，因此
淘汰，不进入默认配置。

另一条保持原 dense multinomial、只把 repetition penalty 改为对最近 16 个
token 做 `unique + gather/scatter` 的精确稀疏实现也被微基准否决：repetition
penalty 单项由约 0.572 ms 增至 0.653 ms，完整采样由 1.199 ms 增至 1.561 ms。
昇腾上这些小张量离散算子的启动成本高于避免 dense 6562 维操作的收益。

在 FULL decode 图下进一步实现的 O21 scatter 版本，其独立微基准反而显示
repetition penalty 由 0.605 ms 降至 0.339 ms、完整 sampling 由 1.232 ms
降至 0.889 ms；但完整服务的 Stage 1 TPOT 从 10.864 ms 增至 11.017 ms，
mean chunk RTF 从 0.29937 增至 0.30549。生成 token、音频帧数保持一致，说明
该小算子微基准收益在整图 replay/调度中没有转化为端到端收益，因此实现保留
为默认关闭的消融路径，不进入最终配置。原始结果：
`results/candidates/full_decode_scatter_repetition/performance/seed_tts_c1_n4.json`。

O22 将 Stage 1 的 `async_scheduling` 从 `true` 改为 `false`，其余配置保持
O20 不变。首次启动因 `/workspace/MiniCPM-o-4_5` 挂载短暂不可见而失败；挂载
恢复后在独立目录重新测试，日志确认 Stage 1 异步调度已关闭且 FULL decode
图正常捕获。4 条主请求的 Stage 1 TPOT 为 10.781、10.494、10.346、
12.426 ms，平均 11.012 ms，相对 O20 的 10.864 ms 回退 1.36%。mean TTFT
402.10 ms、TTFP 666.57 ms、请求级 RTF 0.45796、mean chunk RTF 0.30181、
E2EL 1683.78 ms、吞吐 0.59371 req/s；连续率 100%、underrun 0，生成
350400 音频帧。同步调度未降低 Host 尾延迟，故淘汰。原始结果：
`results/candidates/full_decode_sync_scheduler_retest/performance/seed_tts_c1_n4.json`。

### 当前低延迟晋级候选：Talker runner-local 两步直连 decode

配置：`config/ablations/minicpmo_4_5_full_decode_local2.yaml`。该候选在 O20
基础上，让 Stage 1 runner 每次图 replay 连续执行两个完全相同的自回归 decode
step，再把 codec token 交给原有流式链路；不改变采样参数、模型权重、停止条件、
音频 chunk 大小或 Stage 2 算法。910B4 同周期 c1、4 条 Seed-TTS 结果如下：

| 指标 | O20 FULL_DECODE_ONLY | O23 local decode ×2 | 相对变化 |
|---|---:|---:|---:|
| Mean TTFT | 376.70 ms | 378.27 ms | +0.42% |
| Mean TTFP | 646.28 ms | 642.19 ms | -0.63% |
| 请求级 Mean audio RTF | 0.45273 | 0.40813 | -9.85% |
| Mean steady chunk RTF | 0.29937 | 0.26617 | -11.09% |
| Mean E2EL | 1649.80 ms | 1505.33 ms | -8.76% |
| Throughput | 0.60592 req/s | 0.66405 req/s | +9.59% |
| Streaming continuity / underrun | 100% / 0 s | 100% / 0 s | 持平 |

生成 token 数、音频总帧数与 O20 保持一致；4/4 请求成功。Stage 1 TPOT 约
下降 15.1%，Seed-TTS 3 条小门禁 Mean/Median WER=0。原始结果位于
`results/candidates/full_decode_local2_direct_gate/`。该结果证明 O23 是当前
910B4 上可靠的低延迟方向，但尚未替换 910C 默认 YAML：必须先完成 c4/c8、
完整 Seed-TTS 和官方 910C 复测，避免把单并发收益误当成正式比赛成绩。

### 当前最佳：Euler/local2 + native silence 零墙钟等待

Native duplex 在每个 silence continuation 进入模型前原本固定执行
`await asyncio.sleep(chunk_period_ms / 1000)`；默认 `chunk_period_ms=1000`，
因此一次请求会引入与 NPU 计算无关的整秒级串行等待。此前把覆盖参数放在
`connectors.extra` 并未生效：部署图中只有 Stage 1→2 connector，而创建
continuation 的 API Server 使用的是顶层 session 配置。本轮新增顶层配置：

```yaml
duplex_session:
  native_silence_continuation_delay_ms: 0
```

默认值为 `null` 时仍保持原有实时 pacing；显式设为 `0` 只取消墙钟等待，模型
收到的 1 秒静音 PCM、状态转换和生成终止语义均不改变。配置解析相关测试
14/14 通过。以下为物理 NPU 5、端口 8095、并发 1、1 次预热 + 4 次正式请求，
并严格只统计 `metadata.speak_tail=false` 的 SPEAK 生成 chunk：

| 指标 | RK4/local2 基线 | Euler/local2 | Euler/local2 + O26 | 相对 RK4 基线 |
|---|---:|---:|---:|---:|
| TTFT mean | 3090.497 ms | 2884.738 ms | **1890.822 ms** | **-38.82%** |
| TTFP mean | 3090.193 ms | 2884.431 ms | **1890.509 ms** | **-38.82%** |
| SPEAK 生成 RTF mean | 2.128422 | 2.002694 | **0.788915** | **-62.93%** |

当前最佳的 TTFT median/p99 为 1889.454/1930.985 ms，TTFP median/p99 为
1889.140/1930.678 ms；SPEAK RTF median/p99 为 0.788603/0.901119，范围
0.716372–0.908442，共 12 个有效 SPEAK chunk。4/4 请求均产生 4 个音频包和
3720 ms 完整音频，`response.done=1`、`cancelled=0`、`errors=[]`，终止活动、
终止标点、完整音频与 transcript 检查全部通过。原始结果：
`results/candidates/euler_local2_zero_silence_wait_npu5_n4_v1/result.json`。

同一候选先通过 Seed-TTS 三条本地 Whisper 小门禁（mean/median WER=0），随后
扩大到 10 条：10/10 请求成功，mean WER=0.0253、median WER=0，请求失败、
空 PCM、ASR/WER 失败均为 0，满足当前 `mean WER <= 0.05` 门槛。10 条
`/chat/completions` 辅助性能为 TTFT mean 366.74 ms、TTFP mean 521.53 ms；
其请求级 audio RTF 0.34 不等同于比赛 duplex SPEAK RTF。

Daily-Omni 三条端到端冒烟为 3/3 HTTP 成功、3/3 正确；Video-MME 三条为
3/3 HTTP 成功、2/3 正确。Video-MME 在未给服务配置本地媒体白名单时会被
HTTP 400 拒绝，benchmark 脚本现支持 `VIDEOMME_INLINE_LOCAL_VIDEO=1`，可将
本地视频内联到请求，默认行为保持不变。这些仍是小样本功能/回归门禁，不能
替代三个官方完整集。有效结果分别位于：

- `results/candidates/euler_local2_zero_wait_seed_tts_acc_n10_v1/`
- `results/candidates/euler_local2_zero_wait_daily_omni_smoke_n3_v1/`
- `results/candidates/euler_local2_zero_wait_videomme_smoke_n3_v2_inline/`

零等待候选还完成了 2 个会话、每会话连续 3 轮的 native duplex 稳定性测试。
6/6 音频轮成功，`response.done=6`、`cancelled=0`、错误数 0，无 stale audio、
重复 `response.speak`、跨轮污染或终止语义异常；所有音频均有完整 transcript，
playback history、事件顺序和会话关闭检查通过。该测试得到 TTFT/TTFP mean
1844.35/1844.12 ms，22 个 SPEAK chunk 的 mean/median/p99 RTF 为
0.744630/0.770439/0.894368。结果位于
`results/candidates/euler_local2_zero_wait_multiturn_s2_t3_v1/result.json`。

### 不同参考音频的 TTFP 冷启动归因与淘汰实验

c4 首批不同参考音频的 24–25 s 阻塞已定位到 Stage 2 CosyVoice Conformer
encoder 对新 sequence/cache shape 的首次执行。98-token prompt 的特征提取约
1.52 s，初始状态约 11.03 s，其中 encoder 约 10.85 s、CFM 仅约 0.17 s；同一
shape 第二次初始化约 129 ms。证据位于
`results/candidates/token2wav_cold_prompt_profile.json`、
`token2wav_cold_prompt_repeat_profile.json` 和
`token2wav_cold_prompt_components.json`。

按 16 帧 bucket padding 并在启动时预热 64–224 帧 shape，未覆盖后续每个
prompt cache 长度对应的 live encoder shape，并永久增大 attention cache；c4/n8
得到 TTFP 12.35 s、chunk RTF 0.5272，故淘汰。Ascend fused relative attention
冷 encoder 只改善约 8%，稳态 RTF 反而约回退 9%；动态 `torch.compile` 被
`aten.true_divide.Tensor` converter 缺失和动态 MatMul shape 阻断。CPU encoder
同样不可用：FP32 16 线程约 476 ms/chunk，而 BF16 16 线程实测 encoder
6659.2 ms/chunk、总 RTF 3.4158。对应原始结果均保存在
`results/candidates/token2wav_*profile.json`。因此这些实验不进入正式配置；要
实质改善 c4/c8 TTFP，需要完整的 Conformer ACL static graph/cache bucket，并在
输出后裁剪 cache，或让该 encoder 的相对位置 attention 真正支持动态 shape。

### 权重压缩与低精度消融

以下变化均相对当前最佳 BF16 候选；音频总时长不同的测试不能用 E2EL 或吞吐
单独证明算子加速。

| 候选 | TTFT | TTFP | 请求级 RTF | E2EL | 吞吐 | 音频总时长 | 决策 |
|---|---:|---:|---:|---:|---:|---:|---|
| Stage 1 FP16 | +3.88% | +6.03% | +2.56% | -0.62% | +0.63% | -3.56% | 淘汰 |
| Stage 1 静态 W8A8 | +8.42% | +5.53% | +4.69% | -0.94% | +0.95% | -5.21% | 淘汰 |
| Stage 0 混合 W8A8 | +4.20% | +0.91% | +0.64% | -6.57% | +7.02% | -7.40% | 单请求淘汰 |

Stage 1 W8A8 权重内存由约 0.642 GB 降至 0.470 GB，但 batch=1 的量化/反量化
与 INT8 小矩阵调度开销抵消了带宽收益。Stage 0 混合 W8A8 将已加载权重由
16.84 GB 降至 11.11 GB，KV cache 容量从 56,960 增至 98,688 tokens；因此
它仍值得在 910C c4/c8 下测试容量和吞吐，但不能进入低延迟默认配置。

在 O26 当前最佳链路上又做了独立 FP16 复测，避免历史配置差异干扰结论。
相同物理 NPU 5、1 次预热 + 4 次正式请求下，FP16 的 TTFT/TTFP/SPEAK RTF
分别为 1954.429 ms、1954.122 ms、0.810368，相对 BF16 最佳分别回退
3.36%、3.36%、2.72%。因此最终低延迟主线继续使用 BF16；FP16 配置和原始
结果保留在 `config/ablations/minicpmo_4_5_duplex_full_decode_local2_euler_fp16_npu5.yaml`
及 `results/candidates/euler_local2_zero_wait_fp16_npu5_n4_v1/result.json`，作为
低精度消融证据。

### Stage 2 单请求快路径增量结果

在相同 4 条输入、相同预热和相同低延迟 YAML 上，仅增加 O5：

| 指标 | O1–O4 | O1–O5 | 增量变化 |
|---|---:|---:|---:|
| Mean TTFT | 373.53 ms | 368.41 ms | -1.37% |
| Mean TTFP | 897.16 ms | 846.48 ms | -5.65% |
| Mean audio RTF | 0.7571 | 0.7055 | -6.81% |
| Mean E2EL | 2752.81 ms | 2567.68 ms | -6.73% |
| Throughput | 0.3632 req/s | 0.3894 req/s | +7.21% |
| Streaming continuity OK | 100% | 100% | 持平 |

相对最初官方 YAML，小样本累计变化为：TTFP -44.37%、audio RTF
-8.74%、E2EL -8.33%、吞吐 +9.08%。正式结论仍需 910C 大样本复测。

speaker projection 单独复测的 4 条结果为 TTFT 368.85 ms、TTFP
875.17 ms、audio RTF 0.7439、连续率 100%。与上一轮差异小于这组极小
样本的运行抖动，因此暂不把它单独计为确定收益；O6 的初始 state 零拷贝
复测结果为 TTFT 383.79 ms、TTFP 877.59 ms、audio RTF 0.7273、连续率
100%、underrun 0。Seed-TTS 每条请求使用不同参考音频，默认音色模板不会
命中，因此这组结果仅证明没有回归；O6 的主要受益场景是官方 Demo 的固定
默认音色和重复请求，不计入 Seed-TTS 性能收益。

### 8 步 Flow Matching 候选

在同一版代码、同一 4 条输入和同样 2 次预热下，8 步相对同时段 10 步结果：

| 指标 | 10 steps | 8 steps | 变化 |
|---|---:|---:|---:|
| Mean TTFT | 383.79 ms | 380.34 ms | -0.90% |
| Mean TTFP | 877.59 ms | 823.08 ms | -6.21% |
| Mean audio RTF | 0.7273 | 0.6970 | -4.16% |
| Mean E2EL | 2659.62 ms | 2541.23 ms | -4.45% |
| Throughput | 0.3759 req/s | 0.3934 req/s | +4.66% |
| Streaming continuity | 100% | 100% | 持平 |
| Mean underrun | 0 s | 0 s | 持平 |

该候选的性能方向成立，但降低 CFM 积分步数具有音质风险；只有完整 Seed-TTS
WER 与官方 Demo 主观听感均通过后，才会把 `token2wav_n_timesteps: 8` 合入
最终配置。当前 3 条英文 Seed-TTS 冒烟为 3/3 成功，Mean/Median WER 均为
0，且无空 PCM、ASR 失败；该结果仅用于放行扩大样本测试。

### Stage 2 固定时间嵌入与连续缓存

O8 在 4 条相同 Seed-TTS 输入、并发 1、2 次预热下，相对同时段 O1–O6：

| 指标 | O1–O6 | O1–O6 + O8 | 变化 |
|---|---:|---:|---:|
| Mean TTFT | 383.79 ms | 363.22 ms | -5.36% |
| Mean TTFP | 877.59 ms | 846.43 ms | -3.55% |
| Mean audio RTF | 0.7273 | 0.7014 | -3.57% |
| Mean E2EL | 2659.62 ms | 2564.06 ms | -3.59% |
| Throughput | 0.3759 req/s | 0.3899 req/s | +3.73% |
| Streaming continuity | 100% | 100% | 持平 |
| Mean underrun | 0 s | 0 s | 持平 |

O8 不改变采样、积分步数或音频算法，只消除固定 timestep 的重复计算、单请求
CFG latent 的实体复制以及 estimator cache 的逐步分配/末尾 `stack`，因此作为
低风险默认代码路径保留。收益仍需在 910C 的 32/64/128 请求矩阵中确认。

### 首块 3 帧消融（淘汰）

首块从 4 帧继续降到 3 帧没有降低实际首包：相对 4 帧候选，TTFP +1.11%、
audio RTF +0.48%、E2EL +0.05%，连续率虽仍为 100%，但不存在性能收益。
因此最终配置继续使用 4 帧首块，不再测试更小的 1/2 帧路径。

### 高杠杆 Stage 2 减算候选

O10 将完整 CFG 的每步 2 路 DiT 改为仅计算 conditional 路，并用原始
`1 + cfg_rate` 缩放近似完整 CFG；同时将 Euler steps 10 → 6、稳态 chunk
25 → 50 帧。首块仍为 4 帧，因此不主动牺牲 TTFP。

| 指标 | O1–O8 | O1–O10 | 增量变化 | 相对最初官方 YAML |
|---|---:|---:|---:|---:|
| Mean TTFT | 363.22 ms | 402.68 ms | +10.86% | +7.07% |
| Mean TTFP | 846.43 ms | 768.78 ms | -9.17% | -49.48% |
| Mean audio RTF | 0.7014 | 0.6391 | -8.88% | -17.34% |
| Mean E2EL | 2564.06 ms | 2322.06 ms | -9.44% | -17.10% |
| Throughput | 0.3899 req/s | 0.4305 req/s | +10.42% | +20.62% |
| Streaming continuity | 100% | 100% | 持平 | — |
| Mean underrun | 0 s | 0 s | 持平 | — |

TTFT 的小样本回退与 Stage 2 算法无直接依赖，需要在更大样本上判断是否只是
运行抖动。O10 后续精度门控为 3/3 请求成功但 Mean/Median WER 均为 1.0；
保存 WAV 的 clipping rate 为 3.13%–7.09%，最大相邻跳变达到满幅 2.0。
因此该候选即使性能增益明显也已淘汰，不能进入 Demo 或最终配置。

## 4. 精度结果

| Benchmark | 官方门槛 | 基线 | 优化版 | 差值 | 结论 |
|---|---:|---:|---:|---:|---|
| Daily-Omni accuracy | ≥0.78 | 待填写 | 待填写 | 待填写 | 待填写 |
| Video-MME accuracy | ≥0.68 | 待填写 | 待填写 | 待填写 | 待填写 |
| Seed-TTS mean WER | ≤0.05 | 待填写 | 待填写 | 待填写 | 待填写 |

### 开发机端到端精度冒烟

| Benchmark | 样本 | HTTP/评测成功 | 冒烟结果 | 说明 |
|---|---:|---:|---:|---|
| Daily-Omni | 3 | 3/3 | accuracy 1.0000 | 数据转换、媒体读取、答案归一化均通过 |
| Video-MME | 3 | 3/3 | accuracy 0.6667 | 仅 3 条，不能用于 0.68 正式门槛判断 |
| Seed-TTS EN | 3 | 3/3 | mean/median WER 0.0000 | 本地 Whisper-large-v3，空音频/ASR 失败均为 0 |

冒烟的用途是验证链路与评测实现。正式精度结论必须使用完整官方子集，
并与官方基线执行同样的数据版本、参数和统计口径。

## 5. 稳定性与 Demo

- 连续请求数量/时长：待填写
- 音频中断、重复或空首块：4 请求冒烟中 continuity OK 100%；主观爆音检查待完成
- 峰值 NPU 显存、CPU、主存：待填写
- Demo 视频：待填写

## 6. 复现

参见 `README.md` 和 `scripts/`。所有原始 JSON 与日志保存在 `results/`，并在最终提交时附上 SHA256。
