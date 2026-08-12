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
| O27 | Duplex Talker runner-local decode ×4 | 进一步减少 Stage 1 scheduler/IPC 往返与同卡竞争 | 低 | 晋级候选：相对 local2，SPEAK RTF mean -3.42%、p99 -9.80% |
| O28 | Duplex Talker runner-local decode ×8 | 探索更长本地窗口的调度收益上限 | 低 | RTF 候选：相对 local2 mean -5.02%，但 TTFT/TTFP +0.68% |
| O29 | local4 + fused Token2Wav encoder attention | 正交合并 Stage 1 调度与 Stage 2 首响优化 | 低 | TTFT/TTFP 新低，但 RTF mean/p99 回退，降级为首响专项候选 |
| O30 | Token2Wav Flow BF16 | 验证昇腾低精度 Flow 是否降低 Stage 2 时延 | 中 | 淘汰：TTFT/TTFP +1.51%，SPEAK RTF mean +1.58%、p99 +9.61% |
| O31 | local decode attention metadata 复用 | 消除 runner-local 每 token metadata builder/H2D | 中 | 淘汰并回滚：ACLGraph replay 后首包超时，metadata 含未显式逐步状态 |
| O32 | local64 + fused encoder + continuation 2× | 叠加三个独立热点优化，探索大幅 RTF 收益 | 中 | 淘汰：组合后首包超时，不能简单正交叠加 |
| O33 | Talker codec sampler 独立 ACLGraph | 图内执行 head、repetition、候选过滤和 softmax | 中 | 淘汰并回滚：微基准 3.16–4.13×，但与 Stage 1 FULL_DECODE_ONLY 共存时首次请求停滞 |
| O34 | Top-K-first 精确 Top-P compact sampler | 消除 6562 维全排序与 dense multinomial，保持原请求级 RNG | 中 | 淘汰并回滚：微基准 1.79×，但 FULL_DECODE_ONLY 首次请求同样停滞 |
| O35 | Stage 1 图内 decode-only `head_code` | 只投影 runner 选中的采样行，并入已有模型图 | 中 | 第二版固定驻留 buffer 待 NPU smoke；默认关闭 |
| O36 | Stage 2 稳态整链 NPUGraph | 捕获 53-token 稳态 Conformer + CFM + HiFT，消除 eager 小算子下发 | 中 | 已完成筛选工具及默认关闭的生产实现；Code2Wav CPU 47/47 通过，待物理 NPU 5 数值与性能门禁 |
| O37 | 有界 runtime reference 特征/初始 state 缓存 | 相同参考音频跨请求复用 S3Tokenizer、speaker embedding、prompt features、Conformer/CFM 初始 cache | 低 | 默认关闭的 LRU+启动预载候选；CPU 生命周期/复用/淘汰 47/47 通过，待 NPU 5 TTFT/TTFP A/B |
| O38 | Stage 0 reference embedding 缓存 | 相同参考音频跨会话复用 APM/音频塔 reference embeddings | 低 | 仅缓存无 session state 的直接 encoder 路径；native duplex hooks 64/64 通过，随 O37 做 TTFT/TTFP A/B |
| O39 | 候选性能与 activation 机器门禁 | 只用 SPEAK 生成 RTF，并强制验证图/缓存真实命中及 fatal 日志 | 低 | 5/5 单测通过；已接入 README 与官方 910C 复测流程 |
| O40 | 全纯 decode CPU slot mapping | 用 CPU block table 直接计算 scheduler-visible 与 runner-local 单 token KV slot，跳过全尺寸 NPU slot kernel | 低 | 默认关闭；精确计算并保留 GPU 自动回退，待物理 NPU 5 A/B |
| O41 | FULL graph 内 exact compact codec sampler | 图内 16 槽稀疏 repetition + Top-K/全局 logsumexp 精确重建 Top-P→Top-K，消除 AI-CPU Bincount 和全词表 sort/scatter | 低 | 默认关闭；CPU 分布等价、循环历史与请求压缩测试通过，待 NPU 5 graph smoke/性能/精度门禁 |
| O42 | Talker 二元控制 argmax | codec 已在模型内采样后，直接对确定性的 continue/stop 二元行 argmax，绕过通用 temperature/top-k/top-p sampler | 低 | 随 O41 组合默认关闭；logprobs/allowed IDs/bad words/penalty 自动回退通用 sampler，待 NPU 5 A/B |
| O43 | Stage 0 禁用双进程多模态 LRU | 避免 API sender 与 Engine receiver 在并发/重试/失败请求后出现淘汰顺序分叉，消除长稳 `Expected a cached item` 故障 | 无模型精度风险 | 已进入默认 910C 与 NPU 5 验证 YAML；Daily-Omni 1196/1196、Video-MME 2700/2700 均零请求/解析失败且 fatal gate 通过 |
| O44 | Stage 2 NPUGraph 完整条件回灌 | replay 前同时刷新 codec、speech token、speaker embedding、mel 与 recurrent cache，避免同 shape 不同参考音频误用捕获条件 | 无（修复实验路径正确性） | 定向 CPU replay 单测通过；新增 fixed-13/25 当前最佳候选上的单变量 NPU 5 ablation，待性能与数值门禁 |

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

在完全相同的 Euler、zero-wait、50 帧音频 chunk 配置上，仅将 Talker
runner-local decode 从 2 步扩大到 4 步，4/4 请求仍生成 4 个音频包和
3.72 s 完整音频，错误数为 0，transcript、终止事件和连续性检查全部通过。
TTFT/TTFP mean 为 1867.819/1867.497 ms，SPEAK RTF mean/median/p99 为
0.761914/0.772574/0.812804。相对 local2，TTFT/TTFP 改善约 1.22%，SPEAK
RTF mean 改善 3.42%，p99 改善 9.80%。这说明更长的 runner-local 窗口在
full-duplex 同卡竞争下确实能减少 Host 往返和尾延迟；该候选保留并继续与
continuation 合并路线做正交验证。原始结果：
`results/candidates/euler_local4_zero_wait_npu5_n4_v1/result.json`。

local4 随后通过两项独立准入门禁。Seed-TTS 10 条结果为 10/10 请求成功，
mean/median WER=0.0253/0，请求失败、空 PCM 和 ASR/WER 失败均为 0，流式
连续率 100%；原始结果位于
`results/candidates/euler_local4_zero_wait_seed_tts_acc_n10_v1/`。native duplex
2 会话 × 每会话 3 轮测试也得到 6/6 音频轮成功、错误数 0、无 stale audio、
重复 speak、跨轮污染或生命周期异常；TTFT/TTFP mean 为
1994.442/1994.209 ms，22 个 SPEAK chunk 的 mean/median/p99 RTF 为
0.750323/0.771753/0.873893。原始结果位于
`results/candidates/euler_local4_zero_wait_multiturn_s2_t3_v1/result.json`。
因此 local4 已晋级为当前单卡 910C 三指标均衡默认，最终仍需官方 910C 完整集
复测。

### 当前晋级候选：固定 13/25 包 + reference/runner 启动预热

在 local4、CPU slot mapping、graph sampler 和 binary argmax 组合上，进一步把
非尾部 Code2Wav 调用固定为首包 13 帧、稳态 25 帧，并将官方参考音频的 Stage 0
embedding、Stage 2 prompt/initial state 与完整 Code2Wav runner 13/25 实形状执行
迁移到服务启动期。最终 Thinker turn-end 的精确余量仍单独 flush，并由
`speak_tail=true` 排除在官方 SPEAK 生成阶段 RTF 之外。

物理 NPU 5、端口 8095、并发 1、1 次预热 + 4 次正式请求的同周期结果如下：

| 指标 | local4 同周期基线 | 13/25 + runner prewarm | 相对变化 |
|---|---:|---:|---:|
| TTFT mean / p99 | 1876.714 / 1986.594 ms | **1559.362 / 1596.750 ms** | **-16.91% / -19.62%** |
| TTFP mean / p99 | 1876.384 / 1986.270 ms | **1559.210 / 1596.599 ms** | **-16.90% / -19.62%** |
| SPEAK RTF mean / median / p99 | 0.791225 / 0.788243 / 0.906307 | **0.596573 / 0.595077 / 0.663687** | **-24.60% / -24.51% / -26.77%** |

该表的基线原始文件为
`results/candidates/same_period_local4_baseline_npu5_n4_v3/result.json`，候选原始
文件为 `results/candidates/fixed13_25_boundaryfix_hot_n4_v1/native_duplex_rtf.json`。
`gate_vs_same_period_local4.json` 使用 `combined` 硬门禁重新计算上述变化并得到
`passed=true`；另保留 `gate_vs_local4.json`，用于对更早的 local4 可靠基线做
跨周期保守检查，两份门禁均通过。

4/4 请求全部成功，每条均有 5 个音频包、3 个 SPEAK generation chunk、1 个
SPEAK tail chunk、`response.done=1` 且错误数为 0。由于固定包会使相邻
audio-bearing text window 重叠，data plane 增加同 turn 最长后缀/前缀去重；4 条
transcript 均恢复为完整且不重复的
`The two men hurried back and found the cylinder still lying in the same position`。
零请求预热的真正首请求为 TTFT/TTFP 1738.882/1738.731 ms、SPEAK RTF
0.612476，证明原先约 15 秒的首次 Code2Wav 懒编译已经迁移到启动阶段。

4/25 首包消融在修正 deploy `extra` 浅覆盖问题后，首请求仍为
TTFT/TTFP 1950.712/1950.428 ms、SPEAK RTF 0.702531，全面弱于 13/25，因此淘汰。
晋级结果保存在
`results/candidates/fixed13_25_runner_prewarm_transcriptfix_npu5_v1_cold/` 和
`results/candidates/fixed13_25_boundaryfix_hot_n4_v1/`。
以上仍是 910B4 开发数据，不得替代官方单卡 910C 最终成绩。

固定 13/25 分块最初暴露了一个只在连续 Talker segment/多轮会话出现的连接器
水位问题：resumable 请求复用 external request id，而 scheduler 在 sparse terminal
packet 入队前已把 per-segment `num_computed_tokens` 清零，旧的抢占去重判断会把该
terminal packet 以及下一段误判为重放，最终缺失 `tts_is_last_chunk` / `response.done`。
修复后，仅 resumable segment boundary 可绕过下降水位检查，并在有序 save queue
接收边界后重置水位；普通低水位包仍保持抢占去重。

物理 NPU 5 的稳定性复测为 2 sessions × 3 turns：6/6 audio turns、6/6
`response.done=completed`、0 cancelled、0 stale、0 truncate，所有 transcript 完整。
TTFT mean/p50/p99 为 1657.591/1662.941/1726.047 ms，TTFP 为
1657.458/1662.798/1725.925 ms，20 个 SPEAK generation chunk 的
mean/median/p99 RTF 为 0.663337/0.691060/0.784743。原始结果位于
`results/candidates/fixed13_25_multiturn_boundaryfix_s2_t3_v1/`；连接器日志无
`Enqueue save_async` 下降水位告警，activation gate 全部通过。

离线回归覆盖连接器水位、duplex serving/data plane、Talker local decode、
Code2Wav 固定分块与缓存、NPU runner fast path、部署配置和候选门禁，共
506 项同步与异步测试以标准 pytest 单轮全部通过、0 skipped。测试同时固定校验自包含 910C YAML 与物理 NPU 5 已测候选
解析后的模型/执行热路径完全一致；允许的部署差异仅为设备号 `5 -> 0`，以及为
官方 c8 矩阵把 session admission ceiling 从 1 提升到 8（Stage
`max_num_seqs=4` 不变，多出的会话排队）。当前镜像原先没有异步插件；实际安装
审计发现仓库 dev extra 的 `pytest==9.1.1` / `pytest-asyncio==1.4.0` 与镜像中
`triton-ascend` 对 `pytest==8.3.2` 的硬依赖冲突，因此恢复并保留镜像 pytest，
只安装兼容的 `pytest-asyncio==1.3.0` 完成上述标准复跑。完整 dev extra 如需验证
应放在隔离 venv，不能污染比赛运行时环境。这些路径另有端到端 2 sessions × 3 turns
的真实服务证据。

local4 与 continuation 2× 的组合候选得到 TTFT/TTFP
1898.158/1897.849 ms，SPEAK RTF mean/median/p99 为
0.728925/0.763213/0.916626，请求级音频 RTF 为 0.961111。相对原 local2
零等待最佳，mean RTF 改善 7.60%，但 p99 回退 1.72%；相对 local4 单独候选，
mean RTF 改善 4.33%，p99 则回退 12.77%。4/4 请求均成功且每条保持 4 个包，
但音频时长由 3.72 s 增至 4.40 s。因此该组合只作为需要完整 TTS-Seed 和 Demo
主观连续性门禁的激进均值 RTF 候选，低尾延迟默认仍优先 local4 单独版本。
原始结果：
`results/candidates/euler_local4_zero_wait_silence_units2_npu5_n4_v1/result.json`。

进一步把 local window 从 4 扩至 8 时，输出仍保持 3.72 s、4 个音频包和零错误。
TTFT/TTFP 为 1903.592/1903.282 ms，SPEAK RTF mean/median/p99 为
0.749311/0.762315/0.811865。相对 local4，mean RTF 仅继续改善 1.65%，p99
基本持平，而 TTFT/TTFP 回退约 1.92%；相对原 local2，mean RTF 改善 5.02%，
TTFT/TTFP 则回退约 0.68%。因此 8 步支持作为可回退 RTF 消融能力保留，均衡
默认仍优先 4 步。原始结果：
`results/candidates/euler_local8_zero_wait_npu5_n4_v1/result.json`。

进一步把 runner-local 上限提高到 64，使 50 帧稳态 codec chunk 可以在一次
scheduler 调用中持续解码，仍由真实 codec payload、终止事件和状态边界提前
截断。4/4 请求保持 3.72 s、4 个音频包、零错误，TTFT/TTFP 为
1906.830/1906.516 ms，SPEAK RTF mean/median/p99 为
0.740966/0.748298/0.818287。相对 local4，mean RTF 再改善 2.75%，但首响回退
约 2.09%，p99 RTF 略回退 0.67%；说明 scheduler 往返已接近耗尽，剩余主要是
真实 Talker 计算。该候选保留为 RTF 专项消融，不替换 local4 均衡默认。原始
结果：`results/candidates/euler_local64_zero_wait_npu5_n4_v1/result.json`。

还尝试用 Top-K + 全量 logsumexp 在数学上等价重建 Talker 的 Top-P→Top-K
候选分布，避免每个 codec token 对 6562 维 logits 做完整排序。CPU 分布一致性
测试通过，但在 NPU 图外执行时产生严重的小算子下发/同步开销：local4 warmup
请求 180 秒内未产生 `response.created`，被测试超时中止。因此该实现已回滚，
不会进入默认配置；后续若继续优化采样，必须把完整 sampler 纳入设备图，而不是
在 Python 图外组合更多 NPU 算子。

昇腾还提供 fused TopK/TopP/multinomial 一体算子，独立微基准约
0.278 ms/token，但在 Stage 1 多进程 runner 中传入 NPU generator 后同样导致
warmup 请求 180 秒超时。该一体化路径也不进入配置。去掉 fused multinomial，
仅使用 `npu_top_k_top_p` 过滤并保留原 `torch.multinomial` 后，独立微基准为
0.182 ms/token（原路径约 0.448 ms/token），但真实 local4 warmup 同样在 180 秒
内没有产生 `response.created`。因此三种图外 sampler 变体均已淘汰并回滚；后续
只有将完整 sampler 纳入已捕获设备图，才值得重新验证这一方向。

还尝试把 Stage 1 Talker 的模型加载、FULL decode 图捕获和每次 replay 放到
priority=-1 的昇腾高优先级 stream，希望在单卡 SPEAK 阶段优先于 Stage 0
监听和 Stage 2 音频计算。4/4 请求功能完整，TTFT/TTFP 为
1844.898/1844.520 ms，相对 local4 改善约 1.23%；但 SPEAK RTF
mean/median/p99 为 0.787351/0.796364/0.878903，相对 local4 分别回退约
3.34%/3.08%/8.13%。跨进程同卡竞争没有被该进程内 stream 优先级有效消除，
额外 stream 还放大了尾延迟，因此实现已回滚。原始结果：
`results/candidates/euler_local4_priority_npu5_n4_v1/result.json`。

local4 与 fused Token2Wav encoder attention 的正交组合把 TTFT/TTFP 进一步降至
1821.164/1820.842 ms，相对原 local2 改善 3.68%，相对 local4 改善 2.50%；
但 SPEAK RTF mean/median/p99 为 0.799634/0.799641/1.008936，相对 local4 的
mean/p99 分别回退 4.95%/24.13%。音频仍为 3.72 s、4 包且零错误，因此不是
输出缩短造成的假回退。该组合只保留为 TTFT/TTFP 专项候选，不能替换三指标
均衡的 local4。原始结果：
`results/candidates/euler_local4_zero_wait_fused_encoder_attn_npu5_n4_v1/result.json`。

### 新增高杠杆实验：低精度与控制面整合

在 local4 默认候选上仅将 Token2Wav Flow 权重和计算改为 BF16、HiFT 保持
FP32，启动日志确认 `flow_dtype=torch.bfloat16`。物理 NPU 5、1 次预热 +
4 次正式请求均成功，音频仍为 3.72 s、4 包，但 TTFT/TTFP 为
1896.009/1895.627 ms，SPEAK RTF mean/median/p99 为
0.773944/0.772417/0.890897。相对 local4，TTFT/TTFP 均回退约 1.51%，
RTF mean 回退 1.58%、p99 回退 9.61%，因此 Flow BF16 淘汰。原始结果：
`results/candidates/euler_local4_flow_bf16_npu5_n4_v1/result.json`。

随后尝试复用 Stage 1 runner-local decode 的 Ascend attention metadata：保持
block table、slot mapping 和 tensor 地址不变，仅刷新 Python `seq_lens_list`，
未知 schema 自动回退标准 builder。服务正常启动，Stage 0/1 均进入 ACLGraph
replay，但 duplex 请求无法在客户端超时前形成首包。说明 FULL graph 的 metadata
还包含未显式暴露的逐步状态；该实现已完整回滚，未进入默认代码。

最后组合 local64、Token2Wav fused encoder attention 与 native continuation 2×，
希望叠加完整 codec chunk 的 scheduler 往返消除、Stage 2 attention 融合和两秒
SPEAK continuation 的控制面摊薄。启动日志确认 local64 FULL graph 和 10 层融合
attention 均生效，但组合路径同样在 Stage 0/1 ACLGraph replay 后首包超时，未产生
可计分结果。三个单项不能在当前控制面简单相乘；该 YAML 仅作为失败消融记录，
不进入 910C 候选。

此前 local2 候选先通过 Seed-TTS 三条本地 Whisper 小门禁（mean/median WER=0），随后
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

### Native continuation 合并与 Ascend encoder 融合注意力

为降低每秒 silence continuation 重复经过 serving、orchestrator、connector 和
Stage 0 prefill 的固定开销，新增了默认关闭的部署参数：

```yaml
duplex_session:
  native_silence_continuation_units_per_append: 2
```

默认值仍为 1，不改变官方逐秒决策节奏。物理 NPU 5、并发 1、1 次预热 +
4 次正式请求下，2× 合并单独得到 TTFT/TTFP 1949.339/1949.016 ms，SPEAK
RTF mean 0.703871；相对当前最佳首响回退约 3.1%，但 RTF 改善约 10.8%。
4× 合并的 TTFT/TTFP/RTF 为 1999.398/1999.081 ms/0.801414，全面回退；只在
首个音频包后启用 2× 合并则令音频缩短到 3.76 s，RTF 回退到 0.873296。
因此不再扩大倍率，2× 仅作为需要完整音质门禁的 RTF 候选。

随后启用 `token2wav_npu_fused_encoder_attention`，将 10 层 Token2Wav
Conformer 相对位置注意力的 eager QK/softmax/AV 替换为
`torch_npu.npu_fusion_attention`，并将原始相对位置项作为 PSE 输入。该候选
不改变模型权重、采样参数或 CFM 积分算法，4/4 请求输出完整且无错误：

| 候选 | TTFT mean | TTFP mean | SPEAK RTF mean | 相对当前最佳 |
|---|---:|---:|---:|---|
| 仅 fused encoder attention | **1831.229 ms** | **1830.898 ms** | 0.791304 | TTFT/TTFP -3.15%，RTF +0.30% |
| fused attention + continuation 2× | 1898.110 ms | 1897.788 ms | **0.730806** | TTFT/TTFP +0.39%，RTF **-7.37%** |

组合候选的请求级音频 RTF 为 0.962726，首次低于实时 1.0；但输出音频从当前
最佳的 3.72 s 增至 4.40 s，SPEAK RTF p99 为 0.926848，高于当前最佳的
0.901119。它必须通过完整 TTS-Seed、官方 Demo 主观连续性和 910C 复测后才能
晋级；纯融合注意力候选则是当前低风险 TTFT/TTFP 最优配置。原始结果：

- `results/candidates/euler_local2_zero_wait_fused_encoder_attn_npu5_n4_v1/result.json`
- `results/candidates/euler_local2_zero_wait_fused_encoder_attn_silence_units2_npu5_n4_v1/result.json`

另将融合注意力候选的稳态 codec chunk 从 50 帧增至 75 帧，希望进一步摊薄
Stage 2 固定开销。相同物理 NPU 5、1 次预热 + 4 次正式请求下，TTFT/TTFP
回退到 1927.130/1926.772 ms，SPEAK RTF mean/median 为
0.791668/0.797093，均未优于 50 帧候选；仅 p99 从 0.901119 降至
0.871603，而请求级音频 RTF 回退到 1.067913。音频总时长仍为 3.72 s，排除
输出长度变化造成的假收益。该配置作为负向消融证据保留，不进入正式候选；
原始结果位于
`results/candidates/euler_local2_zero_wait_fused_encoder_attn_chunk75_npu5_n4_v1/result.json`。

另测试了将 Thinker 单段 token 上限从 20 降至 12。两次端到端结果的聚合
Stage 0 token 数仍为 20，TTFT/TTFP 未改善，说明 continuation 累计/控制面覆盖
使该旋钮无法形成有效首段提前交接；相关运行时代码已移除，不进入提交。

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

### Talker codec sampler ACLGraph

Stage 1 profiler 显示 codec sampler 存在大量小算子下发、`bincount`、全词表排序
和 Host wait。O33 尝试将确定性路径
`head_code → repetition penalty → Top-P → Top-K → softmax` 捕获为独立
ACLGraph；`torch.multinomial` 仍使用原 request-owned generator 在图外执行，
因此 RNG 生命周期和抽样顺序不变。Top-P 在数学上改写为 Top-K-first：用全词表
`logsumexp` 与候选的 exclusive cumulative probability 重建相同 nucleus mask，
避免每 token 对 6562 个 codec logits 完整排序。

物理 NPU 5、BF16、hidden 768、vocab 6562 的 500 次微基准结果：

| 路径 | Mean wall time / codec token |
|---|---:|
| Top-K-first eager sampler | 1.1150 ms |
| all-history ACLGraph sampler | 0.2698 ms |
| 加速比 | **4.13×** |

history 长度 0、7、16 与 EOS masked/eligible 共 6 种分支的概率最大绝对误差、
概率总误差均为 0，finite mask 和 candidate ids 完全相同。实验后期通过固定
16 槽 history 与设备端有效权重模板，使首个 codec token 就进入 graph，覆盖
TTFP 关键路径。原始结果：
`results/profiles/talker_sampler_graph_all_history_npu5.json`。

完整三阶段复测严格绑定物理 NPU 5。卡释放后，Stage 1 独立 sampler graph 与
模型 `FULL_DECODE_ONLY/NPUGraph_ex` 均能完成捕获，服务也能正常启动；但首次
native duplex 请求在 Stage 0/1 ACLGraph replay 后停止推进，3 分钟内没有
`response.created`。加入 sampler 输入 stream 同步后，微基准仍为 3.16×，但完整
链路在同一位置再次停滞。因此独立 graph 生产接入和对应 YAML 已完整回滚。

O34 随后只保留不依赖独立 graph 的 Top-K-first 数学改写，在正常 Stage 1 路径
执行。它先取最终可采样的 Top-K 候选，再利用全词表 `logsumexp` 和候选 exclusive
cumulative probability 重建与原 Top-P warper 相同的保留 mask，最后只对至多
25 个候选执行 softmax/multinomial。随机与集中分布 CPU 测试均与原 dense 路径
概率一致，纯 NPU sampler 微基准从 1.428 ms/token 降至 0.798 ms/token（1.79×）。
但 local64 完整服务首次 native duplex 请求仍在 Stage 0/1 ACLGraph replay 后停止
推进，说明该后处理算子序列本身与当前 NPUGraph_ex 流存在兼容性问题。生产代码、
测试开关和 YAML 已完整回滚，仅保留原始性能 JSON 作为后续 Ascend 后端修复证据。

O35 继续尝试把最重的 768→6562 `head_code` 投影并入 Stage 1 已有
`FULL_DECODE_ONLY/NPUGraph_ex` 模型图，同时只对 runner 的 `logits_indices`
采样行做投影，避免 prefill 全序列投影。第一版将 codec logits 作为模型的第二个
graph output 返回；Stage 1 能正常编译和捕获，但物理 NPU 5 首次请求 replay 后，
该额外 output 的地址失效，随后在 repetition penalty 同步时报 MTE DDR 越界，
因此该输出协议已淘汰。当前实验版改为模型唯一输出保持原 hidden states，图内把
codec logits `copy_` 到固定模型驻留 buffer，图外 sampler 读取该 buffer；开关默认
关闭，仅存在于
`config/ablations/minicpmo_4_5_duplex_euler_local4_graph_head_npu5.yaml`。
CPU 等价、采样行路由、静态检查均通过；NPU smoke 和性能结论待复测，未进入默认
竞赛配置。

O36 针对当前剩余的最大结构性瓶颈：Stage 2 仍为完全 eager。默认稳态窗口由
3 帧 codec 左上下文和 50 个新 frame 组成；attention cache 达到截断上限后，
`Conformer forward_chunk → 单步 conditional CFM DiT → HiFT` 的输入和 7 组流式
cache shape 均固定。`scripts/profile_token2wav.py` 新增
`--compare-steady-npugraph --npugraph-only`，在不启动三阶段服务的情况下：

1. 自动运行至连续两次 cache signature 相同；
2. 从同一份 state 分叉 eager 与 NPUGraph 链路；
3. 每次 replay 前把上一轮 graph output cache 回灌到固定输入；
4. 同步统计包含 cache copy 与 replay 的 wall time；
5. 对首轮和最终轮音频及全部 flow/HiFT cache 报告最大绝对误差。

物理 NPU 5 的筛选命令为：

```bash
ASCEND_RT_VISIBLE_DEVICES=5 python \
  competition/minicpmo_b/scripts/profile_token2wav.py \
  --device npu:0 --cfg-mode conditional --solver euler --steps 1 \
  --initial-frames 4 --steady-frames 50 --warmups 4 --iterations 20 \
  --compare-steady-npugraph --npugraph-only \
  --output competition/minicpmo_b/results/profiles/token2wav_steady_npugraph_npu5.json
```

生产侧已加入默认关闭的精确 shape 分派。启用
`token2wav_steady_npugraph: true` 时，启动预热会自动推进到连续两次 cache shape
稳定，再捕获单请求、非 terminal 的固定 packet 整链。每次 replay 前回灌 codec、
speech token、speaker embedding、prompt mel 和全部请求 cache，replay 后把音频与新 cache 克隆为请求私有
存储；首块、尾块、flush、batch>1、prompt 长度或 cache shape 不匹配均自动回退
eager。完整条件回灌很关键：graph key 是 shape bucket，同长度但内容不同的参考音频
不能继续沿用 capture 时的 speech token/mel。进一步审计发现默认 `HT_ref_audio.wav` 为 6.02 s，而 native duplex 脚本
传入的 `system_ref_audio.wav` 为 16.84 s，二者稳定 attention-cache shape 不同；
因此实现已改为按 `(token layout, prompt mel length, state signature)` 保存多个 graph
bucket，并通过 `code2wav_npugraph_prompt_wavs` 在启动时额外捕获 Demo/评测参考音频
shape。当前 fixed-13/25 最佳候选对应的单变量实验 YAML 为
`config/ablations/minicpmo_4_5_duplex_euler_local4_graph_sampler_cpu_slot_fixed13_25_prewarm_ref_steady_npugraph_npu5.yaml`。
完整 Code2Wav CPU 测试 47/47 通过，但在音频/cache 数值等价、端到端收益和多轮
稳定性通过物理 NPU 5 门禁前，不进入默认竞赛配置。

O37 针对同一参考音频在顺序请求间的重复固定开销。原实现会在最后一个 owner
结束时立即删除临时 WAV，并调用 `evict_prompt` 清除 S3Tokenizer 输出、speaker
embedding 和 prompt mel/token features；官方/native duplex 客户端每轮都提交同一
`system_ref_audio.wav`，因此每次请求都会重复 CPU 特征提取和 NPU prompt setup。
新增 `code2wav_runtime_prompt_cache_size` 后，内容 SHA256 相同的未占用参考音频按
LRU 有界保留；再次出现时直接复用已有文件及 prompt features。进一步启用
`code2wav_cache_runtime_initial_state` 后，还会复用该 prompt 的只读 Conformer/CFM
初始 cache，首个 live chunk 返回的新 state 仍按请求隔离；超过容量时统一释放文件、
features 和初始 state。`code2wav_runtime_prompt_preload_wavs` 还可在服务启动时按
native duplex 的 16 kHz mono、整 100 ms frame 规则规范化并预载官方 Demo 音色，
使首个正式交互也能直接命中，而不依赖客户端 warmup。默认容量仍为 0，保持原资源
生命周期；独立实验配置为
`config/ablations/minicpmo_4_5_duplex_euler_local4_runtime_prompt_cache_npu5.yaml`。
该优化不改变权重、codec、CFM 或 HiFT 数值，预期只影响重复音色场景的 TTFT/TTFP；
实际收益必须用一次预热后 4 次正式顺序请求确认。

O38 覆盖同一重复开销在 Stage 0 的另一半：每个新会话原先都会重新运行
`processor.process_audio → get_audio_embedding/get_audio_hidden_states` 处理 16.8 s
参考音频。`stage0_ref_audio_embedding_cache_size` 使用归一化 FP32 waveform SHA256
作为 key，并按 LRU 有界保存只读 embedding tensor。为避免破坏 session-local APM
cache，只有模型暴露不依赖 `state` 的直接 reference encoder 路径时才允许命中；
streaming fallback 始终保持原执行。该开关已加入 O37 独立配置和 O36+O37 组合配置。

O36/O37 的物理 NPU 5 晋级顺序固定为：先独立微基准，再两个单变量端到端 A/B，
最后才测试组合配置。所有服务均绑定 8095，且 deploy config 解析结果必须为
`devices=['5','5','5']`。

1. O36 使用 `system_ref_audio.wav` 执行 production-dispatch 微基准；音频和每组
   cache 最大绝对误差必须在 FP32 NPU 数值抖动范围内，graph wall time 至少改善
   1.5×，否则不启动完整服务。
2. O37 使用 1 次预热 + 4 次正式顺序请求；正式请求应只出现一次 prompt feature/
   initial-state 构建。TTFT、TTFP mean 至少改善 10%，SPEAK RTF 不得回退超过 2%。
3. O36 完整服务要求 SPEAK RTF mean 至少改善 15%，p99 不回退，TTFT/TTFP 不回退
   超过 2%，且日志中必须出现 `Captured steady ... NPUGraph bucket` 和
   `Replayed steady ... NPUGraph`。
4. 两项独立通过后测试
   `config/ablations/minicpmo_4_5_duplex_euler_local4_npugraph_prompt_cache_npu5.yaml`；
   组合结果必须重新执行 Seed-TTS n10 和 native duplex 2 会话 × 3 轮门禁。

### O40/O41：Stage 1 profiler 驱动的等价热路径消除

已有 Stage 1 `FULL_DECODE_ONLY` profiler 的 178 个 decode token 中，设备时间主要为：

- `MatMulV2` 36.19%，`FusedInferAttentionScore` 22.58%；
- `_compute_slot_mapping_kernel` 8.03%；
- codec sampler 的 `Bincount` 6.03%、`ScatterElements` 8.29%，另有 full-vocab
  `Sort/Cumsum/MaskedFill` 约 1.42%。

用 `scripts/summarize_ascend_ops.py` 按完整 sampler 控制组重新汇总后，slot mapping
与 sampler control 合计占设备时间 24.46%，对应 device-only Amdahl 上限 1.324×；
端到端收益仍受 Host wait、阶段重叠和 graph replay 约束。O40 从 CPU block table 按
`block_id * block_size + offset` 直接计算当前请求的 slot，并只复制 `num_reqs` 个值
到 NPU；现在同时覆盖 scheduler-visible 的纯单 token decode 和 runner-local 子步，
而 prefill、speculative、CP 或不支持的 block layout 会保留/自动回退原 GPU kernel。
独立配置为
`config/ablations/minicpmo_4_5_duplex_euler_local4_cpu_slot_npu5.yaml`。

O41 不再像已淘汰的 O33 那样创建第二张 sampler graph，也不再像 O34 那样把新的
算子序列留在 `FULL_DECODE_ONLY` graph 外。runner 在每次初始/local forward 前刷新
固定 `[batch,16]` codec history 与 EOS mask；现有模型 FULL graph 内完成：

1. 仅投影 `logits_indices` 对应的采样行；
2. 用 16×16 的窗口内去重计数、gather 和单次 scatter 精确替代 AI-CPU
   `Bincount`，不再构造 `[batch,16,6562]` 比较张量；
3. 先取 Top-K，再用全词表 `logsumexp` 和候选 exclusive cumulative probability
   精确重建原始 Top-P→Top-K mask；
4. 把 compact FP32 candidate logits/IDs 写入模型驻留 buffer，图外仅保留 compact
   `multinomial`；runner 用循环历史缓存让稳态 local step 只更新一个 codec 标量，
   请求 compaction、prefill 或 step 跳变才重建整行。

O42 继续消除 codec 采样之后的第二套通用 sampler：Talker 的 engine token 只有
`continue/stop` 两列，模型和 min-token logit bias 处理后每行至多一个有效候选，
因此直接 argmax 与原采样分布等价。若请求启用 logprobs、allowed token IDs、bad
words 或 penalty，则自动回退 vLLM 通用 Sampler，避免扩大接口语义风险。

随机与集中 logits 的 CPU 测试均确认最终 finite mask 和归一化概率与原 dense 路径
一致。组合配置为
`config/ablations/minicpmo_4_5_duplex_euler_local4_graph_sampler_cpu_slot_npu5.yaml`。
晋级要求：日志同时证明 FULL graph、CPU slot fast path 和 graph sampler 启用；相对
local4 同周期基线，SPEAK RTF mean 至少改善 12%、p99 不回退，TTFT/TTFP 回退不
超过 2%；随后必须通过 Seed-TTS n10、native duplex 2×3 和 Demo 主观连续性。当前
未获得 NPU 5 实验授权，因此没有填写任何未经实测的性能数字，也未修改默认配置。

## 4. 精度结果

| Benchmark | 官方门槛 | 基线 | 优化版 | 差值 | 结论 |
|---|---:|---:|---:|---:|---|
| Daily-Omni accuracy | ≥0.78 且相对基线降幅 ≤2pp | 待填写 | 待填写 | 待填写 | 待填写 |
| Video-MME accuracy | ≥0.68 且相对基线降幅 ≤2pp | 待填写 | 待填写 | 待填写 | 待填写 |
| Seed-TTS mean WER | ≤0.05 且相对基线增幅 ≤2pp | 待填写 | 待填写 | 待填写 | 待填写 |
| Seed-TTS speaker SIM | 相对基线降幅 ≤2pp，完整 1000/1000 | 待填写 | 待填写 | 待填写 | 待填写 |
| Seed-TTS UTMOS | 相对基线降幅 ≤2pp，完整 1000/1000 | 待填写 | 待填写 | 待填写 | 待填写 |

### 开发机端到端精度冒烟

| Benchmark | 样本 | HTTP/评测成功 | 冒烟结果 | 说明 |
|---|---:|---:|---:|---|
| Daily-Omni | 3 | 3/3 | accuracy 1.0000 | 数据转换、媒体读取、答案归一化均通过 |
| Video-MME | 3 | 3/3 | accuracy 0.6667 | 仅 3 条，不能用于 0.68 正式门槛判断 |
| Seed-TTS EN | 3 | 3/3 | mean/median WER 0.0000 | 本地 Whisper-large-v3，空音频/ASR 失败均为 0 |

冒烟的用途是验证链路与评测实现。正式精度结论必须使用完整官方子集，
并与官方基线执行同样的数据版本、参数和统计口径。

## 5. 稳定性与 Demo

- 连续请求数量/时长：native duplex 2 sessions × 3 turns 已通过；官方 Demo 长稳仍待录制
- 音频中断、重复或空首块：4 请求冒烟中 continuity OK 100%；主观爆音检查待完成
- 峰值 NPU 显存、CPU、主存：待填写
- Demo 视频：待填写

## 6. 复现

参见 `README.md` 和 `scripts/`。所有原始 JSON 与日志保存在 `results/`，并在最终提交时附上 SHA256。
