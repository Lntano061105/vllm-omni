# 子赛道 B 实施计划

## 目标与硬约束

核心指标为逐音频 chunk RTF、TTFT 和 TTFP。优化版本相对官方基线的精度降幅不得超过 2 个百分点，并且必须稳定接入官方 Demo。

精度门槛按仓库官方测试固定为：

- Daily-Omni：准确率不低于 0.78
- Video-MME：准确率不低于 0.68
- Seed-TTS：平均 WER 不高于 0.05

正式性能口径使用 `vllm bench serve --omni`、`openai-chat-omni`、Seed-TTS English 子集和官方请求体；并发 1/4/8 分别测试 32/64/128 个请求。

## 阶段 0：冻结可复现基线

1. 固定代码 commit、镜像、模型校验值、CANN/torch/torch_npu/vLLM/vLLM-Ascend 版本。
2. 使用 `vllm_omni/deploy/minicpmo_4_5.yaml` 在单卡上启动服务。
3. 分别记录冷启动、两次预热后、并发 1/4/8 的性能。
4. 保存服务日志、Benchmark JSON、`npu-smi info` 和资源峰值。

官方 A3 参考基线：并发 1 的平均 TTFT 333.26 ms、音频 TTFP 986.47 ms、音频 RTF 0.4423。

## 阶段 1：低风险首响优化

1. 为 MiniCPM-o 增加 `initial_codec_chunk_frames`，首块使用 4 帧，后续恢复 25 帧。
2. 将共享内存连接器轮询间隔由 10 ms 降到 1 ms，减少阶段交接等待。
3. A/B 测试首块 4/8/12/25 帧，比较 TTFP、逐块 RTF、音频首块是否为空以及拼接连续性。
4. 仅保留精度无变化、Demo 无爆音/断裂的 Pareto 配置。

预期：TTFP 显著下降；稳态 25 帧不变，因此长音频总体 RTF 和质量风险较低。

## 阶段 2：消除每请求重复工作

1. 缓存默认参考音频对应的 Token2Wav prompt features。
2. 缓存并深拷贝默认固定音色的初始 Token2Wav state，避免 Demo 每个请求重复执行 prompt encoder 和 CFM 初始化；一次性 Seed-TTS 参考音频不进入该缓存。
3. 在 Stage 2 启动阶段预热首块与稳态固定 shape，避免首个 Demo 请求触发长时间 NPU 编译。
4. 验证并发请求状态完全隔离，结束/取消请求后缓存可正确回收。

预期：降低 TTFP 和并发下的排队时间；输出算法和采样参数不变。

## 阶段 3：Stage 2 核心性能优化

Stage 2 是当前主要瓶颈。按以下顺序用 Ascend Profiler 分解：

1. Token encoder、10 步 CFM、HiFT vocoder 分段计时。
2. 查找每 chunk 的 CPU/NPU 同步、`.item()`、D2H/H2D 拷贝和动态 shape 重编译。
3. 对 4 帧首块和 25 帧稳态建立固定 shape 路径；评估 NPU graph/compile。
4. 合并可复用张量分配和 cache stack/split，减少 Python 调度和小算子。
5. 以独立 YAML 评估 `token2wav_n_timesteps=8`，必须同时通过 Seed-TTS
   完整 WER 和 Demo 主观音质才允许进入最终配置；随后再评估
   `token2wav_float16`。

当前进展：并发 1 的 Flow/HiFT cache stack/split/clone 已改为零复制快路径，
CFM 时间轴与 speaker projection 改为启动/首次 prompt 时缓存；默认音色的
prompt 初始化 state 也改为只读模板共享，首个 live chunk 后再产生请求私有
state。固定 timestep embedding 已预计算，单请求 CFG latent 使用只读视图，
estimator 的全部 Euler-step cache 改为一次性连续预分配。910B4 小样本中，
前一轮单请求快路径相对 O1–O4 的 RTF 降 6.81%、TTFP 降 5.65%；新增热路径
相对同时段 O1–O6 的 RTF 再降 3.57%、TTFP 再降 3.55%。下一步用 profiler
判断 10 步 CFM 与 HiFT 的剩余占比。

## 阶段 4：调度与并发

1. 并发 1 优先 TTFT/TTFP；并发 4/8 检查 Stage 2 batching 命中率。
2. 调整 `max_num_seqs`、各 Stage 显存比例和 KV cache，避免单卡互相挤压。
3. 评估 prefix cache 对公共 chat template 的收益，确认多模态请求不会误复用。
4. 检查流式 backpressure，避免 Stage 1 生成速度明显快于 Stage 2 时无限积压。

### Stage 1 多步执行主线

逐 chunk 指标复测后，Stage 1 的 50 个串行 codec token 已被证明占据约 99%
的稳态 chunk 间隔。第一优先级 `FULL_DECODE_ONLY` 已完成：同周期 910B4 A/B
中 Stage 1 TPOT 下降 36.07%，mean steady chunk RTF 下降 29.68%，并通过
3/3 Seed-TTS WER=0 门禁，已进入默认低延迟配置。

runner-local 多步执行保留为下一项高杠杆实现，按以下约束推进：

1. 增加仅对纯 decode batch 生效的 `talker_decode_micro_steps`，prefill、混合
   batch、抢占和 KV transfer 路径保持原行为。
2. runner 为活跃请求预留连续的 N 个 KV slot，在一次 engine 调度中顺序执行
   N 个 Talker 子步；每一步仍使用上一步真实采样 codec token 的 embedding，
   保持原 RNG、TopP/TopK、repetition penalty、EOS 和 max-token 语义。
3. 子步只在 runner 内循环，避免每个 codec token 都往返 scheduler、输出构建、
   CPU bookkeeping 和 connector 路由；先实现 N=2，再测试 N=4/8。
4. 对每个请求独立处理提前 EOS，最终一次性返回多个 continue/stop token 和
   已达到 4/50 帧边界的 codec payload；scheduler 的 computed-token、KV block、
   取消/抢占计数必须按实际完成子步数推进。
5. 首先要求固定 seed 下 codec token 序列与全图单步路径逐 token 完全一致，再做
   NPU A/B。若 N=2 不能在 O20 基础上使 Stage 1 TPOT/等效 token 成本再降低 10%，停止
   扩大实现，转向自定义融合采样算子或训练轻量 draft head。

该路径不声称把自回归数学并行化；第一阶段收益来自减少 Python/IPC/调度与
输出构建开销。后续若能把 N 个顺序子步捕获为固定 ACL Graph，才进一步减少
NPU launch 间隔。任何用重复/预测 codec token 替代真实采样的近似方案均不
进入低风险主线。

## 阶段 5：准入回归与提交

每个候选版本先跑单元测试和音频连续性测试，再跑 Seed-TTS 小样本，最后跑完整三项精度测试。最终提交应包含：

- 完整代码、单卡 YAML、启动和 Benchmark 脚本
- Daily-Omni、Seed-TTS、Video-MME 原始输出与汇总
- 并发 1/4/8 的 TTFT、TTFP、逐 chunk RTF 和资源数据
- 官方 Demo 操作说明及连续运行视频
- 优化前后消融表、异常说明和完整复现步骤

## 决策规则

- 任何精度或音频连续性回退优先于性能收益，立即撤销该候选项。
- 所有对比必须使用相同数据、预热次数、并发、请求体和统计脚本。
- 冷启动数据单独报告，不与官方预热性能混合。
- 当前开发机是 910B4；最终结论必须由官方单卡 910C 重测确认。
- 本地 Daily-Omni parquet 需要先转换成 `qa.json + Videos/`；直接由
  `datasets 5.x` 读取会要求额外的 `torchcodec`，且本地字段名不是官方
  `qa.json` 字段。统一使用 `prepare_accuracy_data.py`，避免评测口径漂移。
