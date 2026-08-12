# 子赛道 B 提交完成度审计

只有本表全部变为“通过”后，方案才可声明完成。910B4 开发结果只用于候选筛选，
不能填写为官方单卡 910C 最终成绩。

| 要求 | 当前状态 | 权威证据 / 最终应提交证据 |
|---|---|---|
| 自包含优化 YAML | 通过 | `config/minicpmo_4_5_910c_low_latency.yaml`；单测固定校验其热路径与 NPU 5 已测候选一致 |
| TTFT/TTFP/SPEAK RTF 开发门禁 | 通过（910B4） | `results/candidates/fixed13_25_boundaryfix_hot_n4_v1/native_duplex_rtf.json` 与两份 `gate_*.json` |
| 多轮生命周期与连续性 | 通过（910B4） | `results/candidates/fixed13_25_multiturn_boundaryfix_s2_t3_v1/native_duplex_rtf.json` |
| 优化真实 activation | 通过（910B4） | `results/candidates/fixed13_25_multiturn_boundaryfix_activation_gate.json` |
| Seed-TTS 小样本 | 通过（10/10） | `results/candidates/fixed13_25_runner_prewarm_transcriptfix_npu5_seed_tts_n10_v1/` |
| Daily-Omni 完整集 | 通过（910B4 优化版开发证据） | `results/candidates/fixed13_25_boundaryfix_cacheoff_daily_omni_n1196_v1/`：1196/1196 完成、0 失败、accuracy=0.7801003344、fatal gate 通过；官方 910C 仍须重跑基线和优化版 |
| Video-MME 完整集 | 通过（910B4 优化版开发证据） | `results/candidates/fixed13_25_cacheoff_videomme_n2700_v1/`：2700/2700 完成、0 失败、accuracy=0.6962962963、fatal gate 通过；官方 910C 仍须重跑基线和优化版 |
| Seed-TTS 完整集 | 未完成 | 2026-08-12 优化版已完成 1000/1000 音频请求、0 请求失败、continuity=100%，但 CPU Whisper WER 在用户要求释放资源时中断且未落盘最终 JSON，不能作为精度证据；仍需 mean WER ≤ 0.05、SIM/UTMOS 各 1000/1000、零空 PCM/请求/ASR/质量评测失败；当前镜像缺 WavLM/UTMOS 离线权重 |
| 相对官方基线精度降幅 ≤ 2pp | 未验证 | 三项完整集必须同时保存官方基线与优化结果 |
| 官方 Demo 文本/音频/视频/text+audio | 未完成 | 使用 `finalize_demo_evidence.py` 从运行元数据与四场景逐请求 JSON 重算请求/完成/audio packet 计数并生成全文件 SHA256；带时区起止时间与连续运行时长一致；音频中断/空包/underrun/意外错误均为 0，服务自然退出且日志无 fatal marker；`validate_final_evidence.py --demo-only` 必须通过 |
| 官方单卡 910C c1/c4/c8 | 未完成 | 基线与优化三组原始 JSON、activation gate、同输入/顺序/warmup |
| 官方 910C 多轮稳定性 | 未完成 | `run_duplex_matrix.sh` 已自动强制同一 WebSocket session 的 2 sessions × 3 turns 生命周期字段与服务日志水位告警门禁；仍需官方 910C 原始 JSON/日志 |
| 离线同步与异步回归 | 通过（当前工作树） | 冻结镜像保持 `pytest==8.3.2`（`triton-ascend` 硬依赖）并安装兼容的 `pytest-asyncio==1.3.0` 后，目标集合标准 pytest 单轮 506 passed、0 skipped；仓库 dev extra 的 pytest 9.1.1/pytest-asyncio 1.4.0 需在隔离 venv 使用 |
| 环境/模型/数据冻结 | 910B4 开发快照已采集，910C 待完成 | `results/environment_910b4_fixed13_25_boundaryfix_20260811/`；最终仍需 910C 模型权重 hash、数据索引 hash、镜像 digest |
| 最终报告与复现包 | 未完成 | 填满 `REPORT.md` 的 910C 表格，附代码 SHA、patch、YAML、脚本、日志、视频和 SHA256 |
| 提交包静态审计 | 通过 | 先运行 `python competition/minicpmo_b/scripts/validate_submission_package.py --output /tmp/minicpmo_submission_source_plan.json` 审阅 `planned_source_files`，再运行同脚本的 `--require-tracked` 门禁；所有 `competition/minicpmo_b/` 源码/配置（排除 `results/`）与 `tests/competition/` 新文件均纳入计划，同时拒绝跟踪 `results/`、`extra-info/`、`kernel_meta/`、缺失的 YAML 继承目标或超过 10 MiB 的源文件。根目录 Ascend `extra-info/` 与 `kernel_meta/` 缓存已忽略，不会把源码制品误判为脏工作树 |
| 官方复测编排 | 静态实现通过，待 910C 执行 | `run_official_910c_retest.py` dry-run-first、显式二次确认、干净工作树；冻结输入/镜像/Git commit/tree/26 阶段命令及计划 SHA256，拒绝同目录计划漂移；重跑会保留历史并失效自身及全部下游 marker；最终审计重建规范计划并核对 marker 源码身份、UTC 时间、命令和计划哈希；含单 910C preflight、A/B 矩阵、c1/c4/c8 门禁和三项精度比较 |
| 可复现源码制品 | 开发态生成与 clean-clone 验证通过，官方证据包待生成 | 工作树干净且全部证据完成后，以 `origin/minicpm-challenge` 为 base 运行 `build_submission_artifacts.py`，提交 archive、binary patch、Git SHA/tree 和覆盖整个 HEAD（含 benchmark/E2E 支撑测试）的 SHA256 清单；最终审计会打开归档逐文件核对清单，并绑定环境冻结 commit 与源码制品 commit |
| 最终全证据门禁 | 未通过（当前应失败） | `validate_final_evidence.py` 必须最终得到 `passed=true`；直接核验 910C 环境、原始性能/精度、多轮、Demo 四场景独立证据/计数/媒体头/日志与视频 SHA256、报告与源码 SHA256，当前缺失证据不得被 910B4 开发结果替代 |
| 最终确定性归档 | 未生成 | 全证据通过后运行 `build_final_submission.py`，再用 `validate_final_evidence.py --require-package` 验证归档 SHA256、成员清单与完整性 |

最终执行顺序与命令见 `OFFICIAL_910C_RETEST.md`。任何完整精度、Demo、910C
性能或异步回归证据缺失时，都不得把本表或报告改成“完成”。
