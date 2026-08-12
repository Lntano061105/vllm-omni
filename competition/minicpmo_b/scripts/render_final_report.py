#!/usr/bin/env python3
"""Render the official 910C evidence tree into a submission-ready Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / "competition/minicpmo_b/results/official_910c"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _mean(payload: dict[str, Any], nested: str, legacy: str) -> float:
    value = payload.get(nested)
    if isinstance(value, dict):
        value = value.get("mean")
    if not isinstance(value, (int, float)):
        value = payload.get(legacy)
    if not isinstance(value, (int, float)):
        raise ValueError(f"missing metric {nested}.mean/{legacy}")
    return float(value)


def _improvement(base: float, optimized: float) -> str:
    if base == 0:
        return "N/A"
    return f"{(1.0 - optimized / base) * 100.0:+.2f}%"


def render(result_root: Path) -> str:
    baseline = result_root / "official_910c_baseline"
    optimized = result_root / "official_910c_optimized"
    env = result_root / "environment"
    commit = (env / "git_commit.txt").read_text(encoding="utf-8").strip()
    preflight = _load(env / "preflight.json")
    lines = [
        "# MiniCPM-o 4.5 vLLM-Omni 子赛道 B 官方 910C 最终报告",
        "",
        "## 环境与版本",
        "",
        f"- Git commit：`{commit}`",
        f"- 单卡 910C preflight：`{preflight.get('passed') is True}`",
        f"- 可见设备：`{json.dumps(preflight.get('visible_devices', {}), ensure_ascii=False)}`",
        "- 完整依赖、模型权重和数据 SHA256：见 `environment/`",
        "",
        "## 原始性能瓶颈分析",
        "",
        "- SPEAK 生成阶段的主要稳态瓶颈位于 Stage 1 Talker 串行 codec decode：每个音频包需要多次 scheduler/IPC/Host 下发与小算子执行。",
        "- TTFP 关键路径还包含固定参考音频的 Stage 0 embedding、Stage 2 prompt 特征/初始状态构建以及首个较大 codec packet。",
        "- Stage 2 CFM/HiFT 是次级负载；仅优化 Token2Wav 无法获得与 Stage 1 调度消除同量级的 SPEAK RTF 收益。",
        "- 比赛 RTF 只统计 SPEAK 生成阶段，LISTEN、SPEAK 尾部和请求级全程 RTF 均未用于排名结论。",
        "",
        "## 最终优化方法",
        "",
        "- Stage 1 `FULL_DECODE_ONLY` ACLGraph、runner-local decode ×4、CPU slot mapping、图内 exact sampler 与二元控制 argmax，减少 Talker Host 往返。",
        "- 固定 13/25 codec 分包：缩短首包，同时保持稳态包形状稳定，降低图重捕获和调度抖动。",
        "- Stage 0 reference embedding 与 Stage 2 runtime prompt/initial-state 有界缓存及启动预载，降低固定 Demo 音色的重复首响开销。",
        "- Code2Wav runner 预热、conditional Euler 1 step、共享内存轮询 10 ms→1 ms，降低 Stage 2 首次与阶段交接延迟。",
        "- 取消 native silence continuation 的人工墙钟等待，但保留送入模型的静音 PCM，不修改模型语义。",
        "- Stage 0 禁用双进程多模态 processor LRU，消除并发长跑的缓存淘汰顺序分叉。",
        "- 未采用造成回退或精度风险未闭环的 FP16/W8A8、Flow BF16、激进 continuation 合并和实验性稳态 NPUGraph。",
        "",
        "## TTFT / TTFP（Chat Completions 辅助矩阵）",
        "",
        "| 并发 / 请求 | 基线 TTFT ms | 优化 TTFT ms | 改善 | 基线 TTFP ms | 优化 TTFP ms | 改善 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for concurrency, prompts in ((1, 32), (4, 64), (8, 128)):
        suffix = f"c{concurrency}_n{prompts}"
        base = _load(baseline / "performance" / f"seed_tts_{suffix}.json")
        opt = _load(optimized / "performance" / f"seed_tts_{suffix}.json")
        b_ttft, o_ttft = float(base["mean_ttft_ms"]), float(opt["mean_ttft_ms"])
        b_ttfp, o_ttfp = float(base["mean_audio_ttfp_ms"]), float(opt["mean_audio_ttfp_ms"])
        lines.append(
            f"| c{concurrency} / n{prompts} | {b_ttft:.3f} | {o_ttft:.3f} | "
            f"{_improvement(b_ttft, o_ttft)} | {b_ttfp:.3f} | {o_ttfp:.3f} | "
            f"{_improvement(b_ttfp, o_ttfp)} |"
        )

    lines += [
        "",
        "## 官方目标口径：Realtime SPEAK 生成阶段",
        "",
        "| 并发 / 请求 | 基线 TTFT ms | 优化 TTFT ms | 基线 TTFP ms | 优化 TTFP ms | 基线 SPEAK RTF | 优化 SPEAK RTF | RTF 改善 | Gate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for concurrency, prompts in ((1, 32), (4, 64), (8, 128)):
        suffix = f"c{concurrency}_n{prompts}"
        base = _load(baseline / "duplex_rtf" / f"native_duplex_{suffix}.json")
        opt = _load(optimized / "duplex_rtf" / f"native_duplex_{suffix}.json")
        gate = _load(optimized / "duplex_rtf" / f"gate_c{concurrency}.json")
        b_ttft, o_ttft = _mean(base, "ttft_ms", "mean_ttft_ms"), _mean(opt, "ttft_ms", "mean_ttft_ms")
        b_ttfp, o_ttfp = _mean(base, "ttfp_ms", "mean_audio_ttfp_ms"), _mean(opt, "ttfp_ms", "mean_audio_ttfp_ms")
        b_rtf = _mean(base, "speak_generation_rtf", "mean_audio_speak_generation_rtf")
        o_rtf = _mean(opt, "speak_generation_rtf", "mean_audio_speak_generation_rtf")
        lines.append(
            f"| c{concurrency} / n{prompts} | {b_ttft:.3f} | {o_ttft:.3f} | "
            f"{b_ttfp:.3f} | {o_ttfp:.3f} | {b_rtf:.6f} | {o_rtf:.6f} | "
            f"{_improvement(b_rtf, o_rtf)} | {'PASS' if gate.get('passed') is True else 'FAIL'} |"
        )

    lines += [
        "",
        "说明：排名主指标只使用 `audio_speak_generation_rtf`，未使用请求级或全部 chunk 平均 RTF。",
        "",
        "## 精度准入",
        "",
        "| Benchmark | 指标 | 官方基线 | 优化版 | 退化 | 2pp Gate |",
        "|---|---|---:|---:|---:|:---:|",
    ]
    for suite in ("daily-omni", "videomme", "seed-tts"):
        gate = _load(result_root / f"accuracy_gate_{suite}.json")
        for metric, values in gate.get("metrics", {}).items():
            lines.append(
                f"| {suite} | {metric} | {values.get('baseline'):.6f} | "
                f"{values.get('optimized'):.6f} | {values.get('regression'):.6f} | "
                f"{'PASS' if gate.get('passed') is True else 'FAIL'} |"
            )

    multiturn = _load(optimized / "duplex_rtf/multiturn_s2_t3/multiturn_gate.json")
    activation = _load(optimized / "duplex_rtf/activation_gate.json")
    demo = _load(result_root / "demo/demo_evidence.json")
    lines += [
        "",
        "## 稳定性、Activation 与 Demo",
        "",
        f"- 2 sessions × 3 turns 多轮门禁：`{'PASS' if multiturn.get('passed') is True else 'FAIL'}`",
        f"- 优化 activation 门禁：`{'PASS' if activation.get('passed') is True else 'FAIL'}`",
        f"- Demo 连续运行：`{demo.get('continuous_run_minutes')}` 分钟",
        f"- Demo 非预期错误 / 音频中断 / 空包：`{demo.get('unexpected_error_count')}` / `{demo.get('audio_interruption_count')}` / `{demo.get('empty_audio_packet_count')}`",
        f"- Demo 服务是否正常退出：`{demo.get('service_exit_clean') is True}`",
        "",
        "| Demo 场景 | 结果 |",
        "|---|:---:|",
    ]
    for name in ("text", "audio", "video", "text_audio"):
        passed = demo.get("scenarios", {}).get(name, {}).get("passed") is True
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    paired_path = result_root / "paired_confirmation/paired_confirmation_gate.json"
    if paired_path.is_file():
        paired = _load(paired_path)
        lines += [
            "",
            "### Fresh-start 反序确认（辅助证据）",
            "",
            f"- A1→B1、B2→A2 paired gate：`{'PASS' if paired.get('passed') is True else 'FAIL'}`",
            "- 该确认轮用于排除服务启动顺序与主机漂移，不替代官方 26 阶段主结果。",
            "",
            "| 指标 | A1→B1 改善 | A2→B2 改善 | 两轮改善 | A 漂移 | B 漂移 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for metric, values in paired.get("metrics", {}).items():
            lines.append(
                f"| {metric} | {values.get('pair1_improvement_pct'):.3f}% | "
                f"{values.get('pair2_improvement_pct'):.3f}% | "
                f"{values.get('two_repeat_improvement_pct'):.3f}% | "
                f"{values.get('baseline_repeat_drift_pct'):.3f}% | "
                f"{values.get('optimized_repeat_drift_pct'):.3f}% |"
            )
    lines += [
        "",
        "## 资源使用与异常说明",
        "",
        "- 硬件、HBM 进程快照与峰值证据：见 `environment/npu_smi.txt` 以及每轮结果目录的 `npu_smi.txt`。",
        "- 所有性能与精度阶段均要求服务日志、0 请求失败和对应 activation gate；失败阶段不会继续进入后续排名或归档。",
        "- 若本轮存在异常重试、OOM、编译回退或 Demo 中断，必须在此节追加具体阶段、原因和重跑证据；最终归档不得混入失败轮结果。",
        "",
        "## 完整复现步骤",
        "",
        "1. 使用环境清单中的不可变镜像 digest 启动官方单卡 910C 容器，并检出本报告 Git commit。",
        "2. 准备 `environment/` 清单中 SHA256 一致的模型、Daily-Omni、Video-MME、Seed-TTS 与 Whisper/WavLM/UTMOS。",
        "3. 执行 `run_official_910c_retest.py --image-digest <digest> --execute --confirm-single-910c`；26 个阶段均须通过。",
        "4. 接入官方 Demo，完成四场景录屏并填写 `demo/demo_evidence.json`。",
        "5. 执行 `render_final_report.py`、`validate_final_evidence.py`、`build_final_submission.py` 和最终 `validate_final_evidence.py --require-package`。",
        "",
        "## 复现与制品",
        "",
        "- 官方复测命令：`competition/minicpmo_b/scripts/run_official_910c_retest.py`",
        "- 最终源码制品：`final_submission/source/`",
        "- 全证据门禁：`final_evidence_audit.json`",
        "- Demo 视频与服务日志：`demo/`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.result_root.expanduser().resolve()
    output = (args.output or root / "final_submission/FINAL_REPORT.md").expanduser().resolve()
    rendered = render(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(f"Final report written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
