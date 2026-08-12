from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from competition.minicpmo_b.scripts.gate_duplex_candidate import (
    PROFILES,
    evaluate_candidate,
)
from competition.minicpmo_b.scripts.compare_accuracy_results import (
    _load_result,
    _protocol_expected_requests,
    compare_accuracy,
)
from competition.minicpmo_b.scripts.summarize_ascend_ops import summarize
from competition.minicpmo_b.scripts.validate_candidate_log import validate_log
from vllm_omni.config.stage_config import load_deploy_config
from tests.e2e.accuracy.qwen3_omni.run_qwen_omni_acc_benchmark import (
    _validate_daily_omni,
    _validate_seed_tts,
    _validate_videomme,
)


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_portable_910c_config_matches_validated_npu5_candidate():
    repo_root = Path(__file__).resolve().parents[2]
    config_root = repo_root / "competition" / "minicpmo_b" / "config"
    validated = asdict(
        load_deploy_config(
            config_root
            / "ablations"
            / "minicpmo_4_5_duplex_euler_local4_graph_sampler_cpu_slot_fixed13_25_prewarm_ref_npu5.yaml"
        )
    )
    portable = asdict(load_deploy_config(config_root / "minicpmo_4_5_910c_low_latency.yaml"))

    # The submission YAML is deliberately self-contained.  Its model/runtime
    # hot path matches the measured 910B4 candidate.  Only the portable device
    # id and admission ceiling differ: c8 needs eight sessions to be accepted,
    # while stage max_num_seqs=4 still provides the measured execution batch.
    for stage in validated["stages"]:
        stage["devices"] = "0"
    validated["duplex_session"]["max_sessions"] = 8
    assert portable == validated


def test_competition_candidate_disables_mirrored_multimodal_lru_cache():
    """Long multimodal runs must not depend on two-process LRU ordering."""
    repo_root = Path(__file__).resolve().parents[2]
    config_root = repo_root / "competition" / "minicpmo_b" / "config"

    for config_path in (
        config_root / "minicpmo_4_5_910c_low_latency.yaml",
        config_root
        / "ablations"
        / "minicpmo_4_5_duplex_euler_local4_graph_sampler_cpu_slot_fixed13_25_prewarm_ref_npu5.yaml",
    ):
        config = load_deploy_config(config_path)
        stage0 = next(stage for stage in config.stages if stage.stage_id == 0)
        assert stage0.mm_processor_cache_gb == 0


def test_accuracy_baseline_overlay_only_disables_multimodal_lru_cache():
    repo_root = Path(__file__).resolve().parents[2]
    upstream = asdict(
        load_deploy_config(repo_root / "vllm_omni" / "deploy" / "minicpmo_4_5.yaml")
    )
    overlay = asdict(
        load_deploy_config(
            repo_root
            / "competition"
            / "minicpmo_b"
            / "config"
            / "ablations"
            / "minicpmo_4_5_official_baseline_accuracy_cacheoff.yaml"
        )
    )

    stage0 = next(stage for stage in overlay["stages"] if stage["stage_id"] == 0)
    assert stage0["mm_processor_cache_gb"] == 0
    upstream_stage0 = next(
        stage for stage in upstream["stages"] if stage["stage_id"] == 0
    )
    upstream_stage0["mm_processor_cache_gb"] = 0
    assert overlay == upstream


def test_fixed_packet_npugraph_ablation_preserves_validated_candidate():
    """The graph experiment must not silently alter another hot-path knob."""
    repo_root = Path(__file__).resolve().parents[2]
    config_root = repo_root / "competition" / "minicpmo_b" / "config" / "ablations"
    base = asdict(
        load_deploy_config(
            config_root
            / "minicpmo_4_5_duplex_euler_local4_graph_sampler_cpu_slot_fixed13_25_prewarm_ref_npu5.yaml"
        )
    )
    graph = asdict(
        load_deploy_config(
            config_root
            / "minicpmo_4_5_duplex_euler_local4_graph_sampler_cpu_slot_fixed13_25_prewarm_ref_steady_npugraph_npu5.yaml"
        )
    )

    base_extra = base["connectors"]["connector_of_shared_memory"]["extra"]
    graph_extra = graph["connectors"]["connector_of_shared_memory"]["extra"]
    for key in (
        "token2wav_steady_npugraph",
        "code2wav_npugraph_prompt_wavs",
        "code2wav_npugraph_prompt_sample_rate",
        "code2wav_npugraph_prompt_frame_samples",
    ):
        graph_extra.pop(key)
    assert graph_extra == base_extra
    graph["connectors"] = base["connectors"]
    assert graph == base


@pytest.mark.parametrize(
    ("validator", "result", "expected_fragment"),
    [
        (
            lambda result: _validate_daily_omni(
                result, min_accuracy=0.78, expected_requests=1196
            ),
            {
                "daily_omni_accuracy": 0.90,
                "daily_omni_evaluated": 1196,
                "daily_omni_evaluated_ok": 1195,
                "daily_omni_request_failed": 1,
                "daily_omni_parse_failed": 0,
                "completed": 1195,
                "failed": 1,
            },
            "completed=1195 != expected 1196",
        ),
        (
            lambda result: _validate_videomme(
                result, min_accuracy=0.68, expected_requests=2700
            ),
            {
                "videomme_accuracy": 0.80,
                "videomme_evaluated": 2700,
                "videomme_evaluated_ok": 2699,
                "videomme_request_failed": 1,
                "videomme_parse_failed": 0,
                "completed": 2699,
                "failed": 1,
            },
            "failed=1 != 0",
        ),
        (
            lambda result: _validate_seed_tts(
                result,
                max_mean_wer=0.05,
                min_mean_sim=None,
                min_mean_utmos=None,
                expected_requests=1000,
                expected_turns=1000,
            ),
            {
                "seed_tts_content_evaluated": 999,
                "seed_tts_content_error_mean": 0.01,
                "seed_tts_session_count": 1000,
                "seed_tts_turn_count": 1000,
                "seed_tts_request_failed": 0,
                "seed_tts_asr_failed": 1,
                "seed_tts_no_pcm": 1,
                "seed_tts_save_audio_failed": 0,
                "completed": 1000,
                "failed": 0,
            },
            "seed_tts_content_evaluated=999 != expected 1000",
        ),
    ],
)
def test_complete_accuracy_gate_rejects_partial_success(
    validator, result, expected_fragment
):
    assert expected_fragment in validator(result)


def test_complete_accuracy_gate_rejects_non_finite_score():
    errors = _validate_daily_omni(
        {"daily_omni_accuracy": float("nan")},
        min_accuracy=0.78,
    )

    assert errors == ["daily_omni_accuracy is non-finite: nan"]


def test_complete_seed_tts_gate_rejects_empty_pcm():
    result = {
        "completed": 2,
        "failed": 0,
        "seed_tts_session_count": 2,
        "seed_tts_turn_count": 2,
        "seed_tts_content_evaluated": 2,
        "seed_tts_content_error_mean": 0.0,
        "seed_tts_request_failed": 0,
        "seed_tts_asr_failed": 0,
        "seed_tts_no_pcm": 1,
        "seed_tts_save_audio_failed": 0,
    }

    errors = _validate_seed_tts(
        result,
        max_mean_wer=0.05,
        min_mean_sim=None,
        min_mean_utmos=None,
        expected_requests=2,
        expected_turns=2,
    )

    assert "seed_tts_no_pcm=1 != 0" in errors


def test_complete_seed_tts_quality_gate_requires_all_sim_and_utmos_rows():
    result = {
        "completed": 2,
        "failed": 0,
        "seed_tts_session_count": 2,
        "seed_tts_turn_count": 2,
        "seed_tts_content_evaluated": 2,
        "seed_tts_content_error_mean": 0.0,
        "seed_tts_request_failed": 0,
        "seed_tts_asr_failed": 0,
        "seed_tts_no_pcm": 0,
        "seed_tts_save_audio_failed": 0,
        "seed_tts_sim_mean": 0.8,
        "seed_tts_sim_evaluated": 1,
        "seed_tts_sim_failed": 1,
        "seed_tts_sim_skipped_no_ref": 0,
        "seed_tts_utmos_mean": None,
        "seed_tts_utmos_evaluated": 0,
        "seed_tts_utmos_failed": 2,
    }

    errors = _validate_seed_tts(
        result,
        max_mean_wer=0.05,
        min_mean_sim=0.7,
        min_mean_utmos=3.0,
        expected_requests=2,
        expected_turns=2,
    )

    assert "seed_tts_sim_evaluated=1 != expected 2" in errors
    assert "seed_tts_sim_failed=1 != 0" in errors
    assert any("Missing seed_tts_utmos_mean" in error for error in errors)
    assert "seed_tts_utmos_evaluated=0 != expected 2" in errors
    assert "seed_tts_utmos_failed=2 != 0" in errors


@pytest.mark.parametrize(
    ("suite", "metric", "baseline", "optimized", "passed"),
    [
        ("daily-omni", "daily_omni_accuracy", 0.85, 0.83, True),
        ("daily-omni", "daily_omni_accuracy", 0.85, 0.829, False),
        ("videomme", "videomme_accuracy", 0.704, 0.684, True),
        ("videomme", "videomme_accuracy", 0.704, 0.680, False),
        ("seed-tts", "seed_tts_content_error_mean", 0.02, 0.04, True),
        ("seed-tts", "seed_tts_content_error_mean", 0.02, 0.041, False),
    ],
)
def test_accuracy_comparison_enforces_two_percentage_points(
    suite, metric, baseline, optimized, passed
):
    baseline_result = {metric: baseline}
    optimized_result = {metric: optimized}
    if suite == "seed-tts":
        complete = {
            "completed": 1,
            "failed": 0,
            "seed_tts_quality_complete": True,
            "seed_tts_session_count": 1,
            "seed_tts_turn_count": 1,
            "seed_tts_content_evaluated": 1,
            "seed_tts_sim_evaluated": 1,
            "seed_tts_utmos_evaluated": 1,
            "seed_tts_request_failed": 0,
            "seed_tts_no_pcm": 0,
            "seed_tts_asr_failed": 0,
            "seed_tts_save_audio_failed": 0,
            "seed_tts_sim_failed": 0,
            "seed_tts_sim_skipped_no_ref": 0,
            "seed_tts_utmos_failed": 0,
            "seed_tts_sim_mean": 0.80,
            "seed_tts_utmos_mean": 4.0,
        }
        baseline_result.update(complete)
        optimized_result.update(complete)
    result = compare_accuracy(
        suite,
        baseline_result,
        optimized_result,
    )

    assert result["passed"] is passed


def test_accuracy_comparison_rejects_non_finite_metric():
    result = compare_accuracy(
        "daily-omni",
        {"daily_omni_accuracy": 0.85},
        {"daily_omni_accuracy": float("nan")},
    )

    assert result["passed"] is False
    assert "optimized missing numeric daily_omni_accuracy" in result["failures"]


def test_accuracy_comparison_rejects_below_absolute_gate_even_with_no_regression():
    result = compare_accuracy(
        "daily-omni",
        {"daily_omni_accuracy": 0.77},
        {"daily_omni_accuracy": 0.77},
    )

    assert result["passed"] is False
    assert any("absolute gate 0.780000" in failure for failure in result["failures"])


def test_seed_tts_comparison_rejects_quality_regression_despite_wer_parity():
    complete = {
        "completed": 1,
        "failed": 0,
        "seed_tts_quality_complete": True,
        "seed_tts_session_count": 1,
        "seed_tts_turn_count": 1,
        "seed_tts_content_evaluated": 1,
        "seed_tts_sim_evaluated": 1,
        "seed_tts_utmos_evaluated": 1,
    }
    result = compare_accuracy(
        "seed-tts",
        {
            **complete,
            "seed_tts_content_error_mean": 0.02,
            "seed_tts_sim_mean": 0.80,
            "seed_tts_utmos_mean": 4.0,
        },
        {
            **complete,
            "seed_tts_content_error_mean": 0.02,
            "seed_tts_sim_mean": 0.77,
            "seed_tts_utmos_mean": 4.0,
        },
    )

    assert result["passed"] is False
    assert any("seed_tts_sim_mean regression" in failure for failure in result["failures"])


def test_seed_tts_comparison_rejects_partial_quality_result():
    result = compare_accuracy(
        "seed-tts",
        {
            "seed_tts_content_error_mean": 0.02,
            "seed_tts_sim_mean": 0.80,
            "seed_tts_utmos_mean": 4.0,
        },
        {
            "seed_tts_content_error_mean": 0.02,
            "seed_tts_sim_mean": 0.80,
            "seed_tts_utmos_mean": 4.0,
        },
    )

    assert result["passed"] is False
    assert "baseline seed_tts_quality_complete is not true" in result["failures"]


def test_accuracy_loader_prefers_resumed_seed_tts_result(tmp_path: Path):
    (tmp_path / "qwen_omni_acc_seed_tts_raw.json").write_text(
        json.dumps({"seed_tts_quality_complete": False}), encoding="utf-8"
    )
    (tmp_path / "seed_tts_quality_resumed.json").write_text(
        json.dumps({"seed_tts_quality_complete": True}), encoding="utf-8"
    )

    assert _load_result(tmp_path)["seed_tts_quality_complete"] is True


def test_accuracy_gate_reads_expected_requests_from_protocol(tmp_path: Path):
    (tmp_path / "run_protocol.json").write_text(
        json.dumps(
            {
                "comparison": {
                    "fields": {"num_prompts": 1196},
                }
            }
        ),
        encoding="utf-8",
    )
    assert _protocol_expected_requests(tmp_path) == 1196


def test_seed_tts_comparison_enforces_absolute_wer_and_request_count():
    complete = {
        "completed": 999,
        "failed": 0,
        "seed_tts_quality_complete": True,
        "seed_tts_session_count": 999,
        "seed_tts_turn_count": 999,
        "seed_tts_content_evaluated": 999,
        "seed_tts_sim_evaluated": 999,
        "seed_tts_utmos_evaluated": 999,
        "seed_tts_content_error_mean": 0.051,
        "seed_tts_sim_mean": 0.8,
        "seed_tts_utmos_mean": 4.0,
    }
    result = compare_accuracy(
        "seed-tts", complete, complete, expected_requests=1000
    )

    assert result["passed"] is False
    assert "baseline completed=999 != expected 1000" in result["failures"]
    assert any("mean WER=0.051000" in failure for failure in result["failures"])


def _duplex_result(
    *,
    ttft: float = 1900.0,
    ttfp: float = 1900.0,
    speak_mean: float = 0.80,
    speak_median: float = 0.80,
    speak_p99: float = 0.85,
    chunks: int = 12,
):
    return {
        "audio_turns": 4,
        "audio_speak_generation_chunk_count": chunks,
        "ttft_ms": {"mean": ttft},
        "ttfp_ms": {"mean": ttfp},
        "speak_generation_rtf": {
            "mean": speak_mean,
            "median": speak_median,
            "p99": speak_p99,
        },
        # These fields must never be used by the gate as competition RTF.
        "request_audio_rtf": {"mean": 0.01},
        "mean_audio_chunk_rtf": 0.01,
        "runs": [{"ok": True} for _ in range(4)],
    }


def test_npugraph_gate_uses_speak_phase_and_accepts_large_rtf_gain():
    baseline = _duplex_result()
    candidate = _duplex_result(
        ttft=1920.0,
        ttfp=1920.0,
        speak_mean=0.67,
        speak_median=0.67,
        speak_p99=0.84,
    )

    result = evaluate_candidate(baseline, candidate, PROFILES["npugraph"])

    assert result["passed"] is True
    assert result["metrics"]["speak_rtf_mean"]["improvement_pct"] == pytest.approx(
        16.25
    )


def test_prompt_cache_gate_requires_ten_percent_first_response_gain():
    baseline = _duplex_result()
    candidate = _duplex_result(
        ttft=1750.0,
        ttfp=1700.0,
        speak_mean=0.808,
        speak_median=0.808,
        speak_p99=0.858,
    )

    result = evaluate_candidate(baseline, candidate, PROFILES["prompt-cache"])

    assert result["passed"] is False
    assert "TTFT gate failed" in result["failures"]


def test_stage1_hotpath_gate_requires_double_digit_speak_gain():
    baseline = _duplex_result()
    candidate = _duplex_result(
        ttft=1920.0,
        ttfp=1920.0,
        speak_mean=0.70,
        speak_median=0.70,
        speak_p99=0.84,
    )

    result = evaluate_candidate(baseline, candidate, PROFILES["stage1-hotpath"])

    assert result["passed"] is True
    assert result["metrics"]["speak_rtf_mean"]["improvement_pct"] == pytest.approx(12.5)


def test_gate_allows_packet_boundary_change_but_rejects_failed_runs():
    baseline = _duplex_result()
    candidate = _duplex_result(chunks=9)
    candidate["runs"][2]["ok"] = False

    result = evaluate_candidate(baseline, candidate, PROFILES["npugraph"])

    assert result["passed"] is False
    assert not any("chunk count changed" in failure for failure in result["failures"])
    assert any("runs incomplete" in failure for failure in result["failures"])


def test_gate_rejects_too_few_speak_phase_samples():
    baseline = _duplex_result()
    candidate = _duplex_result(chunks=3)

    result = evaluate_candidate(baseline, candidate, PROFILES["npugraph"])

    assert result["passed"] is False
    assert any("insufficient SPEAK generation chunks" in item for item in result["failures"])


def test_load_accuracy_result_accepts_real_videomme_filename(tmp_path: Path):
    payload = {"videomme_accuracy": 0.7}
    path = tmp_path / "omni_acc_videomme_20260812.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert _load_result(tmp_path) == payload


def test_log_gate_requires_real_capture_replay_and_both_prompt_cache_hits():
    text = "\n".join(
        (
            "CUDAGraphMode.FULL_DECODE_ONLY",
            "enable_npugraph_ex': True",
            "Capturing CUDA graphs (decode, FULL)",
            "Cached MiniCPM-o Stage-0 reference audio embedding",
            "Reused MiniCPM-o Stage-0 reference audio embedding cache",
            "Preloaded MiniCPM-o runtime prompt",
            "Cached MiniCPM-o Token2Wav immutable initial state",
            "Reused MiniCPM-o Token2Wav immutable initial state",
            "runtime_prompt_cache_size=4 cache_runtime_initial_state=True",
            "Captured steady MiniCPM-o Token2Wav NPUGraph bucket",
            "Captured steady MiniCPM-o Token2Wav NPUGraph bucket",
            "Replayed steady MiniCPM-o Token2Wav NPUGraph",
        )
    )

    result = validate_log(
        text,
        require_stage1_full_decode=True,
        require_stage0_ref_cache=True,
        require_stage2_prompt_cache=True,
        require_stage2_npugraph=True,
        min_npugraph_buckets=2,
    )

    assert result["passed"] is True
    assert result["captured_npugraph_buckets"] == 2


def test_log_gate_requires_backend_and_runner_13_25_prewarm():
    text = "\n".join(
        (
            "MiniCPM-o Code2Wav prewarm completed for prompt buckets [], "
            "live codec chunks [13, 25] with 3 left-context frames",
            "MiniCPM-o Code2Wav runner prewarm completed: prompts=1 "
            "live codec chunks=[13, 25] left_context=3",
        )
    )

    result = validate_log(text, require_stage2_runner_prewarm=True)

    assert result["passed"] is True


def test_log_gate_rejects_wrong_runner_prewarm_shapes():
    text = "\n".join(
        (
            "MiniCPM-o Code2Wav prewarm completed for prompt buckets [], "
            "live codec chunks [4, 25] with 3 left-context frames",
            "MiniCPM-o Code2Wav runner prewarm completed: prompts=1 "
            "live codec chunks=[4, 25] left_context=3",
        )
    )

    result = validate_log(text, require_stage2_runner_prewarm=True)

    assert result["passed"] is False
    assert any("13/25" in failure for failure in result["failures"])


def test_log_gate_rejects_fatal_traceback_and_missing_second_bucket():
    text = "\n".join(
        (
            "Captured steady MiniCPM-o Token2Wav NPUGraph bucket",
            "Replayed steady MiniCPM-o Token2Wav NPUGraph",
            "Traceback (most recent call last)",
        )
    )

    result = validate_log(
        text,
        require_stage2_npugraph=True,
        min_npugraph_buckets=2,
    )

    assert result["passed"] is False
    assert result["captured_npugraph_buckets"] == 1
    assert result["fatal_markers"] == ["Traceback (most recent call last)"]


def test_log_gate_rejects_multimodal_sender_receiver_cache_divergence():
    result = validate_log(
        "AssertionError: Expected a cached item for mm_hash='deadbeef'\n"
    )

    assert result["passed"] is False
    assert result["fatal_markers"] == [
        "AssertionError: Expected a cached item for mm_hash="
    ]


def test_log_gate_requires_multimodal_cache_disable_activation_marker():
    missing = validate_log("", require_stage0_mm_cache_disabled=True)
    active = validate_log(
        "[stage_init] Stage 0 multimodal processor cache: disabled "
        "(mm_processor_cache_gb=0)\n",
        require_stage0_mm_cache_disabled=True,
    )

    assert missing["passed"] is False
    assert active["passed"] is True


def test_log_gate_requires_stage1_cpu_slot_and_graph_sampler_activation():
    text = "\n".join(
        (
            "NPU Talker runner-local CPU slot mapping enabled",
            "Used NPU Talker runner-local CPU slot mapping fast path",
            "MiniCPM-o Talker graph-contained exact compact sampler enabled",
            "NPU Talker graph sampler inputs enabled with fixed 16-token history",
            "Used NPU Talker incremental graph sampler history fast path",
            "MiniCPM-o Talker deterministic binary control argmax enabled",
            "Used MiniCPM-o Talker deterministic binary control argmax",
        )
    )

    result = validate_log(
        text,
        require_stage1_cpu_slot_mapping=True,
        require_stage1_graph_sampler=True,
        require_stage1_binary_argmax=True,
    )

    assert result["passed"] is True


def test_log_gate_rejects_cpu_slot_mapping_fallback():
    text = "\n".join(
        (
            "NPU Talker runner-local CPU slot mapping enabled",
            "falling back to the standard GPU kernel",
        )
    )

    result = validate_log(text, require_stage1_cpu_slot_mapping=True)

    assert result["passed"] is False
    assert any("fell back" in failure for failure in result["failures"])


def test_ascend_op_summary_groups_stage1_control_hotspots(tmp_path):
    csv_path = tmp_path / "op_statistic.csv"
    csv_path.write_text(
        "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)\n"
        "0,MatMulV2,AI_CORE,10,500,1,50,60,50\n"
        "0,_compute_slot_mapping_kernel,AI_VECTOR_CORE,5,100,1,20,25,10\n"
        "0,Bincount,AI_CPU,5,150,1,30,35,15\n"
        "0,ScatterElements,AI_CPU,5,250,1,50,55,25\n",
        encoding="utf-8",
    )

    result = summarize([csv_path], top=2)

    assert result["device_total_time_us"] == 1000
    assert result["groups"]["slot_mapping"]["device_share_pct"] == 10
    assert result["groups"]["sampler_control"]["device_share_pct"] == 40
    assert result["slot_plus_sampler"]["device_share_pct"] == 50
    assert result["slot_plus_sampler"]["device_only_max_speedup_if_eliminated"] == 2
