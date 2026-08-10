#!/usr/bin/env python3
"""Build an accuracy-protected Stage-0 ModelSlim W8A8 checkpoint.

Only the 8B Qwen3 language backbone is quantized. Vision, audio, resampler,
multimodal projectors, embeddings, final norm, LM head, and configurable edge
transformer layers remain BF16. The output is a self-contained MiniCPM-o 4.5
model directory intended only for Stage 0 with ``quantization: ascend``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM


_SOURCE_PREFIX = "llm."
_RUNTIME_QUANT_PREFIX = "thinker.llm."
_META_KEYS = frozenset(
    {
        "version",
        "model_quant_type",
        "kv_cache_type",
        "kv_quant_type",
        "fa_quant_type",
        "reduce_quant_type",
        "group_size",
    }
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/workspace/MiniCPM-o-4_5")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--device", choices=("npu", "cpu"), default="npu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--calibration-samples", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument(
        "--preserve-edge-layers",
        type=int,
        default=2,
        help="Keep this many first and last Qwen3 layers in BF16.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-raw-modelslim", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _checkpoint_index(model_path: Path) -> dict[str, Path]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = _read_json(index_path).get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Missing weight_map in {index_path}")
        return {str(name): model_path / str(filename) for name, filename in weight_map.items()}

    result: dict[str, Path] = {}
    for filename in sorted(model_path.glob("*.safetensors")):
        with safe_open(filename, framework="pt", device="cpu") as stream:
            for name in stream.keys():
                if name in result:
                    raise ValueError(f"Duplicate tensor {name!r}")
                result[name] = filename
    if not result:
        raise FileNotFoundError(f"No safetensors checkpoint found in {model_path}")
    return result


def _load_selected(
    index: dict[str, Path],
    predicate: Callable[[str], bool],
) -> dict[str, torch.Tensor]:
    by_file: dict[Path, list[str]] = defaultdict(list)
    for name, filename in index.items():
        if predicate(name):
            by_file[filename].append(name)

    tensors: dict[str, torch.Tensor] = {}
    for filename, names in sorted(by_file.items(), key=lambda row: str(row[0])):
        with safe_open(filename, framework="pt", device="cpu") as stream:
            for name in sorted(names):
                tensors[name] = stream.get_tensor(name)
    return tensors


def _build_float_model(
    model_path: Path,
    index: dict[str, Path],
) -> Qwen3ForCausalLM:
    raw_config = _read_json(model_path / "config.json")
    config = Qwen3Config.from_dict(raw_config)
    config.torch_dtype = torch.bfloat16
    # Avoid spending several minutes randomly initializing an 8B CPU model
    # whose parameters are immediately replaced by checkpoint tensors.
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)

    source = _load_selected(index, lambda name: name.startswith(_SOURCE_PREFIX))
    state = {name[len(_SOURCE_PREFIX) :]: tensor for name, tensor in source.items()}
    missing, unexpected = model.load_state_dict(state, strict=True, assign=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Thinker Qwen3 extraction mismatch: missing={missing} unexpected={unexpected}"
        )
    # Qwen3 keeps RoPE frequencies as non-persistent buffers, so they are not
    # populated by load_state_dict(assign=True). Recreate the two tiny buffers
    # on CPU before moving the fully materialized model to NPU.
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    rotary = Qwen3RotaryEmbedding(config, device=torch.device("cpu"))
    model.model.rotary_emb.inv_freq = rotary.inv_freq
    model.model.rotary_emb.original_inv_freq = rotary.original_inv_freq
    del source, state
    gc.collect()
    return model


def _calibration_texts(model_path: Path, samples: int) -> list[str]:
    prompts = [
        "请先打个招呼，再用一句话介绍 vLLM。",
        "用一句话介绍你自己。",
        "Describe the scene and answer the user's question accurately.",
        "Listen to the reference audio and speak the requested sentence naturally.",
        "Summarize the video, including the main action and important details.",
        "请结合图像、音频和上下文给出简洁准确的回答。",
        "Explain why low latency matters for real-time multimodal interaction.",
        "Generate a fluent response while preserving names, numbers, and punctuation.",
    ]

    # Seed-TTS target text is the fourth field. It supplies realistic English
    # lexical and punctuation distributions without making export depend on
    # the dataset being installed.
    for meta_path in (
        Path("/workspace/seed-tts/en/meta.lst"),
        Path("/tmp/minicpmo_b_seedtts/en/meta.lst"),
    ):
        if not meta_path.is_file():
            continue
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            fields = line.split("|")
            if len(fields) >= 4 and fields[3].strip():
                prompts.append(fields[3].strip())
            if len(prompts) >= samples:
                break
        break

    if not prompts:
        raise RuntimeError(f"No calibration text available for {model_path}")
    return [prompts[i % len(prompts)] for i in range(samples)]


def _build_calibration_data(
    model_path: Path,
    *,
    samples: int,
    sequence_length: int,
    device: torch.device,
) -> list[list[torch.Tensor]]:
    if samples <= 0 or sequence_length < 8:
        raise ValueError("calibration_samples must be positive and sequence_length >= 8")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    result: list[list[torch.Tensor]] = []
    for text_value in _calibration_texts(model_path, samples):
        prompt = (
            "<|im_start|>system\nYou are a helpful multimodal assistant.<|im_end|>\n"
            f"<|im_start|>user\n{text_value}<|im_end|>\n<|im_start|>assistant\n"
        )
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=sequence_length,
            padding=False,
        )
        ids = encoded["input_ids"]
        mask = encoded.get("attention_mask", torch.ones_like(ids))
        result.append([ids.to(device), mask.to(device)])
    return result


def _copy_model_metadata(source: Path, output: Path) -> list[str]:
    copied: list[str] = []
    for src in sorted(source.iterdir(), key=lambda path: path.name):
        if not src.is_file():
            continue
        name = src.name
        if name.endswith(".safetensors") or name == "model.safetensors.index.json":
            continue
        shutil.copy2(src, output / name)
        copied.append(name)
    if "config.json" not in copied:
        raise FileNotFoundError(f"Missing config.json in {source}")
    return copied


def _load_modelslim_tensors(raw_dir: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for filename in sorted(raw_dir.glob("*.safetensors")):
        with safe_open(filename, framework="pt", device="cpu") as stream:
            for name in stream.keys():
                if name in tensors:
                    raise ValueError(f"Duplicate ModelSlim tensor {name!r}")
                tensors[name] = stream.get_tensor(name)
    if not tensors:
        raise FileNotFoundError(f"ModelSlim produced no safetensors in {raw_dir}")
    return tensors


def _package_stage0_checkpoint(
    model_path: Path,
    output_path: Path,
    raw_dir: Path,
    index: dict[str, Path],
    *,
    args: argparse.Namespace,
    disabled_layers: list[int],
) -> dict[str, Any]:
    quant_tensors = _load_modelslim_tensors(raw_dir)
    packaged = {
        f"{_SOURCE_PREFIX}{name}": tensor.contiguous()
        for name, tensor in quant_tensors.items()
    }

    # Preserve all multimodal components exactly. ModelSlim already exports
    # disabled Qwen3 layers in BF16; overlay the sensitive endpoints directly
    # from the official checkpoint as an additional integrity guarantee.
    float_tensors = _load_selected(
        index,
        lambda name: not name.startswith(_SOURCE_PREFIX)
        or name in {
            "llm.model.embed_tokens.weight",
            "llm.model.norm.weight",
            "llm.lm_head.weight",
        },
    )
    for name, tensor in float_tensors.items():
        packaged[name] = tensor.contiguous()

    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "model.safetensors"
    save_file(packaged, checkpoint_path)

    raw_description = _read_json(raw_dir / "quant_model_description.json")
    description: dict[str, Any] = {}
    for name, value in raw_description.items():
        if name in _META_KEYS:
            description[name] = value
        else:
            description[f"{_RUNTIME_QUANT_PREFIX}{name}"] = value
    with (output_path / "quant_model_description.json").open("w", encoding="utf-8") as stream:
        json.dump(description, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")

    metadata_files = _copy_model_metadata(model_path, output_path)
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    manifest = {
        "format": "minicpmo-4.5-thinker-modelslim-mixed-w8a8-v1",
        "source_model": str(model_path),
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": digest,
        "tensor_count": len(packaged),
        "quant_description_entries": len(description),
        "metadata_files": metadata_files,
        "calibration": {
            "samples": args.calibration_samples,
            "sequence_length": args.sequence_length,
            "seed": args.seed,
            "device": args.device,
            "device_id": args.device_id,
        },
        "bf16_preserved": {
            "qwen3_layers": disabled_layers,
            "modules": [
                "thinker vision encoder",
                "thinker audio encoder",
                "resampler and multimodal projectors",
                "llm.model.embed_tokens",
                "llm.model.norm",
                "llm.lm_head",
            ],
        },
    }
    with (output_path / "quantization_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


def main() -> int:
    args = _parse_args()
    model_path = Path(args.model_path).resolve()
    output_path = Path(args.output_path).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if output_path.exists() and any(output_path.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_path}; pass --overwrite")
    output_path.mkdir(parents=True, exist_ok=True)

    index = _checkpoint_index(model_path)
    model = _build_float_model(model_path, index)
    layer_count = int(model.config.num_hidden_layers)
    edge = int(args.preserve_edge_layers)
    if edge < 0 or edge * 2 >= layer_count:
        raise ValueError("preserve_edge_layers must be >= 0 and less than half the layer count")
    disabled_layers = list(range(edge)) + list(range(layer_count - edge, layer_count))

    if args.device == "npu":
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("NPU calibration requested but torch.npu is unavailable")
        torch.npu.set_device(args.device_id)
        torch.npu.set_compile_mode(jit_compile=False)
        device = torch.device(f"npu:{args.device_id}")
    else:
        device = torch.device("cpu")
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    calibration_data = _build_calibration_data(
        model_path,
        samples=args.calibration_samples,
        sequence_length=args.sequence_length,
        device=device,
    )

    summary = {
        "model_path": str(model_path),
        "output_path": str(output_path),
        "layers": layer_count,
        "hidden_size": model.config.hidden_size,
        "calibration_samples": len(calibration_data),
        "disabled_layers": disabled_layers,
        "device": str(device),
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return 0

    from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

    disable_names = ["lm_head"]
    protected_linears = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    disable_names.extend(
        f"model.layers.{index}.{suffix}"
        for index in disabled_layers
        for suffix in protected_linears
    )
    quant_config = QuantConfig(
        w_bit=8,
        a_bit=8,
        disable_names=disable_names,
        dev_type=args.device,
        dev_id=args.device_id,
        act_method=1,
        w_sym=True,
        mm_tensor=False,
        is_dynamic=False,
        disable_last_linear=False,
        open_outlier=True,
    )

    raw_parent = output_path if args.keep_raw_modelslim else Path(tempfile.mkdtemp(prefix="thinker_w8a8_"))
    raw_dir = raw_parent / "modelslim_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    calibrator = Calibrator(model, quant_config, calib_data=calibration_data, disable_level="L0")
    calibrator.run()
    calibrator.save(str(raw_dir), save_type=["ascendV1"])

    manifest = _package_stage0_checkpoint(
        model_path,
        output_path,
        raw_dir,
        index,
        args=args,
        disabled_layers=disabled_layers,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not args.keep_raw_modelslim:
        shutil.rmtree(raw_parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
