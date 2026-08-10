#!/usr/bin/env python3
"""Build a Stage-1-only ModelSlim W8A8 checkpoint for MiniCPM-o 4.5.

The competition pipeline does not need to quantize or duplicate the 8B
Thinker.  This tool extracts the ~200M-parameter Talker Llama backbone,
calibrates it with codec/text-like embeddings, quantizes only its transformer
linears, and packages the result with the small floating-point Talker assets
required by vLLM-Omni Stage 1.

The output directory is intentionally a valid MiniCPM model directory and can
be selected only for Stage 1 with ``model: <output>`` and
``quantization: ascend`` in the deploy YAML.  Stage 0 and Stage 2 continue to
load the original checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM


_SOURCE_PREFIX = "tts."
_BACKBONE_PREFIX = "tts.model."
# The registered Stage-1 wrapper is mounted below ``talker`` while the native
# MiniCPM Talker keeps its Llama backbone below ``tts_obj``. ModelSlim's
# description is queried with that full runtime module prefix.
_RUNTIME_QUANT_PREFIX = "talker.tts_obj."
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
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--condition-tokens", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-raw-modelslim",
        action="store_true",
        help="Keep the intermediate unprefixed ModelSlim checkpoint.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files in an existing output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate extraction/calibration construction without quantizing.",
    )
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

    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors checkpoint found in {model_path}")
    result: dict[str, Path] = {}
    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as stream:
            for name in stream.keys():
                if name in result:
                    raise ValueError(f"Duplicate tensor {name!r} in {filename} and {result[name]}")
                result[name] = filename
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


def _talker_llama_config(model_config: dict[str, Any]) -> LlamaConfig:
    raw = model_config.get("tts_config")
    if not isinstance(raw, dict):
        raise ValueError("MiniCPM config.json does not contain tts_config")
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=int(raw["hidden_size"]),
        intermediate_size=int(raw["intermediate_size"]),
        num_hidden_layers=int(raw["num_hidden_layers"]),
        num_attention_heads=int(raw["num_attention_heads"]),
        num_key_value_heads=int(raw["num_key_value_heads"]),
        hidden_act=str(raw.get("hidden_act", "silu")),
        max_position_embeddings=int(raw["max_position_embeddings"]),
        rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
        tie_word_embeddings=False,
        use_cache=True,
    )
    config.torch_dtype = torch.bfloat16
    return config


def _build_float_calibration_model(
    model_path: Path,
    index: dict[str, Path],
    *,
    seed: int,
) -> LlamaForCausalLM:
    config = _talker_llama_config(_read_json(model_path / "config.json"))
    model = LlamaForCausalLM(config).to(dtype=torch.bfloat16).eval()

    source = _load_selected(index, lambda name: name.startswith(_BACKBONE_PREFIX))
    state = {name[len(_SOURCE_PREFIX) :]: tensor for name, tensor in source.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"lm_head.weight"}
    if set(missing) != allowed_missing or unexpected:
        raise RuntimeError(
            "Talker backbone extraction mismatch: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )

    # Runtime decode consumes emb_code while prefill consumes emb_text plus a
    # semantic projection. ModelSlim's feature collector requires generate(),
    # so expose representative external embeddings through the otherwise
    # unused Llama token embedding table during calibration. The original
    # table is restored when the final checkpoint is packaged.
    extras = _load_selected(
        index,
        lambda name: name in {"tts.emb_code.0.weight", "tts.emb_text.weight"},
    )
    code = extras["tts.emb_code.0.weight"].to(dtype=torch.bfloat16)
    text = extras["tts.emb_text.weight"].to(dtype=torch.bfloat16)
    table = model.model.embed_tokens.weight.data
    code_rows = min(code.shape[0], table.shape[0])
    table[:code_rows].copy_(code[:code_rows])

    remaining = table.shape[0] - code_rows
    if remaining > 0:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        ids = torch.randint(0, text.shape[0], (remaining,), generator=generator)
        table[code_rows:].copy_(text.index_select(0, ids))
    return model


def _build_calibration_data(
    config: LlamaConfig,
    *,
    samples: int,
    sequence_length: int,
    condition_tokens: int,
    code_vocab_size: int,
    seed: int,
    device: torch.device,
) -> list[list[torch.Tensor]]:
    if samples <= 0:
        raise ValueError("calibration_samples must be positive")
    if sequence_length < 2:
        raise ValueError("sequence_length must be >= 2")
    if not 0 <= condition_tokens <= sequence_length:
        raise ValueError("condition_tokens must be in [0, sequence_length]")
    if not 0 < code_vocab_size <= config.vocab_size:
        raise ValueError("Invalid code vocabulary size")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    result: list[list[torch.Tensor]] = []
    for _ in range(samples):
        ids = torch.empty((1, sequence_length), dtype=torch.long)
        if condition_tokens:
            ids[:, :condition_tokens] = torch.randint(
                code_vocab_size,
                config.vocab_size,
                (1, condition_tokens),
                generator=generator,
            )
        decode_tokens = sequence_length - condition_tokens
        if decode_tokens:
            ids[:, condition_tokens:] = torch.randint(
                0,
                code_vocab_size,
                (1, decode_tokens),
                generator=generator,
            )
        attention_mask = torch.ones_like(ids)
        result.append([ids.to(device), attention_mask.to(device)])
    return result


def _copy_model_metadata(source: Path, output: Path) -> list[str]:
    copied: list[str] = []
    # A trust_remote_code model may import sibling modules recursively. Copy
    # every lightweight repository file so the Stage-1-only checkpoint stays
    # a self-contained Hugging Face model directory. The original sharded
    # weights and index are deliberately excluded.
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
    files = sorted(raw_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"ModelSlim produced no safetensors in {raw_dir}")
    tensors: dict[str, torch.Tensor] = {}
    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as stream:
            for name in stream.keys():
                if name in tensors:
                    raise ValueError(f"Duplicate ModelSlim tensor {name!r}")
                tensors[name] = stream.get_tensor(name)
    return tensors


def _package_stage1_checkpoint(
    model_path: Path,
    output_path: Path,
    raw_dir: Path,
    index: dict[str, Path],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    quant_tensors = _load_modelslim_tensors(raw_dir)
    packaged: dict[str, torch.Tensor] = {}
    for name, tensor in quant_tensors.items():
        if name.startswith("lm_head."):
            continue
        packaged[f"{_SOURCE_PREFIX}{name}"] = tensor.contiguous()

    float_tensors = _load_selected(
        index,
        lambda name: name.startswith(_SOURCE_PREFIX) and not name.startswith("tts.model.layers."),
    )
    # Restore the original, unused Llama embedding table and retain the
    # floating-point condition/code embeddings, projectors and codec head.
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
        elif not name.startswith("lm_head."):
            description[f"{_RUNTIME_QUANT_PREFIX}{name}"] = value
    with (output_path / "quant_model_description.json").open("w", encoding="utf-8") as stream:
        json.dump(description, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")

    metadata_files = _copy_model_metadata(model_path, output_path)
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    manifest = {
        "format": "minicpmo-4.5-talker-modelslim-w8a8-v1",
        "source_model": str(model_path),
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": digest,
        "tensor_count": len(packaged),
        "quant_description_entries": len(description),
        "metadata_files": metadata_files,
        "calibration": {
            "samples": args.calibration_samples,
            "sequence_length": args.sequence_length,
            "condition_tokens": args.condition_tokens,
            "seed": args.seed,
            "device": args.device,
            "device_id": args.device_id,
        },
        "float_preserved": [
            "tts.model.embed_tokens",
            "tts.model.norm",
            "tts.emb_text",
            "tts.emb_code",
            "tts.projector_semantic",
            "tts.projector_spk",
            "tts.head_code",
        ],
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
    model = _build_float_calibration_model(model_path, index, seed=args.seed)
    code_vocab_size = int(_read_json(model_path / "config.json")["tts_config"]["num_audio_tokens"])

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
        model.config,
        samples=args.calibration_samples,
        sequence_length=args.sequence_length,
        condition_tokens=args.condition_tokens,
        code_vocab_size=code_vocab_size,
        seed=args.seed,
        device=device,
    )

    print(
        json.dumps(
            {
                "model_path": str(model_path),
                "output_path": str(output_path),
                "layers": model.config.num_hidden_layers,
                "hidden_size": model.config.hidden_size,
                "calibration_samples": len(calibration_data),
                "device": str(device),
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )
    if args.dry_run:
        return 0

    from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

    quant_config = QuantConfig(
        w_bit=8,
        a_bit=8,
        disable_names=["lm_head"],
        dev_type=args.device,
        dev_id=args.device_id,
        act_method=1,
        w_sym=True,
        mm_tensor=False,
        is_dynamic=False,
        disable_last_linear=False,
        open_outlier=True,
    )

    raw_parent = output_path if args.keep_raw_modelslim else Path(tempfile.mkdtemp(prefix="talker_w8a8_"))
    raw_dir = raw_parent / "modelslim_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    calibrator = Calibrator(model, quant_config, calib_data=calibration_data, disable_level="L0")
    calibrator.run()
    calibrator.save(str(raw_dir), save_type=["ascendV1"])

    manifest = _package_stage1_checkpoint(model_path, output_path, raw_dir, index, args=args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not args.keep_raw_modelslim:
        shutil.rmtree(raw_parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
