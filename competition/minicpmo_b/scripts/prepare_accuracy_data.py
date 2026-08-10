#!/usr/bin/env python3
"""Prepare the local challenge datasets for vLLM-Omni accuracy runs.

Daily-Omni's local MTEB mirror stores video/audio bytes inside parquet rows,
while the vLLM benchmark consumes the official ``qa.json`` + ``Videos/``
layout. Video-MME ships parquet plus chunked ZIP archives. This tool converts
both mirrors without downloading data again and supports small smoke subsets.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def _daily_answer_letter(answer: Any, candidates: list[Any]) -> str:
    raw = str(answer or "").strip()
    match = re.match(r"^([A-D])(?:[.、:：)\s]|$)", raw, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    for index, candidate in enumerate(candidates[:4]):
        if raw == str(candidate or "").strip():
            return "ABCD"[index]
    raise ValueError(f"Cannot normalize Daily-Omni answer to A-D: {raw!r}")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size == len(payload):
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}-",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def prepare_daily_omni(source: Path, destination: Path, limit: int) -> None:
    parquet_files = sorted((source / "data").glob("test-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Daily-Omni parquet shards found under {source / 'data'}")

    qa_rows: list[dict[str, Any]] = []
    media_written: set[str] = set()
    done = False
    for parquet_path in parquet_files:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(
            batch_size=1,
            columns=["video_id", "video", "audio", "question", "candidates", "answer"],
        ):
            row = batch.to_pylist()[0]
            video_id = str(row["video_id"])
            candidates = list(row["candidates"] or [])
            qa_rows.append(
                {
                    "video_id": video_id,
                    "Question": row["question"],
                    "Choice": candidates,
                    # The official Daily-Omni evaluator compares the parsed
                    # model letter with raw ``Answer``. The MTEB parquet mirror
                    # stores "B. option text", so normalize it back to "B".
                    "Answer": _daily_answer_letter(row["answer"], candidates),
                }
            )
            if video_id not in media_written:
                video_bytes = (row.get("video") or {}).get("bytes")
                audio_bytes = (row.get("audio") or {}).get("bytes")
                if not video_bytes or not audio_bytes:
                    raise ValueError(f"Daily-Omni row {video_id!r} has no embedded video/audio bytes")
                media_dir = destination / "Videos" / video_id
                _atomic_write_bytes(media_dir / f"{video_id}_video.mp4", video_bytes)
                _atomic_write_bytes(media_dir / f"{video_id}_audio.wav", audio_bytes)
                media_written.add(video_id)
            if limit > 0 and len(qa_rows) >= limit:
                done = True
                break
        if done:
            break

    _atomic_write_json(destination / "qa.json", qa_rows)
    print(
        f"Daily-Omni ready: qa={len(qa_rows)} unique_media={len(media_written)} "
        f"root={destination}"
    )


def _video_ids_from_parquet(parquet_path: Path, limit: int) -> set[str]:
    parquet = pq.ParquetFile(parquet_path)
    required: set[str] = set()
    rows = 0
    for batch in parquet.iter_batches(batch_size=256, columns=["videoID"]):
        for row in batch.to_pylist():
            required.add(str(row["videoID"]))
            rows += 1
            if limit > 0 and rows >= limit:
                return required
    return required


def _extract_videomme_zip(zip_path: Path, destination: Path, required_ids: set[str]) -> tuple[int, int]:
    extracted = 0
    skipped = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir() or Path(info.filename).suffix.lower() not in {".mp4", ".mkv", ".webm", ".avi"}:
                continue
            basename = Path(info.filename).name
            if Path(basename).stem not in required_ids:
                continue
            target = destination / basename
            if target.is_file() and target.stat().st_size == info.file_size:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{basename}-", delete=False) as handle:
                temporary = Path(handle.name)
                with archive.open(info) as source:
                    shutil.copyfileobj(source, handle, length=16 * 1024 * 1024)
            os.replace(temporary, target)
            extracted += 1
    return extracted, skipped


def prepare_videomme(source: Path, destination: Path, limit: int, workers: int) -> None:
    source_parquet = source / "videomme" / "test-00000-of-00001.parquet"
    if not source_parquet.is_file():
        raise FileNotFoundError(f"Video-MME parquet not found: {source_parquet}")
    required_ids = _video_ids_from_parquet(source_parquet, limit)
    zip_paths = sorted(source.glob("videos_chunked_*.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"No Video-MME video archives found under {source}")

    destination_parquet = destination / "videomme" / source_parquet.name
    destination_parquet.parent.mkdir(parents=True, exist_ok=True)
    if not destination_parquet.exists():
        destination_parquet.symlink_to(source_parquet.resolve())

    video_dir = destination / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        results = list(
            executor.map(
                lambda path: _extract_videomme_zip(path, video_dir, required_ids),
                zip_paths,
            )
        )
    extracted = sum(item[0] for item in results)
    skipped = sum(item[1] for item in results)
    available = {path.stem for path in video_dir.iterdir() if path.is_file()}
    missing = sorted(required_ids - available)
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} required Video-MME videos after extraction: {preview}")
    print(
        f"Video-MME ready: required_videos={len(required_ids)} extracted={extracted} "
        f"reused={skipped} root={destination}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    daily = subparsers.add_parser("daily-omni")
    daily.add_argument("--source", type=Path, default=Path("/workspace/MTEB/Daily-Omni"))
    daily.add_argument("--destination", type=Path, default=Path("/tmp/minicpmo_b_daily_omni"))
    daily.add_argument("--limit", type=int, default=0, help="QA rows to prepare; 0 means all rows.")

    video = subparsers.add_parser("videomme")
    video.add_argument("--source", type=Path, default=Path("/workspace/Video-MME"))
    video.add_argument("--destination", type=Path, default=Path("/tmp/minicpmo_b_videomme"))
    video.add_argument("--limit", type=int, default=0, help="QA rows whose videos are needed; 0 means all rows.")
    video.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.dataset == "daily-omni":
        prepare_daily_omni(args.source.resolve(), args.destination.resolve(), args.limit)
    else:
        prepare_videomme(args.source.resolve(), args.destination.resolve(), args.limit, args.workers)


if __name__ == "__main__":
    main()
