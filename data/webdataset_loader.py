"""WebDataset input pipeline for BrainMAP training."""

from __future__ import annotations

import io
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from brainmap.data.modality import ModalityMapper
from brainmap.data.batch_contract import normalize_webdataset_sample


def build_webdataset_pipeline(
    shards: Sequence[str],
    shuffle_buffer: int = 0,
    modality_mapper: ModalityMapper | None = None,
    base_dir: str | Path | None = None,
    distributed: bool = False,
):
    """Build an iterable WebDataset pipeline that emits normalized BrainMAP samples."""
    if not shards:
        raise ValueError("shards must contain at least one WebDataset tar path or pattern")
    if shuffle_buffer < 0:
        raise ValueError("shuffle_buffer must be non-negative")

    try:
        import webdataset as wds
    except ImportError as exc:
        raise ImportError("webdataset is required to read BrainMAP training shards") from exc

    webdataset_kwargs = {"shardshuffle": False, "empty_check": False}
    if distributed:
        webdataset_kwargs["nodesplitter"] = wds.split_by_node
    dataset = wds.WebDataset(_normalize_shards(resolve_shards(shards, base_dir=base_dir)), **webdataset_kwargs)
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer)
    mapper = modality_mapper or ModalityMapper.default()
    return dataset.map(lambda sample: normalize_webdataset_sample(sample, mapper))


def resolve_shards(shards: Sequence[str], base_dir: str | Path | None = None) -> List[str]:
    """Resolve local shard paths or glob patterns while leaving remote URLs unchanged."""
    resolved: List[str] = []
    root = Path(base_dir).resolve() if base_dir is not None else None
    for shard in shards:
        shard_text = str(shard)
        if "://" in shard_text or shard_text.startswith("pipe:"):
            resolved.append(shard_text)
            continue

        path = Path(shard_text)
        if not path.is_absolute() and root is not None and not _path_exists_or_is_glob_match(path):
            path = root / path

        if any(char in str(path) for char in ["*", "?", "["]):
            matches = sorted(Path(match) for match in path.parent.glob(path.name))
            if not matches:
                raise FileNotFoundError(f"No shards matched pattern: {path}")
            resolved.extend(str(match.resolve()) for match in matches)
        else:
            if not path.exists():
                raise FileNotFoundError(f"Shard path not found: {path}")
            resolved.append(str(path.resolve()))
    return resolved


def _path_exists_or_is_glob_match(path: Path) -> bool:
    if path.exists():
        return True
    if any(char in str(path) for char in ["*", "?", "["]):
        return any(path.parent.glob(path.name))
    return False


def _normalize_shards(shards: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for shard in shards:
        path = Path(shard)
        if path.exists():
            normalized.append(f"file:{path.resolve()}")
        else:
            normalized.append(shard)
    return normalized


def build_train_loader(
    shards: Sequence[str],
    batch_size: int,
    num_workers: int,
    shuffle_buffer: int = 1000,
    modality_mapper: ModalityMapper | None = None,
    base_dir: str | Path | None = None,
    distributed: bool = False,
):
    """Build a PyTorch DataLoader over normalized BrainMAP WebDataset samples."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("torch is required to build the BrainMAP training DataLoader") from exc

    mapper = modality_mapper or ModalityMapper.default()
    dataset = build_webdataset_pipeline(
        shards=shards,
        shuffle_buffer=shuffle_buffer,
        modality_mapper=mapper,
        base_dir=base_dir,
        distributed=distributed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_brainmap_batch,
    )


def collate_brainmap_batch(samples: List[Dict[str, object]]) -> Dict[str, object]:
    """Collate normalized samples into a training-ready BrainMAP batch."""
    if not samples:
        raise ValueError("samples must not be empty")

    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required to collate BrainMAP training batches") from exc

    images = np.stack([sample["image"] for sample in samples], axis=0)
    modality_indices = [int(sample["modality_index"]) for sample in samples]
    return {
        "image": torch.as_tensor(images, dtype=torch.float32),
        "modality_index": torch.as_tensor(modality_indices, dtype=torch.long),
        "modality": [str(sample["modality"]) for sample in samples],
        "metadata": [dict(sample["metadata"]) for sample in samples],
        "sample_key": [str(sample["sample_key"]) for sample in samples],
    }


def _write_dummy_shard(
    shard_path: Path,
    sample_key: str,
    image: np.ndarray,
    modality: str,
    metadata: dict,
) -> None:
    with tarfile.open(shard_path, "w") as tar:
        npy_buffer = io.BytesIO()
        np.save(npy_buffer, image)
        _add_bytes(tar, f"{sample_key}.npy", npy_buffer.getvalue())
        _add_bytes(tar, f"{sample_key}.cls", modality.encode("utf-8"))
        _add_bytes(tar, f"{sample_key}.json", json.dumps(metadata).encode("utf-8"))


def _add_bytes(tar: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(content)
    tar.addfile(info, io.BytesIO(content))
