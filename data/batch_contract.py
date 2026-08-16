"""Batch contract utilities for BrainMAP WebDataset samples."""

from __future__ import annotations

import io
import json
from typing import Any, Dict, Mapping

import numpy as np

from brainmap.data.modality import ModalityMapper


REQUIRED_SAMPLE_KEYS = ("__key__", "npy", "cls", "json")

def normalize_webdataset_sample(
    sample: Mapping[str, Any],
    modality_mapper: ModalityMapper | None = None,
) -> Dict[str, Any]:
    """Convert one packed WebDataset sample into the BrainMAP training batch contract."""
    missing_keys = [key for key in REQUIRED_SAMPLE_KEYS if key not in sample]
    if missing_keys:
        raise KeyError(f"Missing WebDataset sample keys: {missing_keys}")

    image = _decode_npy(sample["npy"]).astype(np.float32, copy=False)
    image = _ensure_channel_first_2d(image)
    metadata = _decode_json(sample["json"])
    modality = _decode_text(sample["cls"])

    metadata_modality = metadata.get("modality")
    if metadata_modality is not None and str(metadata_modality) != modality:
        raise ValueError(
            f"Modality mismatch between cls={modality!r} and metadata={metadata_modality!r}"
        )

    normalized = {
        "sample_key": _decode_text(sample["__key__"]),
        "image": image,
        "modality": modality,
        "metadata": metadata,
    }
    if modality_mapper is not None:
        normalized["modality_index"] = modality_mapper.to_index(modality)
    return normalized


def _decode_npy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        with io.BytesIO(bytes(value)) as buffer:
            return np.load(buffer, allow_pickle=False)
    raise TypeError(f"Expected np.ndarray or npy bytes, got {type(value).__name__}")


def _decode_json(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        value = bytes(value).decode("utf-8")
    if isinstance(value, str):
        decoded = json.loads(value)
        if not isinstance(decoded, dict):
            raise TypeError("Expected json metadata to decode into a dict")
        return decoded
    raise TypeError(f"Expected dict, str, or json bytes, got {type(value).__name__}")


def _decode_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8").strip()
    return str(value)


def _ensure_channel_first_2d(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image[None, :, :]
    if image.ndim == 3 and image.shape[0] == 1:
        return image
    raise ValueError(f"Expected 2D slice or 1xHxW image, got shape={image.shape}")

