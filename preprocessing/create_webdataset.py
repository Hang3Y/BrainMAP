"""Create float16 2D-slice WebDataset shards from final MNI MRI volumes."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import nibabel as nib
import numpy as np
import webdataset as wds
from tqdm import tqdm


VALID_MODALITIES = {"t1n", "t1c", "t2w", "t2f"}

# Explicit final MRI MNI input directories, relative to this script directory.
SOURCE_LIST = [
    "1001_abideii/abide_fm_ready_skull_stripping_n4_mni",
    "1002_brats2023/brats_fm_ready_mni",
    "1005_atlasr2/atlas_fm_ready_skull_stripping_n4_mni",
    "1007_isles2022/isles_fm_ready_n4_mni",
    "1010_msd/msd_fm_ready_mni",
    "1014_ixi/ixi_fm_ready_skull_stripping_n4_mni",
    "1016_l2r/l2r_fm_ready_mni",
    "1017_mets/mets_fm_ready_mni",
    "1019_lumiere/lumiere_fm_ready_mni",
    "1020_wmh/wmh_fm_ready_skull_stripping_n4_mni",
    "1021_ucsd_ptgbm/ucsd_ptgbm_fm_ready_mni",
    "1022_upenn_gbm/upenn_gbm_fm_ready_mni",
    "1023_ucsf_pdgm/ucsf_pdgm_fm_ready_mni",
    "1024_remind/remind_fm_ready_n4_mni",
]


def strip_nii_gz_suffix(path: Path) -> str:
    """Return file name without .nii.gz."""
    name = Path(path).name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    return Path(name).stem


def strip_relative_nii_gz_suffix(path: Path) -> str:
    """Return relative path without .nii.gz, preserving nested directories."""
    path = Path(path)
    text = path.as_posix()
    if text.lower().endswith(".nii.gz"):
        return text[:-7]
    return path.with_suffix("").as_posix()


def extract_modality(filename: str) -> str:
    """Extract modality from a standardized MRI filename."""
    return strip_nii_gz_suffix(Path(filename)).split("_")[-1].lower()


def discover_nifti_files(input_dir: Path) -> List[Path]:
    """Find all .nii.gz files recursively in a deterministic order."""
    return sorted(
        [
            path
            for path in Path(input_dir).rglob("*")
            if path.is_file() and path.name.lower().endswith(".nii.gz")
        ],
        key=lambda path: path.relative_to(input_dir).as_posix(),
    )


def build_volume_manifest(base_dir: Path, source_list: List[str]) -> List[Dict[str, Any]]:
    """Collect final MNI MRI volumes from the explicit source directory list."""
    volumes: List[Dict[str, Any]] = []
    missing_sources: List[str] = []

    for source_item in source_list:
        mni_dir = (base_dir / source_item).resolve()
        if not mni_dir.is_dir():
            missing_sources.append(source_item)
            continue

        dataset_name = mni_dir.parent.name
        for volume_path in discover_nifti_files(mni_dir):
            modality = extract_modality(volume_path.name)
            if modality not in VALID_MODALITIES:
                raise ValueError(
                    f"Unsupported modality={modality}: {volume_path}. "
                    f"Expected one of {sorted(VALID_MODALITIES)}"
                )
            relative_path = volume_path.relative_to(mni_dir).as_posix()
            volumes.append(
                {
                    "dataset": dataset_name,
                    "mni_dir": str(mni_dir),
                    "source_item": source_item,
                    "path": str(volume_path),
                    "relative_path": relative_path,
                    "volume_stem": strip_relative_nii_gz_suffix(Path(relative_path)),
                    "modality": modality,
                }
            )

    if missing_sources:
        raise FileNotFoundError(
            "Missing source directories:\n" + "\n".join(f"- {item}" for item in missing_sources)
        )
    return volumes


def load_volume_slices(volume: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield float16 2D slice samples from one 3D NIfTI volume."""
    image = nib.load(volume["path"])
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={data.shape}: {volume['path']}")
    if data.shape[0] != 224 or data.shape[1] != 224:
        raise ValueError(f"Expected 224x224xZ volume, got shape={data.shape}: {volume['path']}")

    data = data.astype(np.float16, copy=False)
    for z_index in range(data.shape[2]):
        slice_2d = np.ascontiguousarray(data[:, :, z_index], dtype=np.float16)
        sample_key = (
            f"{volume['dataset']}__"
            f"{volume['volume_stem'].replace('/', '__')}__"
            f"z{z_index:04d}"
        )
        yield {
            "__key__": sample_key,
            "npy": slice_2d,
            "cls": volume["modality"],
            "json": {
                "dataset": volume["dataset"],
                "relative_path": volume["relative_path"],
                "volume_stem": volume["volume_stem"],
                "modality": volume["modality"],
                "z_index": int(z_index),
                "volume_shape": [int(value) for value in data.shape],
                "dtype": "float16",
            },
        }


def flush_slice_buffer(
    sink: wds.ShardWriter,
    slice_buffer: List[Dict[str, Any]],
    random_seed: int,
    flush_index: int,
) -> int:
    """Shuffle and write buffered slices to WebDataset shards."""
    if not slice_buffer:
        return 0

    rng = random.Random(random_seed + flush_index)
    rng.shuffle(slice_buffer)
    written = 0
    for sample in slice_buffer:
        sink.write(sample)
        written += 1
    slice_buffer.clear()
    return written


def write_summary(output_dir: Path, summary: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "webdataset_build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def create_webdataset(
    base_dir: Path,
    source_list: List[str],
    output_dir: Path,
    shard_size: int,
    slice_buffer_size: int,
    random_seed: int,
    debug: bool,
    debug_max_shards: int,
) -> Dict[str, Any]:
    """Build float16 2D-slice WebDataset shards with bounded slice buffering."""
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if slice_buffer_size <= 0:
        raise ValueError("slice_buffer_size must be positive")
    if debug_max_shards <= 0:
        raise ValueError("debug_max_shards must be positive")

    started_at = time.perf_counter()
    base_dir = Path(base_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    volumes = build_volume_manifest(base_dir, source_list)
    rng = random.Random(random_seed)
    rng.shuffle(volumes)

    shard_pattern = str(output_dir / "FM_shard_%06d.tar")
    sink = wds.ShardWriter(shard_pattern, maxcount=shard_size)

    slice_buffer: List[Dict[str, Any]] = []
    flush_index = 0
    total_slices_written = 0
    total_volumes_processed = 0
    failed_volumes: List[Dict[str, str]] = []
    target_debug_slices = shard_size * debug_max_shards if debug else None

    print(f"Base directory: {base_dir}")
    print(f"Source directories: {len(source_list)}")
    print(f"Output directory: {output_dir}")
    print(f"Volumes discovered: {len(volumes)}")
    print(f"Shard size: {shard_size}")
    print(f"Slice buffer size: {slice_buffer_size}")
    print(f"DEBUG: {debug}")
    if debug:
        print(f"DEBUG max shards: {debug_max_shards}")
        print(f"DEBUG target slices: {target_debug_slices}")

    try:
        for volume in tqdm(volumes, desc="Packing volumes", unit="volume", dynamic_ncols=True):
            if target_debug_slices is not None and total_slices_written >= target_debug_slices:
                break

            try:
                for sample in load_volume_slices(volume):
                    if target_debug_slices is not None:
                        pending_total = total_slices_written + len(slice_buffer)
                        if pending_total >= target_debug_slices:
                            break

                    slice_buffer.append(sample)
                    if len(slice_buffer) >= slice_buffer_size:
                        total_slices_written += flush_slice_buffer(
                            sink,
                            slice_buffer,
                            random_seed=random_seed,
                            flush_index=flush_index,
                        )
                        flush_index += 1

                total_volumes_processed += 1
            except Exception as exc:
                failed_volumes.append(
                    {
                        "path": volume["path"],
                        "error": str(exc),
                    }
                )

        if slice_buffer:
            if target_debug_slices is not None:
                remaining = max(0, target_debug_slices - total_slices_written)
                if len(slice_buffer) > remaining:
                    del slice_buffer[remaining:]
            total_slices_written += flush_slice_buffer(
                sink,
                slice_buffer,
                random_seed=random_seed,
                flush_index=flush_index,
            )
    finally:
        sink.close()

    summary = {
        "base_dir": str(base_dir),
        "source_list": list(source_list),
        "output_dir": str(output_dir),
        "shard_size": int(shard_size),
        "slice_buffer_size": int(slice_buffer_size),
        "random_seed": int(random_seed),
        "debug": bool(debug),
        "debug_max_shards": int(debug_max_shards),
        "volumes_discovered": int(len(volumes)),
        "volumes_processed": int(total_volumes_processed),
        "slices_written": int(total_slices_written),
        "failed_volume_count": int(len(failed_volumes)),
        "failed_volumes": failed_volumes[:200],
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
    }
    write_summary(output_dir, summary)

    print(f"Finished slices: {total_slices_written}")
    print(f"Processed volumes: {total_volumes_processed}")
    print(f"Failed volumes: {len(failed_volumes)}")
    print(f"Summary: {output_dir / 'webdataset_build_summary.json'}")
    return summary


def exit_code_from_summary(summary: Dict[str, Any]) -> int:
    """Return non-zero when any volume failed."""
    return 1 if int(summary.get("failed_volume_count", 0)) > 0 else 0



