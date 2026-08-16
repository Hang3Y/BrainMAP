"""MNI-space preprocessing for segmentation labels using saved MRI transforms."""

from __future__ import annotations

import importlib.util
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


def ensure_antspyx_available() -> None:
    """Fail before multiprocessing when the antspyx package is unavailable."""
    if importlib.util.find_spec("ants") is None:
        raise RuntimeError(
            "ANTsPy is not installed in the current Python environment. "
            "Install package 'antspyx' and verify it with "
            "'python -c \"import ants; print(ants.__version__)\"'."
        )


def discover_nifti_files(input_dir: Path) -> List[Path]:
    """Find all .nii.gz files recursively in a deterministic order."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = [
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.name.lower().endswith(".nii.gz")
    ]
    return sorted(files, key=lambda path: path.relative_to(input_dir).as_posix())


def strip_nii_gz_suffix(path: Path) -> Path:
    """Remove .nii.gz as one suffix when present."""
    path = Path(path)
    if path.name.lower().endswith(".nii.gz"):
        return path.with_name(path.name[:-7])
    return path.with_suffix("")


def derive_output_paths(seg_dir: Path) -> Tuple[Path, Path, Path]:
    """Return output, metadata, and QC directories for one seg directory."""
    seg_dir = Path(seg_dir)
    if seg_dir.name.endswith("_mni"):
        raise ValueError(f"Input directory already has the '_mni' suffix: {seg_dir}")

    output_dir = seg_dir.with_name(f"{seg_dir.name}_mni")
    metadata_dir = output_dir.with_name(f"{output_dir.name}_metadata")
    qc_dir = output_dir.with_name(f"{output_dir.name}_qc")
    return output_dir, metadata_dir, qc_dir


def output_path_for(input_file: Path, input_dir: Path, output_dir: Path) -> Path:
    """Map an input file to the same relative location under output_dir."""
    return Path(output_dir) / Path(input_file).relative_to(input_dir)


def sidecar_path_for(input_file: Path, input_dir: Path, sidecar_dir: Path, suffix: str) -> Path:
    """Map an input NIfTI file to a sidecar path preserving relative structure."""
    relative = strip_nii_gz_suffix(Path(input_file).relative_to(input_dir))
    return Path(sidecar_dir) / relative.with_suffix(suffix)


def _is_valid_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _limit_threads_for_ants(threads: int) -> None:
    """Limit native-library threads inside each worker process."""
    thread_count = str(threads)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "ANTSTools_NUM_THREADS",
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
        "NSLOTS",
    ):
        os.environ[variable] = thread_count


def _append_jsonl(report_path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("a", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
        file_obj.flush()


def _load_existing_success_rows(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load successful rows for resume support."""
    done: Dict[str, Dict[str, Any]] = {}
    if not jsonl_path.is_file():
        return done

    with jsonl_path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") in {"Success", "SkippedExisting"} and row.get("relative_path"):
                done[row["relative_path"]] = row
    return done


def _seg_stem(seg_file: Path) -> str:
    stem = strip_nii_gz_suffix(seg_file).name
    return stem[:-4] if stem.endswith("_seg") else stem


def resolve_reference_metadata(
    seg_file: Path,
    metadata_dir: Path,
    reference_modalities: List[str],
) -> Tuple[Path, str]:
    """Find the reference MRI metadata JSON for one segmentation file."""
    base_stem = _seg_stem(seg_file)
    metadata_dir = Path(metadata_dir)

    exact_path = metadata_dir / f"{base_stem}.json"
    if exact_path.is_file():
        return exact_path, "exact"

    for modality in reference_modalities:
        candidate = metadata_dir / f"{base_stem}_{modality}.json"
        if candidate.is_file():
            return candidate, modality

    fallback_matches = sorted(metadata_dir.glob(f"{base_stem}_*.json"))
    if fallback_matches:
        return fallback_matches[0], "fallback_first_match"

    raise FileNotFoundError(
        "No reference MRI metadata found for "
        f"{seg_file.name}; searched base={base_stem}, metadata_dir={metadata_dir}"
    )


def _make_reference_grid_from_metadata(metadata: Dict[str, Any]) -> Any:
    import ants

    shape = tuple(int(value) for value in metadata["reference_grid_shape"])
    spacing = tuple(float(value) for value in metadata["reference_grid_spacing"])
    origin = tuple(float(value) for value in metadata["reference_grid_origin"])
    direction = np.asarray(metadata["reference_grid_direction"], dtype=np.float64)
    return ants.make_image(
        shape,
        voxval=0,
        spacing=spacing,
        origin=origin,
        direction=direction,
    )


def _crop_or_pad_with_parameters(array: np.ndarray, crop_parameters: Dict[str, Any]) -> np.ndarray:
    """Apply MRI-recorded XY crop/pad parameters to a label array."""
    target_x, target_y = [int(value) for value in crop_parameters["target_xy"]]
    source_x_start = int(crop_parameters["source_x_start"])
    source_x_end = int(crop_parameters["source_x_end"])
    source_y_start = int(crop_parameters["source_y_start"])
    source_y_end = int(crop_parameters["source_y_end"])
    target_x_start = int(crop_parameters["target_x_start"])
    target_y_start = int(crop_parameters["target_y_start"])

    output = np.zeros((target_x, target_y, array.shape[2]), dtype=array.dtype)
    output[
        target_x_start : target_x_start + (source_x_end - source_x_start),
        target_y_start : target_y_start + (source_y_end - source_y_start),
        :,
    ] = array[source_x_start:source_x_end, source_y_start:source_y_end, :]
    return output


def _to_uint8_labels(array: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Round nearest-neighbor output and validate it fits uint8 labels."""
    finite_mask = np.isfinite(array)
    if not np.all(finite_mask):
        raise ValueError("Transformed segmentation contains NaN or Inf.")

    rounded = np.rint(array).astype(np.int64)
    min_label = int(np.min(rounded))
    max_label = int(np.max(rounded))
    if min_label < 0 or max_label > 255:
        raise ValueError(
            f"Segmentation labels are outside uint8 range: min={min_label}, max={max_label}"
        )

    unique_values = np.unique(rounded)
    metadata = {
        "min_label": min_label,
        "max_label": max_label,
        "unique_labels": [int(value) for value in unique_values.tolist()],
        "nonzero_voxel_count": int(np.count_nonzero(rounded)),
    }
    return rounded.astype(np.uint8), metadata


def _process_single_file(task: Tuple[str, str, str, str, Dict[str, Any]]) -> Dict[str, Any]:
    seg_path_text, output_path_text, metadata_path_text, reference_metadata_path_text, parameters = task
    seg_path = Path(seg_path_text)
    output_path = Path(output_path_text)
    metadata_path = Path(metadata_path_text)
    reference_metadata_path = Path(reference_metadata_path_text)
    started_at = time.perf_counter()

    result: Dict[str, Any] = {
        "relative_path": parameters["relative_path"],
        "input_path": str(seg_path),
        "output_path": str(output_path),
        "metadata_path": str(metadata_path),
        "reference_metadata_path": str(reference_metadata_path),
        "reference_modality": parameters["reference_modality"],
        "status": "Success",
        "error": "",
        "original_shape": None,
        "reference_grid_shape": None,
        "output_shape": None,
        "unique_labels": [],
        "nonzero_voxel_count": 0,
        "processing_time_seconds": 0.0,
    }

    if parameters["resume"] and _is_valid_file(output_path) and _is_valid_file(metadata_path):
        result["status"] = "SkippedExisting"
        result["processing_time_seconds"] = round(time.perf_counter() - started_at, 3)
        return result

    try:
        import ants

        reference_metadata = json.loads(reference_metadata_path.read_text(encoding="utf-8"))
        transform_path = Path(reference_metadata["forward_transform_path"])
        if not transform_path.is_file():
            raise FileNotFoundError(f"Reference transform does not exist: {transform_path}")

        seg_image = ants.image_read(str(seg_path))
        if len(seg_image.shape) != 3:
            raise ValueError(f"Expected a 3D segmentation volume, got shape={seg_image.shape}")

        seg_array = seg_image.numpy()
        if seg_array.size == 0:
            raise ValueError("Empty segmentation array.")
        if not np.any(np.isfinite(seg_array)):
            raise ValueError("No finite voxels found in segmentation.")

        reference_grid = _make_reference_grid_from_metadata(reference_metadata)
        warped = ants.apply_transforms(
            fixed=reference_grid,
            moving=seg_image,
            transformlist=[str(transform_path)],
            interpolator="nearestNeighbor",
        )
        warped_array = warped.numpy()
        cropped = _crop_or_pad_with_parameters(
            warped_array,
            reference_metadata["crop_pad_xy"],
        )
        kept_indices = [
            int(index)
            for index in reference_metadata["slice_selection"]["kept_slice_indices"]
        ]
        if not kept_indices:
            raise ValueError("Reference metadata has no kept slice indices.")
        selected = cropped[:, :, kept_indices]
        label_array, label_metadata = _to_uint8_labels(selected)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        output_image = ants.from_numpy(
            label_array,
            spacing=tuple(float(value) for value in reference_metadata["output_spacing"]),
            origin=tuple(float(value) for value in reference_metadata["output_origin"]),
            direction=np.asarray(reference_metadata["output_direction"], dtype=np.float64),
        )
        ants.image_write(output_image, str(output_path))
        if not _is_valid_file(output_path):
            raise IOError(f"Output file is missing or empty: {output_path}")

        metadata = {
            "relative_path": parameters["relative_path"],
            "input_path": str(seg_path),
            "output_path": str(output_path),
            "reference_metadata_path": str(reference_metadata_path),
            "reference_modality": parameters["reference_modality"],
            "reference_transform_path": str(transform_path),
            "seg_interpolator": "nearestNeighbor",
            "original_shape": [int(value) for value in seg_image.shape],
            "original_spacing": [float(value) for value in seg_image.spacing],
            "reference_grid_shape": reference_metadata["reference_grid_shape"],
            "reference_grid_spacing": reference_metadata["reference_grid_spacing"],
            "crop_pad_xy": reference_metadata["crop_pad_xy"],
            "slice_selection": reference_metadata["slice_selection"],
            "output_shape": [int(value) for value in label_array.shape],
            "output_spacing": reference_metadata["output_spacing"],
            "output_origin": reference_metadata["output_origin"],
            "output_direction": reference_metadata["output_direction"],
            "output_dtype": "uint8",
            "labels": label_metadata,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        result["original_shape"] = metadata["original_shape"]
        result["reference_grid_shape"] = metadata["reference_grid_shape"]
        result["output_shape"] = metadata["output_shape"]
        result["unique_labels"] = label_metadata["unique_labels"]
        result["nonzero_voxel_count"] = label_metadata["nonzero_voxel_count"]
    except Exception as exc:
        result["status"] = "Failed"
        result["error"] = str(exc)
        for partial_path in (output_path, metadata_path):
            if partial_path.exists():
                partial_path.unlink()

    result["processing_time_seconds"] = round(time.perf_counter() - started_at, 3)
    return result


def run_seg_mni_preprocessing(
    seg_dir: Path,
    mri_metadata_dir: Path,
    reference_modalities: List[str],
    num_workers: int,
    ants_threads_per_worker: int,
    resume: bool = True,
) -> Dict[str, int]:
    """Preprocess segmentation labels using saved MRI MNI metadata."""
    ensure_antspyx_available()

    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if ants_threads_per_worker < 1:
        raise ValueError("ants_threads_per_worker must be at least 1")

    seg_dir = Path(seg_dir).resolve()
    mri_metadata_dir = Path(mri_metadata_dir).resolve()
    if not mri_metadata_dir.is_dir():
        raise FileNotFoundError(f"MRI metadata directory does not exist: {mri_metadata_dir}")

    output_dir, metadata_dir, qc_dir = derive_output_paths(seg_dir)
    report_path = qc_dir / "seg_mni_preprocessing_report.jsonl"
    summary_path = qc_dir / "seg_mni_preprocessing_summary.json"
    seg_files = discover_nifti_files(seg_dir)
    if not seg_files:
        raise RuntimeError(f"No .nii.gz files found under: {seg_dir}")

    existing_rows = _load_existing_success_rows(report_path) if resume else {}
    tasks: List[Tuple[str, str, str, str, Dict[str, Any]]] = []
    skipped_by_report = 0
    preflight_failures: List[Dict[str, Any]] = []

    for seg_file in seg_files:
        relative_path = seg_file.relative_to(seg_dir).as_posix()
        output_path = output_path_for(seg_file, seg_dir, output_dir)
        metadata_path = sidecar_path_for(seg_file, seg_dir, metadata_dir, ".json")

        existing = existing_rows.get(relative_path)
        if resume and existing and _is_valid_file(output_path) and _is_valid_file(metadata_path):
            skipped_by_report += 1
            continue

        try:
            reference_metadata_path, reference_modality = resolve_reference_metadata(
                seg_file,
                mri_metadata_dir,
                reference_modalities,
            )
        except Exception as exc:
            preflight_failures.append(
                {
                    "relative_path": relative_path,
                    "input_path": str(seg_file),
                    "output_path": str(output_path),
                    "metadata_path": str(metadata_path),
                    "reference_metadata_path": "",
                    "reference_modality": "",
                    "status": "Failed",
                    "error": str(exc),
                    "original_shape": None,
                    "reference_grid_shape": None,
                    "output_shape": None,
                    "unique_labels": [],
                    "nonzero_voxel_count": 0,
                    "processing_time_seconds": 0.0,
                }
            )
            continue

        parameters = {
            "relative_path": relative_path,
            "reference_modality": reference_modality,
            "resume": resume,
        }
        tasks.append(
            (
                str(seg_file),
                str(output_path),
                str(metadata_path),
                str(reference_metadata_path),
                parameters,
            )
        )

    summary = {
        "success": 0,
        "skipped": skipped_by_report,
        "failed": len(preflight_failures),
    }

    print(f"Seg directory: {seg_dir}")
    print(f"MRI metadata directory: {mri_metadata_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Metadata directory: {metadata_dir}")
    print(f"QC report: {report_path}")
    print(f"Seg files: {len(seg_files)}")
    print(f"Already completed from report: {skipped_by_report}")
    print(f"Preflight failures: {len(preflight_failures)}")
    print(f"Pending processing: {len(tasks)}")
    print(
        f"Workers: {num_workers}; "
        f"ANTs threads per worker: {ants_threads_per_worker}"
    )

    if preflight_failures:
        _append_jsonl(report_path, preflight_failures)
        for failure in preflight_failures:
            tqdm.write(f"Failed: {failure['relative_path']}: {failure['error']}")

    if tasks:
        if num_workers == 1:
            _limit_threads_for_ants(ants_threads_per_worker)
            iterator = map(_process_single_file, tasks)
            progress = tqdm(
                iterator,
                total=len(tasks),
                desc=f"SEG {seg_dir.parent.name}",
                unit="label",
                dynamic_ncols=True,
            )
            for result in progress:
                _update_summary_and_report(summary, result, report_path, seg_dir, progress)
        else:
            with Pool(
                processes=num_workers,
                initializer=_limit_threads_for_ants,
                initargs=(ants_threads_per_worker,),
            ) as pool:
                iterator = pool.imap_unordered(_process_single_file, tasks)
                progress = tqdm(
                    iterator,
                    total=len(tasks),
                    desc=f"SEG {seg_dir.parent.name}",
                    unit="label",
                    dynamic_ncols=True,
                )
                for result in progress:
                    _update_summary_and_report(summary, result, report_path, seg_dir, progress)

    qc_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        "Finished: "
        f"success={summary['success']} "
        f"skipped={summary['skipped']} "
        f"failed={summary['failed']}"
    )
    print(f"Summary report: {summary_path}")
    return summary


def _update_summary_and_report(
    summary: Dict[str, int],
    result: Dict[str, Any],
    report_path: Path,
    seg_dir: Path,
    progress: tqdm,
) -> None:
    status = result["status"]
    if status == "Success":
        summary["success"] += 1
    elif status == "SkippedExisting":
        summary["skipped"] += 1
    else:
        summary["failed"] += 1

    _append_jsonl(report_path, [result])
    progress.set_postfix(
        success=summary["success"],
        skipped=summary["skipped"],
        failed=summary["failed"],
        refresh=False,
    )
    if result["error"]:
        relative_input = Path(result["input_path"]).relative_to(seg_dir)
        tqdm.write(f"Failed: {relative_input.as_posix()}: {result['error']}")


def exit_code_from_summary(summary: Dict[str, int]) -> int:
    """Return a non-zero process exit code when any label failed."""
    return 1 if summary.get("failed", 0) > 0 else 0


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    dataset_configs = {
        "1002_brats2023": {
            "seg_dir": "1002_brats2023/brats_seg_ready",
            "mri_metadata_dir": "1002_brats2023/brats_fm_ready_mni_metadata",
            "reference_modalities": ["t1c", "t1n", "t2f", "t2w"],
        },
        "1005_atlasr2": {
            "seg_dir": "1005_atlasr2/atlas_seg_ready",
            "mri_metadata_dir": "1005_atlasr2/atlas_fm_ready_skull_stripping_n4_mni_metadata",
            "reference_modalities": ["t1n", "t1c", "t2f", "t2w"],
        },
        "1010_msd": {
            "seg_dir": "1010_msd/msd_seg_ready",
            "mri_metadata_dir": "1010_msd/msd_fm_ready_mni_metadata",
            "reference_modalities": ["t1c", "t1n", "t2f", "t2w"],
        },
        "1017_mets": {
            "seg_dir": "1017_mets/mets_seg_ready",
            "mri_metadata_dir": "1017_mets/mets_fm_ready_mni_metadata",
            "reference_modalities": ["t1c", "t1n", "t2f", "t2w"],
        },
        "1019_lumiere": {
            "seg_dir": "1019_lumiere/lumiere_seg_ready",
            "mri_metadata_dir": "1019_lumiere/lumiere_fm_ready_mni_metadata",
            "reference_modalities": ["t1c", "t1n", "t2f", "t2w"],
        },
        "1020_wmh": {
            "seg_dir": "1020_wmh/wmh_seg_ready",
            "mri_metadata_dir": "1020_wmh/wmh_fm_ready_skull_stripping_n4_mni_metadata",
            "reference_modalities": ["t1n", "flair", "t2f", "t1c", "t2w"],
        },
        "1021_ucsd_ptgbm": {
            "seg_dir": "1021_ucsd_ptgbm/ucsd_ptgbm_seg_ready",
            "mri_metadata_dir": "1021_ucsd_ptgbm/ucsd_ptgbm_fm_ready_mni_metadata",
            "reference_modalities": ["t1c", "t1n", "t2f", "t2w"],
        },
        "1022_upenn_gbm": {
            "seg_dir": "1022_upenn_gbm/upenn_gbm_seg_ready",
            "mri_metadata_dir": "1022_upenn_gbm/upenn_gbm_fm_ready_mni_metadata",
            "reference_modalities": ["t1c", "t1n", "t2f", "t2w"],
        },
        "1023_ucsf_pdgm": {
            "seg_dir": "1023_ucsf_pdgm/ucsf_pdgm_seg_ready",
            "mri_metadata_dir": "1023_ucsf_pdgm/ucsf_pdgm_fm_ready_mni_metadata",
            "reference_modalities": ["t1c", "t1n", "t2f", "t2w"],
        },
    }

    active_dataset = "1002_brats2023"         # TODO 1251 - DONE
    # active_dataset = "1005_atlasr2"           # TODO 655 - DONE
    # active_dataset = "1010_msd"               # TODO 484 - DONE
    # active_dataset = "1017_mets"              # TODO 105 - DONE
    # active_dataset = "1019_lumiere"           # TODO 400 - DONE
    # active_dataset = "1020_wmh"               # TODO 170 - DONE
    # active_dataset = "1021_ucsd_ptgbm"        # TODO 243 - DONE
    # active_dataset = "1022_upenn_gbm"         # TODO 611 - DONE
    # active_dataset = "1023_ucsf_pdgm"         # TODO 501 - DONE

    selected_config = dataset_configs[active_dataset]
    processing_summary = run_seg_mni_preprocessing(
        seg_dir=script_dir / selected_config["seg_dir"],
        mri_metadata_dir=script_dir / selected_config["mri_metadata_dir"],
        reference_modalities=selected_config["reference_modalities"],
        num_workers=4,
        ants_threads_per_worker=4,
        resume=True,
    )
    raise SystemExit(exit_code_from_summary(processing_summary))

"""
# 保守，适合白天或和别人共享服务器
num_workers=2
ants_threads_per_worker=4   # 约 8 threads

# 平衡，当前推荐
num_workers=4
ants_threads_per_worker=6   # 约 24 threads

# 更快，适合服务器空闲时
num_workers=6
ants_threads_per_worker=12   # 约 60 threads

"""