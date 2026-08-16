
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from tqdm import tqdm


_FIXED_TEMPLATE = None


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


def derive_output_paths(input_dir: Path) -> Tuple[Path, Path, Path, Path]:
    """Return image, transform, metadata, and QC output directories."""
    input_dir = Path(input_dir)
    if input_dir.name.endswith("_mni"):
        raise ValueError(f"Input directory already has the '_mni' suffix: {input_dir}")

    output_dir = input_dir.with_name(f"{input_dir.name}_mni")
    transform_dir = output_dir.with_name(f"{output_dir.name}_transforms")
    metadata_dir = output_dir.with_name(f"{output_dir.name}_metadata")
    qc_dir = output_dir.with_name(f"{output_dir.name}_qc")
    return output_dir, transform_dir, metadata_dir, qc_dir


def output_path_for(input_file: Path, input_dir: Path, output_dir: Path) -> Path:
    """Map an input file to the same relative location under output_dir."""
    return Path(output_dir) / Path(input_file).relative_to(input_dir)


def sidecar_path_for(input_file: Path, input_dir: Path, sidecar_dir: Path, suffix: str) -> Path:
    """Map an input NIfTI file to a sidecar path preserving relative structure."""
    relative = strip_nii_gz_suffix(Path(input_file).relative_to(input_dir))
    return Path(sidecar_dir) / relative.with_suffix(suffix)


def _is_valid_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _limit_threads_and_load_template(template_path: str, threads: int) -> None:
    """Limit native-library threads and load the fixed template once per worker."""
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

    import ants

    global _FIXED_TEMPLATE
    _FIXED_TEMPLATE = ants.image_read(template_path)


def _matrix_to_list(matrix: Any) -> Any:
    if matrix is None:
        return None
    return np.asarray(matrix, dtype=np.float64).round(8).tolist()


def _crop_or_pad_xy(array: np.ndarray, target_xy: Tuple[int, int]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Center crop or pad the first two axes to target_xy."""
    x_size, y_size, z_size = array.shape
    target_x, target_y = target_xy

    source_x_start = max(0, (x_size - target_x) // 2)
    source_y_start = max(0, (y_size - target_y) // 2)
    source_x_end = source_x_start + min(x_size, target_x)
    source_y_end = source_y_start + min(y_size, target_y)

    target_x_start = max(0, (target_x - x_size) // 2)
    target_y_start = max(0, (target_y - y_size) // 2)

    cropped = np.zeros((target_x, target_y, z_size), dtype=array.dtype)
    cropped[
        target_x_start : target_x_start + (source_x_end - source_x_start),
        target_y_start : target_y_start + (source_y_end - source_y_start),
        :,
    ] = array[source_x_start:source_x_end, source_y_start:source_y_end, :]

    parameters = {
        "source_shape_before_crop": [int(x_size), int(y_size), int(z_size)],
        "target_xy": [int(target_x), int(target_y)],
        "source_x_start": int(source_x_start),
        "source_x_end": int(source_x_end),
        "source_y_start": int(source_y_start),
        "source_y_end": int(source_y_end),
        "target_x_start": int(target_x_start),
        "target_y_start": int(target_y_start),
    }
    return cropped, parameters


def _select_valid_slices(
    array: np.ndarray,
    nonzero_threshold: float,
    trim_slices_each_end: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Select valid slices by nonzero ratio and trim both ends."""
    valid_indices: List[int] = []
    nonzero_ratios: List[float] = []

    for z_index in range(array.shape[2]):
        slice_2d = array[:, :, z_index]
        nonzero_ratio = float(np.count_nonzero(slice_2d) / slice_2d.size)
        nonzero_ratios.append(nonzero_ratio)
        if nonzero_ratio >= nonzero_threshold:
            valid_indices.append(z_index)

    if not valid_indices:
        raise ValueError("No slices passed the nonzero threshold.")
    if len(valid_indices) <= 2 * trim_slices_each_end:
        raise ValueError(
            "Not enough valid slices after trimming: "
            f"valid={len(valid_indices)}, trim_each_end={trim_slices_each_end}"
        )

    if trim_slices_each_end > 0:
        kept_indices = valid_indices[trim_slices_each_end:-trim_slices_each_end]
    else:
        kept_indices = valid_indices

    selected = array[:, :, kept_indices]
    contiguous = kept_indices == list(range(kept_indices[0], kept_indices[-1] + 1))
    parameters = {
        "nonzero_threshold": float(nonzero_threshold),
        "trim_slices_each_end": int(trim_slices_each_end),
        "valid_slice_count_before_trim": int(len(valid_indices)),
        "kept_slice_count": int(len(kept_indices)),
        "first_kept_slice_index": int(kept_indices[0]),
        "last_kept_slice_index": int(kept_indices[-1]),
        "kept_slice_indices": [int(index) for index in kept_indices],
        "kept_slice_indices_are_contiguous": bool(contiguous),
        "min_nonzero_ratio": float(np.min(nonzero_ratios)),
        "max_nonzero_ratio": float(np.max(nonzero_ratios)),
    }
    return selected, parameters


def _normalize_foreground(
    array: np.ndarray,
    lower_percentile: float,
    upper_percentile: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Percentile clip and normalize nonzero foreground while keeping background zero."""
    finite_mask = np.isfinite(array)
    foreground_mask = finite_mask & (array > 0)
    if not np.any(foreground_mask):
        raise ValueError("No nonzero finite foreground voxels found.")

    foreground = array[foreground_mask].astype(np.float32)
    lower_value, upper_value = np.percentile(
        foreground, [lower_percentile, upper_percentile]
    )
    clipped = np.clip(foreground, lower_value, upper_value)
    mean_value = float(np.mean(clipped))
    std_value = float(np.std(clipped) + 1.0e-8)
    normalized = (clipped - mean_value) / std_value
    normalized_min = float(np.min(normalized))
    normalized_max = float(np.max(normalized))
    normalized = (normalized - normalized_min) / (
        normalized_max - normalized_min + 1.0e-8
    )

    output = np.zeros(array.shape, dtype=np.float32)
    output[foreground_mask] = normalized.astype(np.float32)

    parameters = {
        "foreground_voxel_count": int(foreground.size),
        "lower_percentile": float(lower_percentile),
        "upper_percentile": float(upper_percentile),
        "lower_value": float(lower_value),
        "upper_value": float(upper_value),
        "foreground_mean_after_clip": mean_value,
        "foreground_std_after_clip": std_value,
        "normalized_min_before_rescale": normalized_min,
        "normalized_max_before_rescale": normalized_max,
    }
    return output, parameters


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


def _copy_affine_transform(transform_list: List[str], target_affine_path: Path) -> str:
    """Copy the affine transform produced by ANTs registration to a stable path."""
    affine_candidates = [
        Path(transform_path)
        for transform_path in transform_list
        if str(transform_path).endswith(".mat")
    ]
    if not affine_candidates:
        raise RuntimeError(f"No affine .mat transform found in: {transform_list}")

    source_affine = affine_candidates[0]
    target_affine_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_affine, target_affine_path)
    if source_affine.resolve() != target_affine_path.resolve():
        source_affine.unlink(missing_ok=True)
    return str(target_affine_path)


def _process_single_file(task: Tuple[str, str, str, str, str, Dict[str, Any]]) -> Dict[str, Any]:
    (
        input_path_text,
        output_path_text,
        transform_path_text,
        metadata_path_text,
        template_path_text,
        parameters,
    ) = task
    input_path = Path(input_path_text)
    output_path = Path(output_path_text)
    transform_path = Path(transform_path_text)
    metadata_path = Path(metadata_path_text)
    started_at = time.perf_counter()

    result: Dict[str, Any] = {
        "relative_path": parameters["relative_path"],
        "input_path": str(input_path),
        "output_path": str(output_path),
        "transform_path": str(transform_path),
        "metadata_path": str(metadata_path),
        "status": "Success",
        "error": "",
        "original_shape": None,
        "original_spacing": None,
        "reference_grid_shape": None,
        "reference_grid_spacing": None,
        "output_shape": None,
        "output_spacing": None,
        "valid_slice_count_before_trim": 0,
        "kept_slice_count": 0,
        "foreground_voxel_count": 0,
        "processing_time_seconds": 0.0,
    }

    if (
        parameters["resume"]
        and _is_valid_file(output_path)
        and _is_valid_file(transform_path)
        and _is_valid_file(metadata_path)
    ):
        result["status"] = "SkippedExisting"
        result["processing_time_seconds"] = round(time.perf_counter() - started_at, 3)
        return result

    try:
        import ants

        global _FIXED_TEMPLATE
        fixed = _FIXED_TEMPLATE
        if fixed is None:
            fixed = ants.image_read(template_path_text)

        moving = ants.image_read(str(input_path))
        if len(moving.shape) != 3:
            raise ValueError(f"Expected a 3D MRI volume, got shape={moving.shape}")
        if any(float(value) <= 0 for value in moving.spacing):
            raise ValueError(f"Invalid spacing: {moving.spacing}")

        moving_array = moving.numpy().astype(np.float32)
        if moving_array.size == 0:
            raise ValueError("Empty image array.")
        if not np.any(np.isfinite(moving_array)):
            raise ValueError("No finite voxels found.")
        if np.count_nonzero(moving_array) == 0:
            raise ValueError("All-zero image.")
        if np.count_nonzero(~np.isfinite(moving_array)) > 0:
            raise ValueError("Image contains NaN or Inf.")

        result["original_shape"] = [int(value) for value in moving.shape]
        result["original_spacing"] = [float(value) for value in moving.spacing]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        transform_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        transform_prefix = str(transform_path.with_suffix("")) + "_"
        registration = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform=parameters["registration_type"],
            outprefix=transform_prefix,
        )
        stable_affine_path = _copy_affine_transform(
            registration["fwdtransforms"], transform_path
        )

        new_spacing = (
            float(fixed.spacing[0]),
            float(fixed.spacing[1]),
            float(moving.spacing[2]),
        )
        new_z_size = int(round((fixed.shape[2] * fixed.spacing[2]) / moving.spacing[2]))
        new_size = (int(fixed.shape[0]), int(fixed.shape[1]), int(new_z_size))
        if new_z_size <= 0:
            raise ValueError(f"Invalid computed Z size: {new_z_size}")

        reference_grid = ants.make_image(
            new_size,
            voxval=0,
            spacing=new_spacing,
            origin=fixed.origin,
            direction=fixed.direction,
        )
        warped = ants.apply_transforms(
            fixed=reference_grid,
            moving=moving,
            transformlist=[stable_affine_path],
            interpolator="linear",
        )

        warped_array = warped.numpy().astype(np.float32)
        cropped_array, crop_parameters = _crop_or_pad_xy(
            warped_array,
            tuple(parameters["target_xy"]),
        )
        selected_array, slice_parameters = _select_valid_slices(
            cropped_array,
            nonzero_threshold=float(parameters["nonzero_threshold"]),
            trim_slices_each_end=int(parameters["trim_slices_each_end"]),
        )
        normalized_array, normalization_parameters = _normalize_foreground(
            selected_array,
            lower_percentile=float(parameters["lower_percentile"]),
            upper_percentile=float(parameters["upper_percentile"]),
        )

        index_shift = np.array(
            [
                crop_parameters["source_x_start"] - crop_parameters["target_x_start"],
                crop_parameters["source_y_start"] - crop_parameters["target_y_start"],
                slice_parameters["first_kept_slice_index"],
            ],
            dtype=np.float64,
        )
        spacing_array = np.array(warped.spacing, dtype=np.float64)
        physical_shift = np.asarray(warped.direction, dtype=np.float64).dot(
            spacing_array * index_shift
        )
        output_origin = tuple(np.asarray(warped.origin, dtype=np.float64) + physical_shift)
        output_spacing = tuple(float(value) for value in warped.spacing)
        output_direction = warped.direction

        processed_image = ants.from_numpy(
            normalized_array.astype(np.float16),
            spacing=output_spacing,
            origin=output_origin,
            direction=output_direction,
        )
        ants.image_write(processed_image, str(output_path))
        if not _is_valid_file(output_path):
            raise IOError(f"Output file is missing or empty: {output_path}")

        metadata = {
            "relative_path": parameters["relative_path"],
            "input_path": str(input_path),
            "output_path": str(output_path),
            "template_path": str(template_path_text),
            "registration_type": parameters["registration_type"],
            "forward_transform_path": stable_affine_path,
            "original_shape": result["original_shape"],
            "original_spacing": result["original_spacing"],
            "original_origin": [float(value) for value in moving.origin],
            "original_direction": _matrix_to_list(moving.direction),
            "reference_grid_shape": [int(value) for value in new_size],
            "reference_grid_spacing": [float(value) for value in new_spacing],
            "reference_grid_origin": [float(value) for value in reference_grid.origin],
            "reference_grid_direction": _matrix_to_list(reference_grid.direction),
            "warped_shape": [int(value) for value in warped.shape],
            "warped_spacing": [float(value) for value in warped.spacing],
            "warped_origin": [float(value) for value in warped.origin],
            "warped_direction": _matrix_to_list(warped.direction),
            "crop_pad_xy": crop_parameters,
            "slice_selection": slice_parameters,
            "normalization": normalization_parameters,
            "output_shape": [int(value) for value in normalized_array.shape],
            "output_spacing": [float(value) for value in output_spacing],
            "output_origin": [float(value) for value in output_origin],
            "output_direction": _matrix_to_list(output_direction),
            "output_dtype": "float16",
            "image_interpolator": "linear",
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        result["reference_grid_shape"] = metadata["reference_grid_shape"]
        result["reference_grid_spacing"] = metadata["reference_grid_spacing"]
        result["output_shape"] = metadata["output_shape"]
        result["output_spacing"] = metadata["output_spacing"]
        result["valid_slice_count_before_trim"] = slice_parameters[
            "valid_slice_count_before_trim"
        ]
        result["kept_slice_count"] = slice_parameters["kept_slice_count"]
        result["foreground_voxel_count"] = normalization_parameters[
            "foreground_voxel_count"
        ]
    except Exception as exc:
        result["status"] = "Failed"
        result["error"] = str(exc)
        for partial_path in (output_path, transform_path, metadata_path):
            if partial_path.exists():
                partial_path.unlink()

    result["processing_time_seconds"] = round(time.perf_counter() - started_at, 3)
    return result


def run_mni_preprocessing(
    input_dir: Path,
    template_path: Path,
    num_workers: int,
    ants_threads_per_worker: int,
    target_xy: Tuple[int, int],
    nonzero_threshold: float,
    trim_slices_each_end: int,
    lower_percentile: float,
    upper_percentile: float,
    registration_type: str = "Affine",
    resume: bool = True,
) -> Dict[str, int]:
    """Run image-only MNI preprocessing with transform and sidecar output."""
    ensure_antspyx_available()

    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if ants_threads_per_worker < 1:
        raise ValueError("ants_threads_per_worker must be at least 1")
    if not 0.0 <= nonzero_threshold <= 1.0:
        raise ValueError("nonzero_threshold must be in [0, 1]")
    if trim_slices_each_end < 0:
        raise ValueError("trim_slices_each_end must be non-negative")
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100")

    input_dir = Path(input_dir).resolve()
    template_path = Path(template_path).resolve()
    if not template_path.is_file():
        raise FileNotFoundError(f"Template image does not exist: {template_path}")

    output_dir, transform_dir, metadata_dir, qc_dir = derive_output_paths(input_dir)
    report_path = qc_dir / "mni_preprocessing_report.jsonl"
    summary_path = qc_dir / "mni_preprocessing_summary.json"
    input_files = discover_nifti_files(input_dir)
    if not input_files:
        raise RuntimeError(f"No .nii.gz files found under: {input_dir}")

    existing_rows = _load_existing_success_rows(report_path) if resume else {}
    tasks: List[Tuple[str, str, str, str, str, Dict[str, Any]]] = []
    skipped_by_report = 0

    for input_file in input_files:
        relative_path = input_file.relative_to(input_dir).as_posix()
        output_path = output_path_for(input_file, input_dir, output_dir)
        transform_path = sidecar_path_for(
            input_file, input_dir, transform_dir, ".mat"
        )
        metadata_path = sidecar_path_for(
            input_file, input_dir, metadata_dir, ".json"
        )
        existing = existing_rows.get(relative_path)
        if (
            resume
            and existing
            and _is_valid_file(output_path)
            and _is_valid_file(transform_path)
            and _is_valid_file(metadata_path)
        ):
            skipped_by_report += 1
            continue

        parameters = {
            "relative_path": relative_path,
            "resume": resume,
            "target_xy": [int(target_xy[0]), int(target_xy[1])],
            "nonzero_threshold": float(nonzero_threshold),
            "trim_slices_each_end": int(trim_slices_each_end),
            "lower_percentile": float(lower_percentile),
            "upper_percentile": float(upper_percentile),
            "registration_type": registration_type,
        }
        tasks.append(
            (
                str(input_file),
                str(output_path),
                str(transform_path),
                str(metadata_path),
                str(template_path),
                parameters,
            )
        )

    summary = {"success": 0, "skipped": skipped_by_report, "failed": 0}

    print(f"Input directory: {input_dir}")
    print(f"Template image: {template_path}")
    print(f"Output directory: {output_dir}")
    print(f"Transform directory: {transform_dir}")
    print(f"Metadata directory: {metadata_dir}")
    print(f"QC report: {report_path}")
    print(f"NIfTI files: {len(input_files)}")
    print(f"Already completed from report: {skipped_by_report}")
    print(f"Pending processing: {len(tasks)}")
    print(
        f"Workers: {num_workers}; "
        f"ANTs threads per worker: {ants_threads_per_worker}"
    )

    if tasks:
        if num_workers == 1:
            _limit_threads_and_load_template(str(template_path), ants_threads_per_worker)
            iterator = map(_process_single_file, tasks)
            progress = tqdm(
                iterator,
                total=len(tasks),
                desc=f"MNI {input_dir.parent.name}",
                unit="volume",
                dynamic_ncols=True,
            )
            for result in progress:
                _update_summary_and_report(summary, result, report_path, input_dir, progress)
        else:
            with Pool(
                processes=num_workers,
                initializer=_limit_threads_and_load_template,
                initargs=(str(template_path), ants_threads_per_worker),
            ) as pool:
                iterator = pool.imap_unordered(_process_single_file, tasks)
                progress = tqdm(
                    iterator,
                    total=len(tasks),
                    desc=f"MNI {input_dir.parent.name}",
                    unit="volume",
                    dynamic_ncols=True,
                )
                for result in progress:
                    _update_summary_and_report(
                        summary, result, report_path, input_dir, progress
                    )

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
    input_dir: Path,
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
        relative_input = Path(result["input_path"]).relative_to(input_dir)
        tqdm.write(f"Failed: {relative_input.as_posix()}: {result['error']}")


def exit_code_from_summary(summary: Dict[str, int]) -> int:
    """Return a non-zero process exit code when any volume failed."""
    return 1 if summary.get("failed", 0) > 0 else 0

