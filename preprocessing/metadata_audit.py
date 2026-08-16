"""Audit NIfTI metadata and basic voxel statistics before MNI preprocessing."""

from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from tqdm import tqdm


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


def derive_report_dir(input_dir: Path) -> Path:
    """Return the sibling report directory for one audited input directory."""
    input_dir = Path(input_dir)
    return input_dir.with_name(f"{input_dir.name}_metadata_audit")


def _json_ready(value: Any) -> Any:
    """Convert numpy scalars and arrays into JSON-serializable Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _matrix_to_list(matrix: Any) -> Any:
    if matrix is None:
        return None
    return np.asarray(matrix, dtype=np.float64).round(8).tolist()


def _safe_float(value: Any) -> Any:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _status_from_issues(issues: List[str]) -> str:
    severe = {
        "load_failed",
        "empty_image",
        "no_finite_voxels",
        "shape_not_3d",
        "zero_or_negative_spacing",
    }
    return "Failed" if any(issue in severe for issue in issues) else "Success"


def audit_single_file(task: Tuple[str, str]) -> Dict[str, Any]:
    """Read one NIfTI image and return metadata plus lightweight QC flags."""
    input_path_text, input_dir_text = task
    input_path = Path(input_path_text)
    input_dir = Path(input_dir_text)
    started_at = time.perf_counter()

    result: Dict[str, Any] = {
        "relative_path": input_path.relative_to(input_dir).as_posix(),
        "input_path": str(input_path),
        "status": "Success",
        "issues": [],
        "error": "",
        "file_size_bytes": 0,
        "file_modified_time": None,
        "shape": None,
        "ndim": None,
        "dtype": None,
        "spacing": None,
        "orientation": None,
        "qform_code": None,
        "sform_code": None,
        "qform_sform_max_abs_diff": None,
        "affine": None,
        "qform": None,
        "sform": None,
        "voxel_count": 0,
        "finite_voxel_count": 0,
        "nan_voxel_count": 0,
        "inf_voxel_count": 0,
        "nonzero_voxel_count": 0,
        "zero_ratio": None,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "p01": None,
        "p99": None,
        "foreground_mean": None,
        "foreground_std": None,
        "processing_time_seconds": 0.0,
    }

    try:
        import nibabel as nib
        from nibabel.orientations import aff2axcodes

        file_stat = input_path.stat()
        result["file_size_bytes"] = file_stat.st_size
        result["file_modified_time"] = file_stat.st_mtime
        image = nib.load(str(input_path))
        header = image.header
        shape = tuple(int(value) for value in image.shape)
        ndim = len(shape)
        spacing = tuple(float(value) for value in header.get_zooms()[:ndim])
        qform, qform_code = image.get_qform(coded=True)
        sform, sform_code = image.get_sform(coded=True)

        issues: List[str] = []
        if ndim != 3:
            issues.append("shape_not_3d")
        if any(value <= 0 for value in spacing):
            issues.append("zero_or_negative_spacing")
        if qform_code == 0:
            issues.append("missing_qform")
        if sform_code == 0:
            issues.append("missing_sform")
        if qform_code > 0 and sform_code > 0:
            max_abs_diff = float(np.max(np.abs(qform - sform)))
            result["qform_sform_max_abs_diff"] = max_abs_diff
            if max_abs_diff > 1.0e-3:
                issues.append("qform_sform_mismatch")

        data = np.asarray(image.dataobj, dtype=np.float32)
        finite_mask = np.isfinite(data)
        finite_count = int(np.count_nonzero(finite_mask))
        voxel_count = int(data.size)
        nan_count = int(np.count_nonzero(np.isnan(data)))
        inf_count = int(np.count_nonzero(np.isinf(data)))
        nonzero_count = int(np.count_nonzero(data))

        if voxel_count == 0:
            issues.append("empty_image")
        if finite_count == 0:
            issues.append("no_finite_voxels")
        if nan_count > 0:
            issues.append("contains_nan")
        if inf_count > 0:
            issues.append("contains_inf")
        if nonzero_count == 0:
            issues.append("all_zero")

        finite_values = data[finite_mask]
        if finite_values.size > 0:
            result["min"] = _safe_float(np.min(finite_values))
            result["max"] = _safe_float(np.max(finite_values))
            result["mean"] = _safe_float(np.mean(finite_values))
            result["std"] = _safe_float(np.std(finite_values))
            result["p01"] = _safe_float(np.percentile(finite_values, 1))
            result["p99"] = _safe_float(np.percentile(finite_values, 99))

            foreground = finite_values[finite_values != 0]
            if foreground.size > 0:
                foreground_std = float(np.std(foreground))
                result["foreground_mean"] = _safe_float(np.mean(foreground))
                result["foreground_std"] = _safe_float(foreground_std)
                if foreground_std < 1.0e-6:
                    issues.append("near_constant_foreground")
            else:
                issues.append("no_nonzero_foreground")

            max_abs = max(abs(float(np.min(finite_values))), abs(float(np.max(finite_values))))
            if max_abs > 1.0e7:
                issues.append("extreme_abs_intensity")

        result.update(
            {
                "issues": issues,
                "status": _status_from_issues(issues),
                "shape": list(shape),
                "ndim": ndim,
                "dtype": str(header.get_data_dtype()),
                "spacing": list(spacing),
                "orientation": "".join(axis or "?" for axis in aff2axcodes(image.affine)),
                "qform_code": int(qform_code),
                "sform_code": int(sform_code),
                "affine": _matrix_to_list(image.affine),
                "qform": _matrix_to_list(qform),
                "sform": _matrix_to_list(sform),
                "voxel_count": voxel_count,
                "finite_voxel_count": finite_count,
                "nan_voxel_count": nan_count,
                "inf_voxel_count": inf_count,
                "nonzero_voxel_count": nonzero_count,
                "zero_ratio": _safe_float(1.0 - (nonzero_count / voxel_count)) if voxel_count else None,
            }
        )
    except Exception as exc:
        result["status"] = "Failed"
        result["issues"] = ["load_failed"]
        result["error"] = str(exc)

    result["processing_time_seconds"] = round(time.perf_counter() - started_at, 3)
    return result


def _append_jsonl(report_path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("a", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False, default=_json_ready) + "\n")
        file_obj.flush()


def _write_csv(report_path: Path, rows: List[Dict[str, Any]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    keys = [
        "relative_path",
        "status",
        "issues",
        "error",
        "file_size_bytes",
        "file_modified_time",
        "shape",
        "spacing",
        "orientation",
        "dtype",
        "qform_code",
        "sform_code",
        "qform_sform_max_abs_diff",
        "voxel_count",
        "finite_voxel_count",
        "nan_voxel_count",
        "inf_voxel_count",
        "nonzero_voxel_count",
        "zero_ratio",
        "min",
        "max",
        "mean",
        "std",
        "p01",
        "p99",
        "foreground_mean",
        "foreground_std",
        "processing_time_seconds",
    ]
    with report_path.open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["issues"] = ";".join(row.get("issues") or [])
            csv_row["shape"] = json.dumps(row.get("shape"), ensure_ascii=False)
            csv_row["spacing"] = json.dumps(row.get("spacing"), ensure_ascii=False)
            writer.writerow({key: csv_row.get(key) for key in keys})


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
            if row.get("status") == "Success" and row.get("relative_path"):
                done[row["relative_path"]] = row
    return done


def _summarize_rows(input_dir: Path, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    issue_counter: Counter[str] = Counter()
    shape_counter: Counter[str] = Counter()
    spacing_counter: Counter[str] = Counter()
    orientation_counter: Counter[str] = Counter()
    dtype_counter: Counter[str] = Counter()
    qform_code_counter: Counter[str] = Counter()
    sform_code_counter: Counter[str] = Counter()

    for row in rows:
        issue_counter.update(row.get("issues") or [])
        shape_counter.update([json.dumps(row.get("shape"), ensure_ascii=False)])
        spacing_counter.update([json.dumps(row.get("spacing"), ensure_ascii=False)])
        orientation_counter.update([str(row.get("orientation"))])
        dtype_counter.update([str(row.get("dtype"))])
        qform_code_counter.update([str(row.get("qform_code"))])
        sform_code_counter.update([str(row.get("sform_code"))])

    success = sum(1 for row in rows if row.get("status") == "Success")
    failed = sum(1 for row in rows if row.get("status") == "Failed")
    return {
        "input_dir": str(input_dir),
        "total_files": len(rows),
        "success": success,
        "failed": failed,
        "issue_counts": dict(issue_counter),
        "shape_counts": dict(shape_counter),
        "spacing_counts": dict(spacing_counter),
        "orientation_counts": dict(orientation_counter),
        "dtype_counts": dict(dtype_counter),
        "qform_code_counts": dict(qform_code_counter),
        "sform_code_counts": dict(sform_code_counter),
    }


def run_metadata_audit(
    input_dir: Path,
    num_workers: int,
    resume: bool = True,
) -> Dict[str, Any]:
    """Run read-only metadata audit for one final MRI directory."""
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")

    input_dir = Path(input_dir).resolve()
    report_dir = derive_report_dir(input_dir)
    jsonl_report = report_dir / "metadata_audit_report.jsonl"
    csv_report = report_dir / "metadata_audit_report.csv"
    summary_report = report_dir / "metadata_audit_summary.json"

    input_files = discover_nifti_files(input_dir)
    if not input_files:
        raise RuntimeError(f"No .nii.gz files found under: {input_dir}")

    existing_rows = _load_existing_success_rows(jsonl_report) if resume else {}
    rows_by_relative_path: Dict[str, Dict[str, Any]] = {}
    tasks: List[Tuple[str, str]] = []

    for input_file in input_files:
        relative_path = input_file.relative_to(input_dir).as_posix()
        existing_row = existing_rows.get(relative_path)
        current_size = input_file.stat().st_size
        if existing_row and int(existing_row.get("file_size_bytes", -1)) == current_size:
            rows_by_relative_path[relative_path] = existing_row
        else:
            tasks.append((str(input_file), str(input_dir)))

    print(f"Input directory: {input_dir}")
    print(f"Report directory: {report_dir}")
    print(f"NIfTI files: {len(input_files)}")
    print(f"Already audited: {len(rows_by_relative_path)}")
    print(f"Pending audit: {len(tasks)}")
    print(f"Workers: {num_workers}")

    if tasks:
        if num_workers == 1:
            iterator = map(audit_single_file, tasks)
            progress = tqdm(
                iterator,
                total=len(tasks),
                desc=f"Audit {input_dir.parent.name}",
                unit="volume",
                dynamic_ncols=True,
            )
            for row in progress:
                rows_by_relative_path[row["relative_path"]] = row
                _append_jsonl(jsonl_report, [row])
        else:
            with Pool(processes=num_workers) as pool:
                iterator = pool.imap_unordered(audit_single_file, tasks)
                progress = tqdm(
                    iterator,
                    total=len(tasks),
                    desc=f"Audit {input_dir.parent.name}",
                    unit="volume",
                    dynamic_ncols=True,
                )
                for row in progress:
                    rows_by_relative_path[row["relative_path"]] = row
                    _append_jsonl(jsonl_report, [row])

    all_rows = [
        rows_by_relative_path[input_file.relative_to(input_dir).as_posix()]
        for input_file in input_files
        if input_file.relative_to(input_dir).as_posix() in rows_by_relative_path
    ]
    summary = _summarize_rows(input_dir, all_rows)

    _write_csv(csv_report, all_rows)
    summary_report.parent.mkdir(parents=True, exist_ok=True)
    summary_report.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_ready),
        encoding="utf-8",
    )

    print(
        "Finished: "
        f"success={summary['success']} "
        f"failed={summary['failed']} "
        f"issues={sum(summary['issue_counts'].values())}"
    )
    print(f"JSONL report: {jsonl_report}")
    print(f"CSV report: {csv_report}")
    print(f"Summary report: {summary_report}")
    return summary


def exit_code_from_summary(summary: Dict[str, Any]) -> int:
    """Return non-zero when any file failed the audit."""
    return 1 if int(summary.get("failed", 0)) > 0 else 0


