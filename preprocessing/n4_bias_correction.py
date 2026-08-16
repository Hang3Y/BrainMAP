"""Apply N4 bias field correction to one explicitly configured MRI directory."""

from __future__ import annotations

import json
import importlib.util
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from tqdm import tqdm


def ensure_antspyx_available() -> None:
    """Fail before multiprocessing when the antspyx package is unavailable."""
    if importlib.util.find_spec("ants") is None:
        raise RuntimeError(
            "ANTsPy is not installed in the current Python environment. "
            "Install package 'antspyx' and verify it with "
            "'python -c \"import ants; print(ants.__version__)\"'."
        )


def exit_code_from_summary(summary: Dict[str, int]) -> int:
    """Return a non-zero process exit code when any volume failed."""
    return 1 if summary.get("failed", 0) > 0 else 0


def derive_output_paths(input_dir: Path) -> Tuple[Path, Path]:
    """Return the sibling output directory and its JSONL QC report path."""
    input_dir = Path(input_dir)
    if input_dir.name.endswith("_n4"):
        raise ValueError(f"Input directory already has the '_n4' suffix: {input_dir}")

    output_dir = input_dir.with_name(f"{input_dir.name}_n4")
    report_path = output_dir.with_name(f"{output_dir.name}_qc") / "n4_processing_report.jsonl"
    return output_dir, report_path


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


def output_path_for(input_file: Path, input_dir: Path, output_dir: Path) -> Path:
    """Map an input file to the same relative location under output_dir."""
    return Path(output_dir) / Path(input_file).relative_to(input_dir)


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


def _is_valid_output(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _process_single_file(task: Tuple[str, str, Dict[str, Any]]) -> Dict[str, Any]:
    input_path_text, output_path_text, n4_parameters = task
    input_path = Path(input_path_text)
    output_path = Path(output_path_text)
    started_at = time.perf_counter()

    result: Dict[str, Any] = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "status": "Success",
        "error": "",
        "processing_time_seconds": 0.0,
    }

    if _is_valid_output(output_path):
        result["status"] = "SkippedExisting"
        result["processing_time_seconds"] = round(
            time.perf_counter() - started_at, 3
        )
        return result

    try:
        import ants

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = ants.image_read(str(input_path))
        brain_mask = ants.get_mask(image)
        corrected = ants.n4_bias_field_correction(
            image,
            mask=brain_mask,
            shrink_factor=int(n4_parameters["shrink_factor"]),
            convergence={
                "iters": [
                    int(value)
                    for value in n4_parameters["convergence_iterations"]
                ],
                "tol": float(n4_parameters["convergence_tolerance"]),
            },
            spline_param=int(n4_parameters["spline_param"]),
            verbose=bool(n4_parameters["verbose"]),
        )
        ants.image_write(corrected, str(output_path))

        if not _is_valid_output(output_path):
            raise IOError(f"Output file is missing or empty: {output_path}")
    except Exception as exc:
        result["status"] = "Failed"
        result["error"] = str(exc)
        if output_path.exists():
            output_path.unlink()

    result["processing_time_seconds"] = round(
        time.perf_counter() - started_at, 3
    )
    return result


def _append_jsonl(report_path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("a", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
        file_obj.flush()


def run_n4_bias_correction(
    input_dir: Path,
    num_workers: int,
    ants_threads_per_worker: int,
    shrink_factor: int,
    convergence_iterations: List[int],
    convergence_tolerance: float,
    spline_param: int,
    verbose: bool,
) -> Dict[str, int]:
    """Run recursive N4 correction with resume support for one dataset."""
    ensure_antspyx_available()

    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if ants_threads_per_worker < 1:
        raise ValueError("ants_threads_per_worker must be at least 1")

    input_dir = Path(input_dir).resolve()
    output_dir, report_path = derive_output_paths(input_dir)
    input_files = discover_nifti_files(input_dir)
    if not input_files:
        raise RuntimeError(f"No .nii.gz files found under: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    n4_parameters = {
        "shrink_factor": shrink_factor,
        "convergence_iterations": convergence_iterations,
        "convergence_tolerance": convergence_tolerance,
        "spline_param": spline_param,
        "verbose": verbose,
    }
    tasks = [
        (
            str(input_file),
            str(output_path_for(input_file, input_dir, output_dir)),
            n4_parameters,
        )
        for input_file in input_files
    ]

    summary = {"success": 0, "skipped": 0, "failed": 0}

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"QC report: {report_path}")
    print(f"NIfTI files: {len(input_files)}")
    print(
        f"Workers: {num_workers}; "
        f"ANTs threads per worker: {ants_threads_per_worker}"
    )

    with Pool(
        processes=num_workers,
        initializer=_limit_threads_for_ants,
        initargs=(ants_threads_per_worker,),
    ) as pool:
        result_iterator = pool.imap_unordered(_process_single_file, tasks)
        progress = tqdm(
            result_iterator,
            total=len(tasks),
            desc=f"N4 {input_dir.parent.name}",
            unit="volume",
            dynamic_ncols=True,
        )
        for result in progress:
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
                tqdm.write(
                    f"Failed: {relative_input.as_posix()}: {result['error']}"
                )

    print(
        "Finished: "
        f"success={summary['success']} "
        f"skipped={summary['skipped']} "
        f"failed={summary['failed']}"
    )
    return summary


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    dataset_inputs = {
        "1001_abideii": "1001_abideii/abide_fm_ready_skull_stripping",
        "1005_atlasr2": "1005_atlasr2/atlas_fm_ready_skull_stripping",
        "1007_isles2022": "1007_isles2022/isles_fm_ready",
        "1014_ixi": "1014_ixi/ixi_fm_ready_skull_stripping",
        "1020_wmh": "1020_wmh/wmh_fm_ready_skull_stripping",
        "1024_remind": "1024_remind/remind_fm_ready",
        "1030_nigerian_clinical_mri": "1030_nigerian_clinical_mri/curated_fm_ready_skull_stripping",
    }

    # active_dataset = "1001_abideii"  # TODO 5
    active_dataset = "1005_atlasr2"  # TODO 1
    # active_dataset = "1007_isles2022"  # TODO 2
    # active_dataset = "1014_ixi"  # TODO 6
    # active_dataset = "1020_wmh"  # TODO 3
    # active_dataset = "1024_remind"  # TODO 4
    # active_dataset = "1030_nigerian_clinical_mri"

    selected_input_dir = script_dir / dataset_inputs[active_dataset]

    processing_summary = run_n4_bias_correction(
        input_dir=selected_input_dir,
        num_workers=4,
        ants_threads_per_worker=6,
        shrink_factor=4,
        convergence_iterations=[50, 50, 50, 50],
        convergence_tolerance=1.0e-7,
        spline_param=200,
        verbose=False,
    )
    raise SystemExit(exit_code_from_summary(processing_summary))
