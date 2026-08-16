"""Run HD-BET recursively for one prepared MRI dataset."""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple


REPORT_FIELDS = (
    "input_path",
    "output_path",
    "status",
    "error",
    "processing_time_seconds",
)


def derive_output_paths(input_dir: Path) -> Tuple[Path, Path, Path]:
    input_dir = Path(input_dir)
    output_dir = input_dir.with_name(f"{input_dir.name}_skull_stripping")
    qc_dir = input_dir.with_name(f"{input_dir.name}_qc")
    report_path = qc_dir / "skull_stripping_report.csv"
    return output_dir, qc_dir, report_path


def discover_nifti_files(input_dir: Path) -> List[Path]:
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
    return Path(output_dir) / Path(input_file).relative_to(input_dir)


def _is_valid_existing_output(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _write_report(report_path: Path, rows: Iterable[Dict[str, str]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = report_path.with_suffix(report_path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, report_path)


def process_files(
    input_dir: Path,
    output_dir: Path,
    report_path: Path,
    predict_file: Callable[[Path, Path], None],
) -> Dict[str, int]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    report_path = Path(report_path)
    input_files = discover_nifti_files(input_dir)
    if not input_files:
        raise RuntimeError(f"No .nii.gz files found under: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    summary = {"success": 0, "skipped": 0, "failed": 0}
    total = len(input_files)

    for index, input_file in enumerate(input_files, start=1):
        output_file = output_path_for(input_file, input_dir, output_dir)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        started_at = time.perf_counter()

        if _is_valid_existing_output(output_file):
            status = "SkippedExisting"
            error = ""
            summary["skipped"] += 1
        else:
            try:
                predict_file(input_file, output_file)
                if not _is_valid_existing_output(output_file):
                    raise RuntimeError("HD-BET did not create a non-empty output file")
                status = "Success"
                error = ""
                summary["success"] += 1
            except Exception as exc:
                if output_file.exists():
                    output_file.unlink()
                status = "Failed"
                error = str(exc)
                summary["failed"] += 1

        elapsed = time.perf_counter() - started_at
        rows.append(
            {
                "input_path": str(input_file),
                "output_path": str(output_file),
                "status": status,
                "error": error,
                "processing_time_seconds": f"{elapsed:.3f}",
            }
        )
        _write_report(report_path, rows)
        print(
            f"[{index}/{total}] {status}: "
            f"{input_file.relative_to(input_dir).as_posix()}"
        )

    return summary


def create_hdbet_predictor(device: str, use_tta: bool):
    try:
        import torch
        from HD_BET.checkpoint_download import maybe_download_parameters
        from HD_BET.hd_bet_prediction import get_hdbet_predictor
    except ImportError as exc:
        raise RuntimeError(
            "HD-BET is not installed. Install it in the server environment with "
            "'pip install hd-bet'."
        ) from exc

    maybe_download_parameters()
    return get_hdbet_predictor(
        use_tta=use_tta,
        device=torch.device(device),
        verbose=False,
    )


def run_skull_stripping(
    input_dir: Path,
    device: str = "cuda",
    use_tta: bool = True,
) -> Dict[str, int]:
    input_dir = Path(input_dir).resolve()
    output_dir, _, report_path = derive_output_paths(input_dir)
    predictor = create_hdbet_predictor(device=device, use_tta=use_tta)

    from HD_BET.hd_bet_prediction import hdbet_predict

    def predict_file(input_file: Path, output_file: Path) -> None:
        hdbet_predict(
            str(input_file),
            str(output_file),
            predictor,
            keep_brain_mask=False,
            compute_brain_extracted_image=True,
        )

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"QC report: {report_path}")
    summary = process_files(
        input_dir=input_dir,
        output_dir=output_dir,
        report_path=report_path,
        predict_file=predict_file,
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
        "1001_abideii": "abide_fm_ready",
        "1005_atlasr2": "atlas_fm_ready",
        "1014_ixi": "ixi_fm_ready",
        "1020_wmh": "wmh_fm_ready",
        "1030_nigerian_clinical_mri": "curated_fm_ready",
    }

    active_dataset = "1001_abideii"
    # active_dataset = "1005_atlasr2"
    # active_dataset = "1014_ixi"
    # active_dataset = "1020_wmh"
    # active_dataset = "1030_nigerian_clinical_mri"
    selected_input_dir = script_dir / active_dataset / dataset_inputs[active_dataset]

    run_skull_stripping(
        input_dir=selected_input_dir,
        device="cuda",
        use_tta=True,
    )


"""
20260622 第一次测试demo
非常厉害 去除颅骨简直是完美的 

接下来是转移到正式颅骨去除数据集中执行。


"""
