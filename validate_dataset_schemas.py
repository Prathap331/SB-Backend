"""
validate_dataset_schemas.py
────────────────────────────────────────────────────────────────────────────
Validate schema analysis quality across all datasets listed in Storybit_Database.xlsx.

What it does:
  1) Reads dataset URLs from spreadsheet
  2) Downloads each Kaggle dataset
  3) Loads supported files
  4) Runs schema analysis (LM Studio + fallback rules from ingest_from_kaggle.py)
  5) Produces JSON + CSV report with ok/warn/error statuses

Usage:
  python validate_dataset_schemas.py
  python validate_dataset_schemas.py --limit 10
  python validate_dataset_schemas.py --only-slug owner/dataset-name
  python validate_dataset_schemas.py --max-rows-per-file 1200
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from ingest_from_kaggle import (
    SPREADSHEET_FILE,
    read_spreadsheet,
    extract_slug,
    download_dataset,
    find_data_files,
    load_dataframe,
    analyse_schema,
)

@dataclass
class FileResult:
    dataset_slug: str
    file: str
    status: str
    reason: str
    rows: int
    cols: int
    primary_count: int
    context_count: int
    confidence: float
    used_llm: bool
    date_col: str
    primary_preview: str
    context_preview: str


@dataclass
class DatasetResult:
    dataset_slug: str
    status: str
    reason: str
    file_count: int
    ok_files: int
    warn_files: int
    error_files: int
    elapsed_sec: float


def _classify_file_result(schema: dict[str, Any], rows: int, cols: int) -> tuple[str, str]:
    primary = schema.get("primary", []) or []
    used_llm = bool(schema.get("used_llm", False))
    confidence = float(schema.get("confidence", 0.0) or 0.0)

    if rows == 0 or cols == 0:
        return "error", "empty dataframe"
    if not primary:
        return "error", "no primary columns"
    if not used_llm:
        return "warn", "llm failed; fallback schema used"
    if confidence < 0.65:
        return "warn", f"low confidence ({confidence:.2f})"
    return "ok", "schema detected"


def _write_reports(
    outdir: Path,
    dataset_results: list[DatasetResult],
    file_results: list[FileResult],
    started_at: str,
) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / f"schema_validation_{started_at}.json"
    csv_path = outdir / f"schema_validation_files_{started_at}.csv"

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": {
            "datasets_total": len(dataset_results),
            "datasets_ok": sum(1 for d in dataset_results if d.status == "ok"),
            "datasets_warn": sum(1 for d in dataset_results if d.status == "warn"),
            "datasets_error": sum(1 for d in dataset_results if d.status == "error"),
            "files_total": len(file_results),
            "files_ok": sum(1 for f in file_results if f.status == "ok"),
            "files_warn": sum(1 for f in file_results if f.status == "warn"),
            "files_error": sum(1 for f in file_results if f.status == "error"),
        },
        "datasets": [asdict(d) for d in dataset_results],
        "files": [asdict(f) for f in file_results],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_slug",
                "file",
                "status",
                "reason",
                "rows",
                "cols",
                "primary_count",
                "context_count",
                "confidence",
                "used_llm",
                "date_col",
                "primary_preview",
                "context_preview",
            ],
        )
        writer.writeheader()
        for row in file_results:
            writer.writerow(asdict(row))

    return json_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate schema extraction across spreadsheet datasets")
    parser.add_argument("--spreadsheet", default=SPREADSHEET_FILE, help="Path to StoryBit spreadsheet")
    parser.add_argument("--limit", type=int, default=0, help="Max number of datasets to validate (0 = all)")
    parser.add_argument("--only-slug", default="", help="Validate only this one slug")
    parser.add_argument("--max-rows-per-file", type=int, default=2000, help="Row sample cap per file")
    parser.add_argument("--output-dir", default="reports", help="Report output directory")
    args = parser.parse_args()

    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    topics = read_spreadsheet(args.spreadsheet)

    slugs: list[str] = []
    for t in topics:
        db_link = t.get("db_link", "")
        if not db_link:
            continue
        slug = extract_slug(db_link)
        if slug:
            slugs.append(slug)

    # preserve order, remove duplicates
    seen = set()
    ordered_slugs = []
    for s in slugs:
        if s not in seen:
            seen.add(s)
            ordered_slugs.append(s)

    if args.only_slug:
        ordered_slugs = [s for s in ordered_slugs if s == args.only_slug]
    if args.limit > 0:
        ordered_slugs = ordered_slugs[: args.limit]

    if not ordered_slugs:
        print("No dataset slugs found to validate.")
        return 1

    print(f"Schema validation started for {len(ordered_slugs)} dataset(s)")
    print(f"Spreadsheet: {args.spreadsheet}")
    print(f"Output dir:  {args.output_dir}\n")

    dataset_results: list[DatasetResult] = []
    file_results: list[FileResult] = []

    for idx, slug in enumerate(ordered_slugs, start=1):
        t0 = time.time()
        print(f"[{idx:02d}/{len(ordered_slugs)}] {slug}")

        path = download_dataset(slug)
        if not path:
            dr = DatasetResult(
                dataset_slug=slug,
                status="error",
                reason="download failed",
                file_count=0,
                ok_files=0,
                warn_files=0,
                error_files=1,
                elapsed_sec=round(time.time() - t0, 2),
            )
            dataset_results.append(dr)
            print("  ✗ download failed")
            continue

        files = find_data_files(path)
        if not files:
            dr = DatasetResult(
                dataset_slug=slug,
                status="error",
                reason="no supported files found",
                file_count=0,
                ok_files=0,
                warn_files=0,
                error_files=1,
                elapsed_sec=round(time.time() - t0, 2),
            )
            dataset_results.append(dr)
            print("  ✗ no supported files")
            continue

        ok_files = warn_files = error_files = 0

        for fp in files:
            fname = Path(fp).name
            try:
                df = load_dataframe(fp)
                if df is None or df.empty:
                    fr = FileResult(
                        dataset_slug=slug,
                        file=fname,
                        status="error",
                        reason="unreadable or empty",
                        rows=0,
                        cols=0,
                        primary_count=0,
                        context_count=0,
                        confidence=0.0,
                        used_llm=False,
                        date_col="",
                        primary_preview="",
                        context_preview="",
                    )
                else:
                    df = df.head(args.max_rows_per_file)
                    schema = analyse_schema(df)
                    status, reason = _classify_file_result(schema, len(df), len(df.columns))
                    primary = schema.get("primary", []) or []
                    context = schema.get("context", []) or []
                    fr = FileResult(
                        dataset_slug=slug,
                        file=fname,
                        status=status,
                        reason=reason,
                        rows=int(len(df)),
                        cols=int(len(df.columns)),
                        primary_count=len(primary),
                        context_count=len(context),
                        confidence=float(schema.get("confidence", 0.0) or 0.0),
                        used_llm=bool(schema.get("used_llm", False)),
                        date_col=str(schema.get("date_col", "") or ""),
                        primary_preview=",".join(primary[:6]),
                        context_preview=",".join(context[:6]),
                    )
            except Exception as e:
                fr = FileResult(
                    dataset_slug=slug,
                    file=fname,
                    status="error",
                    reason=f"exception: {type(e).__name__}",
                    rows=0,
                    cols=0,
                    primary_count=0,
                    context_count=0,
                    confidence=0.0,
                    used_llm=False,
                    date_col="",
                    primary_preview="",
                    context_preview="",
                )

            file_results.append(fr)
            if fr.status == "ok":
                ok_files += 1
            elif fr.status == "warn":
                warn_files += 1
            else:
                error_files += 1

        ds_status = "ok"
        ds_reason = "all files passed"
        if error_files > 0:
            ds_status = "error"
            ds_reason = "one or more files failed"
        elif warn_files > 0:
            ds_status = "warn"
            ds_reason = "one or more files used fallback/low-confidence schema"

        dr = DatasetResult(
            dataset_slug=slug,
            status=ds_status,
            reason=ds_reason,
            file_count=len(files),
            ok_files=ok_files,
            warn_files=warn_files,
            error_files=error_files,
            elapsed_sec=round(time.time() - t0, 2),
        )
        dataset_results.append(dr)
        print(
            f"  ✓ files={len(files)} | ok={ok_files} warn={warn_files} error={error_files} "
            f"| {dr.elapsed_sec}s"
        )

    outdir = Path(args.output_dir)
    json_path, csv_path = _write_reports(outdir, dataset_results, file_results, started_at)

    print("\nValidation completed.")
    print(f"JSON report: {json_path}")
    print(f"CSV report:  {csv_path}")

    ds_err = sum(1 for d in dataset_results if d.status == "error")
    return 0 if ds_err == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
