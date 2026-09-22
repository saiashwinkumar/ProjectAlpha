#!/usr/bin/env python3
"""Build the full historical sector-neutral weekly panel and publish it to S3.

The source is the one-Parquet-file-per-week feature backfill.  The destination
uses a separate versioned prefix and the same weekly partitioning scheme.  S3
writes are conditional, so reruns resume safely and never overwrite a completed
weekly object.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError

from cross_sectional_panel import (
    DEFAULT_FEATURE_DIRECTIONS,
    build_cross_sectional_panel,
)


DEFAULT_BUCKET = "s3-stock-market-project-ashwin"
DEFAULT_SOURCE_PREFIX = (
    "features_parquet/"
    "feature_version=close_features_v1/"
    "data_version=kaggle_2026_02/"
)
PANEL_VERSION = "sector_neutral_active_returns_v2"
DEFAULT_DESTINATION_PREFIX = (
    "panels_parquet/"
    "feature_version=close_features_v1/"
    "data_version=kaggle_2026_02/"
    f"panel_version={PANEL_VERSION}/"
)
DEFAULT_LOCAL_OUTPUT = Path(
    "data/gold/cross_sectional_weekly_panel_full_v2.parquet"
)
WEEK_PATTERN = re.compile(r"week_date=(\d{4}-\d{2}-\d{2})/([^/]+)$")


def list_source_objects(s3, bucket: str, prefix: str) -> list[dict]:
    """Discover and validate exactly one source Parquet object per week."""

    objects = []
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix
    ):
        for item in page.get("Contents", []):
            key = item["Key"]
            match = WEEK_PATTERN.search(key[len(prefix):])
            if match and key.endswith(".parquet"):
                objects.append(
                    {
                        "week_date": pd.Timestamp(match.group(1)),
                        "key": key,
                        "etag": item.get("ETag", "").strip('"'),
                        "size": int(item["Size"]),
                    }
                )

    if not objects:
        raise ValueError(f"No source Parquet objects under s3://{bucket}/{prefix}")

    source = pd.DataFrame(objects).sort_values("week_date")
    duplicates = source["week_date"].duplicated(keep=False)
    if duplicates.any():
        raise ValueError(
            "Multiple source Parquet objects found for weeks: "
            f"{source.loc[duplicates, 'week_date'].dt.strftime('%Y-%m-%d').tolist()}"
        )

    expected = pd.date_range(
        source["week_date"].min(), source["week_date"].max(), freq="W-WED"
    )
    missing = expected.difference(pd.DatetimeIndex(source["week_date"]))
    if len(missing):
        raise ValueError(
            "Historical source has missing Wednesday partitions: "
            f"{missing.strftime('%Y-%m-%d').tolist()}"
        )
    return source.to_dict("records")


def source_fingerprint(objects: list[dict]) -> str:
    stable = [
        {
            "week_date": str(item["week_date"].date()),
            "key": item["key"],
            "etag": item["etag"],
            "size": item["size"],
        }
        for item in objects
    ]
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def download_source_panel(
    s3,
    bucket: str,
    objects: list[dict],
    cache_path: Path,
    *,
    workers: int = 12,
) -> pd.DataFrame:
    if cache_path.exists():
        print(f"Loading historical source cache: {cache_path}")
        return pd.read_parquet(cache_path)

    def download(item: dict) -> pd.DataFrame:
        response = s3.get_object(Bucket=bucket, Key=item["key"])
        body = response["Body"]
        try:
            content = body.read()
        finally:
            body.close()
        frame = pd.read_parquet(io.BytesIO(content))
        if "week_date" not in frame.columns:
            frame["week_date"] = item["week_date"]
        return frame

    frames: list[pd.DataFrame] = []
    completed = 0
    print(f"Downloading {len(objects):,} historical weekly objects...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download, item): item for item in objects}
        for future in as_completed(futures):
            frames.append(future.result())
            completed += 1
            if completed % 100 == 0 or completed == len(objects):
                print(f"Downloaded {completed:,}/{len(objects):,}")

    panel = pd.concat(frames, ignore_index=True)
    panel["week_date"] = pd.to_datetime(
        panel["week_date"], errors="raise"
    ).dt.normalize()
    if "price_date" in panel.columns:
        panel["price_date"] = pd.to_datetime(
            panel["price_date"], errors="raise"
        ).dt.normalize()
    panel["Ticker"] = panel["Ticker"].astype("string").str.strip().str.upper()
    panel = panel.sort_values(["week_date", "Ticker"]).reset_index(drop=True)
    if panel.duplicated(["Ticker", "week_date"]).any():
        raise ValueError("Downloaded source contains duplicate ticker/week rows")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(cache_path, index=False, compression="snappy")
    print(f"Saved historical source cache: {cache_path}")
    return panel


def load_sector_maps(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for priority, path in enumerate(paths):
        frame = pd.read_csv(path, usecols=["Ticker", "sector"])
        frame["priority"] = priority
        frames.append(frame)
    sectors = pd.concat(frames, ignore_index=True)
    sectors["_key"] = (
        sectors["Ticker"].astype("string").str.strip().str.upper()
        .str.replace(".", "-", regex=False)
    )
    conflicts = sectors.groupby("_key")["sector"].nunique().gt(1)
    if conflicts.any():
        raise ValueError(
            "Conflicting sector mappings for: "
            f"{conflicts.index[conflicts].tolist()}"
        )
    return (
        sectors.sort_values("priority")
        .drop_duplicates("_key")
        .loc[:, ["Ticker", "sector"]]
    )


def validate_panel(panel: pd.DataFrame, source_weeks: pd.DatetimeIndex) -> dict:
    features = list(DEFAULT_FEATURE_DIRECTIONS)
    train_features = [f"{name}_train" for name in features]
    target_columns = ["active_return_train", "active_return_fwd"]

    if panel.duplicated(["Ticker", "week_date"]).any():
        raise ValueError("Final panel contains duplicate ticker/week rows")
    if panel["sector"].isna().any():
        raise ValueError("Final panel contains missing sectors")
    if panel[features].isna().any().any():
        raise ValueError("Final panel contains missing current-week feature scores")
    if panel["active_return_train"].isna().any():
        raise ValueError("Final panel contains missing active_return_train scores")
    numeric = panel[features + train_features + target_columns].to_numpy(
        dtype=float
    )
    finite_or_missing = np.isfinite(numeric) | np.isnan(numeric)
    if not finite_or_missing.all():
        raise ValueError("Final panel contains infinite feature values")

    actual_weeks = pd.DatetimeIndex(
        panel["week_date"].drop_duplicates().sort_values()
    )
    if not actual_weeks.equals(source_weeks):
        raise ValueError("Final panel week coverage differs from source coverage")

    weekly_means = panel.groupby("week_date")[features].mean()
    max_weekly_mean = float(weekly_means.abs().max().max())
    sector_means = panel.groupby(["week_date", "sector"])[features].mean()
    max_sector_mean = float(sector_means.abs().max().max())
    if max_weekly_mean > 1e-10 or max_sector_mean > 1e-10:
        raise ValueError(
            "Neutrality validation failed: "
            f"weekly={max_weekly_mean}, sector={max_sector_mean}"
        )

    weekly_std = panel.groupby("week_date")[features].std(ddof=0)
    nonzero = weekly_std.where(weekly_std.gt(1e-12)).stack()
    max_std_error = float((nonzero - 1.0).abs().max())
    if max_std_error > 1e-10:
        raise ValueError(f"Weekly z-score validation failed: {max_std_error}")

    target_weekly_mean = float(
        panel.groupby("week_date")["active_return_train"].mean().abs().max()
    )
    target_sector_mean = float(
        panel.groupby(["week_date", "sector"])["active_return_train"]
        .mean().abs().max()
    )
    target_weekly_std = panel.groupby("week_date")["active_return_train"].std(
        ddof=0
    )
    target_nonzero_std = target_weekly_std[target_weekly_std.gt(1e-12)]
    target_max_std_error = float((target_nonzero_std - 1.0).abs().max())
    if target_weekly_mean > 1e-10 or target_sector_mean > 1e-10:
        raise ValueError(
            "Active-return neutrality validation failed: "
            f"weekly={target_weekly_mean}, sector={target_sector_mean}"
        )
    if target_max_std_error > 1e-10:
        raise ValueError(
            "Active-return z-score validation failed: "
            f"{target_max_std_error}"
        )

    first_week = actual_weeks[0]
    if panel.loc[panel["week_date"].eq(first_week), "prev_week_date"].notna().any():
        raise ValueError("The first historical week must not have a previous week")
    previous_map = pd.Series(actual_weeks[:-1].to_numpy(), index=actual_weeks[1:])
    expected_previous = panel["week_date"].map(previous_map)
    if not panel["prev_week_date"].equals(expected_previous):
        raise ValueError("prev_week_date does not match the preceding source week")

    next_map = pd.Series(actual_weeks[1:].to_numpy(), index=actual_weeks[:-1])
    forward_check = panel.loc[
        :, ["Ticker", "week_date", "active_return_train", "active_return_fwd"]
    ].copy()
    forward_check["_next_week_date"] = forward_check["week_date"].map(next_map)
    next_scores = panel.loc[
        :, ["Ticker", "week_date", "active_return_train"]
    ].rename(
        columns={
            "week_date": "_next_week_date",
            "active_return_train": "_expected_active_return_fwd",
        }
    )
    forward_check = forward_check.merge(
        next_scores,
        on=["Ticker", "_next_week_date"],
        how="left",
        validate="many_to_one",
    )
    if not np.allclose(
        forward_check["active_return_fwd"],
        forward_check["_expected_active_return_fwd"],
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    ):
        raise ValueError(
            "active_return_fwd does not exactly match the ticker's score in "
            "the next source week"
        )

    return {
        "rows": int(len(panel)),
        "tickers": int(panel["Ticker"].nunique()),
        "weeks": int(len(actual_weeks)),
        "first_week": str(actual_weeks[0].date()),
        "last_week": str(actual_weeks[-1].date()),
        "rows_with_any_missing_train_feature": int(
            panel[train_features].isna().any(axis=1).sum()
        ),
        "rows_with_missing_active_return_fwd": int(
            panel["active_return_fwd"].isna().sum()
        ),
        "max_abs_weekly_feature_mean": max_weekly_mean,
        "max_abs_week_sector_feature_mean": max_sector_mean,
        "max_abs_population_std_minus_one": max_std_error,
        "max_abs_weekly_active_return_mean": target_weekly_mean,
        "max_abs_week_sector_active_return_mean": target_sector_mean,
        "max_abs_active_return_population_std_minus_one": (
            target_max_std_error
        ),
    }


def serialize_week(group: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    group.to_parquet(
        buffer,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )
    return buffer.getvalue()


def list_existing_keys(s3, bucket: str, prefix: str) -> set[str]:
    keys: set[str] = set()
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix
    ):
        keys.update(item["Key"] for item in page.get("Contents", []))
    return keys


def upload_partitioned_panel(
    s3,
    bucket: str,
    prefix: str,
    panel: pd.DataFrame,
    *,
    workers: int = 12,
    dry_run: bool = False,
) -> pd.DataFrame:
    existing = list_existing_keys(s3, bucket, prefix)
    jobs = []
    for week, group in panel.groupby("week_date", sort=True):
        date = pd.Timestamp(week).date().isoformat()
        key = f"{prefix}week_date={date}/panel.parquet"
        jobs.append((date, key, group.copy()))

    if dry_run:
        return pd.DataFrame(
            {
                "week_date": [date for date, _, _ in jobs],
                "key": [key for _, key, _ in jobs],
                "status": [
                    "skip_existing" if key in existing else "would_upload"
                    for _, key, _ in jobs
                ],
            }
        )

    def upload(job: tuple[str, str, pd.DataFrame]) -> dict:
        date, key, group = job
        if key in existing:
            return {"week_date": date, "key": key, "status": "skipped_existing"}
        payload = serialize_week(group)
        try:
            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=payload,
                ContentType="application/vnd.apache.parquet",
                IfNoneMatch="*",
            )
        except ClientError as exc:
            if str(exc.response["Error"].get("Code")) in {
                "PreconditionFailed", "412"
            }:
                return {
                    "week_date": date,
                    "key": key,
                    "status": "skipped_concurrent_existing",
                }
            raise
        return {
            "week_date": date,
            "key": key,
            "status": "uploaded",
            "rows": len(group),
            "bytes": len(payload),
        }

    results = []
    completed = 0
    print(f"Publishing {len(jobs):,} weekly panel partitions...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(upload, job): job for job in jobs}
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            if completed % 100 == 0 or completed == len(jobs):
                uploaded = sum(r["status"] == "uploaded" for r in results)
                print(
                    f"Processed {completed:,}/{len(jobs):,}; "
                    f"uploaded {uploaded:,}"
                )

    report = pd.DataFrame(results).sort_values("week_date").reset_index(drop=True)
    expected_keys = {key for _, key, _ in jobs}
    final_keys = list_existing_keys(s3, bucket, prefix)
    missing = sorted(expected_keys - final_keys)
    if missing:
        raise RuntimeError(f"S3 verification found {len(missing)} missing partitions")
    return report


def publish_manifest(
    s3,
    bucket: str,
    prefix: str,
    manifest: dict,
) -> str:
    key = f"{prefix}_manifest.json"
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            ContentType="application/json",
            IfNoneMatch="*",
        )
    except ClientError as exc:
        if str(exc.response["Error"].get("Code")) not in {
            "PreconditionFailed", "412"
        }:
            raise
        existing = json.loads(
            s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode()
        )
        comparable_keys = set(manifest) - {"published_at_utc", "upload"}
        if any(existing.get(name) != manifest.get(name) for name in comparable_keys):
            raise ValueError(
                f"Existing manifest at s3://{bucket}/{key} describes a different panel"
            )
    return key


def run(args: argparse.Namespace) -> dict:
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    # Resolve SSO credentials before concurrent requests to avoid refresh races.
    session.get_credentials().get_frozen_credentials()
    s3 = session.client("s3")

    objects = list_source_objects(s3, args.bucket, args.source_prefix)
    fingerprint = source_fingerprint(objects)
    cache_path = Path("local_cache") / f"historical_features_{fingerprint[:12]}.parquet"
    source_panel = download_source_panel(
        s3, args.bucket, objects, cache_path, workers=args.workers
    )
    sectors = load_sector_maps([Path(path) for path in args.sector_map])
    panel = build_cross_sectional_panel(source_panel, sectors)
    source_weeks = pd.DatetimeIndex(
        [item["week_date"] for item in objects]
    ).sort_values()
    validation = validate_panel(panel, source_weeks)

    output_path = Path(args.local_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False, compression="snappy")
    print(f"Saved validated full panel: {output_path}")
    print(json.dumps(validation, indent=2))

    report = upload_partitioned_panel(
        s3,
        args.bucket,
        args.destination_prefix,
        panel,
        workers=args.workers,
        dry_run=not args.upload,
    )
    status_counts = {
        str(key): int(value)
        for key, value in report["status"].value_counts().to_dict().items()
    }
    manifest = {
        "panel_version": PANEL_VERSION,
        "feature_version": "close_features_v1",
        "data_version": "kaggle_2026_02",
        "source": f"s3://{args.bucket}/{args.source_prefix}",
        "destination": f"s3://{args.bucket}/{args.destination_prefix}",
        "source_fingerprint_sha256": fingerprint,
        "source_objects": len(objects),
        "feature_columns": list(DEFAULT_FEATURE_DIRECTIONS),
        "target_columns": ["active_return_train", "active_return_fwd"],
        "active_return_definition": (
            "raw return_1w minus the arithmetic mean of raw return_1w within "
            "week_date/sector"
        ),
        "target_timing": {
            "active_return_train": "previous observed week to current week",
            "active_return_fwd": "current week to next observed week",
        },
        "feature_directions": DEFAULT_FEATURE_DIRECTIONS,
        "winsor_limits": [0.01, 0.99],
        "zscore_ddof": 0,
        "transform_order": [
            "orient higher-is-better",
            "winsorize within week_date/sector",
            "centered percentile rank within week_date/sector",
            "z-score across week_date",
            "exact previous-observed-week self-join by Ticker",
            "exact next-observed-week active-return self-join by Ticker",
        ],
        "validation": validation,
        "upload": status_counts,
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    manifest_path = output_path.with_suffix(".metadata.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if args.upload:
        manifest_key = publish_manifest(
            s3, args.bucket, args.destination_prefix, manifest
        )
        manifest["manifest_s3_key"] = manifest_key
    else:
        print("Dry run only; pass --upload to publish to S3.")
    print(json.dumps({"upload": status_counts}, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX)
    parser.add_argument("--destination-prefix", default=DEFAULT_DESTINATION_PREFIX)
    parser.add_argument("--profile", default="admin")
    parser.add_argument("--region", default="ap-southeast-2")
    parser.add_argument(
        "--sector-map",
        action="append",
        default=None,
        help="Ticker/sector CSV; repeat to combine mappings",
    )
    parser.add_argument("--local-output", default=str(DEFAULT_LOCAL_OUTPUT))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Publish after validation; without this flag the S3 write is a dry run",
    )
    args = parser.parse_args()
    if args.sector_map is None:
        args.sector_map = ["sector_mapping_completed.csv", "sector_mapping.csv"]
    if args.workers < 1:
        parser.error("--workers must be positive")
    for name in ("source_prefix", "destination_prefix"):
        value = getattr(args, name)
        if not value.endswith("/"):
            setattr(args, name, value + "/")
    return args


if __name__ == "__main__":
    run(parse_args())
