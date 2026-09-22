"""Load the ML panel through Athena without altering the Glue catalog."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import boto3
import pandas as pd

from cross_sectional_panel import DEFAULT_FEATURE_DIRECTIONS

LOGGER = logging.getLogger(__name__)
DEFAULT_DATABASE = "stock_market_test"
DEFAULT_TABLE = "panel_version_sector_neutral_active_returns_v2"
DEFAULT_OUTPUT = "s3://s3-stock-market-project-ashwin/athena-results/ml_training/"


def identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Invalid SQL identifier: {value!r}")
    return f'"{value}"'


def panel_query(database: str, table: str, start: str, end: str | None,
                week_date_type: str = "string") -> str:
    columns = ["ticker", "sector", "week_date", "prev_week_date"]
    columns += list(DEFAULT_FEATURE_DIRECTIONS)
    columns += [f"{name}_train" for name in DEFAULT_FEATURE_DIRECTIONS]
    columns += ["active_return_train", "active_return_fwd"]
    # Works for catalog date, timestamp, or ISO string fields. The actual
    # partition key can be separate (e.g. partition_0) when the file includes
    # week_date. Never infer its meaning or silently rewrite catalog metadata.
    date_expr = 'CAST(substr(CAST("week_date" AS varchar), 1, 10) AS date)'
    literal = "DATE "
    if week_date_type == "string":
        date_expr, literal = '"week_date"', ""
    predicate = f"{date_expr} >= {literal}'{pd.Timestamp(start).date()}'"
    if end:
        predicate += f" AND {date_expr} <= {literal}'{pd.Timestamp(end).date()}'"
    return (
        "SELECT " + ", ".join(identifier(c) for c in columns)
        + f" FROM {identifier(database)}.{identifier(table)} WHERE {predicate}"
        + ' ORDER BY "week_date", "ticker"'
    )


def load_athena_panel(
    *, database: str = DEFAULT_DATABASE, table: str = DEFAULT_TABLE,
    catalog: str = "AwsDataCatalog", start: str = "2017-01-01",
    end: str | None = None, profile: str | None = "admin",
    region: str = "ap-southeast-2", workgroup: str = "primary",
    output_location: str | None = DEFAULT_OUTPUT,
    cache_dir: Path = Path("local_cache"), refresh: bool = False,
    timeout_seconds: int = 900,
) -> tuple[pd.DataFrame, dict]:
    query = panel_query(database, table, start, end)
    request_identity = dict(
        query=query, catalog=catalog, region=region, workgroup=workgroup,
        profile=profile,
    )
    key = hashlib.sha256(json.dumps(request_identity, sort_keys=True).encode()).hexdigest()[:16]
    cache_path = cache_dir / f"athena_panel_{key}.parquet"
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.exists() and metadata_path.exists() and not refresh:
        LOGGER.info("Loading Athena query cache %s (use --refresh-data to query again)", cache_path)
        return pd.read_parquet(cache_path), json.loads(metadata_path.read_text())

    session = boto3.Session(profile_name=profile, region_name=region)
    athena = session.client("athena")
    request = dict(
        QueryString=query,
        QueryExecutionContext={"Catalog": catalog, "Database": database},
        WorkGroup=workgroup,
    )
    config = athena.get_work_group(WorkGroup=workgroup)["WorkGroup"]["Configuration"]
    if not config.get("ManagedQueryResultsConfiguration", {}).get("Enabled"):
        if output_location:
            request["ResultConfiguration"] = {"OutputLocation": output_location}
    query_id = athena.start_query_execution(**request)["QueryExecutionId"]
    LOGGER.info("Athena query %s: loading %s.%s from %s", query_id, database, table, start)
    deadline = time.monotonic() + timeout_seconds
    while True:
        execution = athena.get_query_execution(QueryExecutionId=query_id)["QueryExecution"]
        state = execution["Status"]["State"]
        if state == "SUCCEEDED":
            break
        if state in {"FAILED", "CANCELLED"}:
            raise RuntimeError(f"Athena {query_id} {state}: {execution['Status'].get('StateChangeReason')}")
        if time.monotonic() >= deadline:
            athena.stop_query_execution(QueryExecutionId=query_id)
            raise TimeoutError(f"Stopped Athena query {query_id} after {timeout_seconds}s")
        time.sleep(2)

    location = execution.get("ResultConfiguration", {}).get("OutputLocation")
    if location:
        uri = urlparse(location)
        response = session.client("s3").get_object(Bucket=uri.netloc, Key=uri.path.lstrip("/"))
        with response["Body"] as body:
            panel = pd.read_csv(io.BytesIO(body.read()), dtype={"ticker": "string", "sector": "string"})
    else:
        # Athena-managed results have no customer S3 result location.
        rows, names = [], None
        first = True
        for page in athena.get_paginator("get_query_results").paginate(QueryExecutionId=query_id):
            if names is None:
                names = [c["Name"] for c in page["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]]
            for row in page["ResultSet"]["Rows"]:
                if first:
                    first = False  # SELECT results include a single header row.
                    continue
                rows.append([cell.get("VarCharValue") for cell in row["Data"]])
        panel = pd.DataFrame(rows, columns=names)

    metadata = {
        **request_identity, "query_execution_id": query_id,
        "result_location": location,
        "data_scanned_bytes": execution.get("Statistics", {}).get("DataScannedInBytes"),
        "loaded_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "rows": len(panel),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(cache_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    LOGGER.info("Cached %s Athena rows in %s", f"{len(panel):,}", cache_path)
    return panel, metadata
