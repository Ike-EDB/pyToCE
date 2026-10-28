"""Persist compact package provider and foreground databases.

This module inserts compact package databases in batches within one transaction.
Database processing follows insertion, and scientific identity metadata is
published after processing.
"""

import pickle
import sqlite3
from collections.abc import Iterable, Mapping
from typing import Any, cast


def database_matches(
    bd: Any,
    database_name: str,
    expected_metadata: Mapping[str, object],
) -> bool:
    """Return whether an existing package database has the expected metadata."""

    return database_name in bd.databases and all(
        bd.databases[database_name].get(key) == value for key, value in expected_metadata.items()
    )


def _write_brightway_exchanges(
    data: Mapping[tuple[str, str], Mapping[str, object]],
) -> None:
    """Insert admitted package exchange records in one transaction.

    The compact writer has inserted the activity rows inside its encompassing
    transaction. Rebind exchange outputs to their receiving nodes; processing and completion
    remain with that writer.
    """
    from bw2data.backends.schema import ExchangeDataset  # type: ignore[import-untyped]

    rows: list[tuple[object, ...]] = []
    for (database_name, code), dataset in data.items():
        for source in cast(Iterable[Mapping[str, object]], dataset.get("exchanges", ())):
            exchange = {**source, "output": (database_name, code)}
            input_database, input_code = cast(tuple[str, str], exchange["input"])
            rows.append(
                (
                    sqlite3.Binary(pickle.dumps(exchange, protocol=4)),
                    str(input_code),
                    str(input_database),
                    code,
                    database_name,
                    str(exchange["type"]),
                )
            )
    connection = ExchangeDataset._meta.database.connection()
    with ExchangeDataset._meta.database.atomic():
        connection.executemany(
            "INSERT INTO exchangedataset "
            "(data,input_code,input_database,output_code,output_database,type) "
            "VALUES (?,?,?,?,?,?)",
            rows,
        )


def write_brightway_database(
    bd: Any,
    database_name: str,
    data: dict[tuple[str, str], dict[str, object]],
    metadata: Mapping[str, object],
) -> None:
    """Write and process one compact package-owned database.

    The caller supplies coalesced datasets and their scientific identity metadata.
    Metadata is published after processing succeeds.
    """

    from bw2data.backends.schema import ActivityDataset  # type: ignore[import-untyped]
    from bw2data.snowflake_ids import snowflake_id_generator  # type: ignore[import-untyped]

    if database_name in bd.databases:
        raise RuntimeError(
            f"Brightway database {database_name!r} already exists and cannot be replaced. "
            "Leave this workspace unchanged and create a new workspace."
        )
    database = bd.Database(database_name)
    database.register(
        write_empty=False,
        searchable=False,
    )
    activity_rows: list[tuple[object, ...]] = []
    for (dataset_database, code), dataset in data.items():
        activity = {key: value for key, value in dataset.items() if key != "exchanges"}
        activity["database"] = dataset_database
        activity["code"] = code
        activity_rows.append(
            (
                next(snowflake_id_generator),
                sqlite3.Binary(pickle.dumps(activity, protocol=4)),
                code,
                dataset_database,
                str(activity.get("location", "")),
                str(activity.get("name", "")),
                str(activity.get("reference product", "")),
                str(activity.get("type", "process")),
            )
        )
    activity_sql = (
        "INSERT INTO activitydataset "
        "(id,data,code,database,location,name,product,type) "
        "VALUES (?,?,?,?,?,?,?,?)"
    )
    connection = ActivityDataset._meta.database.connection()
    with ActivityDataset._meta.database.atomic():
        connection.executemany(activity_sql, activity_rows)
        _write_brightway_exchanges(data)
    bd.databases[database_name]["number"] = len(activity_rows)
    bd.databases.set_modified(database_name)
    bd.databases.flush()
    database.process()
    bd.databases[database_name].update(metadata)
    bd.databases.flush()
