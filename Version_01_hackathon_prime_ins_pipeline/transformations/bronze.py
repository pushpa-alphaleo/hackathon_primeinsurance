import dlt
from pyspark.sql import functions as F
from functools import reduce

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
BASE_PATH = ""
CKPT_PATH = "/Volumes/primeinsurance_analytics/source_files/checkpoints"
REGION_RE = r"(Insurance[ _]\d+)"


def _base_read_files(path_glob: str, fmt: str, checkpoint_name: str):
    """
    DLT read_files SETUP:
    ─────────────────────────────────────────────────────────────
    Uses dlt.read_files() instead of spark.readStream.format("cloudFiles")
    because the source data lives in a Unity Catalog MANAGED volume.

    WHY NOT cloudFiles?
      Auto Loader (cloudFiles) throws LOCATION_OVERLAP when the
      source data is in a UC managed volume — it cannot access
      managed volume storage directly via S3 path.

    WHY dlt.read_files()?
      dlt.read_files() is the DLT-native API built specifically
      to work with Unity Catalog managed and external volumes.
      It handles incremental file ingestion the same way Auto
      Loader does — only new files are picked up on each run.

    format               → file format to read (csv / json)
    header               → first row is column header (csv only)
    inferColumnTypes     → auto-detect column types
    schemaEvolutionMode  → addNewColumns: new columns in new files
                           are added automatically without failing
    schemaLocation       → where DLT stores the inferred schema
                           between runs (on dbfs to avoid overlap)

    ** glob              → recursive wildcard picks up files from
                           all Insurance subfolders and root folder
    ─────────────────────────────────────────────────────────────
    """
    options = {
        "inferColumnTypes"   : "true",
        "schemaEvolutionMode": "addNewColumns",
        "schemaLocation"     : f"{CKPT_PATH}/{checkpoint_name}",
    }
    if fmt == "csv":
        options["header"] = "true"

    return (
        dlt.read_files(                                              # <- DLT native API for UC managed volumes
            f"{BASE_PATH}/{path_glob}",
            format=fmt,
            **options
        )
        .withColumn("_source_file",
                    F.col("_metadata.file_path"))                   # <- exact file path per row
        .withColumn("_ingest_time",
                    F.current_timestamp())                          # <- when this row was ingested
        .withColumn("_source_region",
                    F.regexp_extract(F.col("_metadata.file_path"),
                                     REGION_RE, 0))                 # <- e.g. "Insurance 3"
    )


# =============================================================
# BRONZE TABLES
# Schema: bronze
# =============================================================

@dlt.table(
    name             = "bronze.bronze_customers",
    comment          = "Bronze: Raw customer records from all regions. Schema preserved exactly as received.",
    table_properties = {"quality": "bronze", "pipelines.autoOptimize.managed": "true"},
)
def bronze_customers():
    # Picks up: customers_1.csv ... customers_7.csv from all subfolders
    return _base_read_files("**/customers_*.csv", "csv", "customers")


@dlt.table(
    name             = "bronze.bronze_sales",
    comment          = "Bronze: Raw sales records from all regions. Schema preserved exactly as received.",
    table_properties = {"quality": "bronze", "pipelines.autoOptimize.managed": "true"},
)
def bronze_sales():
    # Picks up: Sales_2.csv, sales_1.csv, sales_4.csv
    # *[Ss]ales* handles both uppercase S and lowercase s
    return _base_read_files("**/*[Ss]ales*.csv", "csv", "sales")


@dlt.table(
    name             = "bronze.bronze_claims",
    comment          = "Bronze: Raw claims records from all regions. JSON source.",
    table_properties = {"quality": "bronze", "pipelines.autoOptimize.managed": "true"},
)
def bronze_claims():
    # Picks up: claims_1.json (Insurance 6) + claims_2.json (root)
    return _base_read_files("**/claims_*.json", "json", "claims")


@dlt.table(
    name             = "bronze.bronze_cars",
    comment          = "Bronze: Raw vehicle records from all regions. Schema preserved exactly as received.",
    table_properties = {"quality": "bronze", "pipelines.autoOptimize.managed": "true"},
)
def bronze_cars():
    # Picks up: cars.csv (Insurance 4 only)
    return _base_read_files("**/cars*.csv", "csv", "cars")


@dlt.table(
    name             = "bronze.bronze_policy",
    comment          = "Bronze: Raw policy records from all regions. Schema preserved exactly as received.",
    table_properties = {"quality": "bronze", "pipelines.autoOptimize.managed": "true"},
)
def bronze_policy():
    # Picks up: policy.csv (Insurance 5 only)
    return _base_read_files("**/policy*.csv", "csv", "policy")


# =============================================================
# FILE PROCESSING LOG
# Tracks every file processed across all 5 bronze tables.
# One row per file per entity — human-readable audit log.
# =============================================================

def _file_log_for(table_name: str, entity: str):
    return (
        dlt.read_stream(table_name)
        .groupBy("_source_file", "_source_region")
        .agg(
            F.count("*").alias("row_count"),
            F.min("_ingest_time").alias("first_ingested_at"),
            F.max("_ingest_time").alias("last_seen_at"),
        )
        .withColumn("entity",      F.lit(entity))
        .withColumn("file_name",
            F.regexp_extract(F.col("_source_file"), r"([^/]+)$", 1))
        .withColumn("file_format",
            F.regexp_extract(F.col("_source_file"), r"\.(\w+)$", 1))
        .withColumn("status",      F.lit("processed"))
        .select(
            "entity", "file_name", "file_format",
            "_source_file", "_source_region",
            "row_count", "first_ingested_at", "last_seen_at", "status",
        )
    )


@dlt.table(
    name    = "bronze.bronze_file_log",
    comment = "Bronze: Audit log of every file processed across all bronze tables.",
    table_properties = {"quality": "bronze", "pipelines.autoOptimize.managed": "true"},
)
def bronze_file_log():
    logs = [
        _file_log_for("bronze.bronze_customers", "customers"),
        _file_log_for("bronze.bronze_sales",     "sales"),
        _file_log_for("bronze.bronze_claims",    "claims"),
        _file_log_for("bronze.bronze_cars",      "cars"),
        _file_log_for("bronze.bronze_policy",    "policy"),
    ]
    return reduce(lambda a, b: a.union(b), logs)