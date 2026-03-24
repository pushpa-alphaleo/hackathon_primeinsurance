import dlt
from pyspark.sql import functions as F

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# BASE       — where all source files live (Insurance_1 ... Insurance_6)
# SCHEMA_BASE — where Auto Loader stores its schema per entity
#               Each entity MUST have its own schemaLocation —
#               sharing one location across tables corrupts schema tracking
# ─────────────────────────────────────────────────────────────────

BASE        = "/Volumes/primeinsurance_analytics/source_files/raw_files"
SCHEMA_BASE = "/Volumes/primeinsurance_analytics/source_files/checkpoints/bronze"


# ─────────────────────────────────────────────────────────────────
# WHY THESE OPTIONS ON EVERY TABLE:
#
# cloudFiles.schemaLocation     — Auto Loader saves inferred schema here.
#                                  Without it, schema is re-inferred every
#                                  run, which can cause type conflicts.
#
# cloudFiles.schemaEvolutionMode = addNewColumns
#                                — When a new file arrives with a new column
#                                  (e.g. "Birth_Date"), Auto Loader widens
#                                  the table schema automatically instead of
#                                  failing. Bronze must accept all columns
#                                  exactly as received.
#
# cloudFiles.inferColumnTypes   — Infer proper types (int, date etc.)
#                                  instead of loading everything as string.
#
# _metadata.file_path           — DLT-safe way to get source file path.
#                                  input_file_name() returns empty string
#                                  inside DLT — _metadata.file_path works.
#
# _source_region                — Extracted from path for traceability.
#                                  Tells you which Insurance folder the
#                                  record came from (Insurance_1 ... _6).
# ─────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────
# BRONZE TABLE: customers
#
# Sources  : customers_*.csv across all Insurance_1 to Insurance_6 folders
# Key facts: 7 files with inconsistent schemas across regions —
#            Reg vs Region, Marital_status vs Marital, missing columns etc.
#            addNewColumns handles this — schema widens automatically.
#            All 7 files land in ONE table with NULL for missing columns.
# ─────────────────────────────────────────────────────────────────

@dlt.table(
    name    = "bronze_customers",
    comment = "Bronze: Raw customer records from all regions. Schema preserved exactly as received.",
    table_properties = {
        "quality"                        : "bronze",
        "pipelines.autoOptimize.managed" : "true"
    }
)
def bronze_customers():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format",            "csv")
        .option("cloudFiles.schemaLocation",    f"{SCHEMA_BASE}/customers")  # entity-specific schema folder
        .option("cloudFiles.schemaEvolutionMode","addNewColumns")             # handles new columns like "Birth Date"
        .option("cloudFiles.inferColumnTypes",  "true")                      # infer int/date etc. not just string
        .option("header",                       "true")
        .option("rescuedDataColumn",            "_rescued_data")             # captures unparseable values safely
        .load(f"{BASE}/**/customers_*.csv")
        .withColumn("_source_file",   F.col("_metadata.file_path"))          # which file this row came from
        .withColumn("_ingest_time",   F.current_timestamp())                 # when it was loaded
        .withColumn("_source_region", F.regexp_extract(                      # which Insurance folder
            F.col("_metadata.file_path"), r"(Insurance[_ ]\d+)", 0
        ))
    )


# ─────────────────────────────────────────────────────────────────
# BRONZE TABLE: sales
#
# Sources  : sales_*.csv across all Insurance folders
# Key facts: Multiple region files, different column counts possible.
#            addNewColumns ensures pipeline never breaks on schema drift.
# ─────────────────────────────────────────────────────────────────

@dlt.table(
    name    = "bronze_sales",
    comment = "Bronze: Raw sales records from all regions. Schema preserved exactly as received.",
    table_properties = {
        "quality"                        : "bronze",
        "pipelines.autoOptimize.managed" : "true"
    }
)
def bronze_sales():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format",             "csv")
        .option("cloudFiles.schemaLocation",     f"{SCHEMA_BASE}/sales")
        .option("cloudFiles.schemaEvolutionMode","addNewColumns")
        .option("cloudFiles.inferColumnTypes",   "true")
        .option("header",                        "true")
        .option("rescuedDataColumn",             "_rescued_data")
        .load(f"{BASE}/**/sales_*.csv")
        .withColumn("_source_file",   F.col("_metadata.file_path"))
        .withColumn("_ingest_time",   F.current_timestamp())
        .withColumn("_source_region", F.regexp_extract(
            F.col("_metadata.file_path"), r"(Insurance[_ ]\d+)", 0
        ))
    )


# ─────────────────────────────────────────────────────────────────
# BRONZE TABLE: claims
#
# Sources  : claims_*.json — NOTE: JSON format, not CSV
# Key facts: JSON files have nested/flexible structure.
#            Auto Loader handles JSON natively.
#            addNewColumns handles any new JSON fields added later.
# ─────────────────────────────────────────────────────────────────

@dlt.table(
    name    = "bronze_claims",
    comment = "Bronze: Raw claims records from all regions. JSON source. Schema preserved exactly as received.",
    table_properties = {
        "quality"                        : "bronze",
        "pipelines.autoOptimize.managed" : "true"
    }
)
def bronze_claims():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format",             "json")                     # JSON — not CSV
        .option("cloudFiles.schemaLocation",     f"{SCHEMA_BASE}/claims")
        .option("cloudFiles.schemaEvolutionMode","addNewColumns")
        .option("cloudFiles.inferColumnTypes",   "true")
        .option("rescuedDataColumn",             "_rescued_data")
        .load(f"{BASE}/**/claims_*.json")
        .withColumn("_source_file",   F.col("_metadata.file_path"))
        .withColumn("_ingest_time",   F.current_timestamp())
        .withColumn("_source_region", F.regexp_extract(
            F.col("_metadata.file_path"), r"(Insurance[_ ]\d+)", 0
        ))
    )


# ─────────────────────────────────────────────────────────────────
# BRONZE TABLE: cars
#
# Sources  : cars*.csv across all Insurance folders
# Key facts: Single source entity — car specs per vehicle.
#            addNewColumns protects against future spec columns.
# ─────────────────────────────────────────────────────────────────

@dlt.table(
    name    = "bronze_cars",
    comment = "Bronze: Raw vehicle records from all regions. Schema preserved exactly as received.",
    table_properties = {
        "quality"                        : "bronze",
        "pipelines.autoOptimize.managed" : "true"
    }
)
def bronze_cars():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format",             "csv")
        .option("cloudFiles.schemaLocation",     f"{SCHEMA_BASE}/cars")
        .option("cloudFiles.schemaEvolutionMode","addNewColumns")
        .option("cloudFiles.inferColumnTypes",   "true")
        .option("header",                        "true")
        .option("rescuedDataColumn",             "_rescued_data")
        .load(f"{BASE}/**/cars*.csv")
        .withColumn("_source_file",   F.col("_metadata.file_path"))
        .withColumn("_ingest_time",   F.current_timestamp())
        .withColumn("_source_region", F.regexp_extract(
            F.col("_metadata.file_path"), r"(Insurance[_ ]\d+)", 0
        ))
    )


# ─────────────────────────────────────────────────────────────────
# BRONZE TABLE: policy
#
# Sources  : policy*.csv across all Insurance folders
# Key facts: Policy data links customers to cars.
#            Numeric columns like premium must not be silently miscast —
#            inferColumnTypes handles this at source.
# ─────────────────────────────────────────────────────────────────

@dlt.table(
    name    = "bronze_policy",
    comment = "Bronze: Raw policy records from all regions. Schema preserved exactly as received.",
    table_properties = {
        "quality"                        : "bronze",
        "pipelines.autoOptimize.managed" : "true"
    }
)
def bronze_policy():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format",             "csv")
        .option("cloudFiles.schemaLocation",     f"{SCHEMA_BASE}/policy")
        .option("cloudFiles.schemaEvolutionMode","addNewColumns")
        .option("cloudFiles.inferColumnTypes",   "true")
        .option("header",                        "true")
        .option("rescuedDataColumn",             "_rescued_data")
        .load(f"{BASE}/**/policy*.csv")
        .withColumn("_source_file",   F.col("_metadata.file_path"))
        .withColumn("_ingest_time",   F.current_timestamp())
        .withColumn("_source_region", F.regexp_extract(
            F.col("_metadata.file_path"), r"(Insurance[_ ]\d+)", 0
        ))
    )