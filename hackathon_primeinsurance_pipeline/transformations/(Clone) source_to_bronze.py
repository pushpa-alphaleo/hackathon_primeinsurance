import dlt
from pyspark.sql import functions as F

BASE        = "/Volumes/primeinsurance_analytics/source_files/raw_files"
SCHEMA_BASE = "/Volumes/primeinsurance_analytics/source_files/checkpoints"

# ─────────────────────────────────────────────────────────────────
# BRONZE TABLE: customers
# ─────────────────────────────────────────────────────────────────
@dlt.table(
    name    = "bronze.bronze_customers",
    comment = "Bronze: Raw customer records from all regions.",
    table_properties = {
        "quality"                        : "bronze",
        "pipelines.autoOptimize.managed" : "true"
    }
)
def bronze_customers():
    df = (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format",             "csv")
        .option("cloudFiles.schemaLocation",     f"{SCHEMA_BASE}/customers")
        .option("cloudFiles.schemaEvolutionMode","addNewColumns")
        .option("cloudFiles.inferColumnTypes",   "false")  # ← ALL strings in bronze
        .option("header",                        "true")
        .option("rescuedDataColumn",             "_rescued_data")
        .load(f"{BASE}/**/customers_*.csv")
        .withColumn("_source_file",   F.col("_metadata.file_path"))
        .withColumn("_ingest_time",   F.current_timestamp())
        .withColumn("_source_region", F.regexp_extract(
            F.col("_metadata.file_path"), r"(Insurance[_ ]\d+)", 0
        ))
    )
    data_cols = [c for c in df.columns if not c.startswith("_")]
    return df.filter(F.greatest(*[F.col(c).isNotNull() for c in data_cols]))

# ─────────────────────────────────────────────────────────────────
# BRONZE TABLE: sales
# ─────────────────────────────────────────────────────────────────
@dlt.table(
    name    = "bronze.bronze_sales",
    comment = "Bronze: Raw sales records from all regions.",
    table_properties = {
        "quality"                        : "bronze",
        "pipelines.autoOptimize.managed" : "true"
    }
)
def bronze_sales():
    df = (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format",             "csv")
        .option("cloudFiles.schemaLocation",     f"{SCHEMA_BASE}/sales")
        .option("cloudFiles.schemaEvolutionMode","addNewColumns")
        .option("cloudFiles.inferColumnTypes",   "false")  # ← ALL strings in bronze
        .option("header",                        "true")
        .option("rescuedDataColumn",             "_rescued_data")
        .load(f"{BASE}/**/sales_*.csv")
        .withColumn("_source_file",   F.col("_metadata.file_path"))
        .withColumn("_ingest_time",   F.current_timestamp())
        .withColumn("_source_region", F.regexp_extract(
            F.col("_metadata.file_path"), r"(Insurance[_ ]\d+)", 0
        ))
    )
    data_cols = [c for c in df.columns if not c.startswith("_")]
    return df.filter(F.greatest(*[F.col(c).isNotNull() for c in data_cols]))

# ─────────────────────────────────────────────────────────────────
# BRONZE TABLE: claims
# ─────────────────────────────────────────────────────────────────
@dlt.table(
    name    = "bronze.bronze_claims",
    comment = "Bronze: Raw claims records from all regions. JSON source.",
    table_properties = {
        "quality"                        : "bronze",
        "pipelines.autoOptimize.managed" : "true"
    }
)
def bronze_claims():
    df = (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format",             "json")
        .option("cloudFiles.schemaLocation",     f"{SCHEMA_BASE}/claims")
        .option("cloudFiles.schemaEvolutionMode","addNewColumns")
        .option("cloudFiles.inferColumnTypes",   "false")  # ← ALL strings in bronze
        .option("rescuedDataColumn",             "_rescued_data")
        .load(f"{BASE}/**/claims_*.json")
        .withColumn("_source_file",   F.col("_metadata.file_path"))
        .withColumn("_ingest_time",   F.current_timestamp())
        .withColumn("_source_region", F.regexp_extract(
            F.col("_metadata.file_path"), r"(Insurance[_ ]\d+)", 0
        ))
    )
    data_cols = [c for c in df.columns if not c.startswith("_")]
    return df.filter(F.greatest(*[F.col(c).isNotNull() for c in data_cols]))

# ─────────────────────────────────────────────────────────────────
# BRONZE TABLE: cars
# ─────────────────────────────────────────────────────────────────
@dlt.table(
    name    = "bronze.bronze_cars",
    comment = "Bronze: Raw vehicle records from all regions.",
    table_properties = {
        "quality"                        : "bronze",
        "pipelines.autoOptimize.managed" : "true"
    }
)
def bronze_cars():
    df = (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format",             "csv")
        .option("cloudFiles.schemaLocation",     f"{SCHEMA_BASE}/cars")
        .option("cloudFiles.schemaEvolutionMode","addNewColumns")
        .option("cloudFiles.inferColumnTypes",   "false")  # ← ALL strings in bronze
        .option("header",                        "true")
        .option("rescuedDataColumn",             "_rescued_data")
        .load(f"{BASE}/**/cars*.csv")
        .withColumn("_source_file",   F.col("_metadata.file_path"))
        .withColumn("_ingest_time",   F.current_timestamp())
        .withColumn("_source_region", F.regexp_extract(
            F.col("_metadata.file_path"), r"(Insurance[_ ]\d+)", 0
        ))
    )
    data_cols = [c for c in df.columns if not c.startswith("_")]
    return df.filter(F.greatest(*[F.col(c).isNotNull() for c in data_cols]))

# ─────────────────────────────────────────────────────────────────
# BRONZE TABLE: policy
# ─────────────────────────────────────────────────────────────────
@dlt.table(
    name    = "bronze.bronze_policy",
    comment = "Bronze: Raw policy records from all regions.",
    table_properties = {
        "quality"                        : "bronze",
        "pipelines.autoOptimize.managed" : "true"
    }
)
def bronze_policy():
    df = (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format",             "csv")
        .option("cloudFiles.schemaLocation",     f"{SCHEMA_BASE}/policy")
        .option("cloudFiles.schemaEvolutionMode","addNewColumns")
        .option("cloudFiles.inferColumnTypes",   "false")  # ← ALL strings in bronze
        .option("header",                        "true")
        .option("rescuedDataColumn",             "_rescued_data")
        .load(f"{BASE}/**/policy*.csv")
        .withColumn("_source_file",   F.col("_metadata.file_path"))
        .withColumn("_ingest_time",   F.current_timestamp())
        .withColumn("_source_region", F.regexp_extract(
            F.col("_metadata.file_path"), r"(Insurance[_ ]\d+)", 0
        ))
    )
    data_cols = [c for c in df.columns if not c.startswith("_")]
    return df.filter(F.greatest(*[F.col(c).isNotNull() for c in data_cols]))