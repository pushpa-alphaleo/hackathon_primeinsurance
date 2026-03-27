"""
bronze_to_silver_utils.py
==========================
Reusable data quality and transformation functions for the DLT silver layer.

HOW TO USE IN DATABRICKS:
  1. Upload this file to your Databricks workspace (same folder as silver_pipeline.py)
  2. In your DLT pipeline settings → "Libraries", add this file path
  3. silver_pipeline.py imports from it directly:
       from bronze_to_silver_utils import deduplicate_by_key, coalesce_columns, ...

WHY A SEPARATE FILE?
  Keeping utility functions here means silver_pipeline.py stays readable —
  it only contains the pipeline logic (@dlt.table definitions).
  These functions have NO DLT dependency — they work on plain PySpark DataFrames.
"""

from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql.window import Window


# ========================================
# DATA QUALITY UTILITIES
# ========================================

def deduplicate_by_key(df: DataFrame, key_column: str, order_column: str = "_ingest_time") -> DataFrame:
    """
    Deduplicate a DataFrame by a key column.
    Keeps one row per unique key value (arbitrary row — no ordering guarantee
    in streaming mode, which is why dropDuplicates is used over window functions).

    Args:
        df: Input DataFrame
        key_column: Column to deduplicate on (e.g., "customer_id")
        order_column: Not used in current implementation — reserved for future
                      batch-mode dedup with ordering (e.g., keep latest row)

    Example:
        df = deduplicate_by_key(df, "customer_id")
    """
    return df.dropDuplicates([key_column])


def replace_question_marks_with_null(df: DataFrame, columns: list) -> DataFrame:
    """
    Replace '?' with NULL in specified columns.
    Some source files use '?' as a placeholder for unknown values instead of
    leaving the field empty — this converts them to proper NULLs.

    Args:
        df: Input DataFrame
        columns: List of column names to clean

    Example:
        df = replace_question_marks_with_null(df, ["collision_type", "witnesses"])
        # "?" -> NULL, "rear collision" -> "rear collision" (unchanged)
    """
    for col in columns:
        df = df.withColumn(col,
            F.when(F.col(col) == "?", None)
            .otherwise(F.col(col))
        )
    return df


def replace_null_strings(df: DataFrame, columns: list) -> DataFrame:
    """
    Replace string "NULL" with actual NULL in specified columns.
    Some source files write the word "NULL" as text instead of leaving
    the field empty — this converts them to proper NULLs.

    Args:
        df: Input DataFrame
        columns: List of column names to clean

    Example:
        df = replace_null_strings(df, ["claim_processed_on"])
        # "NULL" -> NULL, "2017-03-15" -> "2017-03-15" (unchanged)
    """
    for col in columns:
        df = df.withColumn(col,
            F.when(F.upper(F.col(col)) == "NULL", None)
            .otherwise(F.col(col))
        )
    return df


def replace_na_strings_with_null(df: DataFrame, columns: list) -> DataFrame:
    """
    Replace string "NA" with actual NULL in specified columns.
    PySpark does NOT auto-convert "NA" strings to null like pandas does —
    this function handles that explicitly.
    Found in: education, job, marital_status columns across customer files.

    Args:
        df: Input DataFrame
        columns: List of column names to clean

    Example:
        df = replace_na_strings_with_null(df, ["education", "job", "marital_status"])
        # "NA" -> NULL, "secondary" -> "secondary" (unchanged)
    """
    for col in columns:
        df = df.withColumn(col,
            F.when(F.trim(F.col(col)) == "NA", None)
            .otherwise(F.col(col))
        )
    return df


def standardize_to_lowercase(df: DataFrame, columns: list) -> DataFrame:
    """
    Convert specified columns to lowercase.
    Used for categorical columns like education, marital_status, job, fuel, transmission.

    Args:
        df: Input DataFrame
        columns: List of column names to lowercase

    Example:
        df = standardize_to_lowercase(df, ["marital_status", "job", "fuel"])
        # "Married" -> "married", "Management" -> "management"
    """
    for col in columns:
        df = df.withColumn(col, F.lower(F.col(col)))
    return df


def standardize_to_uppercase(df: DataFrame, columns: list) -> DataFrame:
    """
    Convert specified columns to uppercase.
    Used for state/region codes that should be consistent (e.g., "OH", "CA").

    Args:
        df: Input DataFrame
        columns: List of column names to uppercase

    Example:
        df = standardize_to_uppercase(df, ["state", "incident_state"])
        # "oh" -> "OH", "ca" -> "CA"
    """
    for col in columns:
        df = df.withColumn(col, F.upper(F.col(col)))
    return df


def lowercase_all_column_names(df: DataFrame) -> DataFrame:
    """
    Lowercase ALL column names in a DataFrame.

    IMPORTANT: Must be called BEFORE coalesce_columns() because column name
    matching in PySpark is case-sensitive — 'CustomerID' != 'customerid'.
    Bronze files arrive with mixed-case column names like 'CustomerID', 'Reg',
    'Marital_status'. Lowercasing first makes all subsequent lookups reliable.

    Args:
        df: Input DataFrame

    Example:
        df = lowercase_all_column_names(df)
        # 'CustomerID' -> 'customerid'
        # 'Reg'        -> 'reg'
        # 'BALANCE'    -> 'balance'
    """
    return df.toDF(*[c.lower() for c in df.columns])


def coalesce_columns(df: DataFrame, target_col: str, source_cols: list, cast_type: str = None) -> DataFrame:
    """
    Coalesce multiple columns into one, taking the first non-null value.

    WHY THIS EXISTS:
      The 7 customer files use different column names for the same field:
        customers_1.csv → 'Reg'         (for region)
        customers_2.csv → 'Region'      (for region)
        customers_3.csv → 'cust_id'     (for customer_id)
      After Auto Loader widens the schema, both 'reg' and 'region' exist
      but only one is populated per row. This function merges them.

    Only includes source columns that actually exist in the DataFrame —
    safely handles files that don't have all columns without errors.

    Args:
        df: Input DataFrame (column names must already be lowercased)
        target_col: Name of output column
        source_cols: List of candidate columns to coalesce (priority order — first non-null wins)
        cast_type: Optional type to cast result to (e.g., "int", "string", "double")

    Example:
        df = coalesce_columns(df, "customer_id", ["customerid", "customer_id", "cust_id"], "int")
        df = coalesce_columns(df, "region",      ["reg", "region"])
    """
    existing = [c for c in source_cols if c in df.columns]

    if not existing:
        # None of the candidate columns exist — add a null column as placeholder
        return df.withColumn(target_col, F.lit(None))

    result = F.coalesce(*[F.col(c) for c in existing])

    if cast_type:
        result = result.cast(cast_type)

    return df.withColumn(target_col, result)


def fix_swapped_columns(df: DataFrame, col_a: str, col_b: str, source_file_col: str, source_file_keyword: str) -> DataFrame:
    """
    Fix two columns that are swapped in a specific source file.
    For all other source files, columns are left unchanged.

    BACKGROUND:
      customers_6.csv has Education and Marital_status accidentally swapped —
      the Education column contains marital values and vice versa.
      This function detects rows from that file and swaps the values back.
      Must be called BEFORE coalesce_columns() so values land in correct columns.

    Args:
        df: Input DataFrame
        col_a: First column name (e.g., "education")
        col_b: Second column name (e.g., "marital_status")
        source_file_col: Column that identifies the source file (e.g., "_source_file")
        source_file_keyword: Keyword to match the affected file path (e.g., "customers_6")

    Example:
        df = fix_swapped_columns(df, "education", "marital_status", "_source_file", "customers_6")
        # Rows from customers_6: education and marital_status values are swapped back
        # All other rows: untouched
    """
    is_affected = F.col(source_file_col).contains(source_file_keyword)
    df = (
        df
        .withColumn("_temp_a",
            F.when(is_affected, F.col(col_b)).otherwise(F.col(col_a))
        )
        .withColumn("_temp_b",
            F.when(is_affected, F.col(col_a)).otherwise(F.col(col_b))
        )
        .drop(col_a, col_b)
        .withColumnRenamed("_temp_a", col_a)
        .withColumnRenamed("_temp_b", col_b)
    )
    return df


def map_region_abbreviations(df: DataFrame, col_name: str = "region") -> DataFrame:
    """
    Map single-letter region codes to full region names.

    BACKGROUND:
      customers_5.csv uses abbreviations C/W/S/E instead of
      Central/West/South/East used by other files.
      Safe to run on ALL rows — the condition only fires on length-1 values,
      so rows from other files with full names like "West" are untouched.

    Args:
        df: Input DataFrame
        col_name: Name of the region column (default: "region")

    Example:
        df = map_region_abbreviations(df, "region")
        # 'W' -> 'West',    'E' -> 'East'
        # 'S' -> 'South',   'C' -> 'Central'
        # 'West' -> 'West'  (unchanged — length > 1)
    """
    region_map = F.create_map(
        F.lit("W"), F.lit("West"),
        F.lit("E"), F.lit("East"),
        F.lit("S"), F.lit("South"),
        F.lit("C"), F.lit("Central"),
    )
    return df.withColumn(col_name,
        F.when(F.length(F.col(col_name)) == 1, region_map[F.col(col_name)])
        .otherwise(F.col(col_name))
    )


def remove_trailing_dot(df: DataFrame, col_name: str) -> DataFrame:
    """
    Remove trailing dot from string values in a column.
    The Job column in customer files contains 'admin.' with a trailing dot
    which creates a duplicate category alongside 'admin'.

    Args:
        df: Input DataFrame
        col_name: Column to clean

    Example:
        df = remove_trailing_dot(df, "job")
        # 'admin.'     -> 'admin'
        # 'management' -> 'management' (unchanged)
    """
    return df.withColumn(col_name,
        F.regexp_replace(F.col(col_name), r"\.$", "")
    )


def add_negative_value_flag(df: DataFrame, col_name: str, flag_col_name: str) -> DataFrame:
    """
    Add a binary flag column indicating whether a numeric column has a negative value.
    The row is KEPT in silver — this only adds a flag for downstream analysis.
    Used for balance column which can validly be negative (overdraft accounts).

    Args:
        df: Input DataFrame
        col_name: Numeric column to check
        flag_col_name: Name of the output flag column (1=negative, 0=non-negative)

    Example:
        df = add_negative_value_flag(df, "balance", "balance_negative_flag")
        # balance=-500  -> balance_negative_flag=1
        # balance=2000  -> balance_negative_flag=0
    """
    return df.withColumn(flag_col_name,
        F.when(F.col(col_name) < 0, 1).otherwise(0)
    )


def add_outlier_flag(df: DataFrame, col_name: str, flag_col_name: str, threshold: float) -> DataFrame:
    """
    Add a binary flag column for values exceeding a threshold.
    The row is KEPT in silver — this only flags it for downstream review.
    Used for km_driven where 1 car has 1,500,000 km (threshold = 500,000).

    Args:
        df: Input DataFrame
        col_name: Numeric column to check
        flag_col_name: Name of the output flag column (1=outlier, 0=normal)
        threshold: Value above which the flag is set to 1

    Example:
        df = add_outlier_flag(df, "km_driven", "km_driven_outlier_flag", 500000)
        # km_driven=1500000 -> km_driven_outlier_flag=1
        # km_driven=45000   -> km_driven_outlier_flag=0
    """
    return df.withColumn(flag_col_name,
        F.when(F.col(col_name) > threshold, 1).otherwise(0)
    )


def fix_negative_to_null(df: DataFrame, col_name: str) -> DataFrame:
    """
    Replace negative values in a numeric column with NULL.
    Used for umbrella_limit in policy.csv — 1 row has -1,000,000 which is
    an invalid value (umbrella limits are always positive or zero).
    Applied AFTER casting the column to its numeric type.

    Args:
        df: Input DataFrame
        col_name: Column to clean (must already be cast to numeric type)

    Example:
        df = fix_negative_to_null(df, "umbrella_limit")
        # -1000000 -> NULL
        # 0        -> 0        (unchanged)
        # 5000000  -> 5000000  (unchanged)
    """
    return df.withColumn(col_name,
        F.when(F.col(col_name) < 0, None)
        .otherwise(F.col(col_name))
    )


def tag_quarantine_reason(df: DataFrame, rules: list) -> DataFrame:
    """
    Add a quarantine_reason column describing why a record failed quality checks.
    Evaluates each rule in order and assigns the FIRST matching reason.
    Used to populate quarantine tables so the compliance team knows exactly
    why each record was rejected.

    Args:
        df: Input DataFrame
        rules: List of (condition_sql_string, reason_string) tuples.
               condition_sql_string must be a valid Spark SQL expression.

    Example:
        df = tag_quarantine_reason(df, [
            ("customer_id IS NULL", "missing customer_id"),
            ("balance IS NULL",     "missing balance"),
        ])
        # Row with customer_id=NULL -> quarantine_reason="missing customer_id"
        # Row with balance=NULL     -> quarantine_reason="missing balance"
        # Row matching neither      -> quarantine_reason="unknown"
    """
    expr = None
    for condition, reason in rules:
        clause = F.when(F.expr(condition), reason)
        if expr is None:
            expr = clause
        else:
            expr = expr.when(F.expr(condition), reason)

    if expr is not None:
        expr = expr.otherwise("unknown")

    return df.withColumn("quarantine_reason", expr)


# ========================================
# PARSING UTILITIES
# ========================================

def extract_numeric_from_string(df: DataFrame, source_col: str, target_col: str, pattern: str = r"[\d.]+") -> DataFrame:
    """
    Extract a numeric value from a string column that contains units.
    The original column is kept; the extracted value goes into a new column.
    Used for mileage ("23.4 kmpl"), engine ("1248 CC"), max_power ("74 bhp").

    Args:
        df: Input DataFrame
        source_col: Source column containing the unit-embedded string
        target_col: New column name for the extracted numeric value
        pattern: Regex pattern to extract (default: any sequence of digits and dots)

    Example:
        df = extract_numeric_from_string(df, "mileage",   "mileage_kmpl")
        df = extract_numeric_from_string(df, "engine",    "engine_cc")
        df = extract_numeric_from_string(df, "max_power", "max_power_bhp")
        # "23.4 kmpl" -> 23.4
        # "1248 CC"   -> 1248.0
        # "74 bhp"    -> 74.0
    """
    return df.withColumn(
        target_col,
        F.regexp_extract(F.col(source_col), pattern, 0).cast("double")
    )


def parse_date_with_format(df: DataFrame, source_col: str, date_format: str) -> DataFrame:
    """
    Parse a date/timestamp string column with a specific format.
    OVERWRITES the source column with the parsed timestamp type.
    Used for ad_placed_on and sold_on in sales files ("dd-MM-yyyy HH:mm").

    Args:
        df: Input DataFrame
        source_col: Column containing the date string (will be overwritten with timestamp)
        date_format: Java/Spark date format string

    Example:
        df = parse_date_with_format(df, "ad_placed_on", "dd-MM-yyyy HH:mm")
        # "10-02-2017 20:22" -> 2017-02-10 20:22:00 (TimestampType)
    """
    return df.withColumn(
        source_col,
        F.to_timestamp(F.col(source_col), date_format)
    )


def parse_csl_limits(df: DataFrame, csl_col: str = "policy_csl") -> DataFrame:
    """
    Parse CSL (Combined Single Limit) format "100/300" into two integer columns.
    Values are multiplied by 1000 (stored as thousands in source).
    Used for policy_csl column in policy files.

    Args:
        df: Input DataFrame
        csl_col: Column containing the "per_person/per_accident" string

    Example:
        df = parse_csl_limits(df, "policy_csl")
        # "100/300" -> csl_bodily_injury_per_person=100000,
        #              csl_bodily_injury_per_accident=300000
    """
    return (
        df
        .withColumn("csl_bodily_injury_per_person",
            F.regexp_extract(F.col(csl_col), r"(\d+)/", 1).cast("int") * 1000
        )
        .withColumn("csl_bodily_injury_per_accident",
            F.regexp_extract(F.col(csl_col), r"/(\d+)", 1).cast("int") * 1000
        )
    )


def clean_numeric_with_invalid_values(df: DataFrame, col_name: str, invalid_values: list, cast_type: str = "double") -> DataFrame:
    """
    Clean a numeric column by replacing invalid placeholder strings with NULL
    before casting to the target numeric type.
    Used for claims columns that contain "?" or "NULL" mixed with real numbers.

    Args:
        df: Input DataFrame
        col_name: Column to clean
        invalid_values: List of string values to replace with NULL (e.g., ["?", "NULL"])
        cast_type: Numeric type to cast to after cleaning (default: "double")

    Example:
        df = clean_numeric_with_invalid_values(df, "witnesses", ["?", "NULL"], "int")
        df = clean_numeric_with_invalid_values(df, "injury",    ["?", "NULL"], "decimal(10,2)")
        # "?"    -> NULL
        # "NULL" -> NULL
        # "3"    -> 3
        # "500"  -> 500.0
    """
    return df.withColumn(col_name,
        F.when(F.col(col_name).isin(invalid_values), None)
        .otherwise(F.col(col_name).cast(cast_type))
    )


def parse_yes_no_to_boolean(df: DataFrame, col_name: str, handle_question_mark: bool = True) -> DataFrame:
    """
    Convert YES/NO string column to boolean type.
    Optionally treats "?" as NULL (default: True).
    Used for police_report_available in claims files.

    Args:
        df: Input DataFrame
        col_name: Column containing YES/NO/? strings
        handle_question_mark: If True, "?" maps to NULL (default: True)

    Example:
        df = parse_yes_no_to_boolean(df, "police_report_available")
        # "YES" -> True, "NO" -> False, "?" -> NULL
    """
    result = (
        F.when(F.upper(F.col(col_name)) == "YES", True)
        .when(F.upper(F.col(col_name)) == "NO", False)
    )
    if handle_question_mark:
        result = result.when(F.col(col_name) == "?", None).otherwise(None)
    else:
        result = result.otherwise(None)
    return df.withColumn(col_name, result)


def parse_y_n_to_boolean(df: DataFrame, col_name: str) -> DataFrame:
    """
    Convert Y/N string column to boolean type.
    Used for claim_rejected column in claims files.

    Args:
        df: Input DataFrame
        col_name: Column containing Y/N strings

    Example:
        df = parse_y_n_to_boolean(df, "claim_rejected")
        # "Y" -> True, "N" -> False, anything else -> NULL
    """
    return df.withColumn(col_name,
        F.when(F.upper(F.col(col_name)) == "Y", True)
        .when(F.upper(F.col(col_name)) == "N", False)
        .otherwise(None)
    )


# ========================================
# DATE UTILITIES
# ========================================

def clean_corrupted_time_format(df: DataFrame, col_name: str, corrupted_pattern: str = r"^\d{2}:\d{2}\.\d$") -> DataFrame:
    """
    Replace corrupted time-offset strings like "34:00.0" with NULL.
    The claims JSON files have date columns stored as numeric time offsets
    (Excel serial date artefact) that cannot be recovered to real dates.
    Applied to: claim_logged_on, claim_processed_on, incident_date.

    Args:
        df: Input DataFrame
        col_name: Column to clean
        corrupted_pattern: Regex to identify corrupted values (default matches "NN:NN.N" format)

    Example:
        df = clean_corrupted_time_format(df, "claim_logged_on")
        # "34:00.0"   -> NULL  (corrupted — cannot recover)
        # "2017-03-15" -> "2017-03-15" (unchanged — valid date)
    """
    return df.withColumn(col_name,
        F.when(F.col(col_name).rlike(corrupted_pattern), None)
        .otherwise(F.col(col_name))
    )


def calculate_date_diff_days(df: DataFrame, end_col: str, start_col: str, result_col: str) -> DataFrame:
    """
    Calculate the difference in days between two date/timestamp columns.
    Returns NULL if either input is NULL — never produces a misleading number.
    Used to derive days_to_sell = sold_on - ad_placed_on in sales data.

    Args:
        df: Input DataFrame
        end_col: Later date column (e.g., "sold_on")
        start_col: Earlier date column (e.g., "ad_placed_on")
        result_col: Name of the output column with day difference

    Example:
        df = calculate_date_diff_days(df, "sold_on", "ad_placed_on", "days_to_sell")
        # sold_on=2017-03-20, ad_placed_on=2017-02-10 -> days_to_sell=38
        # sold_on=NULL,       ad_placed_on=2017-02-10 -> days_to_sell=NULL
    """
    return df.withColumn(result_col,
        F.when(
            F.col(end_col).isNotNull() & F.col(start_col).isNotNull(),
            F.datediff(F.col(end_col), F.col(start_col))
        ).otherwise(None)
    )


# ========================================
# CALCULATED FIELDS
# ========================================

def add_total_amount(df: DataFrame, result_col: str, amount_cols: list) -> DataFrame:
    """
    Sum multiple numeric columns into one total, treating NULL as 0.
    Used to derive total_claim_amount = injury + property + vehicle in claims data.
    NULL-safe: if a component is NULL it contributes 0, not NULL, to the total.

    Args:
        df: Input DataFrame
        result_col: Name of the output total column
        amount_cols: List of columns to sum

    Example:
        df = add_total_amount(df, "total_claim_amount", ["injury", "property", "vehicle"])
        # injury=1000, property=500, vehicle=NULL -> total_claim_amount=1500
        # injury=NULL, property=NULL, vehicle=NULL -> total_claim_amount=0
    """
    total = F.lit(0)
    for col in amount_cols:
        total = total + F.coalesce(F.col(col), F.lit(0))
    return df.withColumn(result_col, total)