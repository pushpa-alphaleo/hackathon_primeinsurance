"""
Silver Layer Utility Functions
================================
Reusable data quality and transformation functions for DLT silver layer
"""

from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql.window import Window


# ========================================
# DATA QUALITY UTILITIES
# ========================================

def deduplicate_by_key(df: DataFrame, key_column: str, order_column: str = "_ingest_time") -> DataFrame:
    return df.dropDuplicates([key_column])


def replace_question_marks_with_null(df: DataFrame, columns: list) -> DataFrame:
    """
    Replace '?' with NULL in specified columns.

    Args:
        df: Input DataFrame
        columns: List of column names to clean

    Returns:
        Cleaned DataFrame

    Example:
        df = replace_question_marks_with_null(df, ["collision_type", "witnesses"])
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

    Args:
        df: Input DataFrame
        columns: List of column names to clean

    Returns:
        Cleaned DataFrame

    Example:
        df = replace_null_strings(df, ["claim_processed_on"])
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

    Args:
        df: Input DataFrame
        columns: List of column names to clean

    Returns:
        Cleaned DataFrame

    Example:
        df = replace_na_strings_with_null(df, ["education", "job", "marital_status"])
        # "NA" -> NULL
        # "secondary" -> "secondary" (unchanged)
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

    Args:
        df: Input DataFrame
        columns: List of column names to lowercase

    Returns:
        Standardized DataFrame

    Example:
        df = standardize_to_lowercase(df, ["marital_status", "job", "fuel"])
    """
    for col in columns:
        df = df.withColumn(col, F.lower(F.col(col)))
    return df


def standardize_to_uppercase(df: DataFrame, columns: list) -> DataFrame:
    """
    Convert specified columns to uppercase.

    Args:
        df: Input DataFrame
        columns: List of column names to uppercase

    Returns:
        Standardized DataFrame

    Example:
        df = standardize_to_uppercase(df, ["state", "incident_state"])
    """
    for col in columns:
        df = df.withColumn(col, F.upper(F.col(col)))
    return df


def lowercase_all_column_names(df: DataFrame) -> DataFrame:
    """
    Lowercase ALL column names in a DataFrame.
    Must be called before coalesce_columns() because column name matching
    is case-sensitive in PySpark — 'CustomerID' != 'customerid'.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with all column names lowercased

    Example:
        df = lowercase_all_column_names(df)
        # 'CustomerID' -> 'customerid'
        # 'Reg' -> 'reg'
    """
    return df.toDF(*[c.lower() for c in df.columns])


def coalesce_columns(df: DataFrame, target_col: str, source_cols: list, cast_type: str = None) -> DataFrame:
    """
    Coalesce multiple columns into one, taking first non-null value.
    Only includes source columns that actually exist in the DataFrame —
    safely handles files that don't have all columns.

    Args:
        df: Input DataFrame
        target_col: Name of output column
        source_cols: List of columns to coalesce (in priority order)
        cast_type: Optional data type to cast to (e.g., "int", "string")

    Returns:
        DataFrame with coalesced column

    Example:
        df = coalesce_columns(df, "customer_id", ["customerid", "customer_id", "cust_id"], "int")
    """
    existing = [c for c in source_cols if c in df.columns]

    if not existing:
        return df.withColumn(target_col, F.lit(None))

    result = F.coalesce(*[F.col(c) for c in existing])

    if cast_type:
        result = result.cast(cast_type)

    return df.withColumn(target_col, result)


def fix_swapped_columns(df: DataFrame, col_a: str, col_b: str, source_file_col: str, source_file_keyword: str) -> DataFrame:
    """
    Fix two columns that are swapped in a specific source file.
    For all other source files, columns are left unchanged.

    Background: customers_6.csv has Education and Marital_status swapped.

    Args:
        df: Input DataFrame
        col_a: First column (e.g., "education")
        col_b: Second column (e.g., "marital_status")
        source_file_col: Column that identifies the source file (e.g., "_source_file")
        source_file_keyword: Keyword to identify the affected file (e.g., "customers_6")

    Returns:
        DataFrame with columns swapped back for the affected file rows

    Example:
        df = fix_swapped_columns(df, "education", "marital_status", "_source_file", "customers_6")
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
    Map single-letter region codes to full names.
    customers_5.csv uses C/W/S/E instead of Central/West/South/East.
    Safe to run on all rows — only triggers on length-1 values.

    Args:
        df: Input DataFrame
        col_name: Name of the region column (default: "region")

    Returns:
        DataFrame with full region names

    Example:
        df = map_region_abbreviations(df, "region")
        # 'W' -> 'West', 'E' -> 'East', 'S' -> 'South', 'C' -> 'Central'
        # 'West' -> 'West' (unchanged)
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
    The Job column contains 'admin.' with a trailing dot.

    Args:
        df: Input DataFrame
        col_name: Column to clean

    Returns:
        DataFrame with trailing dots removed

    Example:
        df = remove_trailing_dot(df, "job")
        # 'admin.' -> 'admin'
        # 'management' -> 'management' (unchanged)
    """
    return df.withColumn(col_name,
        F.regexp_replace(F.col(col_name), r"\.$", "")
    )


def add_negative_value_flag(df: DataFrame, col_name: str, flag_col_name: str) -> DataFrame:
    """
    Add a binary flag column indicating whether a numeric column has a negative value.

    Args:
        df: Input DataFrame
        col_name: Numeric column to check
        flag_col_name: Name of the output flag column (1=negative, 0=non-negative)

    Example:
        df = add_negative_value_flag(df, "balance", "balance_negative_flag")
    """
    return df.withColumn(flag_col_name,
        F.when(F.col(col_name) < 0, 1).otherwise(0)
    )


def add_outlier_flag(df: DataFrame, col_name: str, flag_col_name: str, threshold: float) -> DataFrame:
    """
    Add a binary flag column for values exceeding a threshold.

    Args:
        df: Input DataFrame
        col_name: Numeric column to check
        flag_col_name: Name of the output flag column (1=outlier, 0=normal)
        threshold: Value above which the flag is set to 1

    Example:
        df = add_outlier_flag(df, "km_driven", "km_driven_outlier_flag", 500000)
    """
    return df.withColumn(flag_col_name,
        F.when(F.col(col_name) > threshold, 1).otherwise(0)
    )


def fix_negative_to_null(df: DataFrame, col_name: str) -> DataFrame:
    """
    Replace negative values in a numeric column with NULL.
    Used for umbrella_limit in policy.csv — 1 row has -1,000,000.

    Args:
        df: Input DataFrame
        col_name: Column to clean

    Example:
        df = fix_negative_to_null(df, "umbrella_limit")
        # -1000000 -> NULL, 0 -> 0, 5000000 -> 5000000
    """
    return df.withColumn(col_name,
        F.when(F.col(col_name) < 0, None)
        .otherwise(F.col(col_name))
    )


def tag_quarantine_reason(df: DataFrame, rules: list) -> DataFrame:
    """
    Add a quarantine_reason column describing why a record failed quality checks.
    Evaluates each rule in order and assigns the first matching reason.
    Used to populate quarantine tables so the compliance team knows
    exactly why each record was rejected.

    Args:
        df: Input DataFrame
        rules: List of (condition_expr_string, reason_string) tuples
               condition_expr_string must be a valid SQL expression

    Returns:
        DataFrame with quarantine_reason column added

    Example:
        df = tag_quarantine_reason(df, [
            ("customer_id IS NULL",  "missing customer_id"),
            ("balance IS NULL",      "missing balance"),
        ])
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
    Extract numeric value from string with units.

    Args:
        df: Input DataFrame
        source_col: Source column name (e.g., "mileage")
        target_col: Target column name (e.g., "mileage_kmpl")
        pattern: Regex pattern to extract (default: any number)

    Example:
        df = extract_numeric_from_string(df, "mileage", "mileage_kmpl")
        # "23.4 kmpl" -> 23.4,  "1248 CC" -> 1248.0
    """
    return df.withColumn(
        target_col,
        F.regexp_extract(F.col(source_col), pattern, 0).cast("double")
    )


def parse_date_with_format(df: DataFrame, source_col: str, date_format: str) -> DataFrame:
    """
    Parse date string with specific format (overwrites source column).

    Args:
        df: Input DataFrame
        source_col: Source column name (will be overwritten)
        date_format: Date format string (e.g., "dd-MM-yyyy HH:mm")

    Example:
        df = parse_date_with_format(df, "ad_placed_on", "dd-MM-yyyy HH:mm")
        # "10-02-2017 20:22" -> 2017-02-10 20:22:00
    """
    return df.withColumn(
        source_col,
        F.to_timestamp(F.col(source_col), date_format)
    )


def parse_csl_limits(df: DataFrame, csl_col: str = "policy_csl") -> DataFrame:
    """
    Parse CSL limit format "100/300" into two separate integer columns.

    Example:
        df = parse_csl_limits(df, "policy_csl")
        # "100/300" -> csl_per_person=100000, csl_per_accident=300000
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
    Clean numeric column by replacing invalid string values with NULL before casting.

    Args:
        df: Input DataFrame
        col_name: Column to clean
        invalid_values: List of invalid string values (e.g., ["?", "NULL"])
        cast_type: Type to cast to after cleaning (default: "double")

    Example:
        df = clean_numeric_with_invalid_values(df, "witnesses", ["?", "NULL"], "int")
        # "?" -> NULL, "3" -> 3
    """
    return df.withColumn(col_name,
        F.when(F.col(col_name).isin(invalid_values), None)
        .otherwise(F.col(col_name).cast(cast_type))
    )


def parse_yes_no_to_boolean(df: DataFrame, col_name: str, handle_question_mark: bool = True) -> DataFrame:
    """
    Convert YES/NO string to boolean.

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
    Convert Y/N string to boolean.

    Example:
        df = parse_y_n_to_boolean(df, "claim_rejected")
        # "Y" -> True, "N" -> False
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
    Replace corrupted time formats like "34:00.0" with NULL.
    The claims files have date columns stored as time offsets that
    cannot be recovered.

    Example:
        df = clean_corrupted_time_format(df, "claim_logged_on")
        # "34:00.0" -> NULL, "2017-03-15" -> "2017-03-15"
    """
    return df.withColumn(col_name,
        F.when(F.col(col_name).rlike(corrupted_pattern), None)
        .otherwise(F.col(col_name))
    )


def clean_corrupted_time_to_timestamp(df: DataFrame, col_name: str) -> DataFrame:
    """
    Parse a HH:mm time string into a timestamp, setting NULL for any hour > 23.

    Handles the specific claims data issue where corrupted rows contain
    time offsets like "27:00.0", "34:00.0", "45:00.0" — these cannot be
    parsed as valid times and are set to NULL.

    Valid rows (hour 0–23) are parsed using F.to_timestamp with "HH:mm".
    Invalid rows (hour > 23 or non-numeric) become NULL.

    Unlike the regex-based clean_corrupted_time_format(), this function
    actively parses the value into a proper timestamp type — use this
    for claims date columns that need to be stored as timestamps.

    Args:
        df:       Input DataFrame
        col_name: Column containing "HH:mm" or "HH:mm.0" strings

    Returns:
        DataFrame with col_name replaced by a TimestampType column.
        NULL where the original value was NULL or had an invalid hour.

    Example:
        df = clean_corrupted_time_to_timestamp(df, "claim_logged_on")
        # "08:30.0" -> 1970-01-01 08:30:00  (valid)
        # "27:00.0" -> NULL                 (invalid hour)
        # "34:00.0" -> NULL                 (invalid hour)
        # NULL      -> NULL                 (already null)
    """
    # Strip trailing ".0" suffix and surrounding whitespace
    cleaned = F.regexp_replace(F.trim(F.col(col_name)), r"\.0$", "")
    # Extract the hour portion (everything before the first ":")
    hour_str = F.split(cleaned, ":")[0]

    return df.withColumn(col_name,
        F.when(
            F.col(col_name).isNotNull() &
            F.try_cast(hour_str, "int").isNotNull() &
            F.try_cast(hour_str, "int").between(0, 23),
            F.to_timestamp(cleaned, "HH:mm")
        ).otherwise(None)
    )


def add_time_corruption_warning(df: DataFrame, raw_col: str, parsed_col: str, flag_col_name: str) -> DataFrame:
    """
    Add a binary warning flag that fires when a time column was present in
    the raw data but could NOT be parsed into a valid timestamp.

    Use this AFTER calling clean_corrupted_time_to_timestamp() on the column.
    The raw value must be preserved in a separate column before parsing so
    this function can compare the two.

    Pattern:
        1. Save the raw string:  df = df.withColumn("raw_claim_logged_on", F.col("claim_logged_on"))
        2. Parse the column:     df = clean_corrupted_time_to_timestamp(df, "claim_logged_on")
        3. Add warning flag:     df = add_time_corruption_warning(df, "raw_claim_logged_on",
                                                                      "claim_logged_on",
                                                                      "claim_logged_on_warning")

    Args:
        df:             Input DataFrame
        raw_col:        Column holding the original string value (before parsing)
        parsed_col:     Column holding the parsed timestamp (after parsing)
        flag_col_name:  Name for the output 0/1 flag column

    Returns:
        DataFrame with flag_col_name added.
        flag = 1  →  raw value existed but timestamp is NULL (corruption detected)
        flag = 0  →  timestamp parsed successfully OR raw value was already NULL

    Example:
        df = add_time_corruption_warning(df, "raw_claim_logged_on",
                                             "claim_logged_on",
                                             "claim_logged_on_warning")
        # raw="27:00.0", parsed=NULL  -> flag=1
        # raw="08:30.0", parsed=ts    -> flag=0
        # raw=NULL,      parsed=NULL  -> flag=0
    """
    return df.withColumn(flag_col_name,
        F.when(
            F.col(raw_col).isNotNull() & F.col(parsed_col).isNull(),
            F.lit(1)
        ).otherwise(F.lit(0))
    )


# ========================================
# CALCULATED FIELDS
# ========================================

def calculate_date_diff_days(df: DataFrame, end_col: str, start_col: str, result_col: str) -> DataFrame:
    """
    Calculate difference in days between two date columns.
    Returns NULL if either input date is NULL.

    Example:
        df = calculate_date_diff_days(df, "sold_on", "ad_placed_on", "days_to_sell")
        # sold_on=2017-03-20, ad_placed_on=2017-02-10 -> 38
    """
    return df.withColumn(result_col,
        F.when(
            F.col(end_col).isNotNull() & F.col(start_col).isNotNull(),
            F.datediff(F.col(end_col), F.col(start_col))
        ).otherwise(None)
    )


def add_total_amount(df: DataFrame, result_col: str, amount_cols: list) -> DataFrame:
    """
    Sum multiple amount columns, treating NULL as 0.

    Example:
        df = add_total_amount(df, "total_claim_amount", ["injury", "property", "vehicle"])
        # injury=1000, property=500, vehicle=NULL -> total=1500
    """
    total = F.lit(0)
    for col in amount_cols:
        total = total + F.coalesce(F.col(col), F.lit(0))
    return df.withColumn(result_col, total)