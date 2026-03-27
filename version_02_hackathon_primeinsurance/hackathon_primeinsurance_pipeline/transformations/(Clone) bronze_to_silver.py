import dlt
from pyspark.sql import functions as F
from pyspark.sql import Window

from silver_utils import (
    deduplicate_by_key,
    coalesce_columns,
    lowercase_all_column_names,
    fix_swapped_columns,
    map_region_abbreviations,
    replace_na_strings_with_null,
    remove_trailing_dot,
    add_negative_value_flag,
    add_outlier_flag,
    fix_negative_to_null,
    standardize_to_lowercase,
    standardize_to_uppercase,
    extract_numeric_from_string,
    parse_date_with_format,
    parse_csl_limits,
    clean_numeric_with_invalid_values,
    replace_question_marks_with_null,
    replace_null_strings,
    parse_yes_no_to_boolean,
    parse_y_n_to_boolean,
    clean_corrupted_time_format,
    calculate_date_diff_days,
    add_total_amount
)


# ============================================================
# HELPER: Split duplicates from unique rows
# Returns (deduped_df, duplicates_df)
# ============================================================

def _split_duplicates(df, key_col):
    """
    Splits a DataFrame into unique rows and duplicate rows.
    - Keeps the most recent row (by _ingest_time) for each key.
    - Marks all other rows as duplicates with quarantine_type='drop'.
    Returns: (deduped_df, duplicates_df)
    """
    window = Window.partitionBy(key_col).orderBy(F.col("_ingest_time").desc())
    df_ranked = df.withColumn("_row_num", F.row_number().over(window))

    deduped = df_ranked.filter(F.col("_row_num") == 1).drop("_row_num")

    duplicates = (
        df_ranked
        .filter(F.col("_row_num") > 1)
        .drop("_row_num")
        .withColumn("quarantine_reason", F.lit(f"duplicate {key_col}"))
        .withColumn("quarantine_type", F.lit("drop"))
        .withColumn("quarantine_timestamp", F.current_timestamp())
    )

    return deduped, duplicates


# ============================================================
# MAPPING TABLES
# Used to standardize free-text categoricals via joins
# instead of hardcoded WHEN chains.
# ============================================================

# Incident severity mapping: raw value (any case) -> standard lowercase
INCIDENT_SEVERITY_MAP = {
    "minor damage":   "minor damage",
    "major damage":   "major damage",
    "trivial damage": "trivial damage",
    "total loss":     "total loss",
}

# Incident type mapping: raw value -> standard lowercase
INCIDENT_TYPE_MAP = {
    "single vehicle collision": "single vehicle collision",
    "multi-vehicle collision":  "multi-vehicle collision",
    "vehicle theft":            "vehicle theft",
    "parked car":               "parked car",
}

# Collision type mapping: raw value -> standard lowercase
COLLISION_TYPE_MAP = {
    "rear collision":  "rear collision",
    "front collision": "front collision",
    "side collision":  "side collision",
}

# Authorities contacted mapping
AUTHORITIES_MAP = {
    "police":    "police",
    "ambulance": "ambulance",
    "fire":      "fire",
    "none":      "none",
    "other":     "other",
}

# Fuel type mapping: raw -> standard lowercase
FUEL_MAP = {
    "petrol":   "petrol",
    "diesel":   "diesel",
    "electric": "electric",
    "hybrid":   "hybrid",
    "cng":      "cng",
    "lpg":      "lpg",
}

# Transmission mapping
TRANSMISSION_MAP = {
    "manual":    "manual",
    "automatic": "automatic",
}

# Seller type mapping
SELLER_TYPE_MAP = {
    "individual": "individual",
    "dealer":     "dealer",
    "company":    "company",
}

# Owner type mapping
OWNER_MAP = {
    "first owner":  "first owner",
    "second owner": "second owner",
    "third owner":  "third owner",
}

# Education mapping (includes typo fix for 'terto')
EDUCATION_MAP = {
    "primary":   "primary",
    "secondary": "secondary",
    "tertiary":  "tertiary",
    "terto":     "tertiary",   # typo fix
}

# Marital status mapping
MARITAL_MAP = {
    "married":   "married",
    "single":    "single",
    "divorced":  "divorced",
    "widow":     "widow",
    "separated": "separated",
}

# Region mapping: abbreviations + full names -> standard title case
REGION_MAP = {
    "c":       "Central",
    "w":       "West",
    "s":       "South",
    "e":       "East",
    "n":       "North",
    "central": "Central",
    "west":    "West",
    "south":   "South",
    "east":    "East",
    "north":   "North",
}


def _apply_mapping(df, col_name, mapping_dict, default=None):
    """
    Applies a dictionary mapping to a column using a CASE WHEN chain.
    Comparison is case-insensitive (lowercases input before matching).
    Unmatched values become `default` (None = null).

    Args:
        df:           Input DataFrame.
        col_name:     Column to map.
        mapping_dict: Dict of {raw_lower -> standardized_value}.
        default:      Value for unmatched rows. None means null.
    Returns:
        DataFrame with col_name replaced by standardized values.
    """
    expr = None
    for raw_val, std_val in mapping_dict.items():
        condition = F.lower(F.col(col_name)) == raw_val
        expr = (
            F.when(condition, F.lit(std_val))
            if expr is None
            else expr.when(condition, F.lit(std_val))
        )

    if expr is None:
        return df

    if default is not None:
        expr = expr.otherwise(F.lit(default))
    else:
        expr = expr.otherwise(F.lit(None).cast("string"))

    return df.withColumn(col_name, expr)


# ============================================================
# CUSTOMERS
# ============================================================

def _transform_customers():
    """
    Reads ALL 7 customer files via bronze_customers (which uses
    cloudFiles glob matching customers_*.csv at any depth).

    customers_7.csv is the MASTER file — 1,604 rows, all columns
    standardized, including Job which is missing in customers_1.
    customers_1 through customers_6 are per-Insurance-company files
    with inconsistent column names (Reg, cust_id, Customer_ID etc).

    All 7 files are unified through coalesce_columns so every row
    ends up with the same schema regardless of source file.

    FIX vs OLD CODE:
    - Added REGION_MAP via _apply_mapping (replaces map_region_abbreviations).
    - Added EDUCATION_MAP via _apply_mapping (replaces hardcoded 'terto' fix).
    - Added MARITAL_MAP via _apply_mapping (replaces hardcoded isin check).
    - customers_7 rows are now included — previously the glob pattern
      customers_*.csv already matched it, but coalesce was not handling
      the 'Marital' column name from c7. Now explicitly included.

    WARN flags (rows STAY in silver):
      - marital_status_warning
      - default_warning
      - insurance_warning
      - carloan_warning
      - balance_negative_flag
    """
    df = dlt.read("bronze_customers")

    # Step 1: Lowercase all column names FIRST before any coalesce.
    df = lowercase_all_column_names(df)

    # Step 2: Fix swapped Education / Marital columns in customers_6.
    df = fix_swapped_columns(
        df,
        col_a="education",
        col_b="marital_status",
        source_file_col="_source_file",
        source_file_keyword="customers_6"
    )

    # Step 3: Coalesce column name variants across ALL 7 customer files.
    # customers_7 uses: CustomerID, Region, Marital (no _status suffix)
    # customers_1 uses: Reg (not Region), no Job column
    # customers_2 uses: Customer_ID, City_in_state, Edu
    # customers_3 uses: cust_id
    df = coalesce_columns(df, "customer_id",    ["customerid", "customer_id", "cust_id"], "int")
    df = coalesce_columns(df, "region",         ["reg", "region"])
    df = coalesce_columns(df, "city",           ["city", "city_in_state"])
    df = coalesce_columns(df, "education",      ["education", "edu"])
    # "marital" handles customers_7 which uses 'Marital' (no _status suffix).
    # "marital_status" handles customers_1, 2, 6 which use 'Marital_status'.
    df = coalesce_columns(df, "marital_status", ["marital_status", "marital"])

    # Step 4: Replace "NA" strings with real nulls.
    df = replace_na_strings_with_null(df, ["education", "job", "marital_status"])

    # Step 5: Apply REGION_MAP — handles abbreviations (C/W/S/E/N)
    # AND full names (Central/West etc.) from all 7 files.
    # Replaces old map_region_abbreviations utility.
    df = _apply_mapping(df, "region", REGION_MAP, default=None)

    # Step 6: Apply EDUCATION_MAP — standardizes 'terto' -> 'tertiary'
    # and all other education values to lowercase consistently.
    # Replaces the old hardcoded .when(col == 'terto', ...) fix.
    df = _apply_mapping(df, "education", EDUCATION_MAP, default=None)

    # Step 7: Remove trailing dot from 'admin.' in Job column.
    df = remove_trailing_dot(df, "job")

    # Step 8: Apply MARITAL_MAP — standardizes all marital values to lowercase.
    # Replaces old standardize_to_lowercase for marital_status.
    df = _apply_mapping(df, "marital_status", MARITAL_MAP, default=None)

    # Step 9: Standardize job to lowercase.
    df = standardize_to_lowercase(df, ["job"])

    # Step 10: Cast numeric columns.
    df = (
        df
        .withColumn("default",     F.col("default").cast("int"))
        .withColumn("hhinsurance", F.col("hhinsurance").cast("int"))
        .withColumn("carloan",     F.col("carloan").cast("int"))
        .withColumn("balance",     F.col("balance").cast("decimal(10,2)"))
    )

    # WARN FLAG: Negative balance — may be valid overdraft, keep in silver.
    df = add_negative_value_flag(df, "balance", "balance_negative_flag")

    # WARN FLAG: Marital status not in valid list (catches NULLs from bad mapping).
    df = df.withColumn("marital_status_warning",
        F.when(
            F.col("marital_status").isNull() |
            ~F.col("marital_status").isin("married", "single", "divorced", "widow", "separated"),
            F.lit(1)
        ).otherwise(F.lit(0))
    )

    # WARN FLAG: Default flag not in (0, 1).
    df = df.withColumn("default_warning",
        F.when(
            F.col("default").isNull() | ~F.col("default").isin(0, 1),
            F.lit(1)
        ).otherwise(F.lit(0))
    )

    # WARN FLAG: Home insurance flag not in (0, 1).
    df = df.withColumn("insurance_warning",
        F.when(
            F.col("hhinsurance").isNull() | ~F.col("hhinsurance").isin(0, 1),
            F.lit(1)
        ).otherwise(F.lit(0))
    )

    # WARN FLAG: Car loan flag not in (0, 1).
    df = df.withColumn("carloan_warning",
        F.when(
            F.col("carloan").isNull() | ~F.col("carloan").isin(0, 1),
            F.lit(1)
        ).otherwise(F.lit(0))
    )

    return df.select(
        "customer_id",
        "state",
        "city",
        "region",
        "marital_status",
        "marital_status_warning",
        "education",
        "job",
        F.col("default").alias("has_default"),
        "default_warning",
        "balance",
        "balance_negative_flag",
        F.col("hhinsurance").alias("has_home_insurance"),
        "insurance_warning",
        F.col("carloan").alias("has_car_loan"),
        "carloan_warning",
        "_source_file",
        "_ingest_time"
    )


@dlt.table(
    name    = "silver.silver_customers",
    comment = "Cleaned customer data — all 7 files unified, deduplicated, standardized via mapping model."
)
def silver_customers():
    df = _transform_customers()
    deduped, _ = _split_duplicates(df, "customer_id")

    return deduped.filter(
        F.col("customer_id").isNotNull() &
        F.col("region").isin("Central", "West", "South", "East", "North") &
        F.col("state").isNotNull() &
        (F.trim(F.col("state")) != "")
    )


@dlt.table(
    name    = "silver.quarantine_customers",
    comment = "Customer records removed from silver — quality failures (type=quarantine) + duplicates (type=drop)"
)
def quarantine_customers():
    df = _transform_customers()
    deduped, duplicates = _split_duplicates(df, "customer_id")

    failed = deduped.filter(
        F.col("customer_id").isNull() |
        (F.col("region").isNull() | ~F.col("region").isin("Central", "West", "South", "East", "North")) |
        F.col("state").isNull() |
        (F.trim(F.col("state")) == "")
    )

    quality_failures = (
        failed
        .withColumn("quarantine_reason",
            F.when(F.col("customer_id").isNull(),
                F.lit("missing customer_id"))
            .when(F.col("region").isNull() | ~F.col("region").isin("Central", "West", "South", "East", "North"),
                F.lit("invalid or missing region"))
            .when(F.col("state").isNull() | (F.trim(F.col("state")) == ""),
                F.lit("missing or blank state"))
            .otherwise(F.lit("unknown"))
        )
        .withColumn("quarantine_type", F.lit("quarantine"))
        .withColumn("quarantine_timestamp", F.current_timestamp())
    )

    return quality_failures.unionByName(duplicates, allowMissingColumns=True)


# ============================================================
# CARS
# ============================================================

def _transform_cars():
    """
    Reads bronze_cars from same pipeline using dlt.read().

    FIX vs OLD CODE:
    - Added FUEL_MAP via _apply_mapping — replaces standardize_to_lowercase
      for fuel so unknown fuel types become NULL instead of silently passing.
    - Added TRANSMISSION_MAP via _apply_mapping — same reason.
    - 'model' column renamed to 'manufacturer' — kept as-is, correct.

    WARN flags (rows STAY in silver):
      - km_driven_outlier_flag
      - engine_cc_warning
      - max_power_warning
      - torque_warning
    """
    df = dlt.read("bronze_cars")

    df = lowercase_all_column_names(df)

    df = extract_numeric_from_string(df, "mileage",   "mileage_kmpl")
    df = extract_numeric_from_string(df, "engine",    "engine_cc")
    df = extract_numeric_from_string(df, "max_power", "max_power_bhp")
    df = extract_numeric_from_string(df, "torque",    "torque_nm")

    # Apply FUEL_MAP — unknown fuel types become NULL (caught by quarantine).
    # Replaces old standardize_to_lowercase(df, ["fuel"]).
    df = _apply_mapping(df, "fuel", FUEL_MAP, default=None)

    # Apply TRANSMISSION_MAP — unknown values become NULL.
    # Replaces old standardize_to_lowercase(df, ["transmission"]).
    df = _apply_mapping(df, "transmission", TRANSMISSION_MAP, default=None)

    df = (
        df
        .withColumn("car_id",    F.col("car_id").cast("int"))
        .withColumn("km_driven", F.col("km_driven").cast("int"))
        .withColumn("seats",     F.col("seats").cast("int"))
    )

    df = add_outlier_flag(df, "km_driven", "km_driven_outlier_flag", 500000)

    df = df.withColumn("engine_cc_warning",
        F.when(F.col("engine_cc").isNull(), F.lit(1)).otherwise(F.lit(0))
    )

    df = df.withColumn("max_power_warning",
        F.when(F.col("max_power_bhp").isNull(), F.lit(1)).otherwise(F.lit(0))
    )

    df = df.withColumn("torque_warning",
        F.when(F.col("torque_nm").isNull(), F.lit(1)).otherwise(F.lit(0))
    )

    return df.select(
        "car_id",
        F.trim(F.col("name")).alias("car_name"),
        "name",
        "fuel",
        "transmission",
        "seats",
        "km_driven",
        "km_driven_outlier_flag",
        "mileage_kmpl",
        "engine_cc",
        "engine_cc_warning",
        "max_power_bhp",
        "max_power_warning",
        "torque_nm",
        "torque_warning",
        F.lower(F.col("model")).alias("manufacturer"),
        "_source_file",
        "_ingest_time"
    )


@dlt.table(
    name    = "silver.silver_cars",
    comment = "Cleaned car data — numeric extracted, fuel/transmission mapped, outlier flagged."
)
def silver_cars():
    df = _transform_cars()
    deduped, _ = _split_duplicates(df, "car_id")

    return deduped.filter(
        F.col("car_id").isNotNull() &
        F.col("car_name").isNotNull() &
        (F.trim(F.col("car_name")) != "") &
        F.col("fuel").isin("petrol", "diesel", "electric", "hybrid", "cng", "lpg") &
        F.col("transmission").isin("manual", "automatic") &
        F.col("seats").between(4, 10) &
        (F.col("km_driven") >= 0) &
        (F.col("mileage_kmpl") > 0)
    )


@dlt.table(
    name    = "silver.quarantine_cars",
    comment = "Car records removed from silver — quality failures (type=quarantine) + invalid values (type=drop) + duplicates (type=drop)"
)
def quarantine_cars():
    df = _transform_cars()
    deduped, duplicates = _split_duplicates(df, "car_id")

    failed = deduped.filter(
        F.col("car_id").isNull() |
        F.col("car_name").isNull() |
        (F.trim(F.col("car_name")) == "") |
        (~F.col("fuel").isin("petrol", "diesel", "electric", "hybrid", "cng", "lpg") | F.col("fuel").isNull()) |
        (~F.col("transmission").isin("manual", "automatic") | F.col("transmission").isNull()) |
        (~F.col("seats").between(4, 10) | F.col("seats").isNull()) |
        (F.col("km_driven") < 0) |
        (F.col("mileage_kmpl").isNull() | (F.col("mileage_kmpl") <= 0))
    )

    quality_failures = (
        failed
        .withColumn("quarantine_reason",
            F.when(F.col("car_id").isNull(),
                F.lit("missing car_id"))
            .when(F.col("car_name").isNull() | (F.trim(F.col("car_name")) == ""),
                F.lit("missing or blank car name"))
            .when(F.col("fuel").isNull() | ~F.col("fuel").isin("petrol", "diesel", "electric", "hybrid", "cng", "lpg"),
                F.lit("invalid or missing fuel type"))
            .when(F.col("transmission").isNull() | ~F.col("transmission").isin("manual", "automatic"),
                F.lit("invalid or missing transmission"))
            .when(F.col("seats").isNull() | ~F.col("seats").between(4, 10),
                F.lit("invalid seat count — outside 4-10 range"))
            .when(F.col("km_driven") < 0,
                F.lit("negative km_driven — invalid value"))
            .when(F.col("mileage_kmpl").isNull() | (F.col("mileage_kmpl") <= 0),
                F.lit("zero or missing mileage — physically impossible"))
            .otherwise(F.lit("unknown"))
        )
        .withColumn("quarantine_type",
            F.when(
                (F.col("km_driven") < 0) |
                (F.col("mileage_kmpl").isNull() | (F.col("mileage_kmpl") <= 0)),
                F.lit("drop")
            ).otherwise(F.lit("quarantine"))
        )
        .withColumn("quarantine_timestamp", F.current_timestamp())
    )

    return quality_failures.unionByName(duplicates, allowMissingColumns=True)


# ============================================================
# POLICY
# ============================================================

def _transform_policy():
    """
    Reads bronze_policy.

    FIX vs OLD CODE:
    - No mapping changes needed — policy_state is a free-form US state
      abbreviation (IL, OH etc.) so we just uppercase it.
    - policy_csl format validated with rlike — no mapping needed.
    """
    df = dlt.read("bronze_policy")

    df = lowercase_all_column_names(df)

    df = (
        df
        .withColumn("policy_number", F.col("policy_number").cast("int"))
        .withColumn("car_id",        F.col("car_id").cast("int"))
        .withColumn("customer_id",   F.col("customer_id").cast("int"))
    )

    df = df.withColumn("policy_bind_date",
        F.to_date(F.col("policy_bind_date"), "yyyy-MM-dd")
    )

    df = standardize_to_uppercase(df, ["policy_state"])

    df = parse_csl_limits(df, "policy_csl")

    df = (
        df
        .withColumn("policy_deductable",     F.col("policy_deductable").cast("decimal(10,2)"))
        .withColumn("policy_annual_premium", F.col("policy_annual_premium").cast("decimal(10,2)"))
        .withColumn("umbrella_limit",        F.col("umbrella_limit").cast("bigint"))
    )

    df = df.withColumn("umbrella_limit_warning",
        F.when(F.col("umbrella_limit") < 0, F.lit(1)).otherwise(F.lit(0))
    )

    df = fix_negative_to_null(df, "umbrella_limit")

    df = df.withColumn("is_active",
        F.when(F.col("policy_bind_date").isNotNull(), True).otherwise(False)
    )

    return df.select(
        "policy_number",
        "policy_bind_date",
        "policy_state",
        "policy_csl",
        "csl_bodily_injury_per_person",
        "csl_bodily_injury_per_accident",
        F.col("policy_deductable").alias("deductible"),
        F.col("policy_annual_premium").alias("annual_premium"),
        "umbrella_limit",
        "umbrella_limit_warning",
        "car_id",
        "customer_id",
        "is_active",
        "_source_file",
        "_ingest_time"
    )


@dlt.table(
    name    = "silver.silver_policy",
    comment = "Cleaned policy data — dates parsed, CSL split, umbrella_limit fixed."
)
def silver_policy():
    df = _transform_policy()
    deduped, _ = _split_duplicates(df, "policy_number")

    return deduped.filter(
        F.col("policy_number").isNotNull() &
        F.col("policy_bind_date").isNotNull() &
        F.col("policy_state").isNotNull() &
        (F.trim(F.col("policy_state")) != "") &
        F.col("policy_csl").isNotNull() &
        F.col("policy_csl").rlike(r"^[0-9]+/[0-9]+$") &
        F.col("deductible").isNotNull() &
        (F.col("deductible") >= 0) &
        F.col("annual_premium").isNotNull() &
        (F.col("annual_premium") > 0) &
        F.col("car_id").isNotNull() &
        F.col("customer_id").isNotNull()
    )


@dlt.table(
    name    = "silver.quarantine_policy",
    comment = "Policy records removed from silver — quality failures (type=quarantine) + duplicates (type=drop)"
)
def quarantine_policy():
    df = _transform_policy()
    deduped, duplicates = _split_duplicates(df, "policy_number")

    failed = deduped.filter(
        F.col("policy_number").isNull() |
        F.col("policy_bind_date").isNull() |
        F.col("policy_state").isNull() |
        (F.trim(F.col("policy_state")) == "") |
        F.col("policy_csl").isNull() |
        (~F.col("policy_csl").rlike(r"^[0-9]+/[0-9]+$")) |
        F.col("deductible").isNull() |
        (F.col("deductible") < 0) |
        F.col("annual_premium").isNull() |
        (F.col("annual_premium") <= 0) |
        F.col("car_id").isNull() |
        F.col("customer_id").isNull()
    )

    quality_failures = (
        failed
        .withColumn("quarantine_reason",
            F.when(F.col("policy_number").isNull(),
                F.lit("missing policy_number"))
            .when(F.col("policy_bind_date").isNull(),
                F.lit("missing policy_bind_date"))
            .when(F.col("policy_state").isNull() | (F.trim(F.col("policy_state")) == ""),
                F.lit("missing or blank policy_state"))
            .when(F.col("policy_csl").isNull() | ~F.col("policy_csl").rlike(r"^[0-9]+/[0-9]+$"),
                F.lit("missing or invalid policy_csl format"))
            .when(F.col("deductible").isNull() | (F.col("deductible") < 0),
                F.lit("missing or negative deductible"))
            .when(F.col("annual_premium").isNull() | (F.col("annual_premium") <= 0),
                F.lit("invalid annual premium — zero or negative"))
            .when(F.col("car_id").isNull(),
                F.lit("missing car_id"))
            .when(F.col("customer_id").isNull(),
                F.lit("missing customer_id"))
            .otherwise(F.lit("unknown"))
        )
        .withColumn("quarantine_type", F.lit("quarantine"))
        .withColumn("quarantine_timestamp", F.current_timestamp())
    )

    return quality_failures.unionByName(duplicates, allowMissingColumns=True)


# ============================================================
# SALES
# ============================================================

def _transform_sales():
    """
    Reads bronze_sales.

    FIX vs OLD CODE:
    - Added SELLER_TYPE_MAP via _apply_mapping — unknown seller types
      become NULL and trigger seller_type_warning instead of silently
      passing through as garbage strings.
    - Added OWNER_MAP via _apply_mapping — same reason.
    - region is kept lowercase for sales (sales CSV uses lowercase regions
      unlike customers which uses title case).
    """
    df = dlt.read("bronze_sales")

    df = lowercase_all_column_names(df)

    df = (
        df
        .withColumn("sales_id", F.col("sales_id").cast("int"))
        .withColumn("car_id",   F.col("car_id").cast("int"))
    )

    df = parse_date_with_format(df, "ad_placed_on", "dd-MM-yyyy HH:mm")
    df = parse_date_with_format(df, "sold_on",      "dd-MM-yyyy HH:mm")

    df = calculate_date_diff_days(df, "sold_on", "ad_placed_on", "days_to_sell")

    df = df.withColumn("selling_price",
        F.col("original_selling_price").cast("decimal(12,2)")
    )

    # Apply SELLER_TYPE_MAP — unknown values become NULL.
    # Replaces old standardize_to_lowercase(df, ["seller_type"]).
    df = _apply_mapping(df, "seller_type", SELLER_TYPE_MAP, default=None)

    # Apply OWNER_MAP — unknown values become NULL.
    # Replaces old standardize_to_lowercase(df, ["owner"]).
    df = _apply_mapping(df, "owner", OWNER_MAP, default=None)

    # Lowercase region and state for consistency.
    df = standardize_to_lowercase(df, ["region", "state", "city"])

    df = df.withColumn("sale_status",
        F.when(F.col("sold_on").isNotNull(), "sold")
        .otherwise("listed")
    )

    # WARN FLAG: city null or blank.
    df = df.withColumn("city_warning",
        F.when(
            F.col("city").isNull() | (F.trim(F.col("city")) == ""),
            F.lit(1)
        ).otherwise(F.lit(0))
    )

    # WARN FLAG: seller_type NULL after mapping (means unknown value came in).
    df = df.withColumn("seller_type_warning",
        F.when(F.col("seller_type").isNull(), F.lit(1)).otherwise(F.lit(0))
    )

    # WARN FLAG: owner NULL after mapping (means unknown value came in).
    df = df.withColumn("owner_warning",
        F.when(F.col("owner").isNull(), F.lit(1)).otherwise(F.lit(0))
    )

    return df.select(
        "sales_id",
        "ad_placed_on",
        "sold_on",
        "days_to_sell",
        "selling_price",
        "region",
        "state",
        "city",
        "city_warning",
        "seller_type",
        "seller_type_warning",
        "owner",
        "owner_warning",
        "sale_status",
        "car_id",
        "_source_file",
        "_ingest_time"
    )


@dlt.table(
    name    = "silver.silver_sales",
    comment = "Cleaned sales data — seller_type and owner mapped, dates parsed, sale status derived."
)
def silver_sales():
    df = _transform_sales()
    deduped, _ = _split_duplicates(df, "sales_id")

    return deduped.filter(
        F.col("sales_id").isNotNull() &
        F.col("ad_placed_on").isNotNull() &
        
        F.col("selling_price").isNotNull() &
        (F.col("selling_price") > 0) &
        F.col("region").isNotNull() &
        F.col("region").isin("east", "west", "central", "south", "north") &
        F.col("state").isNotNull() &
        (F.trim(F.col("state")) != "") &
        F.col("car_id").isNotNull()
    )


@dlt.table(
    name    = "silver.quarantine_sales",
    comment = "Sales records removed from silver — quality failures (type=quarantine) + duplicates (type=drop)"
)
def quarantine_sales():
    df = _transform_sales()
    deduped, duplicates = _split_duplicates(df, "sales_id")

    failed = deduped.filter(
        F.col("sales_id").isNull() |
        F.col("ad_placed_on").isNull() |
        F.col("sold_on").isNull() |
        F.col("selling_price").isNull() |
        (F.col("selling_price") <= 0) |
        F.col("region").isNull() |
        (~F.col("region").isin("east", "west", "central", "south", "north")) |
        F.col("state").isNull() |
        (F.trim(F.col("state")) == "") |
        F.col("car_id").isNull()
    )

    quality_failures = (
        failed
        .withColumn("quarantine_reason",
            F.when(F.col("sales_id").isNull(),
                F.lit("missing sales_id — blank or partial row"))
            .when(F.col("ad_placed_on").isNull(),
                F.lit("missing ad placement date"))
            .when(F.col("sold_on").isNull(),
                F.lit("missing sold date"))
            .when(F.col("selling_price").isNull() | (F.col("selling_price") <= 0),
                F.lit("invalid selling price — zero or negative"))
            .when(F.col("region").isNull() | ~F.col("region").isin("east", "west", "central", "south", "north"),
                F.lit("invalid or missing region"))
            .when(F.col("state").isNull() | (F.trim(F.col("state")) == ""),
                F.lit("missing or blank state"))
            .when(F.col("car_id").isNull(),
                F.lit("missing car_id"))
            .otherwise(F.lit("unknown"))
        )
        .withColumn("quarantine_type", F.lit("quarantine"))
        .withColumn("quarantine_timestamp", F.current_timestamp())
    )

    return quality_failures.unionByName(duplicates, allowMissingColumns=True)


# ============================================================
# CLAIMS
# ============================================================

def _transform_claims():
    df = dlt.read("bronze_claims")

    df = lowercase_all_column_names(df)

    df = (
        df
        .withColumn("claim_id",  F.col("claimid").cast("int"))
        .withColumn("policy_id", F.col("policyid").cast("int"))
    )

    # ── EXTRACT NUMBERS BEFORE WIPING CORRUPTED DATES ─────────
    # Source dates look like '34:00.0', '03:00.0' — extract the
    # number before ':' FIRST, then wipe the broken string to NULL.
    # This is the only way to recover days_to_process since Silver
    # has no real dates to diff.
    df = df.withColumn(
        "_logged_num",
        F.regexp_extract(
            F.col("claim_logged_on").cast("string"), r"^(\d+)", 1
        ).cast("int")
    ).withColumn(
        "_processed_num",
        F.regexp_extract(
            F.col("claim_processed_on").cast("string"), r"^(\d+)", 1
        ).cast("int")
    ).withColumn(
        "days_to_process",
        F.when(
            F.col("claim_processed_on").isNotNull() &
            F.col("claim_logged_on").isNotNull() &
            (F.col("_processed_num") >= F.col("_logged_num")) &
            (F.col("_processed_num") > 0) &
            (F.col("_logged_num") > 0),
            F.col("_processed_num") - F.col("_logged_num")
        ).otherwise(None)
    ).drop("_logged_num", "_processed_num")  # drop temp cols

    # NOW wipe the corrupted date strings to NULL (safe — we already
    # saved what we needed from them above)
    df = clean_corrupted_time_format(df, "claim_logged_on")
    df = clean_corrupted_time_format(df, "claim_processed_on")
    df = df.withColumn(
    "incident_minute",
    F.regexp_extract(F.col("incident_date").cast("string"), r"^(\d+)", 1).cast("int")
    )
    df = clean_corrupted_time_format(df, "incident_date")

    df = replace_null_strings(df, ["claim_processed_on", "claim_logged_on"])

    df = df.withColumn("is_rejected",
        F.when(F.upper(F.col("claim_rejected")) == "Y", True)
        .when(F.upper(F.col("claim_rejected")) == "N", False)
        .otherwise(None)
    )

    df = replace_question_marks_with_null(df, ["collision_type"])

    df = _apply_mapping(df, "collision_type",        COLLISION_TYPE_MAP,    default=None)
    df = _apply_mapping(df, "incident_severity",     INCIDENT_SEVERITY_MAP, default=None)
    df = _apply_mapping(df, "incident_type",         INCIDENT_TYPE_MAP,     default=None)
    df = _apply_mapping(df, "authorities_contacted",  AUTHORITIES_MAP,       default=None)

    df = standardize_to_uppercase(df, ["incident_state"])

    df = parse_yes_no_to_boolean(df, "police_report_available")

    df = df.withColumn("property_damage",
        F.when(F.upper(F.col("property_damage")) == "YES", True)
        .when(F.upper(F.col("property_damage")) == "NO",  False)
        .when(F.col("property_damage") == "?", None)
        .otherwise(None)
    )

    df = clean_numeric_with_invalid_values(df, "bodily_injuries", ["?", "NULL"], "int")
    df = clean_numeric_with_invalid_values(df, "injury",          ["?", "NULL"], "decimal(10,2)")
    df = clean_numeric_with_invalid_values(df, "property",        ["?", "NULL"], "decimal(10,2)")
    df = clean_numeric_with_invalid_values(df, "vehicle",         ["?", "NULL"], "decimal(10,2)")
    df = clean_numeric_with_invalid_values(df, "witnesses",       ["?", "NULL"], "int")
    df = df.withColumn("number_of_vehicles_involved",
        F.col("number_of_vehicles_involved").cast("int")
    )

    df = add_total_amount(df, "total_claim_amount", ["injury", "property", "vehicle"])

    # ── WARN FLAGS ────────────────────────────────────────────
    df = df.withColumn("incident_date_warning",
        F.when(F.col("incident_date").isNull(), F.lit(1)).otherwise(F.lit(0))
    )
    df = df.withColumn("claim_logged_on_warning",
        F.when(F.col("claim_logged_on").isNull(), F.lit(1)).otherwise(F.lit(0))
    )
    df = df.withColumn("claim_processed_on_warning",
        F.when(F.col("claim_processed_on").isNull(), F.lit(1)).otherwise(F.lit(0))
    )
    df = df.withColumn("days_to_process_warning",
        F.when(F.col("days_to_process").isNull(), F.lit(1)).otherwise(F.lit(0))
    )
    df = df.withColumn("collision_type_warning",
        F.when(F.col("collision_type").isNull(), F.lit(1)).otherwise(F.lit(0))
    )
    df = df.withColumn("bodily_injuries_warning",
        F.when(
            F.col("bodily_injuries").isNull() | (F.col("bodily_injuries") < 0),
            F.lit(1)
        ).otherwise(F.lit(0))
    )

    return df.select(
        "claim_id",
        "policy_id",
        "claim_logged_on",
        "claim_logged_on_warning",
        "claim_processed_on",
        "claim_processed_on_warning",
        "days_to_process",              # ← NEW: recovered from bronze
        "days_to_process_warning",      # ← NEW: 1 if still NULL
        "incident_date",
        "incident_date_warning",
        "incident_state",
        "incident_severity",
        "incident_type",
        "is_rejected",
        "authorities_contacted",
        "bodily_injuries",
        "bodily_injuries_warning",
        "collision_type",
        "collision_type_warning",
        F.col("incident_city").alias("city"),
        "incident_location",
        F.col("injury").alias("injury_amount"),
        F.col("property").alias("property_amount"),
        "property_damage",
        F.col("vehicle").alias("vehicle_amount"),
        "witnesses",
        "number_of_vehicles_involved",
        "police_report_available",
        "total_claim_amount",
        "_source_file",
        "_ingest_time"
    )
@dlt.table(
    name    = "silver.silver_claims",
    comment = "Cleaned claims — corrupted dates kept as NULL with warn flags, not quarantined."
)
def silver_claims():
    df = _transform_claims()
    deduped, _ = _split_duplicates(df, "claim_id")

    # HARD rules — only things that make a claim genuinely meaningless
    # Dates REMOVED from hard rules — all 1000 rows had corrupted dates
    # which is a source problem, not a record quality problem
    return deduped.filter(
        F.col("claim_id").isNotNull() &
        F.col("policy_id").isNotNull() &
        F.col("incident_state").isNotNull() &
        (F.trim(F.col("incident_state")) != "") &
        F.col("incident_severity").isin(list(INCIDENT_SEVERITY_MAP.values())) &
        (F.col("injury_amount") >= 0) &
        (F.col("property_amount") >= 0) &
        F.col("number_of_vehicles_involved").isNotNull() &
        (F.col("number_of_vehicles_involved") >= 1)
    )


@dlt.table(
    name    = "silver.quarantine_claims",
    comment = "Claims removed from silver — only genuine quality failures + duplicates."
)
def quarantine_claims():
    df = _transform_claims()
    deduped, duplicates = _split_duplicates(df, "claim_id")

    failed = deduped.filter(
        F.col("claim_id").isNull() |
        F.col("policy_id").isNull() |
        F.col("incident_state").isNull() |
        (F.trim(F.col("incident_state")) == "") |
        F.col("incident_severity").isNull() |
        (~F.col("incident_severity").isin(list(INCIDENT_SEVERITY_MAP.values()))) |
        (F.col("injury_amount") < 0) |
        (F.col("property_amount") < 0) |
        F.col("number_of_vehicles_involved").isNull() |
        (F.col("number_of_vehicles_involved") < 1)
    )

    quality_failures = (
        failed
        .withColumn("quarantine_reason",
            F.when(F.col("claim_id").isNull(),   F.lit("missing claim_id"))
            .when(F.col("policy_id").isNull(),   F.lit("missing policy_id"))
            .when(F.col("incident_state").isNull() | (F.trim(F.col("incident_state")) == ""),
                F.lit("missing or blank incident_state"))
            .when(F.col("incident_severity").isNull() |
                ~F.col("incident_severity").isin(list(INCIDENT_SEVERITY_MAP.values())),
                F.lit("invalid or missing incident_severity"))
            .when((F.col("injury_amount") < 0) | (F.col("property_amount") < 0),
                F.lit("negative claim amount"))
            .when(F.col("number_of_vehicles_involved").isNull() |
                (F.col("number_of_vehicles_involved") < 1),
                F.lit("invalid vehicle count — must be at least 1"))
            .otherwise(F.lit("unknown"))
        )
        .withColumn("quarantine_type", F.lit("quarantine"))
        .withColumn("quarantine_timestamp", F.current_timestamp())
    )

    return quality_failures.unionByName(duplicates, allowMissingColumns=True)

# ============================================================
# QUALITY LOG
# ============================================================

@dlt.table(
    name    = "silver.silver_quality_log",
    comment = "Summary of all quality failures — one row per reason per type per entity."
)
def silver_quality_log():
    from functools import reduce

    summaries = []

    for table_name, entity in [
        ("silver.quarantine_customers", "customers"),
        ("silver.quarantine_cars",      "cars"),
        ("silver.quarantine_policy",    "policy"),
        ("silver.quarantine_sales",     "sales"),
        ("silver.quarantine_claims",    "claims"),
    ]:
        summary = (
            dlt.read(table_name)
            .groupBy("quarantine_reason", "quarantine_type")
            .agg(F.count("*").alias("failed_record_count"))
            .withColumn("entity",           F.lit(entity))
            .withColumn("quarantine_table", F.lit(table_name))
            .withColumn("log_timestamp",    F.current_timestamp())
            .select(
                "entity",
                "quarantine_table",
                "quarantine_reason",
                "quarantine_type",
                "failed_record_count",
                "log_timestamp"
            )
        )
        summaries.append(summary)

    return reduce(lambda a, b: a.union(b), summaries)