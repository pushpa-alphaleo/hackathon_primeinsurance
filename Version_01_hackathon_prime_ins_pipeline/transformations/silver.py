import dlt
from pyspark.sql import functions as F
from functools import reduce

from bronze_to_silver_utils import (
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
    clean_corrupted_time_format,
    calculate_date_diff_days,
    add_total_amount,
)


# =============================================================
# CUSTOMERS
# =============================================================

def _transform_customers():
    df = dlt.read_stream("bronze.bronze_customers")
    df = lowercase_all_column_names(df)
    df = fix_swapped_columns(
        df,
        col_a               = "education",
        col_b               = "marital_status",
        source_file_col     = "_source_file",
        source_file_keyword = "customers_6",
    )
    df = coalesce_columns(df, "customer_id",    ["customerid", "customer_id", "cust_id"], "int")
    df = coalesce_columns(df, "region",         ["reg", "region"])
    df = coalesce_columns(df, "city",           ["city", "city_in_state"])
    df = coalesce_columns(df, "education",      ["education", "edu"])
    df = coalesce_columns(df, "marital_status", ["marital_status", "marital"])
    df = replace_na_strings_with_null(df, ["education", "job", "marital_status"])
    df = map_region_abbreviations(df, "region")
    df = df.withColumn(
        "education",
        F.when(F.col("education") == "terto", "tertiary").otherwise(F.col("education")),
    )
    df = remove_trailing_dot(df, "job")
    df = standardize_to_lowercase(df, ["education", "marital_status", "job"])
    df = (
        df.withColumn("default",     F.col("default").cast("int"))
          .withColumn("hhinsurance", F.col("hhinsurance").cast("int"))
          .withColumn("carloan",     F.col("carloan").cast("int"))
          .withColumn("balance",     F.col("balance").cast("decimal(10,2)"))
    )
    df = add_negative_value_flag(df, "balance", "balance_negative_flag")
    return df.select(
        "customer_id", "state", "city", "region", "marital_status", "education", "job",
        F.col("default").alias("has_default"),
        "balance", "balance_negative_flag",
        F.col("hhinsurance").alias("has_home_insurance"),
        F.col("carloan").alias("has_car_loan"),
        "_source_file", "_ingest_time",
    )


@dlt.table(
    name             = "silver.silver_customers",
    comment          = "Silver: Cleaned customer data — all 7 source files unified, deduplicated, standardized",
    table_properties = {"quality": "silver", "pipelines.autoOptimize.managed": "true"},
)
@dlt.expect_or_drop("valid_customer_id", "customer_id IS NOT NULL")
@dlt.expect("valid_region",              "region IN ('Central', 'West', 'South', 'East')")
@dlt.expect("balance_present",           "balance IS NOT NULL")
def silver_customers():
    return deduplicate_by_key(_transform_customers(), "customer_id")


@dlt.table(
    name    = "silver.quarantine_customers",
    comment = "Silver: Customer records that failed quality rules",
)
def quarantine_customers():
    df = _transform_customers()
    failed = df.filter(
        F.col("customer_id").isNull() |
        ~F.col("region").isin("Central", "West", "South", "East") |
        F.col("balance").isNull()
    )
    return (
        failed
        .withColumn(
            "quarantine_reason",
            F.when(F.col("customer_id").isNull(),
                   F.lit("missing customer_id"))
             .when(~F.col("region").isin("Central", "West", "South", "East"),
                   F.lit("invalid or missing region"))
             .when(F.col("balance").isNull(),
                   F.lit("missing balance"))
             .otherwise(F.lit("unknown")),
        )
        .withColumn("quarantine_timestamp", F.current_timestamp())
    )


# =============================================================
# CARS
# =============================================================

def _transform_cars():
    df = dlt.read_stream("bronze.bronze_cars")
    df = lowercase_all_column_names(df)
    df = extract_numeric_from_string(df, "mileage",   "mileage_kmpl")
    df = extract_numeric_from_string(df, "engine",    "engine_cc")
    df = extract_numeric_from_string(df, "max_power", "max_power_bhp")
    df = extract_numeric_from_string(df, "torque",    "torque_nm")
    df = standardize_to_lowercase(df, ["fuel", "transmission"])
    df = (
        df.withColumn("car_id",    F.col("car_id").cast("int"))
          .withColumn("km_driven", F.col("km_driven").cast("int"))
          .withColumn("seats",     F.col("seats").cast("int"))
    )
    df = add_outlier_flag(df, "km_driven", "km_driven_outlier_flag", 500000)
    return df.select(
        "car_id",
        F.trim(F.col("name")).alias("car_name"),
        "km_driven", "km_driven_outlier_flag", "fuel", "transmission",
        "mileage_kmpl", "engine_cc", "max_power_bhp", "torque_nm", "seats",
        F.lower(F.col("model")).alias("manufacturer"),
        "_source_file", "_ingest_time",
    )


@dlt.table(
    name             = "silver.silver_cars",
    comment          = "Silver: Cleaned car data",
    table_properties = {"quality": "silver", "pipelines.autoOptimize.managed": "true"},
)
@dlt.expect_or_drop("valid_car_id",  "car_id IS NOT NULL")
@dlt.expect("km_driven_outlier",     "km_driven_outlier_flag = 0")
def silver_cars():
    return deduplicate_by_key(_transform_cars(), "car_id")


@dlt.table(
    name    = "silver.quarantine_cars",
    comment = "Silver: Car records that failed quality rules",
)
def quarantine_cars():
    df = _transform_cars()
    failed = df.filter(F.col("car_id").isNull())
    return (
        failed
        .withColumn("quarantine_reason",    F.lit("missing car_id"))
        .withColumn("quarantine_timestamp", F.current_timestamp())
    )


# =============================================================
# POLICY
# =============================================================

def _transform_policy():
    df = dlt.read_stream("bronze.bronze_policy")
    df = lowercase_all_column_names(df)
    df = (
        df.withColumn("policy_number", F.col("policy_number").cast("int"))
          .withColumn("car_id",        F.col("car_id").cast("int"))
          .withColumn("customer_id",   F.col("customer_id").cast("int"))
    )
    df = df.withColumn("policy_bind_date", F.to_date(F.col("policy_bind_date"), "yyyy-MM-dd"))
    df = standardize_to_uppercase(df, ["policy_state"])
    df = parse_csl_limits(df, "policy_csl")
    df = (
        df.withColumn("policy_deductable",     F.col("policy_deductable").cast("decimal(10,2)"))
          .withColumn("policy_annual_premium", F.col("policy_annual_premium").cast("decimal(10,2)"))
          .withColumn("umbrella_limit",        F.col("umbrella_limit").cast("bigint"))
    )
    df = fix_negative_to_null(df, "umbrella_limit")
    df = df.withColumn(
        "is_active",
        F.when(F.col("policy_bind_date").isNotNull(), True).otherwise(False),
    )
    return df.select(
        "policy_number", "policy_bind_date", "policy_state",
        "csl_bodily_injury_per_person", "csl_bodily_injury_per_accident",
        F.col("policy_deductable").alias("deductible"),
        F.col("policy_annual_premium").alias("annual_premium"),
        "umbrella_limit", "car_id", "customer_id", "is_active",
        "_source_file", "_ingest_time",
    )


@dlt.table(
    name             = "silver.silver_policy",
    comment          = "Silver: Cleaned policy data — dates parsed, CSL split, negative umbrella fixed",
    table_properties = {"quality": "silver", "pipelines.autoOptimize.managed": "true"},
)
@dlt.expect_or_drop("valid_policy_number", "policy_number IS NOT NULL")
@dlt.expect_or_drop("valid_car_id",        "car_id IS NOT NULL")
@dlt.expect_or_drop("valid_customer_id",   "customer_id IS NOT NULL")
@dlt.expect("valid_premium",               "annual_premium > 0")
@dlt.expect("valid_umbrella_limit",        "umbrella_limit IS NULL OR umbrella_limit >= 0")
def silver_policy():
    return deduplicate_by_key(_transform_policy(), "policy_number")


@dlt.table(
    name    = "silver.quarantine_policy",
    comment = "Silver: Policy records that failed quality rules",
)
def quarantine_policy():
    df = _transform_policy()
    failed = df.filter(
        F.col("policy_number").isNull() |
        F.col("car_id").isNull() |
        F.col("customer_id").isNull() |
        (F.col("annual_premium") <= 0) |
        (F.col("umbrella_limit") < 0)
    )
    return (
        failed
        .withColumn(
            "quarantine_reason",
            F.when(F.col("policy_number").isNull(), F.lit("missing policy_number"))
             .when(F.col("car_id").isNull(),         F.lit("missing car_id"))
             .when(F.col("customer_id").isNull(),    F.lit("missing customer_id"))
             .when(F.col("annual_premium") <= 0,     F.lit("invalid annual premium"))
             .when(F.col("umbrella_limit") < 0,      F.lit("negative umbrella_limit"))
             .otherwise(F.lit("unknown")),
        )
        .withColumn("quarantine_timestamp", F.current_timestamp())
    )


# =============================================================
# SALES
# =============================================================

def _transform_sales():
    df = dlt.read_stream("bronze.bronze_sales")
    df = lowercase_all_column_names(df)
    df = (
        df.withColumn("sales_id", F.col("sales_id").cast("int"))
          .withColumn("car_id",   F.col("car_id").cast("int"))
    )
    df = parse_date_with_format(df, "ad_placed_on", "dd-MM-yyyy HH:mm")
    df = parse_date_with_format(df, "sold_on",      "dd-MM-yyyy HH:mm")
    df = calculate_date_diff_days(df, "sold_on", "ad_placed_on", "days_to_sell")
    df = df.withColumn("selling_price", F.col("original_selling_price").cast("decimal(12,2)"))
    df = standardize_to_lowercase(df, ["seller_type", "owner"])
    df = df.withColumn(
        "sale_status",
        F.when(F.col("sold_on").isNotNull(), "sold").otherwise("listed"),
    )
    return df.select(
        "sales_id", "ad_placed_on", "sold_on", "days_to_sell",
        "selling_price", "region", "state", "city",
        "seller_type", "owner", "sale_status", "car_id",
        "_source_file", "_ingest_time",
    )


@dlt.table(
    name             = "silver.silver_sales",
    comment          = "Silver: Cleaned sales data",
    table_properties = {"quality": "silver", "pipelines.autoOptimize.managed": "true"},
)
@dlt.expect_or_drop("valid_sales_id",  "sales_id IS NOT NULL")
@dlt.expect_or_drop("valid_car_id",    "car_id IS NOT NULL")
@dlt.expect("valid_selling_price",     "selling_price > 0")
@dlt.expect("valid_ad_date",           "ad_placed_on IS NOT NULL")
def silver_sales():
    return deduplicate_by_key(_transform_sales(), "sales_id")


@dlt.table(
    name    = "silver.quarantine_sales",
    comment = "Silver: Sales records that failed quality rules",
)
def quarantine_sales():
    df = _transform_sales()
    failed = df.filter(
        F.col("sales_id").isNull() |
        F.col("car_id").isNull() |
        F.col("selling_price").isNull() |
        (F.col("selling_price") <= 0) |
        F.col("ad_placed_on").isNull()
    )
    return (
        failed
        .withColumn(
            "quarantine_reason",
            F.when(F.col("sales_id").isNull(),
                   F.lit("missing sales_id — blank or partial row"))
             .when(F.col("car_id").isNull(),           F.lit("missing car_id"))
             .when(
                 F.col("selling_price").isNull() | (F.col("selling_price") <= 0),
                 F.lit("invalid selling price"),
             )
             .when(F.col("ad_placed_on").isNull(),     F.lit("missing ad placement date"))
             .otherwise(F.lit("unknown")),
        )
        .withColumn("quarantine_timestamp", F.current_timestamp())
    )


# =============================================================
# CLAIMS
# =============================================================

def _transform_claims():
    df = dlt.read_stream("bronze.bronze_claims")
    df = lowercase_all_column_names(df)
    df = (
        df.withColumn("claim_id",  F.col("claimid").cast("int"))
          .withColumn("policy_id", F.col("policyid").cast("int"))
    )
    df = clean_corrupted_time_format(df, "claim_logged_on")
    df = clean_corrupted_time_format(df, "claim_processed_on")
    df = clean_corrupted_time_format(df, "incident_date")
    df = replace_null_strings(df, ["claim_processed_on"])
    df = df.withColumn(
        "is_rejected",
        F.when(F.upper(F.col("claim_rejected")) == "Y", True)
         .when(F.upper(F.col("claim_rejected")) == "N", False)
         .otherwise(None),
    )
    df = replace_question_marks_with_null(df, ["collision_type"])
    df = parse_yes_no_to_boolean(df, "police_report_available")
    df = df.withColumn(
        "property_damage",
        F.when(F.upper(F.col("property_damage")) == "YES", True)
         .when(F.upper(F.col("property_damage")) == "NO",  False)
         .when(F.col("property_damage") == "?",            None)
         .otherwise(None),
    )
    df = standardize_to_lowercase(df, ["authorities_contacted", "collision_type",
                                        "incident_severity", "incident_type"])
    df = standardize_to_uppercase(df, ["incident_state"])
    df = clean_numeric_with_invalid_values(df, "bodily_injuries", ["?", "NULL"], "int")
    df = clean_numeric_with_invalid_values(df, "injury",          ["?", "NULL"], "decimal(10,2)")
    df = clean_numeric_with_invalid_values(df, "property",        ["?", "NULL"], "decimal(10,2)")
    df = clean_numeric_with_invalid_values(df, "vehicle",         ["?", "NULL"], "decimal(10,2)")
    df = clean_numeric_with_invalid_values(df, "witnesses",       ["?", "NULL"], "int")
    df = df.withColumn("number_of_vehicles_involved",
                       F.col("number_of_vehicles_involved").cast("int"))
    df = add_total_amount(df, "total_claim_amount", ["injury", "property", "vehicle"])
    return df.select(
        "claim_id", "policy_id", "claim_logged_on", "claim_processed_on",
        "incident_date", "is_rejected", "authorities_contacted", "bodily_injuries",
        "collision_type", F.col("incident_city").alias("city"),
        "incident_location", "incident_severity", "incident_state", "incident_type",
        F.col("injury").alias("injury_amount"),
        F.col("property").alias("property_amount"),
        "property_damage",
        F.col("vehicle").alias("vehicle_amount"),
        "witnesses", "number_of_vehicles_involved", "police_report_available",
        "total_claim_amount", "_source_file", "_ingest_time",
    )


@dlt.table(
    name             = "silver.silver_claims",
    comment          = "Silver: Cleaned claims data — corrupted dates nulled, booleans parsed, amounts cast",
    table_properties = {"quality": "silver", "pipelines.autoOptimize.managed": "true"},
)
@dlt.expect_or_drop("valid_claim_id",  "claim_id IS NOT NULL")
@dlt.expect_or_drop("valid_policy_id", "policy_id IS NOT NULL")
@dlt.expect("valid_total_claim",       "total_claim_amount > 0")
@dlt.expect("rejection_status_known",  "is_rejected IS NOT NULL")
def silver_claims():
    return deduplicate_by_key(_transform_claims(), "claim_id")


@dlt.table(
    name    = "silver.quarantine_claims",
    comment = "Silver: Claims records that failed quality rules",
)
def quarantine_claims():
    df = _transform_claims()
    failed = df.filter(
        F.col("claim_id").isNull() |
        F.col("policy_id").isNull() |
        (F.col("total_claim_amount") <= 0) |
        F.col("is_rejected").isNull()
    )
    return (
        failed
        .withColumn(
            "quarantine_reason",
            F.when(F.col("claim_id").isNull(),      F.lit("missing claim_id"))
             .when(F.col("policy_id").isNull(),      F.lit("missing policy_id"))
             .when(F.col("total_claim_amount") <= 0, F.lit("zero or negative total claim amount"))
             .when(F.col("is_rejected").isNull(),    F.lit("unknown rejection status"))
             .otherwise(F.lit("unknown")),
        )
        .withColumn("quarantine_timestamp", F.current_timestamp())
    )


# =============================================================
# QUALITY LOG
# =============================================================

@dlt.table(
    name    = "silver.silver_quality_log",
    comment = "Silver: Summary of all quality failures — one row per failure reason per entity",
)
def silver_quality_log():
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
            .groupBy("quarantine_reason")
            .agg(F.count("*").alias("failed_record_count"))
            .withColumn("entity",           F.lit(entity))
            .withColumn("quarantine_table", F.lit(table_name))
            .withColumn("log_timestamp",    F.current_timestamp())
            .select("entity", "quarantine_table", "quarantine_reason",
                    "failed_record_count", "log_timestamp")
        )
        summaries.append(summary)
    return reduce(lambda a, b: a.union(b), summaries)