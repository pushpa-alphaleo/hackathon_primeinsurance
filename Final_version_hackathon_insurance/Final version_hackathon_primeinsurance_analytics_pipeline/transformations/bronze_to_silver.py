import dlt
from pyspark.sql import functions as F
from pyspark.sql import Window

# ============================================================
# SILVER TRANSFORMATION PIPELINE
# PrimeInsurance Analytics
#
# Bronze → Silver for all 5 entities:
#   customers, cars, policy, sales, claims
#
# Design principles:
#   1. Every quality rule declared INSIDE the pipeline
#   2. Failed records → dedicated quarantine table (never disappear)
#   3. Quarantine type: 'quarantine' (bad data) or 'drop' (corrupt/impossible)
#   4. Warning flags for issues that don't disqualify the record
#   5. silver_quality_log aggregates all failures for audit
# ============================================================


# ============================================================
# HELPER — Deduplicate by key using ingest time
# Keeps the most recently ingested row per key.
# All older duplicates → quarantine with type='drop'
# ============================================================

def _split_duplicates(df, key_col):
    window = Window.partitionBy(key_col).orderBy(F.col("_ingest_time").desc())
    df_ranked = df.withColumn("_row_num", F.row_number().over(window))

    deduped = df_ranked.filter(F.col("_row_num") == 1).drop("_row_num")

    duplicates = (
        df_ranked
        .filter(F.col("_row_num") > 1)
        .drop("_row_num")
        .withColumn("quarantine_reason",    F.lit(f"duplicate {key_col}"))
        .withColumn("quarantine_type",      F.lit("drop"))
        .withColumn("quarantine_timestamp", F.current_timestamp())
    )

    return deduped, duplicates


# ============================================================
# HELPER — Apply a dict mapping to a column (case-insensitive)
# Unmatched values → NULL (or a provided default string)
# ============================================================

def _apply_mapping(df, col_name, mapping_dict, default=None):
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
    expr = expr.otherwise(F.lit(default) if default is not None else F.lit(None).cast("string"))
    return df.withColumn(col_name, expr)


# ============================================================
# MAPPING DICTIONARIES
# Centralised here so changes only need to happen in one place
# ============================================================

REGION_MAP = {
    "c": "Central", "w": "West", "s": "South", "e": "East", "n": "North",
    "central": "Central", "west": "West", "south": "South",
    "east": "East", "north": "North",
}

EDUCATION_MAP = {
    "primary": "primary", "secondary": "secondary",
    "tertiary": "tertiary", "terto": "tertiary",  # typo fix
}

MARITAL_MAP = {
    "married": "married", "single": "single",
    "divorced": "divorced", "widow": "widow", "separated": "separated",
}

FUEL_MAP = {
    "petrol": "petrol", "diesel": "diesel", "electric": "electric",
    "hybrid": "hybrid", "cng": "cng", "lpg": "lpg",
}

TRANSMISSION_MAP = {"manual": "manual", "automatic": "automatic"}

SELLER_TYPE_MAP = {
    "individual": "individual", "dealer": "dealer",
    "trustmark dealer": "dealer", "company": "company",
}

OWNER_MAP = {
    "first owner":  "first owner",
    "second owner": "second owner",
    "third owner":  "third owner",
    "fourth & above owner": "fourth & above owner",
}

INCIDENT_SEVERITY_MAP = {
    "minor damage":   "minor damage",
    "major damage":   "major damage",
    "trivial damage": "trivial damage",
    "total loss":     "total loss",
}

INCIDENT_TYPE_MAP = {
    "single vehicle collision": "single vehicle collision",
    "multi-vehicle collision":  "multi-vehicle collision",
    "vehicle theft":            "vehicle theft",
    "parked car":               "parked car",
}

COLLISION_TYPE_MAP = {
    "rear collision":  "rear collision",
    "front collision": "front collision",
    "side collision":  "side collision",
}

AUTHORITIES_MAP = {
    "police": "police", "ambulance": "ambulance",
    "fire": "fire", "none": "none", "other": "other",
}


# ============================================================
# ENTITY 1 — CUSTOMERS
#
# Bronze issues:
#   - 3 different column names for customer ID across 7 files
#   - Region stored as abbreviation (W, C, E, S, N) in some files
#   - Inconsistent column names: Reg/Region, City/City_in_state,
#     Marital_status/Marital, Education/Edu
#   - Swapped Education/Marital in customers_6
#   - 'terto' typo in education
#   - 'NA' string values instead of real NULL
#   - 'admin.' trailing dot in job
#   - 133 negative balance values
#
# Hard rules (quarantine if broken):
#   - customer_id must not be NULL
#   - region must be in valid set after mapping
#   - state must not be NULL or blank
#
# Warn flags (record stays in Silver):
#   - balance_negative_flag
#   - marital_status_warning
#   - default_warning / insurance_warning / carloan_warning
# ============================================================

def _transform_customers():
    df = dlt.read("bronze_customers")

    # ── Step 1: Lowercase ALL column names first ──────────────
    # Required before coalesce — column names must match exactly
    for c in df.columns:
        df = df.withColumnRenamed(c, c.lower())

    # ── Step 2: Fix swapped Education/Marital in customers_6 ──
    # customers_6 has these two columns swapped at source
    df = df.withColumn("education_fixed",
        F.when(
            F.col("_source_file").contains("customers_6"),
            F.col("marital_status")
        ).otherwise(F.col("education"))
    ).withColumn("marital_status_fixed",
        F.when(
            F.col("_source_file").contains("customers_6"),
            F.col("education")
        ).otherwise(F.col("marital_status"))
    ).drop("education", "marital_status") \
     .withColumnRenamed("education_fixed",    "education") \
     .withColumnRenamed("marital_status_fixed","marital_status")

    # ── Step 3: Coalesce column name variants ─────────────────
    # Handles 7 different file schemas in one table
    df = df.withColumn("customer_id",
        F.coalesce(
            F.col("customerid").cast("int"),
            F.col("customer_id").cast("int"),
            F.col("cust_id").cast("int")
        )
    )
    df = df.withColumn("region",
        F.coalesce(F.col("reg"), F.col("region"))
    )
    df = df.withColumn("city",
        F.coalesce(F.col("city"), F.col("city_in_state"))
    )
    df = df.withColumn("education",
        F.coalesce(F.col("education"), F.col("edu"))
    )
    df = df.withColumn("marital_status",
        F.coalesce(F.col("marital_status"), F.col("marital"))
    )

    # ── Step 4: Replace "NA" strings with real NULLs ──────────
    for col_name in ["education", "job", "marital_status"]:
        df = df.withColumn(col_name,
            F.when(F.upper(F.col(col_name)) == "NA", None)
             .otherwise(F.col(col_name))
        )

    # ── Step 5: Standardise Region — handle abbreviations ─────
    # Converts W → West, C → Central etc.
    df = _apply_mapping(df, "region", REGION_MAP)

    # ── Step 6: Standardise Education — fix 'terto' typo ──────
    df = _apply_mapping(df, "education", EDUCATION_MAP)

    # ── Step 7: Remove trailing dot from 'admin.' in job ──────
    df = df.withColumn("job",
        F.regexp_replace(F.col("job"), r"\.$", "")
    )

    # ── Step 8: Standardise marital_status ────────────────────
    df = _apply_mapping(df, "marital_status", MARITAL_MAP)

    # ── Step 9: Lowercase job ─────────────────────────────────
    df = df.withColumn("job", F.lower(F.col("job")))

    # ── Step 10: Cast numeric columns ─────────────────────────
    df = (
        df
        .withColumn("has_default",      F.col("default").cast("int"))
        .withColumn("has_home_insurance",F.col("hhinsurance").cast("int"))
        .withColumn("has_car_loan",      F.col("carloan").cast("int"))
        .withColumn("balance",           F.col("balance").cast("decimal(10,2)"))
    )

    # ── Warn Flags ────────────────────────────────────────────
    df = df.withColumn("balance_negative_flag",
        F.when(F.col("balance") < 0, F.lit(1)).otherwise(F.lit(0))
    )
    df = df.withColumn("marital_status_warning",
        F.when(
            F.col("marital_status").isNull() |
            ~F.col("marital_status").isin("married","single","divorced","widow","separated"),
            F.lit(1)
        ).otherwise(F.lit(0))
    )
    df = df.withColumn("default_warning",
        F.when(
            F.col("has_default").isNull() | ~F.col("has_default").isin(0,1),
            F.lit(1)
        ).otherwise(F.lit(0))
    )
    df = df.withColumn("insurance_warning",
        F.when(
            F.col("has_home_insurance").isNull() | ~F.col("has_home_insurance").isin(0,1),
            F.lit(1)
        ).otherwise(F.lit(0))
    )
    df = df.withColumn("carloan_warning",
        F.when(
            F.col("has_car_loan").isNull() | ~F.col("has_car_loan").isin(0,1),
            F.lit(1)
        ).otherwise(F.lit(0))
    )

    return df.select(
        "customer_id",
        "state",
        "city",
        "region",
        "marital_status",        "marital_status_warning",
        "education",
        "job",
        "has_default",           "default_warning",
        "balance",               "balance_negative_flag",
        "has_home_insurance",    "insurance_warning",
        "has_car_loan",          "carloan_warning",
        "_source_file",          "_ingest_time"
    )


@dlt.table(
    name    = "silver.silver_customers",
    comment = "Silver: Customers — 7 files unified, deduplicated, region/education standardised."
)
def silver_customers():
    df = _transform_customers()
    deduped, _ = _split_duplicates(df, "customer_id")
    return deduped.filter(
        F.col("customer_id").isNotNull() &
        F.col("region").isin("Central","West","South","East","North") &
        F.col("state").isNotNull() &
        (F.trim(F.col("state")) != "")
    )


@dlt.table(
    name    = "silver.quarantine_customers",
    comment = "Quarantine: Customers that failed silver rules — with reason, type, timestamp."
)
def quarantine_customers():
    df = _transform_customers()
    deduped, duplicates = _split_duplicates(df, "customer_id")
    failed = deduped.filter(
        F.col("customer_id").isNull() |
        F.col("region").isNull() |
        (~F.col("region").isin("Central","West","South","East","North")) |
        F.col("state").isNull() |
        (F.trim(F.col("state")) == "")
    )
    quality_failures = (
        failed
        .withColumn("quarantine_reason",
            F.when(F.col("customer_id").isNull(), F.lit("missing customer_id"))
            .when(
                F.col("region").isNull() |
                ~F.col("region").isin("Central","West","South","East","North"),
                F.lit("invalid or missing region")
            )
            .when(
                F.col("state").isNull() | (F.trim(F.col("state")) == ""),
                F.lit("missing or blank state")
            )
            .otherwise(F.lit("unknown"))
        )
        .withColumn("quarantine_type",      F.lit("quarantine"))
        .withColumn("quarantine_timestamp", F.current_timestamp())
    )
    return quality_failures.unionByName(duplicates, allowMissingColumns=True)


# ============================================================
# ENTITY 2 — CARS
#
# Bronze issues (from ERD: car_id, name, km_driven, fuel,
#   transmission, mileage, engine, max_power, torque, seats, model):
#   - mileage/engine/max_power stored as strings with units embedded
#   - torque has 4+ different formats and units
#   - fuel/transmission as free text — unknown values possible
#   - seats can be outside valid passenger range
#   - negative km_driven possible
#
# Hard rules:
#   - car_id, car_name must not be NULL
#   - fuel must be in known list
#   - transmission must be manual or automatic
#   - seats must be 4-10
#   - km_driven must be >= 0 (DROP if negative)
#   - mileage_kmpl must be > 0 (DROP if zero/missing)
#
# Warn flags:
#   - km_driven_outlier_flag (> 500,000)
#   - engine_cc_warning, max_power_warning, torque_warning
# ============================================================

def _extract_first_number(df, src_col, tgt_col):
    """Extracts leading numeric value from string like '23.4 kmpl' → 23.4"""
    return df.withColumn(tgt_col,
        F.regexp_extract(
            F.regexp_replace(F.col(src_col), ",", ""),
            r"(\d+\.?\d*)", 1
        ).cast("double")
    )


def _transform_cars():
    df = dlt.read("bronze_cars")

    for c in df.columns:
        df = df.withColumnRenamed(c, c.lower())

    # Extract numeric values from string columns
    df = _extract_first_number(df, "mileage",   "mileage_kmpl")
    df = _extract_first_number(df, "engine",    "engine_cc")
    df = _extract_first_number(df, "max_power", "max_power_bhp")
    df = _extract_first_number(df, "torque",    "torque_nm")

    # Map fuel and transmission — unknown → NULL → caught by quarantine
    df = _apply_mapping(df, "fuel",         FUEL_MAP)
    df = _apply_mapping(df, "transmission", TRANSMISSION_MAP)

    df = (
        df
        .withColumn("car_id",    F.col("car_id").cast("int"))
        .withColumn("km_driven", F.col("km_driven").cast("int"))
        .withColumn("seats",     F.col("seats").cast("int"))
    )

    # Warn flags
    df = df.withColumn("km_driven_outlier_flag",
        F.when(F.col("km_driven") > 500000, F.lit(1)).otherwise(F.lit(0))
    )
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
        "fuel",
        "transmission",
        "seats",
        "km_driven",               "km_driven_outlier_flag",
        "mileage_kmpl",
        "engine_cc",               "engine_cc_warning",
        "max_power_bhp",           "max_power_warning",
        "torque_nm",               "torque_warning",
        F.lower(F.col("model")).alias("manufacturer"),
        "_source_file",            "_ingest_time"
    )


@dlt.table(
    name    = "silver.silver_cars",
    comment = "Silver: Cars — numeric extracted, fuel/transmission mapped, outliers flagged."
)
def silver_cars():
    df = _transform_cars()
    deduped, _ = _split_duplicates(df, "car_id")
    return deduped.filter(
        F.col("car_id").isNotNull() &
        F.col("car_name").isNotNull() &
        (F.trim(F.col("car_name")) != "") &
        F.col("fuel").isin("petrol","diesel","electric","hybrid","cng","lpg") &
        F.col("transmission").isin("manual","automatic") &
        F.col("seats").between(4, 10) &
        (F.col("km_driven") >= 0) &
        (F.col("mileage_kmpl") > 0)
    )


@dlt.table(
    name    = "silver.quarantine_cars",
    comment = "Quarantine: Cars that failed silver rules — type=quarantine or type=drop."
)
def quarantine_cars():
    df = _transform_cars()
    deduped, duplicates = _split_duplicates(df, "car_id")
    failed = deduped.filter(
        F.col("car_id").isNull() |
        F.col("car_name").isNull() |
        (F.trim(F.col("car_name")) == "") |
        F.col("fuel").isNull() |
        (~F.col("fuel").isin("petrol","diesel","electric","hybrid","cng","lpg")) |
        F.col("transmission").isNull() |
        (~F.col("transmission").isin("manual","automatic")) |
        F.col("seats").isNull() |
        (~F.col("seats").between(4, 10)) |
        (F.col("km_driven") < 0) |
        F.col("mileage_kmpl").isNull() |
        (F.col("mileage_kmpl") <= 0)
    )
    quality_failures = (
        failed
        .withColumn("quarantine_reason",
            F.when(F.col("car_id").isNull(), F.lit("missing car_id"))
            .when(F.col("car_name").isNull() | (F.trim(F.col("car_name")) == ""), F.lit("missing or blank car name"))
            .when(F.col("fuel").isNull() | ~F.col("fuel").isin("petrol","diesel","electric","hybrid","cng","lpg"), F.lit("invalid or missing fuel type"))
            .when(F.col("transmission").isNull() | ~F.col("transmission").isin("manual","automatic"), F.lit("invalid or missing transmission"))
            .when(F.col("seats").isNull() | ~F.col("seats").between(4, 10), F.lit("invalid seat count — outside 4-10"))
            .when(F.col("km_driven") < 0, F.lit("negative km_driven — physically impossible"))
            .when(F.col("mileage_kmpl").isNull() | (F.col("mileage_kmpl") <= 0), F.lit("zero or missing mileage — physically impossible"))
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
# ENTITY 3 — POLICY
#
# Bronze issues (from ERD: policy_number, policy_bind_date,
#   policy_state, policy_csl, policy_deductable,
#   policy_annual_premium, umbrella_limit, car_id, customer_id):
#   - policy_deductable is misspelled (should be deductible)
#   - policy_csl stored as string '100/300' — needs splitting
#   - policy_state as lowercase — needs uppercasing
#   - 1 negative umbrella_limit — impossible, fix to NULL
#   - umbrella_limit = 0 means no umbrella (valid)
#
# Hard rules:
#   - policy_number must not be NULL
#   - policy_bind_date must parse
#   - policy_state must not be NULL or blank
#   - policy_csl must match number/number pattern
#   - deductible must be >= 0
#   - annual_premium must be > 0
#   - car_id and customer_id must not be NULL
#
# Warn flags:
#   - umbrella_limit_warning (was negative, fixed to NULL)
#   - is_active derived from bind_date presence
# ============================================================

def _transform_policy():
    df = dlt.read("bronze_policy")

    for c in df.columns:
        df = df.withColumnRenamed(c, c.lower())

    df = (
        df
        .withColumn("policy_number", F.col("policy_number").cast("int"))
        .withColumn("car_id",        F.col("car_id").cast("int"))
        .withColumn("customer_id",   F.col("customer_id").cast("int"))
    )

    # Parse bind date
    df = df.withColumn("policy_bind_date",
        F.to_date(F.col("policy_bind_date"), "yyyy-MM-dd")
    )

    # Uppercase state abbreviation: il → IL
    df = df.withColumn("policy_state", F.upper(F.trim(F.col("policy_state"))))

    # Split policy_csl '100/300' → two numeric columns
    df = df.withColumn("csl_bodily_injury_per_person",
        F.split(F.col("policy_csl"), "/").getItem(0).cast("int")
    ).withColumn("csl_bodily_injury_per_accident",
        F.split(F.col("policy_csl"), "/").getItem(1).cast("int")
    )

    # Cast financial columns
    df = (
        df
        .withColumn("deductible",      F.col("policy_deductable").cast("decimal(10,2)"))
        .withColumn("annual_premium",  F.col("policy_annual_premium").cast("decimal(10,2)"))
        .withColumn("umbrella_limit",  F.col("umbrella_limit").cast("bigint"))
    )

    # Flag negative umbrella_limit then fix to NULL
    df = df.withColumn("umbrella_limit_warning",
        F.when(F.col("umbrella_limit") < 0, F.lit(1)).otherwise(F.lit(0))
    )
    df = df.withColumn("umbrella_limit",
        F.when(F.col("umbrella_limit") < 0, None).otherwise(F.col("umbrella_limit"))
    )

    # Derive is_active from bind date
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
        "deductible",                   # renamed from policy_deductable
        "annual_premium",               # renamed from policy_annual_premium
        "umbrella_limit",               "umbrella_limit_warning",
        "car_id",
        "customer_id",
        "is_active",
        "_source_file",                 "_ingest_time"
    )


@dlt.table(
    name    = "silver.silver_policy",
    comment = "Silver: Policy — CSL split, dates parsed, state uppercased, umbrella fixed."
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
    comment = "Quarantine: Policy records that failed silver rules."
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
            F.when(F.col("policy_number").isNull(), F.lit("missing policy_number"))
            .when(F.col("policy_bind_date").isNull(), F.lit("missing or unparseable policy_bind_date"))
            .when(F.col("policy_state").isNull() | (F.trim(F.col("policy_state")) == ""), F.lit("missing or blank policy_state"))
            .when(F.col("policy_csl").isNull() | ~F.col("policy_csl").rlike(r"^[0-9]+/[0-9]+$"), F.lit("missing or invalid policy_csl format"))
            .when(F.col("deductible").isNull() | (F.col("deductible") < 0), F.lit("missing or negative deductible"))
            .when(F.col("annual_premium").isNull() | (F.col("annual_premium") <= 0), F.lit("invalid annual_premium — zero or negative"))
            .when(F.col("car_id").isNull(), F.lit("missing car_id"))
            .when(F.col("customer_id").isNull(), F.lit("missing customer_id"))
            .otherwise(F.lit("unknown"))
        )
        .withColumn("quarantine_type",      F.lit("quarantine"))
        .withColumn("quarantine_timestamp", F.current_timestamp())
    )
    return quality_failures.unionByName(duplicates, allowMissingColumns=True)


# ============================================================
# ENTITY 4 — SALES
#
# Bronze issues (from ERD: sales_id, ad_placed_on, sold_on,
#   original_selling_price, Region, State, City,
#   seller_type, owner, car_id):
#   - 1 fully NULL ghost row
#   - 116 NULL sold_on (unsold listings — NOT bad data)
#   - Dates in DD-MM-YYYY HH:MM format (different from policy)
#   - seller_type has 'Trustmark Dealer' as variant
#   - Region in lowercase
#   - No is_sold flag — must be derived from sold_on
#
# Hard rules:
#   - sales_id must not be NULL
#   - ad_placed_on must parse
#   - selling_price must be > 0
#   - region must be in valid set
#   - state must not be NULL or blank
#   - car_id must not be NULL
#   NOTE: sold_on NULL is ALLOWED — means car is still listed
#
# Warn flags:
#   - seller_type_warning, owner_warning, city_warning
#   - sale_status derived: 'sold' or 'listed'
# ============================================================

def _transform_sales():
    df = dlt.read("bronze_sales")

    for c in df.columns:
        df = df.withColumnRenamed(c, c.lower())

    df = (
        df
        .withColumn("sales_id", F.col("sales_id").cast("int"))
        .withColumn("car_id",   F.col("car_id").cast("int"))
    )

    # Parse dates — DD-MM-YYYY HH:MM format
    df = df.withColumn("ad_placed_on",
        F.to_timestamp(F.col("ad_placed_on"), "dd-MM-yyyy HH:mm")
    )
    df = df.withColumn("sold_on",
        F.to_timestamp(F.col("sold_on"), "dd-MM-yyyy HH:mm")
    )

    # Compute days_to_sell only for sold records
    df = df.withColumn("days_to_sell",
        F.when(
            F.col("sold_on").isNotNull() & F.col("ad_placed_on").isNotNull(),
            F.datediff(F.col("sold_on").cast("date"), F.col("ad_placed_on").cast("date"))
        ).otherwise(None)
    )

    df = df.withColumn("selling_price",
        F.col("original_selling_price").cast("decimal(12,2)")
    )

    # Map seller_type and owner — unknown → NULL
    df = _apply_mapping(df, "seller_type", SELLER_TYPE_MAP)
    df = _apply_mapping(df, "owner",       OWNER_MAP)

    # Lowercase region/state/city for consistency
    df = df.withColumn("region", F.lower(F.trim(F.col("region"))))
    df = df.withColumn("state",  F.lower(F.trim(F.col("state"))))
    df = df.withColumn("city",   F.lower(F.trim(F.col("city"))))

    # Derive sale_status — key business indicator
    df = df.withColumn("sale_status",
        F.when(F.col("sold_on").isNotNull(), "sold").otherwise("listed")
    )

    # Derive is_sold boolean for easy filtering
    df = df.withColumn("is_sold",
        F.when(F.col("sold_on").isNotNull(), True).otherwise(False)
    )

    # Warn flags
    df = df.withColumn("seller_type_warning",
        F.when(F.col("seller_type").isNull(), F.lit(1)).otherwise(F.lit(0))
    )
    df = df.withColumn("owner_warning",
        F.when(F.col("owner").isNull(), F.lit(1)).otherwise(F.lit(0))
    )
    df = df.withColumn("city_warning",
        F.when(
            F.col("city").isNull() | (F.trim(F.col("city")) == ""),
            F.lit(1)
        ).otherwise(F.lit(0))
    )

    return df.select(
        "sales_id",
        "ad_placed_on",
        "sold_on",
        "days_to_sell",
        "selling_price",
        "region",
        "state",
        "city",                 "city_warning",
        "seller_type",          "seller_type_warning",
        "owner",                "owner_warning",
        "sale_status",
        "is_sold",
        "car_id",
        "_source_file",         "_ingest_time"
    )


@dlt.table(
    name    = "silver.silver_sales",
    comment = "Silver: Sales — dates parsed, sale_status derived, is_sold flag added, unsold listings kept."
)
def silver_sales():
    df = _transform_sales()
    deduped, _ = _split_duplicates(df, "sales_id")
    return deduped.filter(
        F.col("sales_id").isNotNull() &
        F.col("ad_placed_on").isNotNull() &
        # sold_on NOT required — NULL means car is still listed (valid)
        F.col("selling_price").isNotNull() &
        (F.col("selling_price") > 0) &
        F.col("region").isNotNull() &
        F.col("region").isin("east","west","central","south","north") &
        F.col("state").isNotNull() &
        (F.trim(F.col("state")) != "") &
        F.col("car_id").isNotNull()
    )


@dlt.table(
    name    = "silver.quarantine_sales",
    comment = "Quarantine: Sales records that failed silver rules."
)
def quarantine_sales():
    df = _transform_sales()
    deduped, duplicates = _split_duplicates(df, "sales_id")
    failed = deduped.filter(
        F.col("sales_id").isNull() |
        F.col("ad_placed_on").isNull() |
        F.col("selling_price").isNull() |
        (F.col("selling_price") <= 0) |
        F.col("region").isNull() |
        (~F.col("region").isin("east","west","central","south","north")) |
        F.col("state").isNull() |
        (F.trim(F.col("state")) == "") |
        F.col("car_id").isNull()
    )
    quality_failures = (
        failed
        .withColumn("quarantine_reason",
            F.when(F.col("sales_id").isNull(), F.lit("missing sales_id — blank or corrupt row"))
            .when(F.col("ad_placed_on").isNull(), F.lit("missing or unparseable ad_placed_on date"))
            .when(F.col("selling_price").isNull() | (F.col("selling_price") <= 0), F.lit("invalid selling price — zero or negative"))
            .when(F.col("region").isNull() | ~F.col("region").isin("east","west","central","south","north"), F.lit("invalid or missing region"))
            .when(F.col("state").isNull() | (F.trim(F.col("state")) == ""), F.lit("missing or blank state"))
            .when(F.col("car_id").isNull(), F.lit("missing car_id"))
            .otherwise(F.lit("unknown"))
        )
        .withColumn("quarantine_type",      F.lit("quarantine"))
        .withColumn("quarantine_timestamp", F.current_timestamp())
    )
    return quality_failures.unionByName(duplicates, allowMissingColumns=True)
# ============================================================
# ENTITY 5 — CLAIMS
# Updated for real synthetic ISO dates (yyyy-MM-dd)
#
# Bronze issues resolved:
#   - Real dates: parse with to_date() — no more corruption
#   - Claim_Processed_On NULL = open claim — keep as NULL
#   - police_report_available / property_damage: YES/NO/?
#   - Claim_Rejected: Y/N → boolean
#   - injury/property/vehicle = dollar amounts (renamed)
#   - collision_type: '?' → NULL
#   - String 'NULL' possible in Claim_Processed_On
#
# Hard rules (quarantine if broken):
#   - claim_id not NULL
#   - policy_id not NULL
#   - incident_date not NULL (real date now — enforceable)
#   - claim_logged_on not NULL (real date — enforceable)
#   - days_to_process not negative (processed before logged)
#   - incident_state not NULL or blank
#   - incident_severity in valid set
#   - injury_amount >= 0
#   - property_amount >= 0
#   - number_of_vehicles_involved >= 1
#
# Warn flags (record stays in Silver):
#   - claim_processed_on_warning  → NULL = open claim (expected)
#   - days_to_process_warning     → NULL = open claim (expected)
#   - collision_type_warning      → valid for single vehicle
#   - bodily_injuries_warning     → NULL or negative
# ============================================================

def _transform_claims():
    df = dlt.read("bronze_claims")

    # ── Step 1: Lowercase all column names ────────────────────
    for c in df.columns:
        df = df.withColumnRenamed(c, c.lower())

    # ── Step 2: Cast IDs ──────────────────────────────────────
    df = (
        df
        .withColumn("claim_id",  F.col("claimid").cast("int"))
        .withColumn("policy_id", F.col("policyid").cast("int"))
    )

    # ── Step 3: Parse real ISO dates ──────────────────────────
    # New data has real dates like '2022-02-12'
    # OLD code: extracted numbers from '34:00.0' — no longer needed
    df = df.withColumn("incident_date",
        F.to_date(F.col("incident_date"), "yyyy-MM-dd")
    )
    df = df.withColumn("claim_logged_on",
        F.to_date(F.col("claim_logged_on"), "yyyy-MM-dd")
    )
    # claim_processed_on:
    #   - NULL = open/unprocessed claim → keep as NULL (valid)
    #   - String "NULL" = also treat as NULL
    #   - Real date = parse normally
    df = df.withColumn("claim_processed_on",
        F.when(
            F.upper(F.col("claim_processed_on").cast("string")) == "NULL",
            None
        ).otherwise(
            F.to_date(F.col("claim_processed_on"), "yyyy-MM-dd")
        )
    )

    # ── Step 4: Calculate days_to_process ─────────────────────
    # Now accurate — real dates, not corrupted strings
    # NULL = open claim (claim_processed_on is NULL)
    df = df.withColumn("days_to_process",
        F.when(
            F.col("claim_processed_on").isNotNull() &
            F.col("claim_logged_on").isNotNull(),
            F.datediff(
                F.col("claim_processed_on"),
                F.col("claim_logged_on")
            )
        ).otherwise(None)
    )

    # ── Step 5: Derive claim_status ───────────────────────────
    df = df.withColumn("claim_status",
        F.when(F.col("claim_processed_on").isNotNull(), "closed")
         .otherwise("open")
    )

    # ── Step 6: Y/N → boolean ─────────────────────────────────
    df = df.withColumn("is_rejected",
        F.when(F.upper(F.col("claim_rejected")) == "Y", True)
        .when(F.upper(F.col("claim_rejected")) == "N", False)
        .otherwise(None)
    )

    # ── Step 7: Replace '?' with NULL before mapping ──────────
    for col_name in ["collision_type",
                     "police_report_available",
                     "property_damage"]:
        df = df.withColumn(col_name,
            F.when(F.col(col_name) == "?", None)
             .otherwise(F.col(col_name))
        )

    # ── Step 8: Apply mapping model ───────────────────────────
    df = _apply_mapping(df, "incident_severity",
                        INCIDENT_SEVERITY_MAP)
    df = _apply_mapping(df, "incident_type",
                        INCIDENT_TYPE_MAP)
    df = _apply_mapping(df, "collision_type",
                        COLLISION_TYPE_MAP)
    df = _apply_mapping(df, "authorities_contacted",
                        AUTHORITIES_MAP)

    # ── Step 9: Uppercase state ───────────────────────────────
    df = df.withColumn("incident_state",
        F.upper(F.trim(F.col("incident_state")))
    )

    # ── Step 10: YES/NO → boolean ─────────────────────────────
    for bool_col in ["police_report_available", "property_damage"]:
        df = df.withColumn(bool_col,
            F.when(F.upper(F.col(bool_col)) == "YES", True)
            .when(F.upper(F.col(bool_col)) == "NO",  False)
            .otherwise(None)
        )

    # ── Step 11: Clean numeric columns ────────────────────────
    for num_col, dtype in [
        ("bodily_injuries", "int"),
        ("injury",          "decimal(10,2)"),
        ("property",        "decimal(10,2)"),
        ("vehicle",         "decimal(10,2)"),
        ("witnesses",       "int"),
    ]:
        df = df.withColumn(num_col,
            F.when(
                F.col(num_col).cast("string").isin("?","NULL","null"),
                None
            ).otherwise(F.col(num_col).cast(dtype))
        )

    df = df.withColumn("number_of_vehicles_involved",
        F.col("number_of_vehicles_involved").cast("int")
    )

    # ── Step 12: Total claim amount ───────────────────────────
    df = df.withColumn("total_claim_amount",
        F.coalesce(F.col("injury"),   F.lit(0)) +
        F.coalesce(F.col("property"), F.lit(0)) +
        F.coalesce(F.col("vehicle"),  F.lit(0))
    )

    # ── Step 13: Warn Flags ───────────────────────────────────
    df = df.withColumn("claim_processed_on_warning",
        F.when(F.col("claim_processed_on").isNull(), F.lit(1))
         .otherwise(F.lit(0))
        # NULL = open claim — expected, not an error
    )
    df = df.withColumn("days_to_process_warning",
        F.when(F.col("days_to_process").isNull(), F.lit(1))
         .otherwise(F.lit(0))
        # NULL = open claim — expected
    )
    df = df.withColumn("collision_type_warning",
        F.when(F.col("collision_type").isNull(), F.lit(1))
         .otherwise(F.lit(0))
        # Valid for single vehicle incidents
    )
    df = df.withColumn("bodily_injuries_warning",
        F.when(
            F.col("bodily_injuries").isNull() |
            (F.col("bodily_injuries") < 0),
            F.lit(1)
        ).otherwise(F.lit(0))
    )

    return df.select(
        "claim_id",
        "policy_id",
        # Real parsed dates ✅
        "incident_date",
        "claim_logged_on",
        "claim_processed_on",      "claim_processed_on_warning",
        # Accurate processing time ✅
        "days_to_process",         "days_to_process_warning",
        # Business status ✅
        "claim_status",
        # Incident details
        "incident_state",
        F.col("incident_city").alias("city"),
        "incident_location",
        "incident_severity",
        "incident_type",
        "collision_type",          "collision_type_warning",
        # Claim details
        "is_rejected",
        "authorities_contacted",
        "bodily_injuries",         "bodily_injuries_warning",
        "number_of_vehicles_involved",
        "witnesses",
        "police_report_available",
        "property_damage",
        # Dollar amounts — renamed for clarity
        F.col("injury").alias("injury_amount"),
        F.col("property").alias("property_amount"),
        F.col("vehicle").alias("vehicle_amount"),
        "total_claim_amount",
        "_source_file",
        "_ingest_time"
    )


@dlt.table(
    name    = "silver.silver_claims",
    comment = "Silver: Claims — real ISO dates parsed, open claims retained, days_to_process accurate."
)
def silver_claims():
    df = _transform_claims()
    deduped, _ = _split_duplicates(df, "claim_id")
    return deduped.filter(
        F.col("claim_id").isNotNull() &
        F.col("policy_id").isNotNull() &
        # ✅ Date rules now enforceable — real dates in source
        F.col("incident_date").isNotNull() &
        F.col("claim_logged_on").isNotNull() &
        # claim_processed_on NOT required — NULL = open claim
        # Negative days = impossible → quarantine
        (
            F.col("days_to_process").isNull() |
            (F.col("days_to_process") >= 0)
        ) &
        F.col("incident_state").isNotNull() &
        (F.trim(F.col("incident_state")) != "") &
        F.col("incident_severity").isin(
            list(INCIDENT_SEVERITY_MAP.values())
        ) &
        (F.col("injury_amount")   >= 0) &
        (F.col("property_amount") >= 0) &
        F.col("number_of_vehicles_involved").isNotNull() &
        (F.col("number_of_vehicles_involved") >= 1)
    )


@dlt.table(
    name    = "silver.quarantine_claims",
    comment = "Quarantine: Claims that failed silver rules + duplicates."
)
def quarantine_claims():
    df = _transform_claims()
    deduped, duplicates = _split_duplicates(df, "claim_id")

    failed = deduped.filter(
        F.col("claim_id").isNull() |
        F.col("policy_id").isNull() |
        F.col("incident_date").isNull() |
        F.col("claim_logged_on").isNull() |
        (
            F.col("days_to_process").isNotNull() &
            (F.col("days_to_process") < 0)
        ) |
        F.col("incident_state").isNull() |
        (F.trim(F.col("incident_state")) == "") |
        F.col("incident_severity").isNull() |
        (~F.col("incident_severity").isin(
            list(INCIDENT_SEVERITY_MAP.values())
        )) |
        (F.col("injury_amount")   < 0) |
        (F.col("property_amount") < 0) |
        F.col("number_of_vehicles_involved").isNull() |
        (F.col("number_of_vehicles_involved") < 1)
    )
    quality_failures = (
        failed
        .withColumn("quarantine_reason",
            F.when(F.col("claim_id").isNull(),
                F.lit("missing claim_id — unidentifiable record"))
            .when(F.col("policy_id").isNull(),
                F.lit("missing policy_id — claim not linked to any policy"))
            .when(F.col("incident_date").isNull(),
                F.lit("missing or unparseable incident_date"))
            .when(F.col("claim_logged_on").isNull(),
                F.lit("missing or unparseable claim_logged_on"))
            .when(
                F.col("days_to_process").isNotNull() &
                (F.col("days_to_process") < 0),
                F.lit("processed_on before logged_on — impossible")
            )
            .when(
                F.col("incident_state").isNull() |
                (F.trim(F.col("incident_state")) == ""),
                F.lit("missing or blank incident_state")
            )
            .when(
                F.col("incident_severity").isNull() |
                ~F.col("incident_severity").isin(
                    list(INCIDENT_SEVERITY_MAP.values())
                ),
                F.lit("invalid or missing incident_severity")
            )
            .when(
                (F.col("injury_amount") < 0) |
                (F.col("property_amount") < 0),
                F.lit("negative claim amount — invalid")
            )
            .when(
                F.col("number_of_vehicles_involved").isNull() |
                (F.col("number_of_vehicles_involved") < 1),
                F.lit("invalid vehicle count — must be at least 1")
            )
            .otherwise(F.lit("unknown"))
        )
        # ✅ DROP if claim_id or policy_id missing — no recovery path
        # QUARANTINE for everything else — could be investigated/fixed
        .withColumn("quarantine_type",
            F.when(
                F.col("claim_id").isNull() |
                F.col("policy_id").isNull(),
                F.lit("drop")           # ← no identity = no recovery
            ).otherwise(
                F.lit("quarantine")     # ← bad data but identifiable
            )
        )
        .withColumn("quarantine_timestamp", F.current_timestamp())
    )

    return quality_failures.unionByName(duplicates, allowMissingColumns=True)


# ============================================================
# QUALITY LOG
#
# Aggregates ALL quarantine tables into a single audit trail.
# One row per entity + quarantine_reason + quarantine_type.
# Answers: what was caught, where, how many records affected.
# ============================================================

@dlt.table(
    name    = "silver.silver_quality_log",
    comment = "Quality audit log — aggregated count of all failures by entity, reason, type, timestamp."
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