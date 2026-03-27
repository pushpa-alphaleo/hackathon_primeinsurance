import dlt
from pyspark.sql import functions as F


SILVER = "primeinsurance_analytics.silver"


# ─────────────────────────────────────────────
# dim_date — no changes needed
# ─────────────────────────────────────────────
@dlt.table(
    name    = "gold.dim_date",
    comment = "Gold: Date dimension covering 2000–2030"
)
def dim_date():
    return (
        spark.range(0, 365 * 31)
        .select(
            F.expr("date_add(date '2000-01-01', CAST(id AS INT))").alias("full_date")
        )
        .select(
            (F.year("full_date") * 10000 +
             F.month("full_date") * 100 +
             F.dayofmonth("full_date")).cast("int").alias("date_key"),
            "full_date",
            F.year("full_date").alias("year"),
            F.month("full_date").alias("month"),
            F.quarter("full_date").alias("quarter"),
            F.dayofmonth("full_date").alias("day"),
            F.date_format("full_date", "EEEE").alias("day_of_week"),
            F.when(F.dayofweek("full_date").isin(1, 7), True)
             .otherwise(False).alias("is_weekend")
        )
    )


# ─────────────────────────────────────────────
# dim_customer
# ─────────────────────────────────────────────
@dlt.table(
    name    = "gold.dim_customer",
    comment = "Gold: Customer dimension"
)
def dim_customer():
    return (
        dlt.read("silver.silver_customers")  
        .select(
            "customer_id",
            "state",
            "city",
            "region",
            "marital_status",
            "education",
            "job",
            "balance",
            "balance_negative_flag",
            "has_home_insurance",
            "has_car_loan"
        )
    )


# ─────────────────────────────────────────────
# dim_car
# ─────────────────────────────────────────────
@dlt.table(
    name    = "gold.dim_car",
    comment = "Gold: Car/vehicle dimension"
)
def dim_car():
    return (
        dlt.read("silver.silver_cars")        
        .select(
            "car_id",
            "car_name",
            "manufacturer",
            "fuel",
            "transmission",
            "mileage_kmpl",
            "engine_cc",
            "max_power_bhp",
            "seats",
            "km_driven",
            "km_driven_outlier_flag"
        )
    )


# ─────────────────────────────────────────────
# dim_policy
# ─────────────────────────────────────────────
@dlt.table(
    name    = "gold.dim_policy",
    comment = "Gold: Policy dimension"
)
def dim_policy():
    return (
        dlt.read("silver.silver_policy")      # ✅ dlt.read()
        .select(
            "policy_number",
            "policy_state",
            "policy_bind_date",
            "csl_bodily_injury_per_person",
            "csl_bodily_injury_per_accident",
            "deductible",
            "annual_premium",
            "umbrella_limit",
            "car_id",
            "customer_id",
            "is_active"
        )
    )


# ─────────────────────────────────────────────
# fact_claims
# Updated: includes claim_status, real dates
# ─────────────────────────────────────────────
@dlt.table(
    name    = "gold.fact_claims",
    comment = "Gold: Claims fact table — one row per claim"
)
def fact_claims():
    claims = dlt.read("silver.silver_claims")          # ✅ dlt.read()
    policy = dlt.read("silver.silver_policy").select(  # ✅ dlt.read()
        "policy_number", "customer_id", "car_id"
    )

    return (
        claims
        .join(policy,
              claims.policy_id == policy.policy_number,
              "left")
        .select(
            "claim_id",
            claims.policy_id,
            F.col("customer_id"),
            F.col("car_id"),

            # Real dates ✅ — now reliable
            "incident_date",
            "claim_logged_on",
            "claim_processed_on",

            # Date key for joining to dim_date
            F.when(F.col("incident_date").isNotNull(),
                (F.year("incident_date") * 10000 +
                 F.month("incident_date") * 100 +
                 F.dayofmonth("incident_date")).cast("int")
            ).alias("incident_date_key"),

            # Accurate days_to_process ✅ — from real dates
            "days_to_process",
            "days_to_process_warning",

            # Claim status ✅ — new column from silver
            "claim_status",             # 'open' or 'closed'

            # Claim financials
            "is_rejected",
            "total_claim_amount",
            "injury_amount",
            "property_amount",
            "vehicle_amount",

            # Incident details
            "incident_type",
            "incident_severity",
            "incident_state",
            "collision_type",
            "bodily_injuries",
            "witnesses",
            "number_of_vehicles_involved",
            "police_report_available",
            "property_damage",
            "authorities_contacted",

            # Warn flags for downstream analysis
            "claim_processed_on_warning",
            "collision_type_warning",

            "_source_file",
            "_ingest_time"
        )
    )


# ─────────────────────────────────────────────
# fact_sales 
# ─────────────────────────────────────────────
@dlt.table(
    name    = "gold.fact_sales",
    comment = "Gold: Sales fact table — one row per car listing"
)
def fact_sales():
    return (
        dlt.read("silver.silver_sales")       # ✅ dlt.read()
        .select(
            "sales_id",
            "car_id",

            # Date keys
            F.when(F.col("ad_placed_on").isNotNull(),
                (F.year("ad_placed_on") * 10000 +
                 F.month("ad_placed_on") * 100 +
                 F.dayofmonth("ad_placed_on")).cast("int")
            ).alias("ad_date_key"),

            F.when(F.col("sold_on").isNotNull(),
                (F.year("sold_on") * 10000 +
                 F.month("sold_on") * 100 +
                 F.dayofmonth("sold_on")).cast("int")
            ).alias("sold_date_key"),

            "selling_price",
            "days_to_sell",
            "is_sold",
            "sale_status",
            "region",
            "state",
            "city",
            "seller_type",
            "owner",
            "_source_file",
            "_ingest_time"
        )
    )
# ─────────────────────────────────────────────
# agg (a): Claim rejection rate by region + month
# Business failure: Regulatory pressure
# ─────────────────────────────────────────────
@dlt.table(
    name    = "gold.agg_claim_rejection_by_region_month",
    comment = "Gold: Rejection rate per region per month — regulatory monitoring"
)
def agg_claim_rejection_by_region_month():
    fact    = dlt.read("gold.fact_claims")
    dim_cust = dlt.read("gold.dim_customer")
    dim_dt   = dlt.read("gold.dim_date")
 
    return (
        fact
        .join(dim_cust, "customer_id", "left")
        .join(dim_dt, fact.incident_date_key == dim_dt.date_key, "left")
        .groupBy("region", "year", "month")
        .agg(
            F.count("claim_id").alias("total_claims"),
            F.sum(F.col("is_rejected").cast("int")).alias("rejected_claims"),
            F.round(
                F.sum(F.col("is_rejected").cast("int")) /
                F.count("claim_id") * 100, 2
            ).alias("rejection_rate_pct")
        )
        .withColumn("agg_timestamp", F.current_timestamp())
    )
 
 
# ─────────────────────────────────────────────
# agg (b): Average days to process by severity
# Business failure: Claims backlog
# ─────────────────────────────────────────────
@dlt.table(
    name    = "gold.agg_claim_processing_time_by_severity",
    comment = "Gold: Avg days to process a claim by incident severity — backlog monitoring"
)
def agg_claim_processing_time_by_severity():
    return (
        dlt.read("gold.fact_claims")
        .filter(F.col("days_to_process").isNotNull())
        .groupBy("incident_severity")
        .agg(
            F.count("claim_id").alias("total_claims"),
            F.round(F.avg("days_to_process"), 1).alias("avg_days_to_process"),
            F.max("days_to_process").alias("max_days_to_process")
        )
        .withColumn("agg_timestamp", F.current_timestamp())
    )
 
 
# ─────────────────────────────────────────────
# agg (c): Unsold inventory by manufacturer
# Business failure: Revenue leakage
# ─────────────────────────────────────────────
@dlt.table(
    name    = "gold.agg_unsold_inventory_by_model_region",
    comment = "Gold: Unsold listings per model and region — revenue leakage"
)
def agg_unsold_inventory_by_model_region():
    sales   = dlt.read("gold.fact_sales")   # ✅ no schema prefix
    dim_car = dlt.read("gold.dim_car")      # ✅ no schema prefix

    return (
        sales
        .filter(F.col("is_sold") == 0)
        .join(dim_car, "car_id", "left")
        .groupBy("manufacturer", "car_name", "region")
        .agg(
            F.count("sales_id").alias("unsold_listings"),
            F.round(F.avg("selling_price"), 2).alias("avg_listed_price"),
            F.round(F.sum("selling_price"), 2).alias("total_listed_value"),
            F.round(F.avg("days_to_sell"), 1).alias("avg_days_listed"),
            F.max("days_to_sell").alias("max_days_listed")
        )
        .orderBy(F.col("unsold_listings").desc())
        .withColumn("agg_timestamp", F.current_timestamp())
    )
@dlt.table(
    name    = "gold.agg_claim_value_by_region_policy",
    comment = "Gold: Total claim value by region and policy type — financial exposure"
)
def agg_claim_value_by_region_policy():
    fact     = dlt.read("gold.fact_claims")
    dim_cust = dlt.read("gold.dim_customer")
    dim_pol  = dlt.read("gold.dim_policy")

    return (
        fact
        .join(dim_cust, "customer_id", "left")
        .join(dim_pol, fact.policy_id == dim_pol.policy_number, "left")
        .groupBy("region", "policy_state")
        .agg(
            F.count("claim_id").alias("total_claims"),
            F.round(F.sum("total_claim_amount"), 2).alias("total_claim_value"),
            F.round(F.avg("total_claim_amount"), 2).alias("avg_claim_value"),
            F.sum(F.col("is_rejected").cast("int")).alias("rejected_claims")
        )
        .withColumn("agg_timestamp", F.current_timestamp())
    )
@dlt.table(
    name    = "gold.agg_customer_count_by_region",
    comment = "Gold: Unique customer count per region — identity deduplication"
)
def agg_customer_count_by_region():
    return (
        dlt.read("gold.dim_customer")
        .groupBy("region")
        .agg(
            F.countDistinct("customer_id").alias("unique_customers"),
            F.count("customer_id").alias("total_records"),
            (F.count("customer_id") - F.countDistinct("customer_id"))
             .alias("duplicate_count")   # should be 0 if silver dedup worked
        )
        .withColumn("agg_timestamp", F.current_timestamp())
    )