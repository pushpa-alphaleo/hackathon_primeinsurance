import dlt
from pyspark.sql import functions as F


# =============================================================
# DIMENSIONS
# =============================================================

@dlt.table(
    name    = "gold.dim_date",
    comment = "Gold: Date dimension covering 2000-2030",
)
def dim_date():
    return spark.sql("""
        WITH date_range AS (
            SELECT date_add(DATE '2000-01-01', pos) AS full_date
            FROM (SELECT explode(sequence(0, 365 * 31)) AS pos)
        )
        SELECT
            CAST(year(full_date) * 10000 + month(full_date) * 100 + day(full_date) AS INT) AS date_key,
            full_date,
            year(full_date)                AS year,
            month(full_date)               AS month,
            quarter(full_date)             AS quarter,
            day(full_date)                 AS day,
            date_format(full_date, 'EEEE') AS day_of_week,
            CASE WHEN dayofweek(full_date) IN (1,7) THEN true ELSE false END AS is_weekend
        FROM date_range
    """)


@dlt.table(
    name    = "gold.dim_customer",
    comment = "Gold: Customer dimension",
)
def dim_customer():
    return dlt.read("silver.silver_customers").select(
        "customer_id", "state", "city", "region",
        "marital_status", "education", "job",
        "balance", "balance_negative_flag",
        "has_home_insurance", "has_car_loan",
    )


@dlt.table(
    name    = "gold.dim_car",
    comment = "Gold: Car/vehicle dimension",
)
def dim_car():
    return dlt.read("silver.silver_cars").select(
        "car_id", "car_name", "manufacturer", "fuel", "transmission",
        "mileage_kmpl", "engine_cc", "max_power_bhp", "seats",
        "km_driven", "km_driven_outlier_flag",
    )


@dlt.table(
    name    = "gold.dim_policy",
    comment = "Gold: Policy dimension",
)
def dim_policy():
    return dlt.read("silver.silver_policy").select(
        "policy_number", "policy_state", "policy_bind_date",
        "csl_bodily_injury_per_person", "csl_bodily_injury_per_accident",
        "deductible", "annual_premium", "umbrella_limit",
        "car_id", "customer_id", "is_active",
    )


# =============================================================
# FACTS
# =============================================================

@dlt.table(
    name    = "gold.fact_claims",
    comment = "Gold: Claims fact table — grain = one row per claim",
)
def fact_claims():
    claims = dlt.read("silver.silver_claims").alias("c")
    policy = dlt.read("silver.silver_policy").alias("p")
    return (
        claims.join(policy, F.col("c.policy_id") == F.col("p.policy_number"), "left")
        .select(
            F.col("c.claim_id"),
            F.col("c.policy_id"),
            F.col("p.customer_id"),
            F.col("p.car_id"),
            F.when(
                F.col("c.incident_date").isNotNull(),
                (F.year("c.incident_date") * 10000
                 + F.month("c.incident_date") * 100
                 + F.dayofmonth("c.incident_date")).cast("int"),
            ).alias("incident_date_key"),
            F.when(
                F.col("c.claim_logged_on").isNotNull() & F.col("c.claim_processed_on").isNotNull(),
                F.datediff("c.claim_processed_on", "c.claim_logged_on"),
            ).alias("days_to_process"),
            F.col("c.is_rejected"),
            F.col("c.total_claim_amount"),
            F.col("c.injury_amount"),
            F.col("c.property_amount"),
            F.col("c.vehicle_amount"),
            F.col("c.incident_type"),
            F.col("c.incident_severity"),
            F.col("c.incident_state"),
            F.col("c.collision_type"),
            F.col("c.bodily_injuries"),
            F.col("c.witnesses"),
            F.col("c.number_of_vehicles_involved"),
            F.col("c.police_report_available"),
            F.col("c.property_damage"),
            F.col("c.authorities_contacted"),
            F.col("c._source_file"),
            F.col("c._ingest_time"),
        )
    )


@dlt.table(
    name    = "gold.fact_sales",
    comment = "Gold: Sales fact table — grain = one row per car listing",
)
def fact_sales():
    return dlt.read("silver.silver_sales").select(
        "sales_id",
        "car_id",
        F.when(
            F.col("ad_placed_on").isNotNull(),
            (F.year("ad_placed_on") * 10000
             + F.month("ad_placed_on") * 100
             + F.dayofmonth("ad_placed_on")).cast("int"),
        ).alias("ad_date_key"),
        F.when(
            F.col("sold_on").isNotNull(),
            (F.year("sold_on") * 10000
             + F.month("sold_on") * 100
             + F.dayofmonth("sold_on")).cast("int"),
        ).alias("sold_date_key"),
        "selling_price",
        "days_to_sell",
        F.when(F.col("sale_status") == "sold", 1).otherwise(0).alias("is_sold"),
        "region", "state", "city", "seller_type", "owner",
        "_source_file", "_ingest_time",
    )


# =============================================================
# AGGREGATES
# =============================================================

@dlt.table(
    name    = "gold.agg_claim_rejection_by_region_month",
    comment = "Gold: Rejection rate per region per month",
)
def agg_claim_rejection_by_region_month():
    fc = dlt.read("gold.fact_claims").alias("fc")
    dc = dlt.read("gold.dim_customer").alias("dc")
    dd = dlt.read("gold.dim_date").alias("dd")
    return (
        fc.join(dc, F.col("fc.customer_id") == F.col("dc.customer_id"), "left")
          .join(dd, F.col("fc.incident_date_key") == F.col("dd.date_key"), "left")
          .groupBy(F.col("dc.region"), F.col("dd.year"), F.col("dd.month"))
          .agg(
              F.count("fc.claim_id").alias("total_claims"),
              F.sum(F.col("fc.is_rejected").cast("int")).alias("rejected_claims"),
              F.round(
                  F.sum(F.col("fc.is_rejected").cast("int")) / F.count("fc.claim_id") * 100, 2
              ).alias("rejection_rate_pct"),
              F.current_timestamp().alias("agg_timestamp"),
          )
    )


@dlt.table(
    name    = "gold.agg_claim_processing_time_by_severity",
    comment = "Gold: Avg days to process a claim by incident severity",
)
def agg_claim_processing_time_by_severity():
    return (
        dlt.read("gold.fact_claims")
        .filter(F.col("days_to_process").isNotNull())
        .groupBy("incident_severity")
        .agg(
            F.count("claim_id").alias("total_claims"),
            F.round(F.avg("days_to_process"), 1).alias("avg_days_to_process"),
            F.max("days_to_process").alias("max_days_to_process"),
            F.current_timestamp().alias("agg_timestamp"),
        )
    )


@dlt.table(
    name    = "gold.agg_unsold_inventory_by_manufacturer",
    comment = "Gold: Count of unsold listings per manufacturer",
)
def agg_unsold_inventory_by_manufacturer():
    fs = dlt.read("gold.fact_sales").alias("fs")
    dc = dlt.read("gold.dim_car").alias("dc")
    return (
        fs.filter(F.col("fs.is_sold") == 0)
          .join(dc, F.col("fs.car_id") == F.col("dc.car_id"), "left")
          .groupBy(F.col("dc.manufacturer"))
          .agg(
              F.count("fs.sales_id").alias("unsold_listings"),
              F.round(F.avg("fs.selling_price"), 2).alias("avg_listed_price"),
              F.round(F.sum("fs.selling_price"), 2).alias("total_listed_value"),
              F.current_timestamp().alias("agg_timestamp"),
          )
          .orderBy(F.col("unsold_listings").desc())
    )