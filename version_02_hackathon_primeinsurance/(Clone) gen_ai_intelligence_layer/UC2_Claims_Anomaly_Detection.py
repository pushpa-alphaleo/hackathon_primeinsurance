# Databricks notebook source
# MAGIC %pip install openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# UC2 — Claims Anomaly Detection
# Reads:  primeins.silver.silver_claims
# Writes: primeins.gold.claim_anomaly_explanations
#
# Purpose: Statistically flags suspicious claims, then uses the LLM to
#          write a consistent, data-driven investigator brief for each one.
#          Scales the fraud triage team without adding headcount.

# COMMAND ----------

# MAGIC %md
# MAGIC ## UC2 · Claims Anomaly Detection
# MAGIC **Who it serves:** Claims operations / fraud investigation team
# MAGIC **The problem:** 3,000 claims, manual review, no consistent write-up standard.
# MAGIC **What this does:**
# MAGIC 1. Uses Spark statistics to score every claim on 5 anomaly signals
# MAGIC 2. Flags claims that cross a threshold
# MAGIC 3. Sends each flagged claim to the LLM which writes an investigator brief:
# MAGIC    what is suspicious, why it matters, and what to check next

# COMMAND ----------

# ── 1. Setup ──────────────────────────────────────────────────────────────────
import json
from datetime import datetime
from openai import OpenAI
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    DoubleType, BooleanType, TimestampType, ArrayType
)

DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
WORKSPACE_URL    = spark.conf.get("spark.databricks.workspaceUrl")

client = OpenAI(
    api_key  = DATABRICKS_TOKEN,
    base_url = f"https://{WORKSPACE_URL}/serving-endpoints"
)

MODEL = "databricks-gpt-oss-20b"

# COMMAND ----------

# ── 2. Parse helper ───────────────────────────────────────────────────────────
def extract_llm_text(response) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        try:
            blocks = json.loads(content)
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text", "").strip()
        except (json.JSONDecodeError, TypeError):
            return content.strip()
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "").strip()
    return str(content).strip()

# COMMAND ----------

# ── 3. Read silver_claims ─────────────────────────────────────────────────────
claims = spark.read.table("primeinsurance_analytics.silver.silver_claims")

print(f"Total claims: {claims.count():,}")
display(claims.limit(5))

# COMMAND ----------

# ── 4. Statistical anomaly scoring ───────────────────────────────────────────
# Five signals — each contributes 1 point to an anomaly_score (0-5).
# We flag anything scoring >= 2. Thresholds are percentile-based so they
# adapt to the actual data distribution rather than hardcoded magic numbers.

# Compute population statistics for threshold-setting
stats = claims.select(
    F.percentile_approx("total_claim_amount", 0.95).alias("p95_amount"),
    F.percentile_approx("total_claim_amount", 0.99).alias("p99_amount"),
    F.mean("total_claim_amount").alias("mean_amount"),
    F.stddev("total_claim_amount").alias("std_amount"),
    F.percentile_approx("witnesses", 0.05).alias("p5_witnesses"),
    F.percentile_approx("bodily_injuries", 0.95).alias("p95_injuries"),
    F.percentile_approx("number_of_vehicles_involved", 0.95).alias("p95_vehicles")
).collect()[0]

p95_amount   = float(stats["p95_amount"]   or 0)
p99_amount   = float(stats["p99_amount"]   or 0)
mean_amount  = float(stats["mean_amount"]  or 0)
std_amount   = float(stats["std_amount"]   or 1)
p95_injuries = float(stats["p95_injuries"] or 0)
p95_vehicles = float(stats["p95_vehicles"] or 0)

print(f"Amount: mean={mean_amount:,.0f}  p95={p95_amount:,.0f}  p99={p99_amount:,.0f}")
print(f"p95 bodily_injuries={p95_injuries}  p95 vehicles={p95_vehicles}")

# COMMAND ----------

# DBTITLE 1,Cell 8
# Score each claim on 5 anomaly signals — weighted, total = 100
scored = claims.withColumn(
    # Signal 1 — weight 30: Rejected claim with non-zero payout
    # HIGHEST weight — data contradiction, almost always indicates fraud
    "sig_rejected_with_payout",
    F.when(
        (F.col("is_rejected") == True) & (F.col("total_claim_amount") > 0), 30
    ).otherwise(0)

).withColumn(
    # Signal 2 — weight 25: Property damage with no police report
    # Very strong fraud indicator — genuine accidents almost always have police reports
    "sig_damage_no_police",
    F.when(
        (F.col("property_damage") == True) &
        (F.col("police_report_available") == False), 25
    ).otherwise(0)

).withColumn(
    # Signal 3 — weight 20: Multi-vehicle incident with zero witnesses
    # Suspicious — hard to believe a multi-car accident with nobody watching
    "sig_no_witnesses_multi_vehicle",
    F.when(
        (F.col("number_of_vehicles_involved") > 1) &
        ((F.col("witnesses") == 0) | F.col("witnesses").isNull()), 20
    ).otherwise(0)

).withColumn(
    # Signal 4 — weight 15: Claim amount in top 5%
    # Moderate signal — some legitimate claims are just expensive
    "sig_high_amount",
    F.when(F.col("total_claim_amount") >= p95_amount, 15).otherwise(0)

).withColumn(
    # Signal 5 — weight 10: Extreme bodily injuries or vehicles (top 5%)
    # Lowest weight — could just be a genuinely bad accident
    "sig_extreme_severity",
    F.when(
        (F.col("bodily_injuries") >= p95_injuries) |
        (F.col("number_of_vehicles_involved") >= p95_vehicles), 10
    ).otherwise(0)

).withColumn(
    # Total anomaly score — 0 to 100
    "anomaly_score",
    F.col("sig_rejected_with_payout") +
    F.col("sig_damage_no_police") +
    F.col("sig_no_witnesses_multi_vehicle") +
    F.col("sig_high_amount") +
    F.col("sig_extreme_severity")

).withColumn(
    # ✅ Priority tier based on score
    "priority_tier",
    F.when(F.col("anomaly_score") >= 50, "HIGH")
     .when(F.col("anomaly_score") >= 25, "MEDIUM")
     .otherwise("LOW")
)

# Flag threshold: score >= 25 (at least one meaningful signal fired)
flagged = scored.filter(F.col("anomaly_score") >= 25).orderBy(F.col("anomaly_score").desc())

flagged_count = flagged.count()
total_claims  = claims.count()
percentage    = (100 * flagged_count / total_claims) if total_claims > 0 else 0.0

print(f"Flagged claims : {flagged_count:,} ({percentage:.1f}% of total)")
print(f"\nPriority breakdown:")
flagged.groupBy("priority_tier").count().orderBy("priority_tier").show()

display(flagged.select(
    "claim_id", "total_claim_amount", "anomaly_score", "priority_tier",
    "sig_rejected_with_payout", "sig_damage_no_police",
    "sig_no_witnesses_multi_vehicle", "sig_high_amount", "sig_extreme_severity"
).limit(20))

# COMMAND ----------

# ── 5. Build signal summary string per claim ──────────────────────────────────
SIGNAL_LABELS = {
    "sig_high_amount":               "Total claim amount is in the top 5% across all claims",
    "sig_rejected_with_payout":      "Claim was marked rejected yet still carries a non-zero payout",
    "sig_no_witnesses_multi_vehicle":"Multi-vehicle incident with no witnesses on record",
    "sig_damage_no_police":          "Property damage reported but no police report filed",
    "sig_extreme_severity":          "Extreme bodily injuries or unusually high vehicle count",
}

def build_signal_summary(row) -> str:
    fired = [label for col, label in SIGNAL_LABELS.items() if getattr(row, col, 0) == 1]
    return "; ".join(fired)

flagged_rows = flagged.collect()

# COMMAND ----------

# ── 6. Generate investigator briefs ───────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior insurance fraud investigator at PrimeInsurance.
You have been asked to write a brief triage memo for a claims investigator reviewing
a potentially suspicious claim.
The memo must:
1. Open with a one-sentence risk summary (HIGH / MEDIUM risk and why)
2. List each red flag in plain English (not variable names)
3. Explain the combined fraud pattern these flags suggest (2–3 sentences)
4. List exactly 3 concrete next steps the investigator should take
Keep the memo under 200 words. Be direct and specific — no filler sentences."""

def build_claim_prompt(row) -> str:
    signals = build_signal_summary(row)
    return f"""CLAIM DETAILS
Claim ID:              {row.claim_id}
Total claim amount:    ${float(row.total_claim_amount or 0):,.2f}
Incident type:         {row.incident_type or 'unknown'}
Incident severity:     {row.incident_severity or 'unknown'}
Collision type:        {row.collision_type or 'unknown'}
State:                 {row.incident_state or 'unknown'}
Bodily injuries:       {row.bodily_injuries or 0}
Vehicles involved:     {row.number_of_vehicles_involved or 0}
Witnesses:             {row.witnesses if row.witnesses is not None else 'none on record'}
Police report filed:   {'Yes' if row.police_report_available else 'No'}
Property damage:       {'Yes' if row.property_damage else 'No'}
Claim rejected flag:   {'Yes' if row.is_rejected else 'No'}
Anomaly score:         {row.anomaly_score}/5
Anomaly signals fired: {signals}

Write the investigator memo now."""

results = []

for i, row in enumerate(flagged_rows):
    user_prompt = build_claim_prompt(row)

    try:
        response = client.chat.completions.create(
            model    = MODEL,
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt}
            ],
            max_tokens  = 400,
            temperature = 0.2
        )
        brief  = extract_llm_text(response)
        status = "success"

    except Exception as e:
        brief  = f"[ERROR: {str(e)}]"
        status = "error"

    results.append({
        "claim_id":              int(row.claim_id),
        "anomaly_score":         int(row.anomaly_score),
        "priority_tier":         str(row.priority_tier or ""), 
        "total_claim_amount":    float(row.total_claim_amount or 0),
        "incident_severity":     str(row.incident_severity or ""),
        "incident_type":         str(row.incident_type or ""),
        "signals_fired":         build_signal_summary(row),
        "investigator_brief":    brief,
        "generation_status":     status,
        "generated_at":          datetime.utcnow()
    })

    if (i + 1) % 25 == 0:
        print(f"  Progress: {i+1}/{len(flagged_rows)} briefs written")

print(f"\n✓ Generated {len(results)} investigator briefs")

# COMMAND ----------

# ── 7. Preview top 3 ──────────────────────────────────────────────────────────
for r in results[:3]:
    print(f"\n{'='*65}")
    print(f"CLAIM {r['claim_id']}  |  Score {r['anomaly_score']}/5  |  ${r['total_claim_amount']:,.2f}")
    print(f"Signals: {r['signals_fired']}")
    print(f"{'─'*65}")
    print(r["investigator_brief"])

# COMMAND ----------

# ── 8. Write to gold ──────────────────────────────────────────────────────────
schema = StructType([
    StructField("claim_id",            IntegerType(),   False),
    StructField("anomaly_score",       IntegerType(),   True),
    StructField("priority_tier",      StringType(),    True),
    StructField("total_claim_amount",  DoubleType(),    True),
    StructField("incident_severity",   StringType(),    True),
    StructField("incident_type",       StringType(),    True),
    StructField("signals_fired",       StringType(),    True),
    StructField("investigator_brief",  StringType(),    True),
    StructField("generation_status",   StringType(),    True),
    StructField("generated_at",        TimestampType(), True),
])

output_df = spark.createDataFrame(results, schema=schema)

(
    output_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("primeinsurance_analytics.gold.claim_anomaly_explanations")
)

print(f"\n✅ Wrote {output_df.count()} rows → primeinsurance_analytics.gold.claim_anomaly_explanations")
display(
    spark.read.table("primeinsurance_analytics.gold.claim_anomaly_explanations")
    .orderBy(F.col("anomaly_score").desc())
    .select("claim_id","anomaly_score","total_claim_amount","signals_fired","investigator_brief")
    .limit(20)
)
