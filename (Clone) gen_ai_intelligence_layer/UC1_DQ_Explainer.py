# Databricks notebook source
# MAGIC %pip install openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# UC1 — Data Quality Explainer
# Reads:  primeins.silver.silver_quality_log
# Writes: primeins.gold.dq_explanation_report
#
# Purpose: Turns technical DQ failure logs into plain-English
#          memos the compliance team can act on — no analyst needed.

# COMMAND ----------

# MAGIC %md
# MAGIC ## UC1 · Data Quality Explainer
# MAGIC **Who it serves:** Compliance team
# MAGIC **The problem:** `silver_quality_log` is full of field names, rule names, and row counts.
# MAGIC The compliance team does not know what any of it means for their work.
# MAGIC **What this does:** Sends each entity's DQ failures to the LLM and gets back a
# MAGIC plain-English memo: what broke, how many records were affected, and what to do about it.

# COMMAND ----------

# ── 1. Setup ──────────────────────────────────────────────────────────────────
import json
from datetime import datetime
from openai import OpenAI
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType

# Databricks auth — no external keys needed
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
WORKSPACE_URL    = spark.conf.get("spark.databricks.workspaceUrl")

client = OpenAI(
    api_key    = DATABRICKS_TOKEN,
    base_url   = f"https://{WORKSPACE_URL}/serving-endpoints"
)

MODEL = "databricks-gpt-oss-20b"

# COMMAND ----------

# ── 2. Helper: parse the model's structured response ──────────────────────────
# databricks-gpt-oss-20b returns a JSON list:
#   [{"type": "reasoning", "summary": [...]}, {"type": "text", "text": "..."}]
# We only want the "text" block.

def extract_llm_text(response) -> str:
    """Extract the plain-text answer from a databricks-gpt-oss-20b response."""
    content = response.choices[0].message.content

    # Sometimes content arrives as a string that wraps JSON, sometimes as a list
    if isinstance(content, str):
        try:
            blocks = json.loads(content)
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text", "").strip()
        except (json.JSONDecodeError, TypeError):
            return content.strip()          # Already plain text — return as-is
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "").strip()

    return str(content).strip()

# COMMAND ----------

# ── 3. Read the DQ log ────────────────────────────────────────────────────────
dq_log = spark.read.table("primeinsurance_analytics.bronze.silver_quality_log")

print(f"Total DQ failure groups: {dq_log.count()}")
display(dq_log.orderBy("entity", "failed_record_count"))

# COMMAND ----------

# ── 4. Build per-entity context strings ───────────────────────────────────────
# Collapse all failure rows for each entity into a readable bullet list,
# then send one LLM call per entity. This keeps prompts focused and short.

dq_by_entity = (
    dq_log
    .groupBy("entity")
    .agg(
        F.collect_list(
            F.struct("quarantine_reason", "failed_record_count")
        ).alias("issues"),
        F.sum("failed_record_count").alias("total_failed"),
        F.max("log_timestamp").alias("latest_run")
    )
    .collect()
)

print(f"Entities to explain: {[r['entity'] for r in dq_by_entity]}")

# COMMAND ----------

# ── 5. Generate explanations ──────────────────────────────────────────────────
# System prompt sets the persona once; user prompt carries the data per entity.

SYSTEM_PROMPT = """You are a data quality analyst writing briefing memos for a compliance team
at an automotive insurance company called PrimeInsurance.
The compliance team are not technical — they do not know what field names, rule names, or
quarantine tables are.
Your job is to read a list of data quality failures and write a SHORT, CLEAR memo (4–8 sentences)
that explains:
1. What type of records were affected (customers / cars / policies / sales / claims)
2. What was wrong with them in plain English (not field names)
3. How many records were affected
4. The likely business risk if the issue is left unresolved
5. One concrete recommended action for the compliance team
Do NOT use technical jargon. Do NOT mention column names, table names, or rule names.
Write in plain English a non-technical business person can act on immediately."""

def build_entity_prompt(entity: str, issues: list, total_failed: int) -> str:
    bullet_lines = "\n".join(
        f"  • {row['quarantine_reason'].replace('_', ' ')}: {row['failed_record_count']:,} records"
        for row in issues
    )
    return (
        f"Entity: {entity.upper()}\n"
        f"Total failed records: {total_failed:,}\n"
        f"Failure breakdown:\n{bullet_lines}\n\n"
        f"Write the compliance memo now."
    )

results = []

for row in dq_by_entity:
    entity       = row["entity"]
    issues       = row["issues"]
    total_failed = row["total_failed"]
    run_ts       = row["latest_run"]

    user_prompt = build_entity_prompt(entity, issues, total_failed)

    try:
        response = client.chat.completions.create(
            model    = MODEL,
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt}
            ],
            max_tokens = 512,
            temperature = 0.3   # Low temp = consistent, factual tone
        )
        explanation = extract_llm_text(response)
        status      = "success"

    except Exception as e:
        explanation = f"[ERROR generating explanation: {str(e)}]"
        status      = "error"

    results.append({
        "entity":           entity,
        "total_failed":     int(total_failed),
        "explanation":      explanation,
        "generation_status": status,
        "generated_at":     datetime.utcnow()
    })

    print(f"✓ {entity}: {len(explanation)} chars")

# COMMAND ----------

# ── 6. Preview results ────────────────────────────────────────────────────────
for r in results:
    print(f"\n{'='*60}")
    print(f"ENTITY: {r['entity'].upper()}  |  {r['total_failed']:,} failed records")
    print(f"{'='*60}")
    print(r["explanation"])

# COMMAND ----------

# ── 7. Write to gold ──────────────────────────────────────────────────────────
schema = StructType([
    StructField("entity",            StringType(),    False),
    StructField("total_failed",      IntegerType(),   True),
    StructField("explanation",       StringType(),    True),
    StructField("generation_status", StringType(),    True),
    StructField("generated_at",      TimestampType(), True),
])

output_df = spark.createDataFrame(results, schema=schema)

(
    output_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("primeinsurance_analytics.gold.dq_explanation_report")
)

print(f"\n✅ Wrote {output_df.count()} rows → primeins.gold.dq_explanation_report")
display(spark.read.table("primeinsurance_analytics.gold.dq_explanation_report"))
