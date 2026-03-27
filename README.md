# PrimeInsurance Analytics — Medallion Pipeline + AI Use Cases

End-to-end Databricks pipeline covering three data layers (Source → Bronze → Silver → Gold) and four AI/ML use cases built on top of the Gold layer.

---

## Repository Structure

```
├── source_to_bronze_.py       # Layer 1 — Raw CSV/JSON ingestion into Bronze
├── bronze_to_silver_.py       # Layer 2 — Cleaning, standardisation, quarantine
└── silver_to_gold_.py         # Layer 3 — Dimensions, facts, aggregations
```

---

## Architecture Overview

```
Raw Files (CSV / JSON)
        │
        ▼
  [ Bronze Layer ]   ──  All strings, no transforms, rescued data column
        │
        ▼
  [ Silver Layer ]   ──  Typed, cleaned, deduplicated, quarantine tables
        │
        ▼
  [ Gold Layer ]     ──  Dims, facts, aggregations, ready for BI
```

**Catalog:** `primeinsurance_analytics`  
**Schemas:** `bronze` · `silver` · `gold`  
**Source volume:** `/Volumes/primeinsurance_analytics/source_files/raw_files`

---

## Entities

| Entity | Source Format | Bronze Table | Silver Table | Gold Table |
|---|---|---|---|---|
| Customers | CSV | `bronze_customers` | `silver_customers` | `dim_customer` |
| Cars | CSV | `bronze_cars` | `silver_cars` | `dim_car` |
| Policy | CSV | `bronze_policy` | `silver_policy` | `dim_policy` |
| Sales | CSV | `bronze_sales` | `silver_sales` | `fact_sales` |
| Claims | JSON (multiLine array) | `bronze_claims` | `silver_claims` | `fact_claims` |

---

## Layer 1 — Source to Bronze (`source_to_bronze_.py`)

Ingests raw files using **Auto Loader** (`cloudFiles`). All columns land as strings — no type casting at this layer.

**Key behaviours:**
- Schema evolution mode: `addNewColumns` — new columns are added automatically, never dropped
- `_rescued_data` column captures any fields that don't match the schema
- Three metadata columns added to every table: `_source_file`, `_ingest_time`, `_source_region`
- Ghost rows (all-null data columns) are filtered out before landing
- Claims loaded with `multiLine: true` to handle JSON array format

**File patterns:**

| Table | Glob Pattern |
|---|---|
| customers | `**/customers_*.csv` |
| sales | `**/sales_*.csv` |
| claims | `**/claims_*.json` |
| cars | `**/cars*.csv` |
| policy | `**/policy*.csv` |

---

## Layer 2 — Bronze to Silver (`bronze_to_silver_.py`)

Applies all cleaning, standardisation, and quality rules. Every entity has a **silver table** (clean records) and a **quarantine table** (failed records with reasons).

### Quality Design

- **Hard rules** → record goes to quarantine if broken
- **Warn flags** → record stays in silver, flag column set to `1`
- **Quarantine type `drop`** → corrupt/impossible data, no recovery path
- **Quarantine type `quarantine`** → bad but identifiable, can be investigated
- **Deduplication** → most recent `_ingest_time` wins; older duplicates go to quarantine as `type=drop`
- **`silver_quality_log`** → single audit table aggregating all quarantine counts across all entities

### Cleaning Summary by Entity

**Customers**
- Unified 3 different customer ID column names (`customerid` / `customer_id` / `cust_id`)
- Region abbreviations expanded: `W/C/E/S/N` → `West/Central/East/South/North`
- Education typo fixed: `terto` → `tertiary`
- `admin.` trailing dot removed from job column
- `"NA"` strings replaced with real `NULL` in education, job, marital_status
- Swapped Education/Marital columns in `customers_6` source file corrected
- 133 negative balance values flagged (`balance_negative_flag`) but retained

**Cars**
- Embedded units stripped from string columns: `"23.4 kmpl"` → `mileage_kmpl` DOUBLE; same for `engine_cc`, `max_power_bhp`, `torque_nm`
- Fuel and transmission free text mapped to controlled vocabulary
- Seats outside range 4–10 quarantined
- Negative `km_driven` → quarantine with `type=drop`

**Policy**
- Column renamed: `policy_deductable` → `deductible` (typo fix); `policy_annual_premium` → `annual_premium`
- `policy_csl` string `"100/300"` split into `csl_bodily_injury_per_person` and `csl_bodily_injury_per_accident` (INT)
- `policy_state` lowercased values uppercased (e.g. `il` → `IL`)
- Negative `umbrella_limit` flagged (`umbrella_limit_warning=1`) and replaced with NULL
- `is_active` column derived from presence of `policy_bind_date`

**Sales**
- NULL `sold_on` treated as valid unsold listing — retained in silver with `sale_status='listed'`
- Dates parsed from `DD-MM-YYYY HH:MM` format to TIMESTAMP
- `days_to_sell` derived (NULL for unsold listings)
- `"Trustmark Dealer"` normalised to `dealer`
- Region/state/city lowercased for consistency
- `original_selling_price` renamed to `selling_price` (DECIMAL)
- `is_sold` boolean and `sale_status` derived columns added

**Claims**
- `claimid` / `policyid` renamed to `claim_id` / `policy_id` (INT)
- ISO dates parsed with `to_date("yyyy-MM-dd")` for `incident_date`, `claim_logged_on`, `claim_processed_on`
- String `"NULL"` in `claim_processed_on` treated as real NULL (open claim)
- `days_to_process` derived from real dates; NULL for open claims
- `claim_status` derived: `'open'` / `'closed'`
- `claim_rejected` Y/N → `is_rejected` BOOLEAN
- `"?"` values replaced with real NULL across `collision_type`, `police_report_available`, `property_damage`
- `incident_severity`, `incident_type`, `collision_type`, `authorities_contacted` mapped to controlled vocabulary
- `incident_state` uppercased and trimmed
- `police_report_available` / `property_damage` YES/NO → BOOLEAN
- `injury` / `property` / `vehicle` renamed to `injury_amount` / `property_amount` / `vehicle_amount`
- `total_claim_amount` derived as sum of the three amount columns

### Silver Tables

| Table | Purpose |
|---|---|
| `silver_customers` | Clean customers |
| `silver_cars` | Clean vehicles |
| `silver_policy` | Clean policies |
| `silver_sales` | Clean sales listings |
| `silver_claims` | Clean claims |
| `quarantine_customers` | Failed customer records |
| `quarantine_cars` | Failed car records |
| `quarantine_policy` | Failed policy records |
| `quarantine_sales` | Failed sales records |
| `quarantine_claims` | Failed claims records |
| `silver_quality_log` | Aggregated failure audit across all entities |

---

## Layer 3 — Silver to Gold (`silver_to_gold_.py`)

Builds the star schema and pre-aggregated tables for analytics and BI consumption.

### Dimension Tables

| Table | Grain | Source |
|---|---|---|
| `dim_date` | One row per calendar day (2000–2030) | Generated |
| `dim_customer` | One row per customer | `silver_customers` |
| `dim_car` | One row per vehicle | `silver_cars` |
| `dim_policy` | One row per policy | `silver_policy` |

### Fact Tables

| Table | Grain | Key joins |
|---|---|---|
| `fact_claims` | One row per claim | `dim_customer`, `dim_car`, `dim_policy`, `dim_date` |
| `fact_sales` | One row per sales listing | `dim_car`, `dim_date` |

Date keys (`incident_date_key`, `ad_date_key`, `sold_date_key`) are integer keys in `YYYYMMDD` format for joining to `dim_date`.

### Aggregation Tables

| Table | Business Question |
|---|---|
| `agg_claim_rejection_by_region_month` | Rejection rate per region per month — regulatory monitoring |
| `agg_claim_processing_time_by_severity` | Average days to process by incident severity — backlog monitoring |
| `agg_unsold_inventory_by_model_region` | Unsold listings by model and region — revenue leakage |
| `agg_claim_value_by_region_policy` | Total claim value by region and policy state — financial exposure |
| `agg_customer_count_by_region` | Unique customer count per region — deduplication verification |

---

## Running the Pipeline

Deploy each file as a separate **Delta Live Tables pipeline** in Databricks, in this order:

```
1. source_to_bronze_.py     →  target schema: bronze
2. bronze_to_silver_.py     →  target schema: silver
3. silver_to_gold_.py       →  target schema: gold
```

Each pipeline reads from the previous layer using `dlt.read()` — no cross-pipeline reads required within a single layer.

---

## Monitoring & Data Quality

After each silver pipeline run, query the quality log:

```sql
SELECT entity, quarantine_reason, quarantine_type, failed_record_count
FROM primeinsurance_analytics.silver.silver_quality_log
ORDER BY entity, failed_record_count DESC;
```

To check quarantine records for a specific entity:

```sql
SELECT quarantine_reason, quarantine_type, COUNT(*) AS cnt
FROM primeinsurance_analytics.silver.quarantine_claims
GROUP BY 1, 2
ORDER BY 3 DESC;
```

---

## AI Use Cases (Notebooks)

Four notebooks extend the pipeline with LLM-powered intelligence. All use the **Databricks-hosted `databricks-gpt-oss-20b`** model via the OpenAI-compatible client — no external API keys required.

| Notebook | Who it serves | Reads from | Writes to |
|---|---|---|---|
| `UC1_DQ_Explainer.ipynb` | Compliance team | `silver.silver_quality_log` | `gold.dq_explanation_report` |
| `UC2_claims_anomaly_detection.ipynb` | Fraud investigation team | `silver.silver_claims` | `gold.claim_anomaly_explanations` |
| `uc3_rag_policy_assistant.ipynb` | Claims adjusters | `gold.dim_policy` | `gold.rag_query_history` |
| `uc4_ai_business_insights.ipynb` | Executives | All gold tables | `gold.ai_business_insights` |

---

### UC1 — Data Quality Explainer

**Problem:** `silver_quality_log` contains field names, rule names, and row counts. The compliance team cannot interpret these into actions.

**Solution:** One LLM call per entity. The notebook collapses all failure rows for an entity into a bullet list and asks the model to produce a plain-English briefing memo — what failed, why it matters, and what to do next.

**How it works:**
1. Read `silver_quality_log`, group failures by entity
2. Build a bullet-list context string per entity (failure reason + count)
3. Send system prompt (compliance analyst persona) + context to the LLM
4. Parse the structured JSON response — extract only the `"text"` block from the reasoning model's output
5. Write entity + total_failed + explanation + timestamp to `gold.dq_explanation_report`

---

### UC2 — Claims Anomaly Detection

**Problem:** 3,000 claims, manual review, no consistent write-up standard. Investigators have no triage priority.

**Solution:** Statistical scoring in Spark flags suspicious claims; the LLM then writes a consistent investigator brief for every flagged claim.

**Anomaly scoring — 5 signals (weighted, total = 100):**

| Signal | Weight | Description |
|---|---|---|
| Rejected with non-zero payout | 30 | Data contradiction — highest fraud indicator |
| Claim amount > 95th percentile | 25 | Unusually high financial exposure |
| High witness count | 20 | Potential staged incident |
| Multiple vehicles involved | 15 | Elevated complexity |
| Rapid processing (< 5th percentile days) | 10 | Suspiciously fast turnaround |

Claims scoring ≥ 25 are flagged. Priority tiers: **Critical (≥ 70) · High (≥ 40) · Medium (≥ 25)**.

**How it works:**
1. Compute percentile thresholds from the actual data distribution (no hardcoded magic numbers)
2. Score each claim across 5 signals; sum weighted scores
3. Build a signal summary string listing which signals fired and why
4. LLM writes a structured triage memo per flagged claim (incident summary, red flags, recommended action)
5. Write claim_id + score + priority_tier + signals_fired + brief to `gold.claim_anomaly_explanations`

> **Bug fixed (v2):** `databricks-gpt-oss-20b` sometimes returns only a `reasoning` block with no `text` block. The parser was updated to fall back to the reasoning block content when no `text` block is present. Signal columns store weighted scores (10–30), never `1` — signal detection check corrected from `== 1` to `> 0`.

---

### UC3 — RAG Policy Assistant

**Problem:** Claims adjusters spend 10–15 minutes per claim looking up policy details. At 3,000 claims/year that is 500+ analyst hours lost to manual lookups.

**Solution:** True vector RAG pipeline — every policy row is embedded locally and indexed in FAISS. Adjusters ask natural-language questions; the system retrieves the relevant policies and generates a grounded answer.

**Stack:**
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, runs fully locally on the cluster — zero external API calls)
- Vector index: FAISS `IndexFlatIP` with L2-normalised vectors (inner product = cosine similarity)
- LLM: `databricks-gpt-oss-20b` for answer generation

**How it works:**
1. Read `gold.dim_policy`; convert every row to a natural-language sentence (structured columns → readable text)
2. Chunk strategy: one chunk = one policy row (rows are ~120–180 tokens, well within the 256-token sweet spot)
3. Embed all chunks locally with `all-MiniLM-L6-v2`
4. Build FAISS flat index; normalize embeddings to unit vectors
5. At query time: embed the question → FAISS top-k search → build context block with policy numbers cited → LLM generates answer
6. Log every query + retrieved context + answer to `gold.rag_query_history`

---

### UC4 — AI Business Insights

**Problem:** Executives receive dashboards but not synthesis. A dashboard shows Ohio has 312 claims averaging $28,400 — it does not say whether that is alarming, expected, or an opportunity.

**Solution:** The notebook aggregates KPIs from all gold tables into compact summaries, then asks the LLM to generate narrative executive insights across three business domains. Also demonstrates Databricks-native `ai_query()` inside SQL.

**Three business domains:**

| Domain | Business area |
|---|---|
| Policy Portfolio | Underwriting, coverage exposure, premium book |
| Claims Performance | Processing efficiency, rejection rates, backlog |
| Customer Profile | Customer health, cross-sell opportunity, risk segments |

**How it works:**
1. Collect pre-aggregated KPIs from all gold tables (COUNT, AVG, SUM, distributions) — raw rows are never sent to the LLM
2. One LLM call per domain: KPI block → narrative summary + recommended actions
3. `ai_query()` examples show SQL-native LLM calls directly inside `SELECT` statements (per-row enrichment without Python)
4. Write domain_name + summary_title + ai_generated_summary + kpi_json + timestamp to `gold.ai_business_insights`

> **Design rule:** KPIs are aggregated in Spark before being passed to the LLM. Sending 10 computed KPIs produces a better summary than 10,000 raw rows — and avoids context window limits.

---

## Full Table Inventory

| Layer | Table | Description |
|---|---|---|
| Bronze | `bronze_customers` | Raw customer CSV |
| Bronze | `bronze_cars` | Raw vehicle CSV |
| Bronze | `bronze_policy` | Raw policy CSV |
| Bronze | `bronze_sales` | Raw sales CSV |
| Bronze | `bronze_claims` | Raw claims JSON |
| Silver | `silver_customers` | Clean customers |
| Silver | `silver_cars` | Clean vehicles |
| Silver | `silver_policy` | Clean policies |
| Silver | `silver_sales` | Clean sales |
| Silver | `silver_claims` | Clean claims |
| Silver | `quarantine_customers` | Failed customer records |
| Silver | `quarantine_cars` | Failed car records |
| Silver | `quarantine_policy` | Failed policy records |
| Silver | `quarantine_sales` | Failed sales records |
| Silver | `quarantine_claims` | Failed claims records |
| Silver | `silver_quality_log` | Aggregated DQ audit |
| Gold | `dim_date` | Date dimension (2000–2030) |
| Gold | `dim_customer` | Customer dimension |
| Gold | `dim_car` | Vehicle dimension |
| Gold | `dim_policy` | Policy dimension |
| Gold | `fact_claims` | Claims fact table |
| Gold | `fact_sales` | Sales fact table |
| Gold | `agg_claim_rejection_by_region_month` | Rejection rate by region/month |
| Gold | `agg_claim_processing_time_by_severity` | Processing time by severity |
| Gold | `agg_unsold_inventory_by_model_region` | Unsold inventory by model/region |
| Gold | `agg_claim_value_by_region_policy` | Claim value by region/policy |
| Gold | `agg_customer_count_by_region` | Customer count by region |
| Gold | `dq_explanation_report` | UC1 — plain-English DQ memos |
| Gold | `claim_anomaly_explanations` | UC2 — fraud triage briefs |
| Gold | `rag_query_history` | UC3 — RAG query log |
| Gold | `ai_business_insights` | UC4 — executive narratives |
