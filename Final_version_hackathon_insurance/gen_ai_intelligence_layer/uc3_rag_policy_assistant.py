# Databricks notebook source
# MAGIC %pip install openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# UC3 — RAG Policy Assistant (COMPLETE — Vector Search Edition)
# Reads : primeinsurance_analytics.gold.dim_policy
# Writes: primeinsurance_analytics.gold.rag_query_history
#
# HOW TO IMPORT INTO DATABRICKS:
#   Workspace → Import → select this .py file → Import
#   Then attach to a cluster with ML runtime and run cell by cell

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 1 — Install Dependencies
# Run once. Restart kernel after if prompted.
# ─────────────────────────────────────────────────────────────

%pip install sentence-transformers faiss-cpu --quiet

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 2 — Imports and LLM Client Setup
# ─────────────────────────────────────────────────────────────

import uuid
import numpy as np
from datetime import datetime

from openai import OpenAI
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType,
    FloatType, BooleanType, TimestampType
)

from sentence_transformers import SentenceTransformer
import faiss

# ── Auto-fetch Databricks credentials — no manual key needed ──
DATABRICKS_TOKEN = (
    dbutils.notebook.entry_point
    .getDbutils().notebook().getContext()
    .apiToken().get()
)

WORKSPACE_URL = spark.conf.get("spark.databricks.workspaceUrl")

client = OpenAI(
    api_key  = DATABRICKS_TOKEN,
    base_url = f"https://{WORKSPACE_URL}/serving-endpoints"
)

MODEL = "databricks-gpt-oss-20b"

print("✅ LLM client ready")
print(f"   Workspace : {WORKSPACE_URL}")
print(f"   Model     : {MODEL}")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 3 — LLM Response Helper
#
# databricks-gpt-oss-20b returns TWO blocks:
#   Block 1: {"type": "reasoning"} — model's internal chain-of-thought
#   Block 2: {"type": "text"}      — the actual answer
#
# This function extracts only the clean text block.
# ─────────────────────────────────────────────────────────────

def get_llm_response(prompt: str, system: str = None, max_tokens: int = 600) -> str:
    """
    Calls the LLM and returns a clean text answer.
    Handles the reasoning-block response format of databricks-gpt-oss-20b.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model      = MODEL,
        messages   = messages,
        max_tokens = max_tokens
    )

    raw = response.choices[0].message.content

    # Handle two-block response format
    if isinstance(raw, list):
        for block in raw:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "").strip()

    return str(raw).strip()


# Quick connection test
test = get_llm_response("Reply with exactly: UC3 RAG pipeline ready")
print(f"LLM test: {test}")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 4 — Preview dim_policy
# ─────────────────────────────────────────────────────────────

policy_table = spark.read.table("primeinsurance_analytics.gold.dim_policy")

print(f"Total policies in dim_policy: {policy_table.count():,}")
print(f"\nColumns:")
for col in policy_table.columns:
    print(f"  → {col}")

print("\nSample rows:")
policy_table.show(5, truncate=False)

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 5 — Convert Every Policy Row to Natural Language
# ─────────────────────────────────────────────────────────────

def format_currency(value) -> str:
    """Safely format a value as a dollar amount."""
    if value is None or str(value).lower() in ("none", "null", ""):
        return "not specified"
    try:
        return f"${float(value):,.2f}"
    except:
        return str(value)


def policy_row_to_text(row: dict) -> str:
    """
    Converts one dim_policy row dict into a natural language document.

    This is the critical conversion step that makes structured data
    compatible with vector embedding models.

    The resulting text reads like a policy summary a human would write,
    which allows the embedding model to understand semantic relationships
    (e.g., "umbrella coverage" maps to the umbrella_limit field).
    """
    policy_num   = row.get("policy_number", "Unknown")
    state        = row.get("policy_state", "Unknown")
    bind_date    = row.get("policy_bind_date", "Unknown")
    is_active    = row.get("is_active", False)
    premium      = format_currency(row.get("annual_premium"))
    deductible   = format_currency(row.get("deductible"))
    bi_person    = format_currency(row.get("csl_bodily_injury_per_person"))
    bi_accident  = format_currency(row.get("csl_bodily_injury_per_accident"))
    umbrella     = format_currency(row.get("umbrella_limit"))
    customer_id  = row.get("customer_id", "Unknown")
    car_id       = row.get("car_id", "Unknown")

    active_str   = "currently active and in force" if is_active else "currently inactive"
    umbrella_str = (
        f"This policy includes umbrella coverage with a limit of {umbrella}."
        if row.get("umbrella_limit") and float(row.get("umbrella_limit") or 0) > 0
        else "This policy does not include umbrella coverage."
    )

    return (
        f"Policy number {policy_num} is registered in {state} and was bound on {bind_date}. "
        f"The policy is {active_str}. "
        f"The annual premium for this policy is {premium}. "
        f"The deductible — the amount the customer pays out of pocket before coverage kicks in — is {deductible}. "
        f"Bodily injury coverage limits are {bi_person} per person and {bi_accident} per accident. "
        f"{umbrella_str} "
        f"This policy is linked to customer ID {customer_id} and vehicle ID {car_id}."
    )


# Load ALL policies from the Gold table
print("Loading all policies from dim_policy...")
all_policy_rows = spark.read.table("primeinsurance_analytics.gold.dim_policy").collect()

# Convert every row to a natural language document
documents = []
for row in all_policy_rows:
    row_dict = row.asDict()
    documents.append({
        "policy_number" : row_dict.get("policy_number"),
        "policy_state"  : row_dict.get("policy_state"),
        "is_active"     : row_dict.get("is_active"),
        "annual_premium": float(row_dict.get("annual_premium") or 0),
        "deductible"    : float(row_dict.get("deductible") or 0),
        "umbrella_limit": float(row_dict.get("umbrella_limit") or 0),
        "text"          : policy_row_to_text(row_dict)
    })

print(f"✅ Converted {len(documents):,} policy rows to natural language documents")
print(f"\nSample document:")
print("-" * 60)
print(documents[0]["text"])
print("-" * 60)

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 6 — Generate Local Embeddings with sentence-transformers
#
# Model: all-MiniLM-L6-v2
#   - 384-dimensional embeddings
#   - Runs fully locally on the cluster — zero external API calls
#   - Fast: ~14,000 sentences/second on CPU
#   - Optimized for semantic similarity tasks
# ─────────────────────────────────────────────────────────────

print("Loading embedding model: all-MiniLM-L6-v2 (local, no API)...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print(f"✅ Embedding model loaded")
print(f"   Embedding dimension: {embedding_model.get_sentence_embedding_dimension()}")

# Extract all document texts
texts = [doc["text"] for doc in documents]

print(f"\nGenerating embeddings for {len(texts):,} policy documents...")
print("(Running locally on cluster — no external API calls)")

embeddings = embedding_model.encode(
    texts,
    batch_size     = 64,
    show_progress_bar = True,
    convert_to_numpy  = True
)

print(f"\n✅ Embeddings generated")
print(f"   Shape : {embeddings.shape}  ({embeddings.shape[0]} docs × {embeddings.shape[1]} dims)")
print(f"   Memory: {embeddings.nbytes / 1024 / 1024:.1f} MB")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 7 — Build FAISS Index
# ─────────────────────────────────────────────────────────────

embedding_dim = embeddings.shape[1]  # 384 for all-MiniLM-L6-v2

# Normalize to unit vectors so inner product = cosine similarity
embeddings_norm = embeddings.copy().astype(np.float32)
faiss.normalize_L2(embeddings_norm)

# Build flat index (exact search — suitable for up to ~1M vectors)
faiss_index = faiss.IndexFlatIP(embedding_dim)
faiss_index.add(embeddings_norm)

print(f"✅ FAISS index built")
print(f"   Vectors indexed : {faiss_index.ntotal:,}")
print(f"   Embedding dim   : {embedding_dim}")
print(f"   Index type      : IndexFlatIP (exact cosine similarity)")

# Quick sanity check — search for a test query
test_query = "umbrella coverage"
test_vec   = embedding_model.encode([test_query], convert_to_numpy=True).astype(np.float32)
faiss.normalize_L2(test_vec)
scores, indices = faiss_index.search(test_vec, 3)
print(f"\nSanity check — top 3 results for '{test_query}':")
for score, idx in zip(scores[0], indices[0]):
    print(f"  Policy {documents[idx]['policy_number']} | score={score:.4f} | {documents[idx]['text'][:80]}...")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 8 — RAG Query Functions
# ─────────────────────────────────────────────────────────────

def retrieve_top_k(question: str, k: int = 4) -> list:
    """
    Embeds the question and retrieves the k most relevant policy chunks
    from the FAISS index using cosine similarity.

    Args:
        question : Natural language question from the adjuster
        k        : Number of policy chunks to retrieve

    Returns:
        List of dicts with policy metadata, text, and similarity score
    """
    # Embed and normalize the question
    q_vec = embedding_model.encode([question], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_vec)

    # Search FAISS index
    scores, indices = faiss_index.search(q_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        doc = documents[idx]
        results.append({
            "policy_number" : doc["policy_number"],
            "policy_state"  : doc["policy_state"],
            "is_active"     : doc["is_active"],
            "annual_premium": doc["annual_premium"],
            "deductible"    : doc["deductible"],
            "umbrella_limit": doc["umbrella_limit"],
            "text"          : doc["text"],
            "confidence"    : float(score)
        })
    return results


def answer_with_rag(question: str, k: int = 4) -> dict:
    """
    Full RAG pipeline:
      Retrieve → Augment → Generate

    Args:
        question : Adjuster's natural language question
        k        : Number of policy chunks to retrieve

    Returns:
        Dict with answer, source policy citations, confidence score,
        and all metadata needed for rag_query_history
    """
    # ── RETRIEVE: find most relevant policy chunks ────────────
    chunks = retrieve_top_k(question, k=k)

    if not chunks:
        return {
            "query_id"        : str(uuid.uuid4())[:8].upper(),
            "question"        : question,
            "answer"          : "No relevant policies found for this query.",
            "source_policies" : "",
            "confidence_score": 0.0,
            "chunks_retrieved": 0,
            "policy_found"    : False
        }

    source_policies  = [str(c["policy_number"]) for c in chunks]
    avg_confidence   = round(sum(c["confidence"] for c in chunks) / len(chunks), 4)
    top_confidence   = round(chunks[0]["confidence"], 4)

    # ── AUGMENT: build grounded context from retrieved chunks ─
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        context_blocks.append(
            f"[Source {i} — Policy {chunk['policy_number']} "
            f"(relevance score: {chunk['confidence']:.4f})]\n{chunk['text']}"
        )
    context = "\n\n".join(context_blocks)

    # ── GENERATE: LLM answer grounded in retrieved context ────
    system = """You are a policy lookup assistant for insurance claims adjusters.
You are given exact policy records retrieved from the database via semantic search.
Rules:
  1. Answer ONLY based on the policy data provided — never guess or make up numbers
  2. Always cite specific policy numbers in your answer (e.g. "Policy 10001 has...")
  3. If comparing multiple policies, address each one
  4. If the answer is not in the retrieved data, say "Not available in retrieved records"
  5. Be concise and direct — adjusters need fast, accurate answers"""

    prompt = f"""The following policy records were retrieved as most relevant to the question:

{context}

ADJUSTER QUESTION: {question}

Provide a direct, accurate answer based only on the policy records above.
Cite policy numbers explicitly in your answer."""

    answer = get_llm_response(prompt, system=system, max_tokens=600)

    return {
        "query_id"        : str(uuid.uuid4())[:8].upper(),
        "question"        : question,
        "answer"          : answer,
        "source_policies" : ", ".join(source_policies),
        "confidence_score": top_confidence,
        "avg_confidence"  : avg_confidence,
        "chunks_retrieved": len(chunks),
        "policy_found"    : True
    }


print("✅ RAG query functions defined")
print("   retrieve_top_k() — FAISS semantic search")
print("   answer_with_rag() — full Retrieve→Augment→Generate pipeline")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 9 — Run 10 Test Queries
#
# Covers all required query types:
#   1. Specific policy lookup
#   2. Filter-based questions (umbrella, active status, state)
#   3. Comparative queries
#   4. Coverage scenario questions
#   5. Financial summary questions
# ─────────────────────────────────────────────────────────────

# Get a few real policy numbers for specific lookups
sample_nums = [
    row["policy_number"]
    for row in spark.read.table("primeinsurance_analytics.gold.dim_policy")
        .select("policy_number").limit(5).collect()
]

p1, p2, p3 = sample_nums[0], sample_nums[1], sample_nums[2]

# ── Test questions covering all required query types ──────────
test_questions = [
    # Type 1 — Specific policy lookup (2 questions)
    f"What is the deductible for policy {p1}?",
    f"Is policy {p2} currently active and what is the annual premium?",

    # Type 2 — Filter-based questions (3 questions)
    "Which policies include umbrella coverage? List the policy numbers and their umbrella limits.",
    "Which policies are currently inactive?",
    "Which policies are registered in Texas?",

    # Type 3 — Comparative queries (2 questions)
    f"Compare the bodily injury limits of policy {p1} and policy {p3}. Which has better coverage?",
    "Among the retrieved policies, which has the highest annual premium?",

    # Type 4 — Coverage scenario questions (2 questions)
    f"If a claim is filed under policy {p2}, how much will the customer pay out of pocket before coverage applies?",
    "Which of the retrieved policies would provide the most protection in a serious multi-vehicle accident?",

    # Type 5 — Full summary (1 question)
    f"Give me a complete financial summary of policy {p1} including all limits, premiums, and deductibles.",
]

# ── Run every question through the RAG pipeline ───────────────
rag_results = []

print(f"\n{'='*65}")
print("RUNNING RAG QUERIES — Full Vector Search Pipeline")
print(f"{'='*65}\n")

for i, question in enumerate(test_questions, 1):
    print(f"{'─'*65}")
    print(f"Q{i}: {question}")

    result = answer_with_rag(question, k=4)

    print(f"\n📋 Sources  : {result['source_policies']}")
    print(f"📊 Confidence: {result['confidence_score']:.4f} (top match)")
    print(f"🔢 Chunks   : {result['chunks_retrieved']} retrieved")
    print(f"\n💬 Answer:")
    print(result["answer"])
    print()

    rag_results.append(result)

print(f"{'='*65}")
print(f"✅ Completed {len(rag_results)} RAG queries")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 10 — Display 3 Sample Q&A Pairs for Submission
# ─────────────────────────────────────────────────────────────

print("=" * 70)
print("SAMPLE RAG OUTPUT — 3 Q&A PAIRS WITH CITATIONS")
print("=" * 70)

for i in range(3):
    r = rag_results[i]
    print(f"\n{'─'*70}")
    print(f"QUERY {i+1} (ID: {r['query_id']})")
    print(f"{'─'*70}")
    print(f"❓ Question  : {r['question']}")
    print(f"📄 Sources   : Policy numbers [{r['source_policies']}]")
    print(f"📊 Confidence: {r['confidence_score']:.4f}")
    print(f"\n✅ Answer:")
    print(r["answer"])

print(f"\n{'='*70}")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 11 — Save All Results to gold.rag_query_history
# ─────────────────────────────────────────────────────────────

# Define schema explicitly for clean Delta table
schema = StructType([
    StructField("query_id",         StringType(),  False),
    StructField("question",         StringType(),  True),
    StructField("answer",           StringType(),  True),
    StructField("source_policies",  StringType(),  True),   # cited policy numbers
    StructField("confidence_score", FloatType(),   True),   # top cosine similarity (0–1)
    StructField("avg_confidence",   FloatType(),   True),   # avg across k chunks
    StructField("chunks_retrieved", FloatType(),   True),   # k chunks used
    StructField("policy_found",     BooleanType(), True),
])

# Build rows list with matching schema
rows = []
for r in rag_results:
    rows.append((
        str(r.get("query_id", "")),
        str(r.get("question", "")),
        str(r.get("answer", "")),
        str(r.get("source_policies", "")),
        float(r.get("confidence_score", 0.0)),
        float(r.get("avg_confidence", 0.0)),
        float(r.get("chunks_retrieved", 0)),
        bool(r.get("policy_found", False)),
    ))

rag_df = spark.createDataFrame(rows, schema=schema)

# Add audit timestamp
rag_df = rag_df.withColumn("queried_at", F.current_timestamp())

# Write to Gold layer (append so history accumulates across runs)
(
    rag_df
    .write
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable("primeinsurance_analytics.gold.rag_query_history")
)

print("✅ Saved to primeinsurance_analytics.gold.rag_query_history")
print(f"   Rows written: {rag_df.count()}")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 12 — Verify Output Table
# ─────────────────────────────────────────────────────────────

output = spark.read.table("primeinsurance_analytics.gold.rag_query_history")

print(f"Total rows in rag_query_history: {output.count()}")
print(f"\nSchema:")
output.printSchema()

print("\nPreview (latest queries):")
(
    output
    .orderBy(F.col("queried_at").desc())
    .select(
        "query_id",
        "question",
        "source_policies",
        "confidence_score",
        "policy_found",
        "queried_at"
    )
    .show(10, truncate=55)
)

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 13 — Interactive: Ask Your Own Question
#
# Change the question below and re-run this cell.
# No policy number needed — the vector search finds relevant policies.
# ─────────────────────────────────────────────────────────────

my_question = "Which policies have the lowest deductible and are currently active?"

print(f"Question: {my_question}\n")
result = answer_with_rag(my_question, k=4)

print(f"Sources   : {result['source_policies']}")
print(f"Confidence: {result['confidence_score']:.4f}")
print(f"\nAnswer:\n{result['answer']}")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from primeinsurance_analytics.gold.rag_query_history;
