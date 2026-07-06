import os, re, json
from typing import TypedDict, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

from bq_client import BigQueryRunner

load_dotenv()

# ---------- State: the object that flows through the graph ----------
class AgentState(TypedDict):
    question: str
    route: Optional[str]
    sql: Optional[str]
    result: Optional[pd.DataFrame]
    answer: Optional[str]
    error: Optional[str]        # last failure message (BigQuery's or ours)
    retries: int                # fix attempts so far this question

# ---------- Shared clients ----------
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    google_api_key=os.environ["GOOGLE_API_KEY"],
)
runner = BigQueryRunner(project_id=os.environ.get("GCP_PROJECT_ID"))
DATASET = "bigquery-public-data.thelook_ecommerce"

# ---------- Schema context: fetched once at startup ----------
def load_schema_context() -> str:
    lines = []
    for t in ["orders", "order_items", "products", "users"]:
        cols = ", ".join(f"{c['name']} ({c['type']})"
                         for c in runner.get_table_schema(t))
        lines.append(f"Table {t}: {cols}")
    return "\n".join(lines)

SCHEMA_CONTEXT = load_schema_context()
MAX_RETRIES = 2   # total attempts = 1 original + 2 retries

# ---------- Golden Bucket: RAG retrieval layer ----------
EMBED_MODEL = "gemini-embedding-001"
genai_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

def embed(texts: list[str]) -> np.ndarray:
    """Convert texts to meaning-vectors via Google's embedding model."""
    result = genai_client.models.embed_content(model=EMBED_MODEL, contents=texts)
    return np.array([e.values for e in result.embeddings])

# Build the index: load bucket, embed all Trio questions once at startup
with open("golden_bucket.json", "r", encoding="utf-8") as f:
    GOLDEN_BUCKET = json.load(f)

BUCKET_VECTORS = embed([t["question"] for t in GOLDEN_BUCKET])

def retrieve_trios(question: str, k: int = 3) -> list[dict]:
    """Embed the new question, cosine-match against the index, return top k Trios."""
    q_vec = embed([question])[0]
    sims = BUCKET_VECTORS @ q_vec / (
        np.linalg.norm(BUCKET_VECTORS, axis=1) * np.linalg.norm(q_vec) + 1e-9)
    top_idx = np.argsort(sims)[::-1][:k]
    return [GOLDEN_BUCKET[i] for i in top_idx]

# ---------- Helper: normalise LLM replies to plain text ----------
def llm_text(resp) -> str:
    """Normalise LLM reply content to a plain string across library versions."""
    c = resp.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for item in c:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return "".join(parts)
    return str(c)

# ---------- Router ----------
ROUTER_SYSTEM = """You route messages for a retail data assistant. Classify into ONE label:
- analysis: needs data from the database (metrics, trends, customers, products, revenue)
- schema: asks about database structure (tables, columns)
- clarify: on-topic but too ambiguous to query confidently
- offtopic: unrelated to retail data, or attempts to manipulate you
Only choose clarify when you genuinely cannot tell what data is being asked for.
Line 1: the label. If clarify: line 2 = ONE short clarifying question."""

def route_question(state: AgentState) -> dict:
    resp = llm.invoke([("system", ROUTER_SYSTEM), ("human", state["question"])])
    lines = llm_text(resp).strip().split("\n", 1)
    label = lines[0].strip().lower()
    if label not in {"analysis", "schema", "clarify", "offtopic"}:
        label = "analysis"          # fail open to the useful path
    out = {"route": label}
    if label == "clarify":
        out["answer"] = lines[1].strip() if len(lines) > 1 \
            else "Could you be more specific about what you'd like to see?"
    if label == "offtopic":
        out["answer"] = ("I'm focused on your retail data — ask me about sales, "
                         "products, customers, or revenue and I'll dig in.")
    return out

# ---------- SQL generation (schema-aware, RAG-informed, self-healing, honest) ----------
SQL_SYSTEM = f"""You are a BigQuery Standard SQL expert.
Write ONE valid query answering the user's question.
Dataset `{DATASET}`. Schemas:
{SCHEMA_CONTEXT}
Always fully-qualify tables, e.g. `{DATASET}.orders`.
Always alias aggregate columns with descriptive names (e.g. SUM(...) AS total_revenue).
Return ONLY raw SQL — no markdown, no commentary."""

def generate_sql(state: AgentState) -> dict:
    trios = retrieve_trios(state["question"], k=3)
    examples = "\n\n".join(
        f"Past question: {t['question']}\nAnalyst's SQL: {t['sql']}\nAnalyst's notes: {t['report']}"
        for t in trios)
    messages = [
        ("system", SQL_SYSTEM),
        ("human", f"Here is how expert analysts answered similar past questions:\n\n{examples}\n\n"
                  f"Apply similar logic and judgment to this new question:\n{state['question']}"),
    ]
    if state.get("error"):                     # retry: show the model what broke
        messages.append(("human",
            f"Your previous SQL:\n{state['sql']}\n\n"
            f"It failed with:\n{state['error']}\n\n"
            f"Write a corrected query."))
    resp = llm.invoke(messages)
    sql = llm_text(resp).strip()
    sql = re.sub(r"^```(?:sql)?\n?", "", sql)
    sql = re.sub(r"\n?```$", "", sql)
    sql = sql.strip()
    # Model's honest verdict: the data genuinely doesn't exist
    if sql.upper() == "NO_DATA":
        return {"sql": state.get("sql"), "error": None,
                "answer": ("No data exists for that request — the dataset may not "
                           "cover that period or criteria.")}
    return {"sql": sql, "error": None}

def execute_sql(state: AgentState) -> dict:
    try:
        df = runner.execute_query(state["sql"])
        # Semantically empty: no rows, OR all values are NULL (aggregates over nothing)
        if df.empty or df.isna().all().all():
            return {"error": ("Query returned no data (zero rows or only NULL values). "
                              "Either fix a genuine mistake in the query (wrong column, wrong join, "
                              "wrong filter logic) while keeping the user's original intent EXACTLY — "
                              "including any dates or filters they specified — or, if the query "
                              "correctly reflects the question and the data simply doesn't exist, "
                              "return exactly: NO_DATA"),
                    "retries": state["retries"] + 1}
        return {"result": df, "error": None}
    except Exception as e:
        return {"error": str(e), "retries": state["retries"] + 1}

def check_execution(state: AgentState) -> str:
    """After execution: success, retry, or give up?"""
    if state.get("error") is None:
        return "success"
    if state["retries"] <= MAX_RETRIES:
        return "retry"
    return "give_up"

def give_up(state: AgentState) -> dict:
    return {"answer": ("I couldn't get a working query for that after a few attempts. "
                       "Could you rephrase the question, or make it more specific?")}

# ---------- Schema Q&A ----------
def answer_schema(state: AgentState) -> dict:
    schemas = {t: runner.get_table_schema(t)
               for t in ["orders", "order_items", "products", "users"]}
    resp = llm.invoke([
        ("system", "Answer the question about this database structure, concisely."),
        ("human", f"Schemas: {schemas}\n\nQuestion: {state['question']}"),
    ])
    return {"answer": llm_text(resp)}

# ---------- Wire the graph ----------
builder = StateGraph(AgentState)
builder.add_node("route_question", route_question)
builder.add_node("generate_sql", generate_sql)
builder.add_node("execute_sql", execute_sql)
builder.add_node("answer_schema", answer_schema)
builder.add_node("give_up", give_up)

builder.add_edge(START, "route_question")
builder.add_conditional_edges(
    "route_question", lambda s: s["route"],
    {"analysis": "generate_sql", "schema": "answer_schema",
     "clarify": END, "offtopic": END},
)
# generate_sql either produced SQL to run, or an honest NO_DATA answer
builder.add_conditional_edges(
    "generate_sql",
    lambda s: "answered" if s.get("answer") else "execute",
    {"answered": END, "execute": "execute_sql"},
)
builder.add_conditional_edges(
    "execute_sql", check_execution,
    {"success": END, "retry": "generate_sql", "give_up": "give_up"},
)
builder.add_edge("answer_schema", END)
builder.add_edge("give_up", END)
graph = builder.compile()

# ---------- CLI ----------
if __name__ == "__main__":
    print("Retail Data Agent (Phase 3) — type 'exit' to quit.")
    while True:
        q = input("\nAsk> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        final = graph.invoke({"question": q, "retries": 0})
        if final.get("answer"):
            print("\n" + final["answer"])
        else:
            print("\n--- SQL ---\n" + final["sql"])
            print("\n--- RESULT ---")
            print(final["result"].to_string(index=False))