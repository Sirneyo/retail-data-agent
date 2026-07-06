import os, re
from typing import TypedDict, Optional

import pandas as pd
from dotenv import load_dotenv
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

# ---------- Shared clients ----------
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0,
    google_api_key=os.environ["GOOGLE_API_KEY"],
)
runner = BigQueryRunner(project_id=os.environ.get("GCP_PROJECT_ID"))
DATASET = "bigquery-public-data.thelook_ecommerce"

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

# ---------- Router: stamps one label on every message ----------
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

# ---------- Worker nodes ----------
SQL_SYSTEM = f"""You are a BigQuery Standard SQL expert.
Write ONE valid query answering the user's question.
Dataset `{DATASET}`, tables: orders, order_items, products, users.
Always fully-qualify tables, e.g. `{DATASET}.orders`.
Return ONLY raw SQL — no markdown, no commentary."""

def generate_sql(state: AgentState) -> dict:
    resp = llm.invoke([("system", SQL_SYSTEM), ("human", state["question"])])
    sql = llm_text(resp).strip()
    sql = re.sub(r"^```(?:sql)?\n?", "", sql)
    sql = re.sub(r"\n?```$", "", sql)
    return {"sql": sql.strip()}

def execute_sql(state: AgentState) -> dict:
    df = runner.execute_query(state["sql"])
    return {"result": df}

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

builder.add_edge(START, "route_question")
builder.add_conditional_edges(
    "route_question", lambda s: s["route"],
    {"analysis": "generate_sql", "schema": "answer_schema",
     "clarify": END, "offtopic": END},
)
builder.add_edge("generate_sql", "execute_sql")
builder.add_edge("execute_sql", END)
graph = builder.compile()

# ---------- CLI ----------
if __name__ == "__main__":
    print("Retail Data Agent (Phase 1) — type 'exit' to quit.")
    while True:
        q = input("\nAsk> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        final = graph.invoke({"question": q})
        if final.get("answer"):
            print("\n" + final["answer"])
        else:
            print("\n--- SQL ---\n" + final["sql"])
            print("\n--- RESULT ---")
            print(final["result"].to_string(index=False))