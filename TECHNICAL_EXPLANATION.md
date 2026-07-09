# Retail Data Agent — Technical Explanation

This document explains the design of the retail data analysis agent: the stack
choices, how data flows through the system, how each of the eight requirements
is addressed, and where the prototype's honest boundaries lie. The companion
architecture diagram is in the README.

---

## 1. Stack & reasoning

| Layer | Choice | Why |
|---|---|---|
| Orchestration | **LangGraph** | The requirements demand graph-native features a linear chain cannot express: a retry *cycle* (self-heal), a mid-execution *pause* (`interrupt()` for delete confirmation), and checkpointed state. LangGraph provides all three as first-class primitives. |
| LLM | **Gemini 3.1 Flash-Lite** (chat) + **gemini-embedding-001** (retrieval) | Free-tier accessible for a reviewer, high daily quota suited to iterative development, sufficient capability for routing/SQL/report tasks over a four-table schema. Model choice is a single-line config; the development/production inversion is discussed in §4. |
| Warehouse | **BigQuery** (read-only, public dataset) | Mandated by the brief; accessed through a thin client (`bq_client.py`) so the rest of the system never touches BigQuery APIs directly — one file to change if the warehouse changes. |
| Knowledge & app stores | **Local JSON files** | Golden Bucket, saved reports, promotion candidates, user preferences: all JSON. Deliberate prototype choice — zero setup for the reviewer, human-inspectable, and the enforcement logic (ownership scoping, precedence) is identical to what a database version would run. Production mapping in §4. |
| Retrieval index | **In-memory NumPy cosine similarity** | The bucket holds 8 vectors; exhaustive exact search is optimal at this scale. A vector database adds infrastructure without adding capability below ~10⁴ items. Production path: Vertex AI Vector Search once the bucket grows past that. |
| Interface | **CLI** | Per the brief ("UIs won't gain additional points"). The graph is interface-agnostic: the CLI loop is ~100 lines that a web backend would replace without touching the agent core. |

**Guiding principle used throughout:** *probabilistic components (LLMs) for
interpretation; deterministic code for consequences.* The LLM routes, writes SQL,
and drafts prose. Plain Python enforces PII masking, retry caps, ownership
scoping, format precedence, and command dispatch. Wherever an action has
consequences, the decision point is code, not a prompt.

## 2. System overview & data flow

A question's lifecycle (the analysis path):

1. **CLI** receives input. Exact-match commands (`save that`, `prefer X`,
   `metrics`) dispatch deterministically without any LLM call. Questions
   proceed, carrying the rolling **conversation window** (last 3 Q&A pairs,
   RAM-only) as context.
2. **Router** (LLM, context-aware) stamps one of six labels: analysis, schema,
   clarify, pii_request, delete_reports, offtopic. Refusals and clarifications
   end here — cheap exits before any data work. Unrecognised router output
   fails open to `analysis` (a wasted query is cheaper than a blocked
   executive).
3. **SQL generation** (LLM) receives: live table schemas (fetched once at
   startup), the top-3 most similar Golden Bucket **Trios** (retrieved by
   embedding similarity — the Hybrid Intelligence mechanism), the conversation
   window, and on retries, the previous SQL plus the exact failure message.
4. **Execution** against BigQuery. Failures don't raise — they write an error
   into state. Two failure classes are detected: hard (query rejected) and
   soft (zero rows *or* all-NULL aggregate results, which are semantically
   empty). Either loops back to step 3 with the error as feedback, capped at
   2 retries; a NO_DATA verdict or graceful give-up ends the loop honestly.
5. **PII masking** (deterministic) scrubs the result frame — the *only*
   version of the data that anything downstream (report LLM or screen) ever
   sees.
6. **Report generation** (LLM) writes the analyst narrative, voiced by the
   externally editable persona, formatted per resolved user preference, with
   retrieved Trio reports as style exemplars.
7. **Observability** wraps the whole path: a correlation ID stamps every trace
   line; one JSONL event records the outcome, latency, retries, maskings, and
   LLM-call count.

Two side paths: `delete_reports` flows to owner-scoped matching and the
`interrupt()` confirmation gate; `save that` writes to both the user's report
library and the Golden Bucket candidate queue (two files, two lifecycles —
users may delete their library entries; candidates await analyst review and
have no CLI delete path).

---

## 3. The eight requirements: implementation & decisions

### 3.1 Hybrid Intelligence (Golden Bucket)

**Built:** The bucket is `golden_bucket.json` — 8 hand-authored, BigQuery-verified
Trios (Question → SQL → Report) covering the brief's stated capability areas
(customer behaviour, product performance, time-based metrics, business metrics).
At startup, every Trio question is embedded once (`gemini-embedding-001`); at
query time the incoming question is embedded and cosine-matched, and the top-3
Trios are injected into the SQL prompt as worked expert examples. Trio *reports*
are additionally injected into report generation as style exemplars — the same
knowledge serving both halves of the pipeline.

**Why it matters:** schemas describe what data exists; Trios encode *business
judgment* schemas cannot express. Demonstrably: a question phrased "most valuable
customers" produced SQL excluding `Cancelled/Returned` order statuses — a
judgment appearing nowhere in the schema or question, inherited from a retrieved
Trio whose wording shared almost no vocabulary with the query. Retrieval steers
generation by meaning, not keywords.

**Updating the bucket over time (the learning loop):** every user `save that`
action writes a promotion candidate (full Trio + provenance: who, when,
status=pending) to `bucket_candidates.json`. Promotion into the bucket itself
requires analyst approval — deliberately *not* implemented as a self-serve
command, because self-approval would make curation theatre; the bucket's value
is that a qualified second human vouches for each entry. Production adds:
failure-driven authoring (observability logs reveal question classes the agent
fumbles → analysts author Trios precisely there), lifecycle maintenance
(revision on schema change, retirement of stale judgments, deduplication), and
implicit-signal mining as a future enhancement.

### 3.2 Safety & PII Masking

**Three layers, one guarantee:**

1. *Steer (soft):* the SQL prompt instructs against selecting contact columns
   unless the analysis requires reading them — reduces PII in flight.
2. *Refuse (front door):* the router's `pii_request` label catches explicit
   contact-detail requests with an explanation and an alternative offer;
   manipulation attempts route to `offtopic`.
3. *Scrub (the guarantee):* `mask_pii()` — deterministic Python, two sweeps
   (known PII column names masked wholesale; regex scan of remaining text cells
   for email/phone patterns) — runs on every result frame **between execution
   and everything downstream**. The masked frame is the only version the report
   LLM or the screen ever receives. Code cannot be prompt-injected; this layer
   satisfies the brief's "even if the SQL query retrieves it" clause directly.

**Decisions of note:** names remain visible — the brief scopes masking to phones
and emails, and its own flagship use case ("top customers") requires identifying
customers; the PII column list is config-driven so extending scope (names,
addresses) is a one-line governance decision, not engineering. Testing surfaced
a precision bug — the initial phone regex masked product style codes
(`404309-109`) — fixed by tightening to phone-shaped structures and exempting
product-descriptor columns from cell scanning. The masking bias is deliberately
toward recall (an over-masked SKU is cosmetic; a leaked phone number is a
breach). Reports mention masking **only when it occurred** (conditional prompt
injection), after testing showed the model hallucinating compliance notices
when primed unconditionally.

### 3.3 High-Stakes Oversight (Destructive Ops)

**Built:** a Saved Reports library (`saved_reports.json`, every record
owner-stamped) and a delete flow: natural-language criteria ("mentioning X",
"made today") are parsed by an LLM against the user's *own* reports only —
ownership filtering happens in code *before* criteria matching, so cross-user
deletion is structurally impossible, not merely discouraged. Matches trigger
LangGraph's `interrupt()`: the graph pauses mid-execution (checkpointer-backed),
displays exactly what will be deleted (count + questions + timestamps), and
resumes only on user input. Only the exact string `CONFIRM` executes; anything
else cancels. Informed consent without breaking conversational UX — the
confirmation shows the blast radius, and a lazy "y" cannot destroy data.

Bucket candidates are deliberately outside the delete path: a user tidying
their library must not silently drain the curation pipeline.

### 3.4 Continuous Improvement (Learning Loop)

**User level:** per-user format preferences (`user_preferences.json`), set via a
deterministic `prefer <format>` command, persisted across sessions, injected
into report generation. Precedence is resolved in code, not LLM arbitration:
*inline request > stored preference > default* — testing showed a soft-prompt
version losing the arbitration (stored preference steamrolling explicit
one-off asks), so format words are detected in Python and the model receives
exactly one unambiguous instruction. Session-scoped conversation memory
(rolling 3-turn window) additionally lets executives ask follow-ups ("what
about by units sold?") that resolve against prior context — the brief's
"discuss about it".

**System level:** identical mechanism to §3.1's bucket update — quality-signalled
interactions become curated retrieval exemplars, so per-question competence
grows with usage. No model retraining is involved anywhere; the model stays
fixed while its retrieved context improves.

### 3.5 Resilience & Graceful Error Handling

**SQL self-heal:** execution failures (hard: rejected queries; soft: zero-row or
all-NULL results) write the exact error into state and loop back to generation,
where the retry prompt contains the previous SQL and the failure text — each
attempt is better-informed, not a blind re-roll. Hard cap of 2 retries (3 total
attempts) bounds worst-case cost per question — the brief's "without inflating
costs" made concrete.

**Fidelity constraint:** early testing produced a subtle failure — asked for
1905 revenue (no data exists), the retry loop "fixed" the empty result by
silently substituting 2019–2023, presenting a confidently wrong answer. The
retry prompt now permits exactly two moves: fix a genuine query mistake while
preserving the user's stated intent *exactly*, or return a NO_DATA verdict,
which surfaces as an honest "no data exists for that request." A confidently
wrong answer is worse than no answer.

**Failure UX:** exhausted retries produce a polite in-chat message and a fresh
prompt — no tracebacks, no crashed UI. Empty input is guarded before reaching
any API.

**Third-party resilience:** during development the system encountered the full
set of provider failures live — 503 (model overloaded), 429 (daily quota
exhausted), 404 (invalid model name). The design response: all chat calls flow
through a single funnel (`llm_invoke`), positioned to carry retry-with-backoff
honouring server retry hints, and model fallback (Flash-Lite ↔ Flash have
independent quotas). The funnel currently implements call accounting; the
backoff/fallback policy is specified for production and acknowledged as
partially implemented in the prototype (§6).

### 3.6 Quality Assurance

`eval_agent.py` fires 10 cases through the *actual compiled graph* (not a
copy) and grades three deterministic tiers — routing (analysis routes to
analysis, PII probes refuse, manipulation refuses, ambiguity clarifies),
safety (a regex scan asserts no raw PII in any user-visible output — the
masking invariant, tested on every analysis case), and execution (SQL ran,
expected tables touched, non-empty results) — plus an optional `--judge` mode
where a second LLM scores whether reports answer intent (kept optional and
separate because judge reliability is itself an open problem; the doc does not
pretend it is solved). The test set is the 8 Trios' coverage plus adversarial
cases discovered during development (the 1905 honesty case, PII probes, an
accidental source-code paste that became an offtopic stress test). Exit code
is nonzero on any failure — CI-gateable by construction. Current scorecard:
**10/10**.

**Verifying reports answer intent** (the requirement's second question) is
addressed at four levels: retrieval grounds interpretation in expert-verified
examples; the fidelity constraint prevents silent intent drift during retries;
deterministic checks verify structural correctness; and human signals (saves,
and in production thumbs-down feedback) close the loop.

### 3.7 Observability

Three instruments: a **live trace** (every node announces itself with a
per-question correlation ID — route chosen, retrieved Trio IDs, row counts,
timings, retry attempts, mask counts); a **structured event log**
(`agent_log.jsonl`, one line per question: outcome class, latency, retries,
maskings, LLM-call count — append-only JSONL for crash-safety and trivial
parsing); and a **`metrics` command** rendering the agent-level dashboard:
outcome distribution, route distribution, self-heal frequency, PII maskings,
average latency, calls per question. The correlation ID answers the brief's
"what the message correspondence is": one question's complete journey is
grep-able from a busy log.

This paid off during development: an accidental paste of source code into the
CLI produced ~20 malformed inputs; the event log captured every one, showed
the router correctly refusing them all, and surfaced an unhandled empty-input
path — which was then fixed. Incident, diagnosis, and fix, all from the log.

Production mapping: events ship to Cloud Logging/BigQuery, dashboards in
Looker, alerting on give-up-rate and refusal-rate anomalies; LangSmith is the
LangGraph-native tracing option for token-level LLM call inspection.

### 3.8 Agility (Persona Management)

The agent's report voice lives in `persona.txt` — plain text, outside the
code — and is **re-read on every report call**, so an edit takes effect on the
very next question with no restart and no redeployment. Demonstrated live: the
persona was edited mid-session from analyst-voice to blunt-minimal, and
consecutive identical questions returned visibly different registers.
Deliberate boundary: safety instructions (PII handling) are appended in code
*after* the persona loads — the CEO can change the voice, never the guardrails.
Production shape: the file becomes a config-store row behind an admin textbox,
optionally versioned and A/B tested; the mechanism — *instructions as data,
not code* — is identical.

---

## 4. Production path

The prototype's architecture is the production architecture with cheaper parts.
The mapping:

| Prototype component | Production replacement | What stays identical |
|---|---|---|
| `--user` flag (asserted identity) | SSO (Okta / Google Workspace) via the chat frontend | The ownership-scoping code — it just receives a verified name instead of a claimed one |
| JSON stores (reports, candidates, prefs) | Firestore / Cloud SQL, with row-level security as a second net | Every enforcement rule (owner filtering, precedence, two-lifecycle split) |
| In-memory cosine retrieval | Vertex AI Vector Search (trigger: bucket > ~10⁴ Trios) | The retrieve-inject pattern and Trio schema |
| CLI loop | Web chat frontend + streaming API over the same compiled graph | The entire agent core — it is interface-agnostic |
| `MemorySaver` checkpointer | Postgres/Redis-backed checkpointer | The interrupt/resume confirmation flow (state is already fully serialisable — enforced during development when the checkpointer rejected raw DataFrames) |
| Per-session RAM conversation window | Checkpointer-backed session threads with summarisation for long conversations | The context-injection points |
| Flash-Lite as primary model | Stronger model primary, Lite as economical fallback — the development inversion reversed | The single-funnel `llm_invoke` boundary |
| JSONL log + `metrics` | Cloud Logging → BigQuery → Looker dashboards + alerting; LangSmith traces | Event schema and correlation IDs |
| Manual eval runs | Eval harness gating CI/CD (nonzero exit fails the deploy); golden-set growth from bucket promotions | The harness itself |

Extension points the brief asks about (graphs, email delivery, new data
sources) slot in as: new router labels + nodes (a `chart` node consuming the
same masked result frame; an email tool behind the same interrupt-confirmation
pattern as deletes — any side-effectful action gets a gate), and new data
sources as additional thin clients beside `bq_client.py` with their schemas
joining the SQL context.

## 5. Setup & example run

Full setup instructions, the verification script (`connection_test.py`), usage
table, and a captured example session live in the [README](README.md). The
short version: clone → venv → `pip install -r requirements.txt` → gcloud auth →
`.env` with two values → `python connection_test.py` → `python agent.py`.

## 6. Known limitations (honest boundaries)

1. **LLM-call resilience is specified, partially implemented.** The single-funnel
   boundary exists; automatic backoff/fallback policy is production work. A
   quota exhaustion mid-session currently surfaces as a graceful error message,
   not a silent model switch.
2. **Eval invocations bypass the event log** — logging wraps the interaction
   layer; the harness calls the graph directly. Production moves emission
   inside the graph (or LangSmith callbacks) with synthetic traffic tagged out
   of production dashboards.
3. **Empty-result retry can be wasteful when zero rows is the true answer**
   (e.g. a customer who bought nothing) — one retry is spent confirming. The
   NO_DATA verdict mitigates; a plausibility pre-check would refine.
4. **Reference resolution in chained follow-ups is best-effort** — "which of
   those" resolves by LLM interpretation over a 3-turn window; scoping is
   occasionally looser than a human would intend.
5. **The masking column-sweep over-masks derived non-identifying columns**
   (e.g. `email_domain` aggregations) — a deliberate recall-biased tradeoff;
   production adds a governance-reviewed allowlist.
6. **Synthetic-data oddities are narrated credulously in places** (e.g.
   $557 socks in `thelook`'s cost column), though the report layer has begun
   flagging outliers unprompted; a sanity-check Trio would systematise this.
7. **Compound intents** ("run this analysis *and* change my default") execute
   the analysis and silently skip the settings change — settings are standalone
   commands by design; a production UI resolves this with dedicated controls.
8. **Dataset drift:** Google regenerates `thelook_ecommerce` continuously;
   absolute numbers in captured examples will differ between runs.

Each of these was discovered through hands-on testing during the build, and each
has a named production remedy — the boundary of the prototype is documented
rather than hidden.

---

*Built with LangGraph, Gemini, and BigQuery. See the README for the
architecture diagram and quick start.*