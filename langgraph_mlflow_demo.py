# Databricks notebook source
# MAGIC %md
# MAGIC # Building Custom Agents on Databricks: LangGraph + MLflow
# MAGIC
# MAGIC A minimal end-to-end pattern for building a custom agent with **LangGraph** and
# MAGIC deploying it via the **Databricks Agent Framework**.
# MAGIC
# MAGIC **Roadmap:**
# MAGIC 1. **LangGraph fundamentals** — state, nodes, edges, compile
# MAGIC 2. **Conditional edges** — branching control flow (the JokeReview agent)
# MAGIC 3. **Checkpointing** — persisting state across turns with `thread_id`
# MAGIC 4. **Human-in-the-loop** — pause with `interrupt()`, resume with `Command`
# MAGIC 5. **MLflow + UC + Serving** — log → register → deploy

# COMMAND ----------

# MAGIC %pip install -qqqq -U langgraph langchain-core "mlflow[databricks]>=2.20.0" databricks-langchain databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1 — LangGraph fundamentals
# MAGIC
# MAGIC LangGraph models an agent as a **state machine**:
# MAGIC
# MAGIC | Concept | What it is |
# MAGIC |---|---|
# MAGIC | **State** | A typed dict describing what flows between steps |
# MAGIC | **Nodes** | Functions that read state and return updates |
# MAGIC | **Edges** | How control moves between nodes (linear or conditional) |
# MAGIC | **Compile** | Produces a runnable graph |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the state
# MAGIC
# MAGIC `add_messages` is a built-in **reducer** — when a node returns `{"messages": [new_msg]}`,
# MAGIC LangGraph appends rather than overwrites. Without it, each node would clobber the
# MAGIC message history.

# COMMAND ----------

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AnyMessage


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    turn_count: int

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define nodes
# MAGIC
# MAGIC Two trivial nodes for illustration:
# MAGIC - `increment_turn` — bumps a counter (shows non-message state)
# MAGIC - `respond` — calls an LLM via `databricks-langchain`

# COMMAND ----------

from databricks_langchain import ChatDatabricks

LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT)


def increment_turn(state: AgentState) -> dict:
    return {"turn_count": state.get("turn_count", 0) + 1}


def respond(state: AgentState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build and compile the graph
# MAGIC
# MAGIC `START` and `END` are sentinel nodes. Edges define the control flow.

# COMMAND ----------

builder = StateGraph(AgentState)
builder.add_node("increment", increment_turn)
builder.add_node("respond", respond)
builder.add_edge(START, "increment")
builder.add_edge("increment", "respond")
builder.add_edge("respond", END)

graph = builder.compile()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize and invoke

# COMMAND ----------

from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))

# COMMAND ----------

result = graph.invoke({
    "messages": [HumanMessage(content="In one sentence, what is photosynthesis?")],
    "turn_count": 0,
})
print(f"Turn count: {result['turn_count']}")
print(f"Response:   {result['messages'][-1].content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2 — Conditional edges (the JokeReview agent)
# MAGIC
# MAGIC Real agents branch. LangGraph expresses branching with **conditional edges**:
# MAGIC a routing function reads the current state and returns the name of the next
# MAGIC node (or `END`).
# MAGIC
# MAGIC We'll build a three-node graph:
# MAGIC
# MAGIC ```
# MAGIC START → judge ─┬─ funny ─────→ END
# MAGIC                └─ not funny ─→ rewriter → END
# MAGIC ```
# MAGIC
# MAGIC - **judge** decides if the joke is funny (structured output → `bool`)
# MAGIC - **rewriter** runs only if the joke flopped, and returns a critique + rewrite

# COMMAND ----------

# MAGIC %md
# MAGIC ### Structured output for deterministic routing
# MAGIC
# MAGIC We use `with_structured_output(<pydantic model>)` so the judge returns a
# MAGIC typed object — not free-form text we'd have to parse. The `is_funny` field
# MAGIC drives the routing decision.

# COMMAND ----------

from pydantic import BaseModel, Field


class JokeVerdict(BaseModel):
    is_funny: bool = Field(description="Whether the joke is genuinely funny")
    reasoning: str = Field(description="Brief explanation of the verdict")


class JokeRewrite(BaseModel):
    critique: str = Field(description="What makes the original joke fall flat")
    rewritten_joke: str = Field(description="An improved version of the joke")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extend the state with the routing flag
# MAGIC
# MAGIC `MessagesState` is LangGraph's pre-built convenience — it's already a
# MAGIC `TypedDict` with a `messages` field using the `add_messages` reducer. We
# MAGIC subclass it to add `is_funny`.

# COMMAND ----------

from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, SystemMessage
from typing import Any


class JokeReviewState(MessagesState):
    is_funny: bool


JUDGE_LLM = llm.with_structured_output(JokeVerdict)
REWRITER_LLM = llm.with_structured_output(JokeRewrite)


def judge(state: JokeReviewState) -> dict[str, Any]:
    system = SystemMessage(
        content="You are a discerning comedy critic. Judge whether the following joke is genuinely funny."
    )
    verdict: JokeVerdict = JUDGE_LLM.invoke([system] + state["messages"])
    return {
        "is_funny": verdict.is_funny,
        "messages": [AIMessage(
            content=f"{'Funny!' if verdict.is_funny else 'Not funny.'} {verdict.reasoning}"
        )],
    }


def rewriter(state: JokeReviewState) -> dict[str, Any]:
    system = SystemMessage(
        content="You are a comedy writer. The joke was judged not funny. Critique it and write an improved version."
    )
    result: JokeRewrite = REWRITER_LLM.invoke([system] + state["messages"])
    return {
        "messages": [AIMessage(
            content=f"Critique: {result.critique}\n\nRewritten joke: {result.rewritten_joke}"
        )],
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ### The routing function + conditional edge
# MAGIC
# MAGIC `add_conditional_edges(source, router, mapping)` says: after `source` runs,
# MAGIC call `router(state)`; the return value is looked up in `mapping` to decide
# MAGIC the next node.

# COMMAND ----------

def route_after_judge(state: JokeReviewState) -> str:
    return END if state.get("is_funny") else "rewriter"


joke_builder = StateGraph(JokeReviewState)
joke_builder.add_node("judge", judge)
joke_builder.add_node("rewriter", rewriter)
joke_builder.add_edge(START, "judge")
joke_builder.add_conditional_edges(
    "judge",
    route_after_judge,
    {END: END, "rewriter": "rewriter"},
)
joke_builder.add_edge("rewriter", END)

joke_graph = joke_builder.compile()
display(Image(joke_graph.get_graph().draw_mermaid_png()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run it — the rewriter only fires for bad jokes
# MAGIC
# MAGIC Same graph, two inputs, two different paths.

# COMMAND ----------

# A genuinely funny joke — should short-circuit at `judge`
funny = joke_graph.invoke({
    "messages": [HumanMessage(content="I told my wife she was drawing her eyebrows too high. She looked surprised.")]
})
print("=== FUNNY PATH ===")
for m in funny["messages"]:
    print(f"[{m.type}] {m.content}\n")

# COMMAND ----------

# A flat joke — should route through `rewriter`
flat = joke_graph.invoke({
    "messages": [HumanMessage(content="Why did the chicken cross the road? Because it wanted to.")]
})
print("=== NOT FUNNY PATH ===")
for m in flat["messages"]:
    print(f"[{m.type}] {m.content}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3 — Checkpointing
# MAGIC
# MAGIC Without a checkpointer, every `invoke` starts from a fresh state. Add one and
# MAGIC the graph **persists state per `thread_id`**, so the next call resumes the
# MAGIC same conversation.
# MAGIC
# MAGIC - `MemorySaver` — in-process, ephemeral. Fine for dev.
# MAGIC - For production, swap in a durable backend (Databricks Lakebase is an ideal solution for this and can be swapped directly for `MemorySaver`!).

# COMMAND ----------

from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph_with_memory = builder.compile(checkpointer=checkpointer)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Two-turn conversation on the same thread
# MAGIC
# MAGIC Notice we only pass **the new message** on turn 2 — the checkpointer rehydrates
# MAGIC prior state from `thread_id`.

# COMMAND ----------

config = {"configurable": {"thread_id": "user-123"}}

# Turn 1
graph_with_memory.invoke(
    {"messages": [HumanMessage(content="My name is Brennan.")]},
    config=config,
)

# Turn 2 — same thread_id, model has memory of turn 1
result = graph_with_memory.invoke(
    {"messages": [HumanMessage(content="What's my name?")]},
    config=config,
)
print(f"Response:   {result['messages'][-1].content}")
print(f"Turn count: {result['turn_count']}  # counter persisted across turns")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inspect saved state
# MAGIC
# MAGIC `get_state(config)` returns the full snapshot for a thread. This is what
# MAGIC powers human-in-the-loop, time-travel, and debugging.

# COMMAND ----------

snapshot = graph_with_memory.get_state(config)
print(f"Messages stored:  {len(snapshot.values['messages'])}")
print(f"Turn count:       {snapshot.values['turn_count']}")
print(f"Next node to run: {snapshot.next}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4 — Human-in-the-loop with `interrupt()`
# MAGIC
# MAGIC A checkpointer doesn't just remember conversation history — it lets the
# MAGIC graph **pause mid-run** and resume later. The mechanic:
# MAGIC
# MAGIC 1. Inside a node, call `interrupt(payload)`. Execution halts; the payload
# MAGIC    is surfaced back to the caller.
# MAGIC 2. The graph state is saved at that point (so the checkpointer is
# MAGIC    **required** — `interrupt()` won't work without one).
# MAGIC 3. To continue, call `graph.invoke(Command(resume=<human_answer>), config=...)`.
# MAGIC    The interrupted node re-runs from the top, but `interrupt()` now
# MAGIC    returns `<human_answer>` instead of pausing.
# MAGIC
# MAGIC In a notebook this maps cleanly to cell boundaries: one cell triggers the
# MAGIC pause, the next inspects it, the next resumes with an answer.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build a graph that asks the human for a topic

# COMMAND ----------

from langgraph.types import interrupt, Command


class TopicJokeState(TypedDict):
    topic: str
    joke: str


def ask_for_topic(state: TopicJokeState) -> dict:
    # IMPORTANT: when the graph is resumed, this entire node re-runs from the top.
    # On the first pass, interrupt() halts execution. On resume, interrupt() does
    # NOT pause — it returns the value passed via Command(resume=...). So any code
    # before this line runs twice. Keep it cheap and side-effect-free.
    human_topic = interrupt({"prompt": "What topic should the joke be about?"})
    return {"topic": human_topic}


def write_joke(state: TopicJokeState) -> dict:
    response = llm.invoke([HumanMessage(content=f"Write a short, clever joke about {state['topic']}.")])
    return {"joke": response.content}


hitl_builder = StateGraph(TopicJokeState)
hitl_builder.add_node("ask_for_topic", ask_for_topic)
hitl_builder.add_node("write_joke", write_joke)
hitl_builder.add_edge(START, "ask_for_topic")
hitl_builder.add_edge("ask_for_topic", "write_joke")
hitl_builder.add_edge("write_joke", END)

# Checkpointer is REQUIRED for interrupt() — it's how state is preserved across the pause.
hitl_graph = hitl_builder.compile(checkpointer=MemorySaver())
display(Image(hitl_graph.get_graph().draw_mermaid_png()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 1 — kick off the run, hit the interrupt
# MAGIC
# MAGIC The graph runs `ask_for_topic`, calls `interrupt(...)`, and returns. The
# MAGIC result dict contains an `__interrupt__` key with the payload we sent.

# COMMAND ----------

hitl_config = {"configurable": {"thread_id": "hitl-demo-1"}}

paused = hitl_graph.invoke({}, config=hitl_config)
print(paused)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 2 — inspect the pause point
# MAGIC
# MAGIC `get_state` shows us *exactly* where execution stopped and what was
# MAGIC waiting on a human. Useful for building UIs that surface pending requests.

# COMMAND ----------

snap = hitl_graph.get_state(hitl_config)
print(f"Next node:        {snap.next}")
print(f"Pending interrupts: {snap.tasks[0].interrupts}")
print(f"State so far:     {snap.values}  # 'joke' not set yet — write_joke hasn't run")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 3 — resume with a human answer
# MAGIC
# MAGIC `Command(resume=...)` is the magic: it re-enters `ask_for_topic`, but
# MAGIC `interrupt()` now returns the value we passed in. Execution continues
# MAGIC straight through to `write_joke` and `END`.

# COMMAND ----------

result = hitl_graph.invoke(Command(resume="distributed systems"), config=hitl_config)
print(f"Topic chosen: {result['topic']}")
print(f"Joke:\n{result['joke']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Patterns to avoid (and what to do instead)
# MAGIC
# MAGIC Because the **entire node re-runs on resume**, anything above `interrupt()`
# MAGIC executes twice. This is where teams get burned:
# MAGIC
# MAGIC | Anti-pattern | Why it breaks | Do this instead |
# MAGIC |---|---|---|
# MAGIC | Side effects before `interrupt()` (send email, write to DB, charge card, call external API) | Fires on the original pause AND on every resume — duplicate emails, double-writes | Put side effects in a **downstream node** that runs only after resume |
# MAGIC | Expensive computation before `interrupt()` (LLM call, big query, embedding) | Wasted spend — the work is thrown away and redone on resume | Move the work to a node **after** the interrupt, or cache the result in state on the first pass |
# MAGIC | Non-deterministic locals before `interrupt()` (`uuid4()`, `datetime.now()`, random) | Original-run values are lost; resume sees fresh values, leading to subtle state mismatches | Generate IDs/timestamps **after** resume, or write them to state before pausing |
# MAGIC | Long-lived resources opened above the interrupt (DB connection, file handle) | Leaked on the first pass; reopened on resume | Open resources after the interrupt, or use `with` blocks that close before pausing |
# MAGIC
# MAGIC ### The right shape for an interrupting node
# MAGIC
# MAGIC ```
# MAGIC def interrupting_node(state):
# MAGIC     # 1. Cheap, pure prep only — runs twice
# MAGIC     payload = build_payload(state)
# MAGIC
# MAGIC     # 2. Pause. On resume, this returns the human's answer.
# MAGIC     answer = interrupt(payload)
# MAGIC
# MAGIC     # 3. Anything below the interrupt runs ONCE (after resume).
# MAGIC     #    Safe place for side effects, expensive calls, fresh IDs, etc.
# MAGIC     return {"answer": answer}
# MAGIC ```
# MAGIC
# MAGIC ### Where this fits in real agents
# MAGIC
# MAGIC - **Approval gates** — agent drafts an action, pauses for sign-off before executing
# MAGIC - **Clarifying questions** — agent realizes it lacks info and asks the user
# MAGIC - **Tool-call review** — surface a proposed tool call to a human before invocation
# MAGIC
# MAGIC In a notebook the "UI" is the next cell. In a Databricks App or the Review
# MAGIC App, it's a rendered form. The agent-side primitive is identical.

# COMMAND ----------

# MAGIC %md
# MAGIC    
# MAGIC ## Part 5 — Deploy with MLflow + Databricks Agent Framework
# MAGIC
# MAGIC To get this on a Model Serving endpoint we:
# MAGIC
# MAGIC 1. Wrap the graph in MLflow's `ResponsesAgent` interface
# MAGIC 2. **Log** the agent to an MLflow experiment as a tracked artifact
# MAGIC 3. **Register** the logged model to **Unity Catalog**
# MAGIC 4. **Deploy** with `databricks.agents.deploy` (provisions a serving endpoint + review app)

# COMMAND ----------

# MAGIC %md
# MAGIC    
# MAGIC ### Step 1 — Define the agent in a Python file
# MAGIC
# MAGIC MLflow uses **code-based logging**: we point it at a `.py` file that builds the
# MAGIC agent and calls `mlflow.models.set_model(...)`. This makes the model
# MAGIC reproducible — no pickling, no closure-capture surprises, and the file is what
# MAGIC gets executed at serving time.
# MAGIC
# MAGIC **`ResponsesAgent` vs `ChatAgent`:** The `ChatAgent` interface is now legacy.
# MAGIC The current recommended wrapper is `ResponsesAgent` (`mlflow.pyfunc.ResponsesAgent`),
# MAGIC which uses the OpenAI Responses API schema and provides helper utilities like
# MAGIC `to_chat_completions_input` and `output_to_responses_items_stream` for seamless
# MAGIC conversion between the Responses format and LangGraph's message dicts.

# COMMAND ----------

# `%%writefile` drops the file in the kernel's
# CWD, but on Databricks notebooks that directory **isn't automatically on
# `sys.path`** — so `from agent import AGENT` would fail. We fix that by writing
# to a known absolute path and putting that path on `sys.path` ourselves.

import os
import sys

WORK_DIR = "/tmp/langgraph_demo"
os.makedirs(WORK_DIR, exist_ok=True)
if WORK_DIR not in sys.path:
    sys.path.insert(0, WORK_DIR)

# COMMAND ----------

# MAGIC %%writefile /tmp/langgraph_demo/agent.py
# MAGIC from typing import Generator
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks_langchain import ChatDatabricks
# MAGIC from langchain_core.messages import AnyMessage
# MAGIC from langgraph.graph import START, END, StateGraph
# MAGIC from langgraph.graph.message import add_messages
# MAGIC from mlflow.models import set_model
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest,
# MAGIC     ResponsesAgentResponse,
# MAGIC     ResponsesAgentStreamEvent,
# MAGIC     output_to_responses_items_stream,
# MAGIC     to_chat_completions_input,
# MAGIC )
# MAGIC from typing import Annotated, TypedDict
# MAGIC
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
# MAGIC
# MAGIC
# MAGIC class AgentState(TypedDict):
# MAGIC     messages: Annotated[list[AnyMessage], add_messages]
# MAGIC
# MAGIC
# MAGIC def _build_graph():
# MAGIC     llm = ChatDatabricks(endpoint=LLM_ENDPOINT)
# MAGIC
# MAGIC     def respond(state: AgentState) -> dict:
# MAGIC         return {"messages": [llm.invoke(state["messages"])]}
# MAGIC
# MAGIC     builder = StateGraph(AgentState)
# MAGIC     builder.add_node("respond", respond)
# MAGIC     builder.add_edge(START, "respond")
# MAGIC     builder.add_edge("respond", END)
# MAGIC     return builder.compile()
# MAGIC
# MAGIC
# MAGIC class LangGraphResponsesAgent(ResponsesAgent):
# MAGIC     """Wraps a compiled LangGraph as an MLflow ResponsesAgent for serving."""
# MAGIC
# MAGIC     def __init__(self, graph):
# MAGIC         self.graph = graph
# MAGIC
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         outputs = [
# MAGIC             event.item
# MAGIC             for event in self.predict_stream(request)
# MAGIC             if event.type == "response.output_item.done"
# MAGIC         ]
# MAGIC         return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         request: ResponsesAgentRequest,
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         cc_msgs = to_chat_completions_input([i.model_dump() for i in request.input])
# MAGIC         for _, events in self.graph.stream({"messages": cc_msgs}, stream_mode=["updates"]):
# MAGIC             for node_data in events.values():
# MAGIC                 yield from output_to_responses_items_stream(node_data["messages"])
# MAGIC
# MAGIC
# MAGIC AGENT = LangGraphResponsesAgent(_build_graph())
# MAGIC set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Smoke test the agent module before logging
# MAGIC
# MAGIC Catch import / runtime errors here, not after a 5-minute deploy.

# COMMAND ----------

import importlib
import agent
importlib.reload(agent)

from mlflow.types.responses import ResponsesAgentRequest
from agent import AGENT

request = ResponsesAgentRequest(
    input=[{"role": "user", "content": "hello"}]
)
response = AGENT.predict(request)
print(response.output)

# COMMAND ----------

# MAGIC %md
# MAGIC    
# MAGIC ### Step 2 — Log to an MLflow experiment
# MAGIC
# MAGIC Before we log, it helps to understand which MLflow primitives live **where** and
# MAGIC **why** — in larger repos these are often scattered across files, and the
# MAGIC relationship between them becomes confusing.
# MAGIC
# MAGIC | Primitive | Where it lives | What it does |
# MAGIC |---|---|---|
# MAGIC | `mlflow.langchain.autolog()` | **Agent file** (`agent.py`) | Instruments LangGraph nodes as spans in MLflow Traces. At serving time, every request auto-generates a trace — no extra code needed. |
# MAGIC | `mlflow.models.set_model(agent)` | **Agent file** (`agent.py`) | Tells MLflow which object is the inference entry point. This is what `log_model` will serialize and what the serving endpoint will call. |
# MAGIC | `mlflow.set_experiment(path)` | **Driver notebook** (this file) | Sets the experiment that subsequent `start_run()` calls write to. Without it, runs go to the implicit notebook experiment — fine for personal work, confusing in shared repos. |
# MAGIC | `mlflow.set_registry_uri("databricks-uc")` | **Driver notebook** (this file) | Routes `register_model` to Unity Catalog (not the legacy workspace registry). Required for `agents.deploy`. |
# MAGIC | `mlflow.start_run()` | **Driver notebook** (this file) | Creates a tracked run. Each run captures the model artifact, code snapshot, dependencies, and any params/metrics you log. |
# MAGIC
# MAGIC **The mental model:** The *agent file* is what runs at serving time (so
# MAGIC autolog + set_model go there). The *driver notebook* is what runs at
# MAGIC development time to log, register, and deploy (so experiment config + run
# MAGIC management go here).
# MAGIC
# MAGIC **Tracing at serving time:** Because `autolog()` is in the agent file, every
# MAGIC inference request to the deployed endpoint automatically produces an MLflow
# MAGIC Trace — a hierarchical view of each node's inputs, outputs, and latency. These
# MAGIC traces are persisted to the inference table attached to your endpoint, queryable
# MAGIC via SQL for monitoring and evaluation.
# MAGIC
# MAGIC `resources=[...]` declares the downstream Databricks resources the agent
# MAGIC needs at serving time. The platform uses this to provision automatic
# MAGIC authentication (no PATs in code).

# COMMAND ----------

import mlflow
from mlflow.models.resources import DatabricksServingEndpoint
from pkg_resources import get_distribution

LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

# Derive the current user so each person gets their own experiment
username = spark.sql("SELECT current_user()").first()[0]
experiment_path = f"/Users/{username}/langgraph_demo_experiment"

# Be explicit about where runs are tracked and where models are registered
mlflow.set_experiment(experiment_path)
mlflow.set_registry_uri("databricks-uc")

# ResponsesAgent expects "input" (Responses API schema), not "messages" (legacy ChatAgent schema)
input_example = {
    "input": [{"role": "user", "content": "What is Lakeflow?"}]
}

with mlflow.start_run():
    logged = mlflow.pyfunc.log_model(
        python_model=os.path.join(WORK_DIR, "agent.py"),
        name="agent",
        input_example=input_example,
        pip_requirements=[
            f"langgraph=={get_distribution('langgraph').version}",
            f"langchain-core=={get_distribution('langchain-core').version}",
            f"mlflow=={get_distribution('mlflow').version}",
            "databricks-langchain",
        ],
        resources=[DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT)],
    )

print(f"Logged model URI: {logged.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 — Register to Unity Catalog
# MAGIC
# MAGIC UC is the single governed home for models. `agents.deploy` requires a
# MAGIC UC-registered model — the workspace registry is not supported.

# COMMAND ----------

CATALOG = "agentbuilder_serverless_stable_catalog"
SCHEMA = "agent_builder"
MODEL_NAME = "langgraph_demo_agent"
uc_model = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

registered = mlflow.register_model(model_uri=logged.model_uri, name=uc_model)
print(f"Registered as {uc_model} version {registered.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 — Deploy
# MAGIC
# MAGIC `agents.deploy` does the heavy lifting:
# MAGIC - Provisions a Model Serving endpoint
# MAGIC - Wires up automatic auth for declared resources
# MAGIC - Sets up the **Review App** (subject-matter-expert feedback UI)
# MAGIC - Enables inference tables for trace logging
# MAGIC
# MAGIC First call takes ~5–10 minutes (cold endpoint). Subsequent versions deploy
# MAGIC into the same endpoint and are much faster.

# COMMAND ----------

from databricks import agents

deployment = agents.deploy(
    endpoint_name="langgraph-demo-agent",
    model_name=uc_model,
    model_version=int(registered.version),
)
print(f"Endpoint:   {deployment.endpoint_name}")
print(f"Review app: {deployment.review_app_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 — Call the deployed endpoint
# MAGIC
# MAGIC Once the endpoint is **READY** (check the Serving UI or poll
# MAGIC `WorkspaceClient().serving_endpoints.get`), invoke it like any other
# MAGIC chat-style model.

# COMMAND ----------

from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
response = client.predict(
    endpoint=deployment.endpoint_name,
    inputs={"input": [{"role": "user", "content": "Give me one sentence on Unity Catalog."}]},
)
print(response)

# COMMAND ----------

# DBTITLE 1,Part 6 — RAG in LangGraph
# MAGIC %md
# MAGIC    
# MAGIC ## Part 6 — RAG in LangGraph: Tool vs Function Node
# MAGIC
# MAGIC Retrieval-Augmented Generation (RAG) grounds the LLM in your data. In LangGraph
# MAGIC there are two clean ways to wire retrieval into the graph:
# MAGIC
# MAGIC | Pattern | How it works | When to use |
# MAGIC |---|---|---|
# MAGIC | **Retriever as a Tool** | The LLM decides *whether and when* to retrieve. You define the retriever as a tool; a `ToolNode` executes it when the LLM emits a tool call. | Multi-turn agents where retrieval is one of several capabilities. The LLM can answer from memory, retrieve, or use other tools depending on the query. |
# MAGIC | **Retriever as a Function Node** | Retrieval *always* runs as a fixed step in the graph, before the LLM responds. | Simple Q&A over a corpus where every question needs context. Deterministic, lower latency (no extra LLM call to decide). |
# MAGIC
# MAGIC We'll build both below using the same retriever so you can compare the graph
# MAGIC shapes side by side.

# COMMAND ----------

# DBTITLE 1,Set up the retriever
# MAGIC %md
# MAGIC    
# MAGIC ### Set up the retriever
# MAGIC
# MAGIC We use **Databricks Vector Search** via `databricks-langchain`. Point this at
# MAGIC your own index — the retriever interface is the same regardless of embedding
# MAGIC model or index type (Delta Sync or Direct Access).
# MAGIC
# MAGIC > **Note:** If you don't have an index yet, the `DatabricksVectorSearch` below
# MAGIC > will fail at query time. Swap in any LangChain-compatible retriever (FAISS,
# MAGIC > Chroma, etc.) — the LangGraph patterns are identical.

# COMMAND ----------

# DBTITLE 1,Configure retriever and tool
from databricks_langchain import DatabricksVectorSearch
from langchain_core.tools import tool

# ─── Configure your Vector Search index ───────────────────────────────────────
VS_INDEX_NAME = "your_catalog.your_schema.your_index"  # TODO: replace with your index
VS_COLUMNS = ["content", "url"]  # columns to return from the index

# The retriever object — works like any LangChain retriever (.invoke(query) → list[Document])
vs_retriever = DatabricksVectorSearch(
    index_name=VS_INDEX_NAME,
    columns=VS_COLUMNS,
).as_retriever(search_kwargs={"k": 3})


@tool
def retrieve_docs(query: str) -> str:
    """Search the knowledge base for information relevant to the user's question."""
    docs = vs_retriever.invoke(query)
    return "\n\n---\n\n".join(
        f"[{doc.metadata.get('url', 'source')}]\n{doc.page_content}" for doc in docs
    )


# NOTE: @tool just adds metadata (name, description, JSON schema for the LLM).
# The function is still callable via retrieve_docs.invoke(query) — so we reuse
# the same object in both the tool-based and function-node approaches below.

print(f"Retriever configured for index: {VS_INDEX_NAME}")
print(f"Tool name: '{retrieve_docs.name}' — description: '{retrieve_docs.description}'")

# COMMAND ----------

# DBTITLE 1,Approach 1 — Tool-based RAG
# MAGIC %md
# MAGIC    
# MAGIC ### Approach 1 — Retriever as a Tool (Agentic RAG)
# MAGIC
# MAGIC The LLM has the retriever bound as a tool. On each turn it decides:
# MAGIC - Answer directly (no tool call → route to END)
# MAGIC - Retrieve first (tool call → route to `ToolNode` → loop back to LLM)
# MAGIC
# MAGIC This is the **ReAct** pattern applied to retrieval. The graph looks like:
# MAGIC
# MAGIC ```
# MAGIC START → agent ─┬─ tool_call ──→ tools → agent (loop)
# MAGIC                └─ no tool_call → END
# MAGIC ```
# MAGIC
# MAGIC Key pieces:
# MAGIC - `llm.bind_tools([retrieve_docs])` — makes the LLM aware of the tool
# MAGIC - `ToolNode([retrieve_docs])` — executes tool calls and returns results as `ToolMessage`
# MAGIC - `tools_condition` — LangGraph built-in that checks if the last message has tool calls

# COMMAND ----------

# DBTITLE 1,Build tool-based RAG graph
from langgraph.prebuilt import ToolNode, tools_condition

# Bind the retriever tool to the LLM
tools = [retrieve_docs]
llm_with_tools = llm.bind_tools(tools)


class RAGToolState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def agent_node(state: RAGToolState) -> dict:
    """The LLM decides whether to call the retriever tool or respond directly."""
    system = SystemMessage(
        content=(
            "You are a helpful assistant with access to a knowledge base. "
            "Use the retrieve_docs tool when the user asks a question that "
            "requires specific information from the documentation. "
            "If you can answer confidently without retrieval, do so directly."
        )
    )
    response = llm_with_tools.invoke([system] + state["messages"])
    return {"messages": [response]}


# Build the graph
tool_rag_builder = StateGraph(RAGToolState)
tool_rag_builder.add_node("agent", agent_node)
tool_rag_builder.add_node("tools", ToolNode(tools))

tool_rag_builder.add_edge(START, "agent")
tool_rag_builder.add_conditional_edges(
    "agent",
    tools_condition,  # routes to "tools" if tool_calls present, else END
)
tool_rag_builder.add_edge("tools", "agent")  # after tool execution, loop back to LLM

tool_rag_graph = tool_rag_builder.compile()
display(Image(tool_rag_graph.get_graph().draw_mermaid_png()))

# COMMAND ----------

# DBTITLE 1,Invoke tool-based RAG
# The LLM decides to retrieve because this requires specific knowledge
result = tool_rag_graph.invoke({
    "messages": [HumanMessage(content="What is Unity Catalog and how does it handle data governance?")]
})
print("=== TOOL-BASED RAG ===")
print(f"Response: {result['messages'][-1].content}")

# COMMAND ----------

# DBTITLE 1,Approach 2 — Function node RAG
# MAGIC %md
# MAGIC    
# MAGIC ### Approach 2 — Retriever as a Function Node (Deterministic RAG)
# MAGIC
# MAGIC Retrieval is a **fixed step** in the graph — it always runs before the LLM.
# MAGIC No tool-calling overhead, no extra LLM decision. Simpler and faster when you
# MAGIC *know* every query needs context.
# MAGIC
# MAGIC The graph looks like:
# MAGIC
# MAGIC ```
# MAGIC START → retrieve → generate → END
# MAGIC ```
# MAGIC
# MAGIC We add a `context` field to state so the retriever and generator communicate
# MAGIC via state rather than message history. This keeps the retrieval results
# MAGIC separate from the conversation (cleaner for multi-turn).

# COMMAND ----------

# DBTITLE 1,Build function-node RAG graph
class RAGNodeState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    context: str  # retrieved documents live here, separate from messages


def retrieve_node(state: RAGNodeState) -> dict:
    """Always retrieves — no LLM decision involved."""
    # Same @tool function, just called directly via .invoke() instead of through ToolNode.
    # This works because @tool is just metadata — the underlying function is still callable.
    user_query = state["messages"][-1].content
    context = retrieve_docs.invoke(user_query)
    return {"context": context}


def generate_node(state: RAGNodeState) -> dict:
    """Generates a response grounded in the retrieved context."""
    system = SystemMessage(
        content=(
            "You are a helpful assistant. Answer the user's question using ONLY "
            "the provided context. If the context doesn't contain the answer, "
            "say so.\n\n"
            f"Context:\n{state['context']}"
        )
    )
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}


# Build the graph
node_rag_builder = StateGraph(RAGNodeState)
node_rag_builder.add_node("retrieve", retrieve_node)
node_rag_builder.add_node("generate", generate_node)

node_rag_builder.add_edge(START, "retrieve")
node_rag_builder.add_edge("retrieve", "generate")
node_rag_builder.add_edge("generate", END)

node_rag_graph = node_rag_builder.compile()
display(Image(node_rag_graph.get_graph().draw_mermaid_png()))

# COMMAND ----------

# DBTITLE 1,Invoke function-node RAG
# Retrieval always fires — no LLM deciding whether to retrieve
result = node_rag_graph.invoke({
    "messages": [HumanMessage(content="What is Unity Catalog and how does it handle data governance?")],
    "context": "",
})
print("=== FUNCTION-NODE RAG ===")
print(f"Response: {result['messages'][-1].content}")

# COMMAND ----------

# DBTITLE 1,Comparison — when to choose which
# MAGIC %md
# MAGIC    
# MAGIC ### When to choose which
# MAGIC
# MAGIC | Consideration | Tool-based (Approach 1) | Function-node (Approach 2) |
# MAGIC |---|---|---|
# MAGIC | **LLM autonomy** | LLM decides when to retrieve | Always retrieves |
# MAGIC | **Latency** | +1 LLM call to decide | No decision overhead |
# MAGIC | **Multi-tool agents** | Natural fit — retrieval is one tool among many | Awkward if you also need other tools |
# MAGIC | **Determinism** | Non-deterministic (LLM might skip retrieval) | Deterministic — every query gets context |
# MAGIC | **Cost** | Higher (extra LLM reasoning turn) | Lower (one LLM call total) |
# MAGIC | **Best for** | Complex agents with multiple capabilities | Focused Q&A / search assistants |
# MAGIC
# MAGIC **Hybrid pattern:** You can combine both — use a function node for an initial
# MAGIC retrieval pass, then give the LLM tools for *follow-up* retrieval if the first
# MAGIC pass wasn't sufficient. This gives you a guaranteed baseline context with
# MAGIC optional agentic refinement.

# COMMAND ----------

# MAGIC %md
# MAGIC    
# MAGIC ## Where to go next
# MAGIC
# MAGIC - **More tools** — add web search, SQL execution, or API tools alongside the retriever (same `ToolNode` pattern from Part 6)
# MAGIC - **Durable checkpointing** — swap `MemorySaver` for Databricks Lakebase (drop-in replacement, production-ready persistence)
# MAGIC - **Evaluation** — `mlflow.evaluate` with built-in agent metrics (relevance, groundedness, safety) on a labeled dataset
# MAGIC - **Tracing deep-dive** — open the MLflow experiment; `autolog()` already captured per-node spans with inputs, outputs, and latency
# MAGIC - **Multi-agent** — compose graphs as nodes inside a parent graph (supervisor pattern) for complex workflows
