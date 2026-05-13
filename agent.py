from typing import Generator

import mlflow
from databricks_langchain import ChatDatabricks
from langchain_core.messages import AnyMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from mlflow.models import set_model
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from typing import Annotated, TypedDict

mlflow.langchain.autolog()

LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def _build_graph():
    llm = ChatDatabricks(endpoint=LLM_ENDPOINT)

    def respond(state: AgentState) -> dict:
        return {"messages": [llm.invoke(state["messages"])]}

    builder = StateGraph(AgentState)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    builder.add_edge("respond", END)
    return builder.compile()


class LangGraphResponsesAgent(ResponsesAgent):
    """Wraps a compiled LangGraph as an MLflow ResponsesAgent for serving."""

    def __init__(self, graph):
        self.graph = graph

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(
        self,
        request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        cc_msgs = to_chat_completions_input([i.model_dump() for i in request.input])
        for _, events in self.graph.stream({"messages": cc_msgs}, stream_mode=["updates"]):
            for node_data in events.values():
                yield from output_to_responses_items_stream(node_data["messages"])


AGENT = LangGraphResponsesAgent(_build_graph())
set_model(AGENT)
