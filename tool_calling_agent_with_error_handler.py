"""Tool-Calling Agent with Error Handling in LangGraph.

This module demonstrates defensive tool design and error recovery patterns in LangGraph,
enabling an LLM agent to handle execution errors (e.g. division by zero) gracefully.
"""

from typing import Literal
from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

# Initialize the chat model using Google GenAI provider
model = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash-lite")


@tool
def divide(a: float, b: float) -> str:
    """Divide two numbers safely.

    Args:
        a: Numerator value.
        b: Denominator value.

    Returns:
        String result of division or error message if denominator is zero.
    """
    if b == 0:
        return "Error: Division by zero is undefined."
    result = a / b
    return f"Result of {a} divided by {b} is {result}"


def tool_with_errors():
    """Builds and compiles the tool-calling StateGraph agent configured with error handling.

    Returns:
        CompiledStateGraph: A compiled LangGraph workflow with error handling support.
    """
    tools = [divide]
    model_with_tools = model.bind_tools(tools)

    class AgentState(TypedDict):
        """State dictionary for tracking conversation messages."""

        messages: Annotated[list[BaseMessage], add_messages]

    def agent_node(state: AgentState) -> dict:
        """Agent node invoking the model with available tools."""
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        """Router deciding whether to execute tools or complete the workflow."""
        last_message = state["messages"][-1]

        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return "end"
        return "tools"

    tool_node = ToolNode(tools)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END}
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


def tool_call_with_error_handling():
    """Executes a sample division by zero query to demonstrate error handling."""
    agent = tool_with_errors()
    print("Tool Error Handling Agent")

    query = "Divide 10 by 0"
    print(f"Query: {query}")

    response = agent.invoke({"messages": [HumanMessage(content=query)]})
    last_msg_content = response["messages"][-1].content
    if (
        isinstance(last_msg_content, list)
        and len(last_msg_content) > 0
        and isinstance(last_msg_content[0], dict)
    ):
        text_out = last_msg_content[0].get("text", str(last_msg_content))
    else:
        text_out = str(last_msg_content)
    print(f"Final response: {text_out}")
    print(f"Total messages: {len(response['messages'])}")


if __name__ == "__main__":
    tool_call_with_error_handling()
