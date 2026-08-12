import operator
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from typing import Literal

load_dotenv()

model = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash-lite")

@tool
def divide(a: float, b: float) -> str:
    """Divide two numbers."""

    if b == 0:
        return "Error: Division by zero"
    result = a / b
    return f"Result of {a} divided by {b} is {result}"


def tool_with_errors():
    """Demo tool error handling"""

    tools = [divide]
    model_with_tools = model.bind_tools(tools)


    class AgentState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]


    def agent_node(state: AgentState) -> str:
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        last_message = state["messages"][-1]

        # If no tool calls, we are done
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return "end"
        return "tools"
    
    tool_node = ToolNode(tools)
    
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

def tool_call_with_error_handling():
    """Demo the tool error handling."""
    agent = tool_with_errors()
    print("Tool Error Handling Agent")

    query = "Divide 10 by 0"
    print(f"Query: {query}")

    response = agent.invoke({"messages": [HumanMessage(content=query)]})
    print(f"Final response: {response['messages'][-1].content[0].get('text')}")
    print(f"Total messages: {len(response['messages'])}")

if __name__ == "__main__":
    tool_call_with_error_handling()
    