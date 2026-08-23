"""Tool-Calling Agent Pattern in LangGraph.

This module demonstrates binding custom tools (calculator, weather lookup, web search)
to an LLM using `model.bind_tools()` and orchestrating execution via LangGraph's
`ToolNode` and conditional state transitions.
"""

import json
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
model = init_chat_model(model="gemini-3.5-flash-lite", model_provider="google_genai")


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression safely.

    Args:
        expression: A mathematical expression string, e.g., '2 + 2' or '15 * 20'.

    Returns:
        String containing the calculated result or error message.
    """
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating: {e}"


@tool
def get_weather(city: str) -> str:
    """Get current weather conditions for a specified city.

    Args:
        city: The target city name, e.g., 'New York', 'London', 'Tokyo', 'Paris'.

    Returns:
        String weather report for the requested city.
    """
    weather_data = {
        "new_york": "72F Sunny",
        "london": "65F Cloudy",
        "tokyo": "78F Humid",
        "paris": "65F, Partly Cloudy",
    }
    city_lower = city.lower()
    if city_lower in weather_data:
        return f"Weather in {city}: {weather_data[city_lower]}"
    return f"Weather data not available for {city}"


@tool
def search_web(query: str) -> str:
    """Search web knowledge for a query topic.

    Args:
        query: Search query text.

    Returns:
        String search result excerpt.
    """
    search_results = {
        "python programming": "Python is a high-level programming language known for its versatility.",
        "latest news": "Today's top news: AI continues to advance, impacting various industries.",
        "best restaurants in new york": "Top restaurants in New York include Le Bernardin.",
    }
    query_lower = query.lower()
    if query_lower in search_results:
        return f"Search results for '{query}': {search_results[query_lower]}"
    return f"No search results found for '{query}'"


class AgentState(TypedDict):
    """State dictionary for tracking message history in tool calling.

    Attributes:
        messages: Sequence of conversation messages accumulated between agent and tools.
    """
    messages: Annotated[list[BaseMessage], add_messages]


def create_tool_agent():
    """Builds and compiles the tool-calling StateGraph agent.

    Returns:
        CompiledStateGraph: A compiled LangGraph workflow with bound tool execution nodes.
    """
    tools = [calculate, get_weather, search_web]
    model_with_tools = model.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        """Agent node that passes message history to the tool-bound model."""
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        """Conditional routing function determining whether to invoke tools or end."""
        last_message = state["messages"][-1]

        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return "end"

        return "tools"

    # Create the tool node
    tool_node = ToolNode(tools)

    # Create the graph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END}
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


def calling_tool_agent():
    """Executes sample queries using the tool-calling agent."""
    agent = create_tool_agent()

    queries = [
        "What's 25*19?",
        "What is the weather in Tokyo?",
        "What's 100/4 and what's the weather in London?",
    ]

    print("Tool-calling Agent")

    for query in queries:
        print(f"Query: {query}")
        response = agent.invoke({"messages": [HumanMessage(content=query)]})
        last_msg_content = response['messages'][-1].content
        if isinstance(last_msg_content, list) and len(last_msg_content) > 0 and isinstance(last_msg_content[0], dict):
            text_out = last_msg_content[0].get('text', str(last_msg_content))
        else:
            text_out = str(last_msg_content)
        print(f"Final response: {text_out}")
        print(f"Total Messages: {len(response['messages'])}")
        print("=" * 40)


def tool_execution_trace():
    """Executes a sample query and prints step-by-step tool invocation traces."""
    agent = create_tool_agent()
    print("\nTool Execution Trace:\n")

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Calculate 15% of 300 and check the weather in Paris"
                )
            ]
        }
    )

    for i, msg in enumerate(result["messages"]):
        print(f"Step {i+1} ({type(msg).__name__}):")
        if isinstance(msg, HumanMessage):
            print(f"  Content: {msg.content}")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                print(f"  Tool Calls: {len(msg.tool_calls)}")
                for tc in msg.tool_calls:
                    print(f"    {tc['name']}({tc['args']})")
            else:
                print(f"  Content: {msg.content}")
        elif isinstance(msg, ToolMessage):
            print(f"  Tool: {msg.name}")
            print(f"  Result: {msg.content}")
        print("-" * 40)


if __name__ == "__main__":
    tool_execution_trace()

