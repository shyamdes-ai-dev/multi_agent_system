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
import json

load_dotenv()

model = init_chat_model(model="gemini-3.5-flash-lite", model_provider="google_genai")


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression. Example: calculate('2 + 2')"""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating: {e}"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""

    # Simulate the current weather for a city

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
    """Search the web for a query."""

    search_results = {
        "python programming": "Python is a high-level programming language known for its versatality",
        "latest news": "Today's top news: AI continues to advance , impacting various industry",
        "best restaurants in new york": "Top restaurants in New York include Le Bernardin",
    }
    query_lower = query.lower()
    if query_lower in search_results:
        return f"Search results for '{query}' {search_results[query_lower]}"
    return f"No search results found for '{query}'"


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def create_tool_agent():
    """Create a basic tool-calling agent."""

    tools = [calculate, get_weather, search_web]
    model_with_tools = model.bind_tools(tools)  # this is the secret!

    def agent_node(state: AgentState) -> str:
        # Generate a response usning the LLM with tool access
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        """Check if we should continue to tools or end."""
        last_message = state["messages"][-1]

        # If no tool calls, we are done
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return "end"

        return "tools"

    # Create the tool node
    tool_node = ToolNode(tools)

    # create the graph
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
    """Demo the tool-calling agent"""

    agent = create_tool_agent()

    queries = [
        "What's 25*19? ",
        "What is the weather in Tokyo ",
        "What's 100/4 and what's the weather in London?",
    ]

    print("Tool-calling Agent")

    for query in queries:
        print(f"Query: {query} ")
        response = agent.invoke({"messages": [HumanMessage(content=query)]})
        print(f"Final response : {response['messages'][-1].content[0].get('text')}")
        print(f"Total Messages: {len(response['messages'])}")
        print("=" * 40)


def tool_execution_trace():
    """Show detailed tool execution trace."""

    agent = create_tool_agent()
    print("\n Too l Execution Trace:\n")

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
        print(f"Step {i+1} ({type(msg).__name__}): ")
        if isinstance(msg, HumanMessage):
            print(f"  content: {msg.content}")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                print(f"Tool Calls: {len(msg.tool_calls)}")
                for tc in msg.tool_calls:
                    print(f"  {tc['name']}({tc['args']})")
            else:
                print(f"  Content: {msg.content}")
        elif isinstance(msg, ToolMessage):
            print(f" Tool: {msg.name}")
            print(f" Result: {msg.content}")
        print("-" * 40)


tool_execution_trace()
