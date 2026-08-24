"""Message-Passing Communication Pattern in LangGraph.

This module demonstrates a sequential pipeline pattern where multiple agents
(Researcher -> Fact Checker -> Summarizer) communicate by reading and appending
to a shared message history state.
"""

import json
from typing import Literal
from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

# Initialize the chat model using Google GenAI provider
model = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash-lite")


class MessagePassingState(TypedDict):
    """State dictionary for tracking sequential message passing.

    Attributes:
        messages: Accumulated conversation message trace passed across pipeline stages.
        current_phase: Current phase string tracking pipeline progression.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    current_phase: str


def create_message_passing_pipeline():
    """Builds and compiles the sequential message-passing StateGraph pipeline.

    Returns:
        CompiledStateGraph: A compiled LangGraph workflow for sequential message passing.
    """

    def researcher(state: MessagePassingState) -> dict:
        """Researches the prompt topic and posts findings into message history."""
        response = model.invoke(
            [
                SystemMessage(
                    content="You are a helpful assistant that researches topics. Read the user's question and provide your findings in 2-3 sentences."
                ),
                *state["messages"],
            ]
        )
        content_text = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        return {
            "messages": [
                HumanMessage(content=f"[Researcher]: {content_text}", name="researcher")
            ],
            "current_phase": "fact_checker",
        }

    def fact_checker(state: MessagePassingState) -> dict:
        """Verifies findings from message history and posts fact-check conclusions."""
        response = model.invoke(
            [
                SystemMessage(
                    content="You are a helpful assistant that checks facts. Read the user's question and research findings, then verify their accuracy in 2-3 sentences."
                ),
                *state["messages"],
            ]
        )
        content_text = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        return {
            "messages": [
                HumanMessage(
                    content=f"[Fact Checker]: {content_text}", name="fact_checker"
                )
            ],
            "current_phase": "summarizer",
        }

    def summarizer(state: MessagePassingState) -> dict:
        """Summarizes research and fact-check findings into a concise report."""
        response = model.invoke(
            [
                SystemMessage(
                    content="You are a helpful assistant that synthesizes past messages into a clear summary report."
                ),
                *state["messages"],
            ]
        )
        content_text = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        return {
            "messages": [
                HumanMessage(
                    content=f"[Summarized Report]: {content_text}", name="summarizer"
                )
            ],
            "current_phase": "done",
        }

    graph = StateGraph(MessagePassingState)
    graph.add_node("researcher", researcher)
    graph.add_node("fact_checker", fact_checker)
    graph.add_node("summarizer", summarizer)

    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "fact_checker")
    graph.add_edge("fact_checker", "summarizer")
    graph.add_edge("summarizer", END)

    return graph.compile()


def message_passing_demo():
    """Executes a demonstration of the message-passing pipeline workflow."""
    print("=== Message Passing Demo ===")
    workflow = create_message_passing_pipeline()
    inputs = {
        "messages": [
            HumanMessage(
                content="What are the main benefits of artificial intelligence in healthcare?"
            )
        ],
        "current_phase": "start",
    }
    final_state = workflow.invoke(inputs)

    print("\n--- Pipeline Message Trace ---")
    for msg in final_state["messages"]:
        print(f"\n{msg.content}")


if __name__ == "__main__":
    message_passing_demo()
