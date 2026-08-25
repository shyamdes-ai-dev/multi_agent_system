"""Blackboard Communication Pattern in LangGraph.

This module demonstrates the Blackboard communication pattern where multiple agents
(a Drafter and a Critic) read from and write to a shared workspace state (the Blackboard).
The agents iterate through drafts and critiques until content is approved or max iterations
are reached.
"""

import json
import operator
from typing import Literal
from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

# Initialize the chat model using Google GenAI provider
model = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash-lite")


class BlackBoardState(TypedDict):
    """State dictionary representing the shared Blackboard workspace.

    Attributes:
        messages: Accumulated conversation message trace.
        topic: Research or writing topic.
        drafts: History of generated drafts (appended via operator.add).
        critiques: History of critic feedback (appended via operator.add).
        iteration: Current loop iteration counter.
        is_approved: Boolean indicating whether the draft meets quality standards.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    topic: str
    drafts: Annotated[list[str], operator.add]
    critiques: Annotated[list[str], operator.add]
    iteration: int
    is_approved: bool


def create_blackboard_system(max_iterations: int = 3):
    """Builds and compiles the Blackboard iterative draft/critique StateGraph workflow.

    Args:
        max_iterations: Maximum number of revision cycles allowed before terminating.

    Returns:
        CompiledStateGraph: A compiled LangGraph workflow for blackboard collaboration.
    """

    class ApprovalDecision(BaseModel):
        """Structured decision output from the critic agent."""

        approved: bool = Field(description="Whether the content is approved.")
        feedback: str = Field(description="Feedback for improvement if not approved.")

    critic_llm = model.with_structured_output(ApprovalDecision)

    def drafter(state: BlackBoardState) -> dict:
        """Reads critiques from blackboard workspace and writes/revises content."""
        current_iteration = state.get("iteration", 0)
        context_parts = [f"Topic: {state['topic']}"]

        if state.get("drafts"):
            context_parts.append(f"Previous draft:\n{state['drafts'][-1]}")
        if state.get("critiques"):
            context_parts.append(f"Feedback to address:\n{state['critiques'][-1]}")

        context = "\n\n".join(context_parts)
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a skilled writer. Write or revise a concise, high-quality paragraph "
                        "(3-4 sentences) based on the topic and any feedback provided. "
                        "If feedback exists, directly address all points in your revision."
                    )
                ),
                HumanMessage(content=context),
            ]
        )

        content_text = (
            response.content[0].get("text")
            if isinstance(response.content[0].get("text"), str)
            else str(response.content[0].get("text"))
        )

        return {
            "drafts": [content_text],
            "messages": [
                HumanMessage(
                    content=f"[Drafter] Iteration {current_iteration + 1}:\n{content_text}",
                    name="drafter",
                )
            ],
            "iteration": current_iteration + 1,
        }

    def critic(state: BlackBoardState) -> dict:
        """Reviews the latest draft on the blackboard and provides structured feedback."""
        latest_draft = (
            state["drafts"][-1] if state.get("drafts") else "No draft available."
        )

        decision = critic_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a critical editor. Review the latest draft and decide:\n"
                        "- Is it clear, compelling, and well-written?\n"
                        "- Does it address the topic and any past feedback?\n"
                        "- Should it be approved or returned for revision?"
                    )
                ),
                HumanMessage(
                    content=f"Topic: {state['topic']}\nLatest Draft:\n{latest_draft}"
                ),
            ]
        )

        if decision.approved:
            return {
                "is_approved": True,
                "messages": [
                    HumanMessage(
                        content=f"[Critic] Approved! Final Draft:\n{latest_draft}",
                        name="critic",
                    )
                ],
            }
        else:
            return {
                "is_approved": False,
                "critiques": [decision.feedback],
                "messages": [
                    HumanMessage(
                        content=f"[Critic] Feedback: {decision.feedback}", name="critic"
                    )
                ],
            }

    def decide_continue(state: BlackBoardState) -> str:
        """Determines whether to continue drafting or finalize the workflow."""
        if state.get("is_approved") or state.get("iteration", 0) >= max_iterations:
            return "end"
        return "drafter"

    graph = StateGraph(BlackBoardState)
    graph.add_node("drafter", drafter)
    graph.add_node("critic", critic)

    graph.add_edge(START, "drafter")
    graph.add_edge("drafter", "critic")
    graph.add_conditional_edges(
        "critic",
        decide_continue,
        {
            "drafter": "drafter",
            "end": END,
        },
    )

    return graph.compile()


def blackboard_demo():
    """Executes a demonstration of the Blackboard pattern with Drafter and Critic."""
    print("=== Blackboard Communication Pattern Demo ===")
    workflow = create_blackboard_system(max_iterations=3)

    topic = "The importance of artificial intelligence ethics in modern software development"
    print(f"Topic: {topic}\n")

    initial_state = {
        "topic": topic,
        "messages": [],
        "drafts": [],
        "critiques": [],
        "iteration": 0,
        "is_approved": False,
    }

    final_state = workflow.invoke(initial_state)

    print("\n" + "=" * 50)
    print("BLACKBOARD WORKFLOW COMPLETE")
    print("=" * 50)
    print(f"Total Iterations: {final_state.get('iteration')}")
    print(f"Approved: {final_state.get('is_approved')}")
    if final_state.get("drafts"):
        print(f"\nFinal Approved Draft:\n{final_state['drafts'][-1]}")


if __name__ == "__main__":
    blackboard_demo()
