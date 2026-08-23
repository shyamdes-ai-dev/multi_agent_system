"""Supervisor Architecture Pattern in LangGraph.

This module demonstrates a Supervisor Agent system where a central coordinator LLM
routes incoming user tasks across a team of specialist worker agents (Researcher,
Writer, Critic) until task completion.
"""

from typing import Literal
import operator
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated

load_dotenv()


class SuperVisorState(TypedDict):
    """State dictionary for tracking supervisor workflow progression.

    Attributes:
        messages: Accumulated conversation state between supervisor and specialist agents.
        next_agent: Target worker agent to route to, or FINISH when complete.
        task_completed: Boolean flag indicating if the overall task is finalized.
        final_response: Polished output response extracted upon task completion.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str
    task_completed: bool
    final_response: str


def create_supervisor_system():
    """Builds and compiles the Supervisor-Worker StateGraph orchestration workflow.

    Returns:
        CompiledStateGraph: A compiled LangGraph workflow ready for task orchestration.
    """

    model = init_chat_model(
        model_provider="google_genai", model="gemini-3.5-flash-lite"
    )

    class RouteDecision(BaseModel):
        """Structured routing decision schema produced by the supervisor agent."""
        next: Literal["researcher", "writer", "critic", "FINISH"] = Field(
            description="The next specialist agent to invoke, or FINISH if task is complete."
        )
        reasoning: str = Field(description="Reasoning behind the routing decision.")

    supervisor_llm = model.with_structured_output(RouteDecision)

    def supervisor(state: SuperVisorState) -> dict:
        """Central coordinator node that inspects progress and decides the next agent or FINISH."""
        system_prompt = """
            You are the Supervisor Agent for a team of specialists. Your job is to coordinate them to complete a task.

            Specialist roles:
            1. researcher - Gathers information, background, and facts.
            2. writer - Creates and refines text content.
            3. critic - Reviews and improves work.

            Based on the conversation history, decide which agent should act next.
            If the task is complete, respond with FINISH.
            Do not include markdown formatting in your decision.
            """

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        decision = supervisor_llm.invoke(messages)

        if decision.next == "FINISH":
            return {
                "next_agent": "FINISH",
                "task_completed": True,
            }

        return {
            "next_agent": decision.next,
            "messages": [
                AIMessage(
                    content=f"[SUPERVISOR] Routing to {decision.next}: {decision.reasoning}"
                )
            ],
        }

    def researcher(state: SuperVisorState) -> dict:
        """Research specialist node that gathers relevant factual background."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a research specialist. Gather facts and information relevant to the topic.",
                ),
                (
                    "human",
                    "Task Context:\n{context}\n\nProvide your research findings.",
                ),
            ]
        )
        task = next(
            (m.content for m in state["messages"] if isinstance(m, HumanMessage)), ""
        )
        response = model.invoke(prompt.format_messages(context=task))
        content_text = response.content if isinstance(response.content, str) else str(response.content)
        return {
            "messages": [
                HumanMessage(content=f"[Researcher]: {content_text}")
            ]
        }

    def writer(state: SuperVisorState) -> dict:
        """Writing specialist node that crafts polished draft content."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a writing specialist. Create clear and concise content based on the provided context.",
                ),
                (
                    "human",
                    "Previous work:\n{context}\n\nWrite a polished version of this content.",
                ),
            ]
        )

        task = next(
            (m.content for m in state["messages"] if isinstance(m, HumanMessage)), ""
        )
        response = model.invoke(prompt.format_messages(context=task))
        content_text = response.content if isinstance(response.content, str) else str(response.content)
        return {
            "messages": [
                HumanMessage(content=f"[Writer]: {content_text}")
            ]
        }

    def critic(state: SuperVisorState) -> dict:
        """Critic specialist node that reviews draft content for quality improvements."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a critic. Review the following content for accuracy, clarity, and improvements.",
                ),
                (
                    "human",
                    "Content to review:\n{context}\n\nProvide your critique and suggestions for improvement.",
                ),
            ]
        )

        context = "\n".join([str(m.content) for m in state["messages"][-3:]])
        response = model.invoke(prompt.format_messages(context=context))
        content_text = response.content if isinstance(response.content, str) else str(response.content)
        return {
            "messages": [
                HumanMessage(content=f"[Critic]: {content_text}")
            ]
        }

    def finalize(state: SuperVisorState) -> dict:
        """Finalization node extracting the completed writer response."""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage) and "[Writer]" in str(msg.content):
                content = str(msg.content).replace("[Writer]: ", "")
                return {"final_response": content}
        return {"final_response": "Task Completed"}

    def route_to_agent(state: SuperVisorState) -> str:
        """Conditional routing function switching from supervisor to specialist or finalize."""
        if state.get("task_completed"):
            return "FINISH"
        return state["next_agent"]

    graph = StateGraph(SuperVisorState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_node("critic", critic)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "researcher": "researcher",
            "writer": "writer",
            "critic": "critic",
            "FINISH": "finalize",
        },
    )
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("critic", "supervisor")
    graph.add_edge("finalize", END)

    return graph.compile()


def run_supervisor():
    """Executes the supervisor-worker orchestration demo across sample tasks."""
    print("Supervisor Agent System")
    print("-" * 50)

    graph = create_supervisor_system()

    tasks = [
        "Write a short blog post about the benefits of AI",
        "Explain the concept of quantum computing in simple terms",
        "Summarize the latest trends in renewable energy",
    ]

    for task in tasks:
        print(f"\nTask: {task}")
        print("-" * 20)
        response = graph.invoke(
            {
                "messages": [HumanMessage(content=task)],
                "next_agent": "",
                "task_completed": False,
                "final_response": "",
            }
        )
        print(f"\nFinal Response: {response['final_response']}")
        print(f"Total Messages: {len(response['messages'])}")
        print("=" * 50)


if __name__ == "__main__":
    run_supervisor()

