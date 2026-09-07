"""Hierarchical Multi-Agent System Architecture in LangGraph.

This module demonstrates a multi-level hierarchical agent organization:
1. Top-Level Supervisor (CEO Node): Evaluates incoming user requests and routes them to dedicated specialized departments.
2. Subgraph Departments:
   - Research Department: Parallel web researcher & paper reviewer agents synthesized by a Research Lead.
   - Content Department: Sequential Content Writer -> Content Editor workflow.
   - Analysis Department: Sequential Data Analyst -> Strategy Advisor workflow.
3. Nested StateGraph Compilation: Subgraphs are compiled and included as single nodes in the parent supervisor graph.
"""

from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Load environment configuration (API keys, settings)
load_dotenv()

# Initialize primary chat model
model = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash-lite")


def extract_text(content) -> str:
    """Safely extracts a plain string from LLM response content fields."""
    if isinstance(content, str):
        return content
    elif (
        isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict)
    ):
        return content[0].get("text", str(content))
    return str(content)


# ============================================================================
# Shared State Schema
# ============================================================================


class TeamState(TypedDict):
    """Shared state schema used across both parent graph and department subgraphs.

    Attributes:
        messages: Message history log appended via add_messages operator.
        final_answer: Polished final response produced by department leads.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    final_answer: str


# ============================================================================
# Department 1: Research Team Subgraph
# ============================================================================


def build_research_team() -> StateGraph:
    """Builds the Research Department subgraph featuring parallel worker nodes.

    Returns:
        StateGraph: Uncompiled StateGraph for the Research Team.
    """

    def web_researcher(state: TeamState) -> dict:
        """Worker Node: Gathers broad web research and factual key points."""
        query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = str(msg.content)
                break

        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a web researcher. Find key facts and data about the topic. "
                        "Provide 3-4 bullet points of findings. Be specific."
                    )
                ),
                HumanMessage(content=query),
            ]
        )
        text = extract_text(response.content)
        return {
            "messages": [
                AIMessage(
                    name="web_researcher",
                    content=f"[Web Researcher]: {text}",
                )
            ]
        }

    def paper_reviewer(state: TeamState) -> dict:
        """Worker Node: Reviews academic and technical literature for deep insights."""
        query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = str(msg.content)
                break

        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are an academic paper reviewer. Analyze academic perspective on the topic in 2-3 sentences."
                    )
                ),
                HumanMessage(content=query),
            ]
        )
        text = extract_text(response.content)
        return {
            "messages": [
                AIMessage(
                    name="paper_reviewer",
                    content=f"[Paper Reviewer]: {text}",
                )
            ]
        }

    def research_lead(state: TeamState) -> dict:
        """Lead Node: Synthesizes findings from both web researcher and paper reviewer."""
        # Extract findings from preceding worker AIMessages
        findings = "\n\n".join(
            f"{msg.name or 'Researcher'}: {msg.content}"
            for msg in state["messages"]
            if isinstance(msg, AIMessage)
        )

        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a research lead. Synthesize the web researcher's and paper reviewer's findings "
                        "into a cohesive research brief. Keep it to one short paragraph."
                    )
                ),
                HumanMessage(
                    content=f"Here are the research findings to synthesize:\n\n{findings}"
                ),
            ]
        )
        text = extract_text(response.content)
        return {
            "messages": [
                AIMessage(
                    name="research_lead",
                    content=f"[Research Lead]: {text}",
                )
            ],
            "final_answer": text,
        }

    # Assemble Subgraph Nodes & Edges
    builder = StateGraph(TeamState)

    builder.add_node("web_researcher", web_researcher)
    builder.add_node("paper_reviewer", paper_reviewer)
    builder.add_node("research_lead", research_lead)

    # Parallel dispatch from START to researchers
    builder.add_edge(START, "web_researcher")
    builder.add_edge(START, "paper_reviewer")

    # Fan-in convergence at research_lead
    builder.add_edge("web_researcher", "research_lead")
    builder.add_edge("paper_reviewer", "research_lead")

    # Complete team output
    builder.add_edge("research_lead", END)

    return builder


# ============================================================================
# Department 2: Content Team Subgraph
# ============================================================================


def build_content_team() -> StateGraph:
    """Builds the Content Department subgraph featuring sequential writing & editing.

    Returns:
        StateGraph: Uncompiled StateGraph for the Content Team.
    """

    def content_writer(state: TeamState) -> dict:
        """Worker Node: Drafts engaging article/blog post content based on prompt."""
        query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = str(msg.content)
                break

        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a content writer. Write a detailed, engaging blog post based on the topic requested. "
                        "Structure with title, introduction, body, and conclusion."
                    )
                ),
                HumanMessage(content=query),
            ]
        )
        text = extract_text(response.content)
        return {
            "messages": [
                AIMessage(
                    name="content_writer",
                    content=f"[Content Writer]: {text}",
                )
            ]
        }

    def content_editor(state: TeamState) -> dict:
        """Editor Node: Polishes and proofreads writer draft for tone and quality."""
        draft_text = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.name == "content_writer":
                draft_text = str(msg.content)
                break

        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a content editor. Proofread and refine the blog post draft. "
                        "Ensure tone consistency, perfect grammar, and strong readability."
                    )
                ),
                HumanMessage(content=f"Draft to edit:\n\n{draft_text}"),
            ]
        )
        text = extract_text(response.content)
        return {
            "messages": [
                AIMessage(
                    name="content_editor",
                    content=f"[Content Editor]: {text}",
                )
            ],
            "final_answer": text,
        }

    content_builder = StateGraph(TeamState)
    content_builder.add_node("content_writer", content_writer)
    content_builder.add_node("content_editor", content_editor)

    content_builder.add_edge(START, "content_writer")
    content_builder.add_edge("content_writer", "content_editor")
    content_builder.add_edge("content_editor", END)

    return content_builder


# ============================================================================
# Department 3: Analysis Team Subgraph
# ============================================================================


def build_analysis_team() -> StateGraph:
    """Builds the Analysis Department subgraph featuring data analytics & strategic advisory.

    Returns:
        StateGraph: Uncompiled StateGraph for the Analysis Team.
    """

    def data_analyst(state: TeamState) -> dict:
        """Analyst Node: Extracts quantitative trends and analytical observations."""
        query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = str(msg.content)
                break

        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a data analyst. Extract 3-4 data-driven insights and key analytical trends for the request."
                    )
                ),
                HumanMessage(content=query),
            ]
        )
        text = extract_text(response.content)
        return {
            "messages": [
                AIMessage(
                    name="data_analyst",
                    content=f"[Data Analyst]: {text}",
                )
            ]
        }

    def strategy_advisor(state: TeamState) -> dict:
        """Advisor Node: Translates data insights into actionable strategic advice."""
        analysis_text = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.name == "data_analyst":
                analysis_text = str(msg.content)
                break

        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a strategy advisor. Provide actionable strategic recommendations based on the analytical insights."
                    )
                ),
                HumanMessage(content=f"Analytical findings:\n\n{analysis_text}"),
            ]
        )
        text = extract_text(response.content)
        return {
            "messages": [
                AIMessage(
                    name="strategy_advisor",
                    content=f"[Strategy Advisor]: {text}",
                )
            ],
            "final_answer": text,
        }

    analysis_builder = StateGraph(TeamState)
    analysis_builder.add_node("data_analyst", data_analyst)
    analysis_builder.add_node("strategy_advisor", strategy_advisor)

    analysis_builder.add_edge(START, "data_analyst")
    analysis_builder.add_edge("data_analyst", "strategy_advisor")
    analysis_builder.add_edge("strategy_advisor", END)

    return analysis_builder


# ============================================================================
# Top-Level Parent Supervisor Graph (CEO Node)
# ============================================================================


def create_hierarchical_system():
    """Builds the top-level parent graph orchestrating compiled department subgraphs as nodes.

    Returns:
        CompiledStateGraph: Complete hierarchical multi-agent graph ready for invocation.
    """
    # Compile department subgraphs into reusable runnable units
    research_team = build_research_team().compile()
    content_team = build_content_team().compile()
    analysis_team = build_analysis_team().compile()

    # Routing Schema for Structured Decision Output
    class DepartmentRoute(BaseModel):
        department: Literal["research", "content", "analysis"] = Field(
            description="Which department should handle this request."
        )
        reasoning: str = Field(
            description="Brief reasoning for selecting this department."
        )

    router_llm = model.with_structured_output(DepartmentRoute)

    def ceo_supervisor(state: TeamState) -> dict:
        """Parent Supervisor (CEO) Node: Analyzes user request and selects target department."""
        query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = str(msg.content)
                break

        response = router_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are the CEO of a multi-agent organization. "
                        "Determine which department (research, content, analysis) is best suited for the request."
                    )
                ),
                HumanMessage(content=f"User Request: {query}"),
            ]
        )
        return {
            "messages": [
                AIMessage(
                    name="ceo",
                    content=f"[CEO]: Routing to {response.department} department. Reasoning: {response.reasoning}",
                )
            ]
        }

    def route_to_department(state: TeamState) -> str:
        """Router Function: Evaluates CEO decision and returns node identifier."""
        last_ai = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.name == "ceo":
                last_ai = msg
                break

        if last_ai:
            text = str(last_ai.content).lower()
            if "research" in text:
                return "research_team"
            elif "content" in text:
                return "content_team"
            elif "analysis" in text:
                return "analysis_team"

        return "research_team"

    # Build Parent Graph
    parent = StateGraph(TeamState)

    parent.add_node("ceo", ceo_supervisor)
    parent.add_node("research_team", research_team)
    parent.add_node("content_team", content_team)
    parent.add_node("analysis_team", analysis_team)

    parent.add_edge(START, "ceo")
    parent.add_conditional_edges(
        "ceo",
        route_to_department,
        {
            "research_team": "research_team",
            "content_team": "content_team",
            "analysis_team": "analysis_team",
        },
    )
    parent.add_edge("research_team", END)
    parent.add_edge("content_team", END)
    parent.add_edge("analysis_team", END)

    return parent.compile()


# ============================================================================
# Execution & Demonstration Entry Point
# ============================================================================


def hierarchical_routing():
    """Runs a multi-query demonstration of the hierarchical multi-agent system."""
    system = create_hierarchical_system()

    print(
        "================================================================================"
    )
    print(" HIERARCHICAL MULTI-AGENT SYSTEM (CEO -> SUBGRAPHS) ")
    print(
        "================================================================================"
    )
    queries = [
        "What are the latest developments and technical trends in LLM fine-tuning?",
        "Write an engaging blog post about RAG (Retrieval-Augmented Generation).",
        "Should our startup invest in custom AI model training this year?",
    ]

    for query in queries:
        print(f"\nUser Query: {query}")
        print("-" * 80)
        result = system.invoke(
            {"messages": [HumanMessage(content=query)], "final_answer": ""}
        )

        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and msg.name == "ceo":
                print(f"CEO Routing: {msg.content}")

        print(f"\nFinal Answer Output:\n{result['final_answer']}")
        print("=" * 80)


if __name__ == "__main__":
    hierarchical_routing()
