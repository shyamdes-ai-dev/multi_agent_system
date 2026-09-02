from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict, Annotated
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash-lite")


# ==========================================
# Shared State Schema
# ==========================================


class TeamState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    final_answer: str


# ==========================================
#       BUILD RESEARCH TEAM
# ==========================================


def build_research_team() -> StateGraph:

    def web_researcher(state: TeamState) -> dict:
        """
        Searches the web for information
        """
        query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break

        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a web researcher. Find key facts and data about the topic."
                        "Provide 3-4 bullet points of findings. Be specific"
                    )
                ),
                HumanMessage(content=query),
            ]
        )
        return {
            "messages": [
                AIMessage(
                    name="web_researcher",
                    content=f"[Web Researcher]: {response.content[0].get('text')}",
                )
            ]
        }

    def paper_reviewer(state: TeamState) -> dict:
        """
        Reviews academic papers for key insights
        """
        query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are an academic paper reviewer. Analyze the paper related to the topic in 2-3 sentences only"
                    )
                ),
                HumanMessage(content=query),
            ]
        )
        return {
            "messages": [
                AIMessage(
                    name="paper_reviewer",
                    content=f"[Paper Reviewer]: {response.content[0].get('text')}",
                )
            ]
        }

    def research_lead(state: TeamState) -> dict:
        """
        Synthesizes findings from both researchers
        """
        # Extract findings from preceding researcher AIMessages
        findings = "\n\n".join(
            f"{msg.name or 'Researcher'}: {msg.content}"
            for msg in state["messages"]
            if isinstance(msg, AIMessage)
        )

        response = model.invoke(
            [
                SystemMessage(content="""
                You are a research lead. Synthesize the web researchers and paper reviewer's findings into a cohesive research brief"
                Keep it to one short paragraph and highlight the key insights and their relevance to the original query.
            """),
                HumanMessage(
                    content=f"Here are the research findings to synthesize:\n\n{findings}"
                ),
            ]
        )
        return {
            "messages": [
                AIMessage(
                    name="research_lead",
                    content=f"[Research Lead]: {response.content[0].get('text')}",
                )
            ],
            "final_answer": response.content[0].get("text"),
        }

    # ==========================================
    # Build the Research Team Graph
    # ==========================================

    builder = StateGraph(TeamState)

    builder.add_node("web_researcher", web_researcher)
    builder.add_node("paper_reviewer", paper_reviewer)
    builder.add_node("research_lead", research_lead)

    # Define edges
    builder.add_edge(START, "web_researcher")
    builder.add_edge(START, "paper_reviewer")

    # Parallel paths merge at research_lead
    builder.add_edge("web_researcher", "research_lead")
    builder.add_edge("paper_reviewer", "research_lead")

    # Research lead signals done
    builder.add_edge("research_lead", END)

    return builder


# ==============================================
# Build Content team (subgraph)
# ==============================================


def build_contetnt_team() -> StateGraph:
    """Build the content department subgraph"""

    def content_writer(state: TeamState) -> dict:
        """Writes content based on available context"""

        findings = "\n\n".join(
            f"{msg.name or 'Researcher'}: {msg.content}"
            for msg in state["messages"]
            if isinstance(msg, AIMessage)
        )

        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a content writer. Write a detailed, engaging blog post based on the research findings provided."
                        "The tone should be accessible and informative, suitable for a general audience."
                        "Structure your response with a catchy title, an introduction, body paragraphs, and a conclusion."
                        "Expand on the key insights from the research and make them easy to understand."
                    )
                ),
                HumanMessage(
                    content=f"Here are the research findings to work with:\n\n{findings}"
                ),
            ]
        )
        return {
            "messages": [
                AIMessage(
                    name="content_writer",
                    content=f"[Content Writer]: {response.content[0].get('text')}",
                )
            ]
        }

    def content_editor(state: TeamState) -> dict:
        """Edits and polishes the writer's output."""
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a content editor. Proofread and refine the blog post written by the content writer."
                        "Check for clarity, coherence, grammar, spelling, and punctuation errors."
                        "Ensure the tone is consistent and the message is clear and impactful."
                        "Return the final polished version."
                    )
                ),
                HumanMessage(
                    content=f"Here is the blog post to edit:\n\n{state['messages']}"
                ),
            ]
        )
        return {
            "messages": [
                AIMessage(
                    name="content_editor",
                    content=f"[Content Editor]: {response.content[0].get('text')}",
                )
            ],
            "final_answer": response.content[0].get("text"),
        }

    content_builder = StateGraph(TeamState)
    content_builder.add_node("content_writer", content_writer)
    content_builder.add_node("content_editor", content_editor)

    content_builder.add_edge(START, "content_writer")
    content_builder.add_edge("content_writer", "content_editor")
    content_builder.add_edge("content_editor", END)

    return content_builder


def build_analysis_team() -> StateGraph:
    """Build the analysis department subgraph"""

    def data_analyst(state: TeamState) -> dict:
        """Provide Data Driven Analysis"""
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a data analyst. Your goal is to extract key insights and trends from the research findings provided. Provide 3-4 data driven insights"
                    )
                ),
                HumanMessage(
                    content=f"Here are the research findings to analyze:\n\n{state['messages']}"
                ),
            ]
        )

        return {
            "messages": [
                AIMessage(
                    name="data_analyst",
                    content=f"[Data Analyst]: {response.content[0].get('text')}",
                )
            ]
        }

    def strategy_advisor(state: TeamState) -> dict:
        """Provides Strategic Recommendations"""
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a strategy advisor. Provide actionable strategic recommendations based on the data analyst's insights and the original query. Be specific and practicle"
                    )
                ),
                HumanMessage(
                    content=f"Here are the research findings to analyze:\n\n{state['messages']}"
                ),
            ]
        )
        return {
            "messages": [
                AIMessage(
                    name="strategy_advisor",
                    content=f"[Strategy Advisor]: {response.content[0].get('text')}",
                )
            ],
            "final_answer": response.content[0].get("text"),
        }

    analysis_builder = StateGraph(TeamState)
    analysis_builder.add_node("data_analyst", data_analyst)
    analysis_builder.add_node("strategy_advisor", strategy_advisor)

    analysis_builder.add_edge(START, "data_analyst")
    analysis_builder.add_edge("data_analyst", "strategy_advisor")
    analysis_builder.add_edge("strategy_advisor", END)

    return analysis_builder


# ============================================
# Build Top-Level Supervisor (CEO as Parent graph)
# ============================================


def create_hierarchical_system():
    """
    Top-level supervisor that routes to departments subgraphs.
    Each department is compiled subgraph added as a single node.
    """

    # Compile depratment subgraphs
    research_team = build_research_team().compile()
    content_team = build_contetnt_team().compile()
    analysis_team = build_analysis_team().compile()

    # Create TOP-LEVELsupervisor (CEO)
    class DepartmentRoute(BaseModel):
        department: Literal["research", "content", "analysis"] = Field(
            description="Which department should handle this request"
        )
        reasoning: str = Field(
            description="Brief reasoning for the department selection"
        )

    router_llm = model.with_structured_output(DepartmentRoute)

    def ceo_supervisor(state: TeamState) -> dict:
        """
        CEO decides which department to route the request to.
        """

        response = router_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are the CEO. Decide which department should handle this request."
                    )
                ),
                HumanMessage(content=f"Here is the request:\n\n{state['messages']}"),
            ]
        )
        return {
            "messages": [
                AIMessage(
                    name="ceo",
                    content=f"[CEO]: Routing to {response.department} - {response.reasoning}",
                )
            ]
        }

    def route_to_department(state: TeamState) -> dict:
        """
        Routes the request to the appropriate department subgraph
        """

        last_ai = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.name == "ceo":
                last_ai = msg
                break

        if last_ai and "research" in last_ai.content.lower():
            return "research_team"
        elif last_ai and "content" in last_ai.content.lower():
            return "content_team"
        elif last_ai and "analysis" in last_ai.content.lower():
            return "analysis_team"
        else:
            return "research_team"

    # Build parant graph - departments are compiled subgraphs as nodes

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
            "research_team": "research_team",  # Compiled subgraph
            "content_team": "content_team",  # Complied subgraph
            "analysis_team": "analysis_team",  # Complied subgraph
        },
    )
    parent.add_edge("research_team", END)
    parent.add_edge("content_team", END)
    parent.add_edge("analysis_team", END)

    return parent.compile()


def hierarchical_routing():
    """Demo the full hierarchical system with routing"""

    system = create_hierarchical_system()

    print("=" * 80)
    print("HIERARCHICAL ROUTING SYSTEM")
    print("=" * 80)
    queries = [
        "What are the latest trends in LLM?",
        "Write a blog post about RAG?",
        "Should my startup invest in building AI features this year",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 80)
        result = system.invoke(
            {"messages": [HumanMessage(content=query)], "final_answer": ""}
        )

        # Show the CEO routing decision
        for msg in result["messages"]:
            if isinstance(msg, AIMessage):
                print("\nCEO Routing:", msg.content)

        # Print final answer
        print(f"\nFinal Answer: {result['final_answer']}")


if __name__ == "__main__":
    hierarchical_routing()
