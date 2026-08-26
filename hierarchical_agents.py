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
        
        response = model.invoke([
            SystemMessage(content=(
                "You are a web researcher. Find key facts and data about the topic."
                "Provide 3-4 bullet points of findings. Be specific"
            )),
            HumanMessage(content=query)
        ])
        return {"messages": [AIMessage(
            name="web_researcher",
            content=f"[Web Researcher]: {response.content[0].get('text')}"
        )]}  

    def paper_reviewer(state: TeamState) -> dict:
        """
        Reviews academic papers for key insights
        """
        query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break
        response = model.invoke([
            SystemMessage(content=(
                "You are an academic paper reviewer. Analyze the paper related to the topic in 2-3 sentences only"
            )),
            HumanMessage(content=query)
        ])
        return {"messages": [AIMessage(
            name="paper_reviewer",
            content=f"[Paper Reviewer]: {response.content[0].get('text')}"
        )]}

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

        response = model.invoke([
            SystemMessage(content="""
                You are a research lead. Synthesize the web researchers and paper reviewer's findings into a cohesive research brief"
                Keep it to one short paragraph and highlight the key insights and their relevance to the original query.
            """),
            HumanMessage(content=f"Here are the research findings to synthesize:\n\n{findings}")
        ])
        return {"messages": [AIMessage(
            name="research_lead",
            content=f"[Research Lead]: {response.content[0].get('text')}"
        )], "final_answer": response.content[0].get('text')}


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


def run_research_team():
    """
    Build and run the research team
    """
    research_team = build_research_team().compile()
    result = research_team.invoke({
        "messages": [HumanMessage(content="What is the retrieval-augmented generation (RAG)?")],
        "final_answer": ""
    })
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(msg.content[:200] + "...\n\n")
    
    print(f"Research Brief: \n{result['final_answer']}")

if __name__ == "__main__":
   run_research_team()