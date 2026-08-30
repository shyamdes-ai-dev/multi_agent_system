from langchain_core.prompts import AIMessagePromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, BaseMessage, AIMessage, HumanMessage
from typing import Literal
from pydantic import BaseModel, Field
import operator
from typing_extensions import TypedDict, Annotated
from dotenv import load_dotenv
import json

load_dotenv()
model = init_chat_model(model_provider="google_genai", model="gemini-2.5-flash-lite", temperature=0)
creative_llm = init_chat_model(model_provider="google_genai", model="gemini-2.5-flash-lite", temperature=0.7)


# ======================================
# State Schema
# ======================================

class ResearchState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    topic: str
    search_queries: list[str]
    findings: Annotated[list[str], operator.add]
    analysis: str
    report: str
    quality_score: float
    quality_feedback: str
    iteration: int


# State for individual search tasks (Used with Send API)
class SearchTaskState(TypedDict):
    search_query: str
    findings: Annotated[list[dict], operator.add]


# ======================================
# Node: Supervisor - Plans the research
# ======================================

def supervisor(state: ResearchState) -> dict:
    """ Plans research by generating targeted search queries."""

    response = model.invoke([
        SystemMessage(content="You are a research supervisor. Given a topic, generate exactly 3 specific search queries that will cover different angles of the topic. Return ONLY a JSON array of strings. No markdwon formatting"),
        HumanMessage(content=f"Research topic: {state['topic']}")
    ])

    try:
        queries = json.loads(response.content[0].get("text"))
    except json.JSONDecodeError:
        # fallback: Split by newlines
        queries = [
            f"{state['topic']} overview",
            f"{state['topic']} latest developments",
            f"{state['topic']} practical applications"
        ]

    return {
        "search_queries": queries[:3],
        "messages" : [
            AIMessage(content=f"[SUPERVISOR]: Planned {len(queries)} search queries: {queries}", name="supervisor")
        ]
    }