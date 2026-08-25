from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, AIMessage, HumanMessage, BaseMessage
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



