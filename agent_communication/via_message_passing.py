from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict
from typing_extensions import Annotated
from pydantic import BaseModel, Field
from typing import Literal
import json

load_dotenv()

model = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash-lite")


#=======================================
# Pattern 1: Message Passing
# Agents communicte through a shared message list
#=======================================

class MessagePassingState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_phase: str


def create_message_passing_pipeline():
    """ Agents communicate by appending messages that others can read"""
    def researcher(state: MessagePassingState) -> dict:
        """Researchers the topic and posts finding as a message"""
        response = model.invoke(
            [
                SystemMessage(content="You are a helpful assistant that researches topics. Read the user's question and provide your findings in 2-3 sentences."),
                *state['messages']
            ]
        )
        return {"messages": [HumanMessage(content=f"[Researcher]: {response.content[0].get('text')}", name="researcher")], "current_phase": "fact_checker"}

    def fact_checker(state: MessagePassingState) -> dict:
        """Checks the facts and posts findings"""
        response = model.invoke(
            [
                SystemMessage(content="You are a helpful assistant that checks facts. Read the user's question and provide your findings in 2-3 sentences."),
                *state['messages']
            ]
        )
        return {"messages": [HumanMessage(content=f"[Fact Checker]: {response.content[0].get('text')}", name="fact_checker")], "current_phase": "summerizer"}
    
    def summarizer(state: MessagePassingState) -> dict:
        """Summarizes the findings"""
        response = model.invoke(
            [
                SystemMessage(content="You are a helpful assistant that summarizes the findings"),
                *state['messages']
            ]
        )
        return {"messages": [HumanMessage(content=f"[Summarized Report]: {response.content[0].get('text')}", name="summarizer")], "current_phase": "done"}
    
    
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
    print("=== Message Passing Demo ===")
    workflow = create_message_passing_pipeline()
    inputs = {"messages": [HumanMessage(content="What are the benefits of AI?")]}
    final_state = workflow.invoke(inputs)
    print(final_state['messages'][-1].content)

if __name__ == "__main__":
    message_passing_demo()