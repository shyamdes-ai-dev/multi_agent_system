"""

Supervisor Architecture in LangGraph
One agent coordinates multiple specialist agents

"""

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatMessagePromptTemplate
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from typing import Literal
from pydantic import BaseModel, Field
import operator
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

class SuperVisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str
    task_completed: bool
    final_response: str


def create_supervisor_system():
    """Create a supervisor with specialist agents"""
    
    model = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash-lite")
    
    class RouteDecision(BaseModel):
        next: Literal["researcher", "writer", "critic", "FINISH"] = Field(description="The next agent to call, or FINISH if task ifs complete")
        reasoning: str = Field(description="Reason for the routing decision")

    supervisor_llm = model.with_structured_output(RouteDecision)

    # Supervisor Node

    def supervisor(state:SuperVisorState) -> dict:
        system_prompt = """
            You are the Supervisor Agent for a team of specialists. Your job is to coordinate them to complete a task.

            1. researcher - Gathers information and facts
            2. writer - Creates and refines text content
            3. critic - Reviews and improves work

            Based on the conversation, decide which agent shoiuld act next.
            If the task is complete, respond with FINISH
            No markdown response

            Current conversation shows the pregress so far.
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
            "messages": [AIMessage(content=f"[SUPERVISOR] Routing to {decision.next}: {decision.reasoning}")],
        }

    # Define specialist agents (For Demo purposes, they just echo the task)
    def researcher(state: SuperVisorState) -> dict:
        prompt = ChatMessagePromptTemplate.from_messages(
            [
                ("system", "You are a research specialist. Gather facts and information relevant to the topic"),
                ("human", "Task Context: \n{context}\n\n Provide your research findings.")
            ]
        )
        task = next((m.content for m in state["messages"] if isinstance(m, HumanMessage)), "")

        response = model.invoke(prompt.format_messages(context=task))
        return {
            "messages": AIMessage(content=f"[Researcher]: {response.content[0].get('text')}")
        }
    
    def writer(state: SuperVisorState) -> dict:
        prompt = ChatMessagePromptTemplate.format_messages([
            ("system", "You are a writing specialist. Create clear and concise content based on the provided context."),
            ("human", "Previous work:\n{context}\n\nWrite a polished version of this content.")
        ])

        task = next((m.content for m in state["messages"] if isinstance(m, HumanMessage)), "")
        response = model.invoke(prompt.format_messages(context=task))
        return {
            "messages": AIMessage(content=f"[Writer]: {response.content[0].get('text')}")
        }

    
    
    def critic(state: SuperVisorState) -> dict:
        prompt = ChatMessagePromptTemplate.format_messages([
            ("system", "You are a critic. Review the following content for accuracy, clarity, and improvements."),
            ("human", "Content to review:\n{context}\n\nProvide your critique and suggestions for improvement.")
        ])

        context = "\n".join([m.content for m in state["messages"][-3:]])
        response = model.invoke(prompt.format_messages(context=context))
        return {
            "messages": AIMessage(content=f"[Critic]: {response.content[0].get('text')}")
        }

    def finalize(state: SuperVisorState) -> dict:
        #Get the last substantial response

        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and "[Writer]" in msg.content:
                content = msg.content.replace("[Writer]: ", "")
                return {
                    "final_response": content
                }
        return {
            "final_response": "Task Completed"
        }
    
    def route_to_agent(state: SuperVisorState) -> dict:
        if state.get("task_complete"):
            return "finalize"
        return state["next_agent"]

    
    graph = StateGraph(SuperVisorState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_node("critic", critic)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_to_agent, {
        "researcher": "researcher",
        "writer": "writer",
        "critic": "critic",
        "FINISH": "finalize"
    })
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("critic", "supervisor")
    graph.add_edge("finalize", END)

    return graph.compile()


def run_supervisor():
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
        response = graph.invoke({
            "messages": [HumanMessage(content=task)],
            "next_agent": "",
            "task_complete": False,
            "final_response": ""  
        })
        print(f"\nFinal Response: {response['final_response']}")
        print(f"Total Messages: {len(response['messages'])}")
        print("=" * 50)

if __name__ == "__main__":
    run_supervisor()


# run_supervisor()