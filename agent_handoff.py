"""
agent Handoffs in LangGraph
Passing control and context between agents
"""

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from typing_extensions import TypedDict, Annotated
from typing import Literal
from pydantic import BaseModel, Field
import operator
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(model="gemini-3.5-flash-lite", model_provider="google_genai")


class HandoffState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: str
    handoff_reason: str
    context_summary: str


class HandoffDecision(BaseModel):
    handoff_to: Literal["sales", "support", "billing", "stay", "end"] = Field(
        description="Which agent to hand off to"
    )
    reason: str = Field(descritpion="Reason for handoff")
    context: str = Field(description="key onctext to pass to next agent")


def create_customer_service_system():

    def triage_agent(state: HandoffState) -> HandoffState:
        """Initial triage to route customers"""
        system = """
            You are the initial contact for our customer service system.
            Your goal is to quickly understand the customer's issue and route them to the correct agent.
            
            Available agents:
            - sales: For questions about pricing, demos, or new purchases
            - support: For technical issues, bugs, or how-to questions
            - billing: For questions about invoices, payments, or subscriptions
        """

        handoff_llm = model.with_structured_output(HandoffDecision)
        messages = [SystemMessage(content=system)] + state["messages"]
        decision = handoff_llm.invoke(messages)

        if decision.handoff_to == "end":
            response = model.invoke(
                [
                    SystemMessage(
                        content="Provide a brief ,helpful response to the customer"
                    ),
                    *state["messages"],
                ]
            )
            return {
                "messages": [HumanMessage(content=f"[Triage] {response.content}")],
                "current_agent": "end",
            }

        return {
            "current_agent": decision.handoff_to,
            "handoff_reason": decision.reason,
            "context_summary": decision.context,
            "messages": [
                HumanMessage(
                    content=f"[Handoff to {decision.handoff_to}] {decision.context}"
                )
            ],
        }

    def sales_agent(state: HandoffState):
        """Sales Specialist"""

        system = f"""
            You are a sales specialist. Context from the triage: {state.get("context_summary")}
            
            Your goal is to help customer interested in purchasing our product.
            Answer sales related questions, provide demos and pricing information.
            Your are not authorized to provide technical or billing information.
        """

        response = model.invoke([SystemMessage(content=system), *state["messages"]])

        return {
            "messages": [HumanMessage(content=f"[Sales] {response.content}")],
            "current_agent": "sales_complete",
        }

    def billing_agent(state: HandoffState):
        """Billing Agents"""

        system = """
            You are a billing specialist. Context from the triage: {state.get("context_summary")}

            Your goal is to help customers with billing related questions.
            Answer billing related questions and provide billing information.
            You are not authorized to provide technical or sales information.
        """
        response = model.invoke([SystemMessage(content=system), *state["messages"]])

        return {
            "messages": [HumanMessage(content=f"[Billing] {response.content}")],
            "current_agent": "billing_complete",
        }

    def support_agent(state: HandoffState):
        """Support Agents"""

        system = """
            You are a support specialist. Context from the triage: {state.get("context_summary")}

            Your goal is to help customers with technical issues.
            Answer technical questions and provide technical information.
            You are not authorized to provide sales or billing information.
        """
        response = model.invoke([SystemMessage(content=system), *state["messages"]])

        return {
            "messages": [HumanMessage(content=f"[Support] {response.content}")],
            "current_agent": "support_complete",
        }

    def route_from_triage(state: HandoffState):
        """Router from triage to appropriate agent"""
        current_agent = state["current_agent"]
        if current_agent in ["sales", "support", "billing"]:
            return current_agent
        return "end"

    # Initialize the graph
    builder = StateGraph(HandoffState)

    # Add nodes
    builder.add_node("triage_agent", triage_agent)
    builder.add_node("sales_agent", sales_agent)
    builder.add_node("billing_agent", billing_agent)
    builder.add_node("support_agent", support_agent)

    # Add edges
    builder.add_edge(START, "triage_agent")
    builder.add_conditional_edges(
        "triage_agent",
        route_from_triage,
        {
            "sales": "sales_agent",
            "billing": "billing_agent",
            "support": "support_agent",
            "end": END,
        },
    )

    # Add edges from specialist agents to end
    builder.add_edge("sales_agent", END)
    builder.add_edge("billing_agent", END)
    builder.add_edge("support_agent", END)

    return builder.compile()


def handoff_function():
    """Demo customer service handoffs"""

    agent = create_customer_service_system()
    print("Customer Service handoff Demo:\n")

    queries = [
        "My app keeps crashing when I try to upload photos",
        "I want to upgrade to the premium plan",
        "I was charged tweice for my subscription",
        "What time do you close?",
    ]

    for query in queries:
        print(f"Customer: {query}")

        result = agent.invoke(
            {
                "messages": [HumanMessage(content=query)],
                "current_agent": "",
                "handoff_reason": "",
                "context_summary": "",
            }
        )

        for msg in result["messages"]:
            print(f"{msg.content[:150]}")

        print("-" * 50)


handoff_function()
