from langchain_core import documents
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from typing import Dict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from dotenv import load_dotenv
import asyncio

load_dotenv()

model = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash-lite")


class ParallelState(TypedDict):
    query: str
    research_result: str
    creative_result: str
    technical_result: str
    final_synthesis: str


def create_parallel_research():
    """Three research agents working in paralle."""


    def research_agent(state: ParallelState):
        """Academic or factual Research"""

        response = model.invoke([
            SystemMessage(content="""
                You are a academic Researcher
                gather information and facts
            """),
            HumanMessage(content=f"Query: {state['query']}\n Provide detailed research on this topic")
        ])
        return {"research_result": response.content[0].get('text')}

    def creative_agent(state: ParallelState):
        """Creative Research"""

        response = model.invoke([
            SystemMessage(content="""
                You are a creative researcher
                gather creative ideas and concepts
            """),
            HumanMessage(content=f"Query: {state['query']}\n Provide detailed creative research on this topic")
        ])
        return {"creative_result": response.content[0].get('text')}

    def technical_agent(state: ParallelState):
        """Technical Research"""

        response = model.invoke([
            SystemMessage(content="""
                You are a technical researcher
                gather technical information and facts
            """),
            HumanMessage(content=f"Query: {state['query']}\n Provide detailed technical research on this topic")
        ])
        return {"technical_result": response.content[0].get('text')}

    def synthesize(state: ParallelState) -> dict:
        """ Combine all result"""

        synthesis_prompt = f""" Synthesize these4 three perspectives into a comprehensive result

        RESEARCH: {state['research_result']}
        CREATIVE: {state['creative_result']}
        TECHNICAL: {state['technical_result']}

        Create unified, well-structured response.
        """

        response = model.invoke([
            SystemMessage(content="You are an expert synthesizer. Combine multiple perspective coherently"),
            HumanMessage(content=synthesis_prompt)
        ])

        return {"final_synthesis": response.content[0].get('text')}


    graph = StateGraph(ParallelState)

    graph.add_node("research_agent", research_agent)
    graph.add_node("creative_agent", creative_agent)
    graph.add_node("technical_agent", technical_agent)
    graph.add_node("synthesize", synthesize)

    graph.add_edge(START, "research_agent")
    graph.add_edge(START, "creative_agent")
    graph.add_edge(START, "technical_agent")

    # Run all agents in parallel
    graph.add_edge("research_agent", "synthesize")
    graph.add_edge("creative_agent", "synthesize")
    graph.add_edge("technical_agent", "synthesize")

    graph.add_edge("synthesize", END)

    return graph.compile()



def run_parallel_research():
    print("Starting parallel agents...")
    
    graph = create_parallel_research()
    
    query = "What are the latest AI trends and their business implications?"
    print(f"\nQuery: {query}")

    # Start the graph with the query
    response = graph.invoke({"query": query})

    print("\n" + "="*50)
    print("PARALLEL RESEARCH RESULTS")
    print("="*50)
    print(f"\nRESEARCH: {response['research_result'][:300]}")
    print(f"\nCREATIVE: {response['creative_result'][:300]}")
    print(f"\nTECHNICAL: {response['technical_result'][:300]}")
    print(f"\n" + "="*50)
    print(f"\nSYNTHESIS: {response['final_synthesis'][:300]}")



class MapReduceState(TypedDict):
    documents: list[str]
    summaries: list[str]
    final_summary: str


def create_map_reduce_summurizer():
    """ Summarize multiple documents in parallel """

    def map_summarizer(state: MapReduceState) -> dict:
        """Summaries each document (runs in paralle for each)"""

        summaries = []
        for doc in state["documents"]:
            response = model.invoke([
                SystemMessage(content="You are a helpful assistant that summarizes in 2 lines in non markdown state."),
                HumanMessage(content=f"Summarize this document:\n\n{doc}")
            ])
            summaries.append(response.content[0].get('text'))
            
        return {"summaries": summaries}
        
    def reduce_combine(state: MapReduceState) -> dict: 
        """ Combine all summaries into one """
        all_summaries = "\n\n".join([f"Summary {i+1}: {s}" for i, s in enumerate(state['summaries'])])
        response = model.invoke([
            SystemMessage(content="You are a helpful assistant that combines multiple summaries into one coherent summary in non markdown state."),
            HumanMessage(content=f"Combine these summaries:\n\n{all_summaries}")
        ])
        return {"final_summary": response.content[0].get('text')}
        


    graph = StateGraph(MapReduceState)

    graph.add_node("map", map_summarizer)
    graph.add_node("reduce", reduce_combine)

    graph.add_edge(START, "map")
    graph.add_edge("map", "reduce")
    graph.add_edge("reduce", END)

    return graph.compile()

if __name__ == "__main__":
    """ Map Reduce pattern"""

    agent = create_map_reduce_summurizer()
    documents = [
        "Python is highlevel programming language know for its simplicity and readability",
        "Java is object oriendated language known for its performance and scalability",
        "C++ is highperformance language know for its efficiency",
        "Javascript is programming language used for web development",
        "Go is programming language used for system programming"
    ]
    response = agent.invoke({"documents": documents, 
                            "summaryies":[], 
                            "final_summary":""})

    print("\n" + "="*50)
    print("MAP REDUCE RESULTS")
    print("="*50)
    print(f"\nFINAL SUMMARY: {response['final_summary']}")


    

