"""Parallel Execution & Map-Reduce Architecture Patterns in LangGraph.

This module demonstrates two parallel execution patterns:
1. Parallel Research Workflow: Invokes multiple specialized researchers (Academic,
   Creative, Technical) concurrently from START and synthesizes their perspectives.
2. Map-Reduce Summarization Workflow: Maps over a set of documents to summarize each
   individually in parallel, then reduces the summaries into a master document summary.
"""

from typing import Dict, Any
from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

# Initialize the chat model using Google GenAI provider
model = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash-lite")


class ParallelState(TypedDict):
    """State dictionary for tracking parallel research findings and final synthesis.

    Attributes:
        query: User research topic or prompt.
        research_result: Findings from the academic research agent.
        creative_result: Ideas and concepts from the creative research agent.
        technical_result: Technical information from the technical research agent.
        final_synthesis: Integrated synthesis of all three perspectives.
    """
    query: str
    research_result: str
    creative_result: str
    technical_result: str
    final_synthesis: str


def create_parallel_research():
    """Builds and compiles the parallel research StateGraph workflow.

    Returns:
        CompiledStateGraph: A compiled LangGraph workflow ready for parallel execution.
    """

    def research_agent(state: ParallelState) -> dict:
        """Academic research node for gathering factual and academic information."""
        response = model.invoke([
            SystemMessage(content="""
                You are an academic Researcher.
                Gather information, facts, and academic context.
            """),
            HumanMessage(content=f"Query: {state['query']}\nProvide detailed research on this topic.")
        ])
        content_text = response.content if isinstance(response.content, str) else str(response.content)
        return {"research_result": content_text}

    def creative_agent(state: ParallelState) -> dict:
        """Creative research node for generating novel concepts and creative ideas."""
        response = model.invoke([
            SystemMessage(content="""
                You are a creative researcher.
                Gather creative ideas, design concepts, and novel angles.
            """),
            HumanMessage(content=f"Query: {state['query']}\nProvide detailed creative research on this topic.")
        ])
        content_text = response.content if isinstance(response.content, str) else str(response.content)
        return {"creative_result": content_text}

    def technical_agent(state: ParallelState) -> dict:
        """Technical research node for gathering technical specifications and implementation details."""
        response = model.invoke([
            SystemMessage(content="""
                You are a technical researcher.
                Gather technical information, architecture details, and technical facts.
            """),
            HumanMessage(content=f"Query: {state['query']}\nProvide detailed technical research on this topic.")
        ])
        content_text = response.content if isinstance(response.content, str) else str(response.content)
        return {"technical_result": content_text}

    def synthesize(state: ParallelState) -> dict:
        """Synthesizer node combining academic, creative, and technical findings."""
        synthesis_prompt = f"""Synthesize these three perspectives into a comprehensive result:

RESEARCH: {state['research_result']}
CREATIVE: {state['creative_result']}
TECHNICAL: {state['technical_result']}

Create a unified, well-structured response.
"""

        response = model.invoke([
            SystemMessage(content="You are an expert synthesizer. Combine multiple perspectives coherently."),
            HumanMessage(content=synthesis_prompt)
        ])
        content_text = response.content if isinstance(response.content, str) else str(response.content)
        return {"final_synthesis": content_text}

    graph = StateGraph(ParallelState)

    graph.add_node("research_agent", research_agent)
    graph.add_node("creative_agent", creative_agent)
    graph.add_node("technical_agent", technical_agent)
    graph.add_node("synthesize", synthesize)

    # Fan out to all research agents simultaneously
    graph.add_edge(START, "research_agent")
    graph.add_edge(START, "creative_agent")
    graph.add_edge(START, "technical_agent")

    # Fan in all agents to the synthesis node
    graph.add_edge("research_agent", "synthesize")
    graph.add_edge("creative_agent", "synthesize")
    graph.add_edge("technical_agent", "synthesize")

    graph.add_edge("synthesize", END)

    return graph.compile()


def run_parallel_research():
    """Executes the parallel research workflow demo."""
    print("Starting parallel agents...")
    
    graph = create_parallel_research()
    
    query = "What are the latest AI trends and their business implications?"
    print(f"\nQuery: {query}")

    # Start the graph with the query
    response = graph.invoke({"query": query})

    print("\n" + "=" * 50)
    print("PARALLEL RESEARCH RESULTS")
    print("=" * 50)
    print(f"\nRESEARCH: {response['research_result'][:300]}")
    print(f"\nCREATIVE: {response['creative_result'][:300]}")
    print(f"\nTECHNICAL: {response['technical_result'][:300]}")
    print("\n" + "=" * 50)
    print(f"\nSYNTHESIS: {response['final_synthesis'][:300]}")


class MapReduceState(TypedDict):
    """State dictionary for Map-Reduce document summarization.

    Attributes:
        documents: Raw text documents to summarize.
        summaries: Individual document summaries output by the map phase.
        final_summary: Combined master summary output by the reduce phase.
    """
    documents: list[str]
    summaries: list[str]
    final_summary: str


def create_map_reduce_summarizer():
    """Builds and compiles the Map-Reduce document summarizer StateGraph workflow.

    Returns:
        CompiledStateGraph: A compiled LangGraph workflow for document summarization.
    """

    def map_summarizer(state: MapReduceState) -> dict:
        """Map step: Summarizes each document individually."""
        summaries = []
        for doc in state["documents"]:
            response = model.invoke([
                SystemMessage(content="You are a helpful assistant that summarizes in 2 lines without markdown."),
                HumanMessage(content=f"Summarize this document:\n\n{doc}")
            ])
            content_text = response.content if isinstance(response.content, str) else str(response.content)
            summaries.append(content_text)
            
        return {"summaries": summaries}
        
    def reduce_combine(state: MapReduceState) -> dict: 
        """Reduce step: Combines all individual document summaries into a final master summary."""
        all_summaries = "\n\n".join([f"Summary {i+1}: {s}" for i, s in enumerate(state['summaries'])])
        response = model.invoke([
            SystemMessage(content="You are a helpful assistant that combines multiple summaries into one coherent summary without markdown."),
            HumanMessage(content=f"Combine these summaries:\n\n{all_summaries}")
        ])
        content_text = response.content if isinstance(response.content, str) else str(response.content)
        return {"final_summary": content_text}

    graph = StateGraph(MapReduceState)

    graph.add_node("map", map_summarizer)
    graph.add_node("reduce", reduce_combine)

    graph.add_edge(START, "map")
    graph.add_edge("map", "reduce")
    graph.add_edge("reduce", END)

    return graph.compile()


if __name__ == "__main__":
    run_parallel_research()

    print("\n" + "#" * 50)
    print("RUNNING MAP REDUCE SUMMARIZER")
    print("#" * 50)

    agent = create_map_reduce_summarizer()
    documents = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Java is an object-oriented language known for its performance and scalability.",
        "C++ is a high-performance language known for its efficiency.",
        "JavaScript is a programming language used for web development.",
        "Go is a programming language used for system programming."
    ]
    response = agent.invoke({
        "documents": documents, 
        "summaries": [], 
        "final_summary": ""
    })

    print("\n" + "=" * 50)
    print("MAP REDUCE RESULTS")
    print("=" * 50)
    print(f"\nFINAL SUMMARY:\n{response['final_summary']}")


    

