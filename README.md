# Multi-Agent Systems in LangGraph & LangChain

A comprehensive repository demonstrating standard design patterns for building, orchestrating, and scaling multi-agent artificial intelligence systems using **LangGraph** and **LangChain**.

---

## 📋 Overview

Multi-agent architectures allow complex tasks to be decomposed across specialized LLM-driven agents. This project implements seven core multi-agent interaction patterns:

1. **Agent Handoffs**: Dynamic context and control delegation between specialized agents.
2. **Parallel Agents & Map-Reduce**: Concurrent agent execution and synthesis of multi-perspective results.
3. **Supervisor Architecture**: Centralized supervisor model routing tasks to specialist domain agents.
4. **Tool-Calling Agents**: Binding tools directly to LLMs with execution state graphs.
5. **Tool Error Handling Agents**: Resilient execution patterns for catching and handling tool errors gracefully.
6. **Blackboard Pattern (Communication)**: Shared workspace pattern where agents iterate on drafts and critiques.
7. **Message Passing (Communication)**: Sequential pipeline communication through a shared message history.

---

## 📁 Repository Structure

```
.
├── README.md                                  # Project overview and usage documentation
├── pyproject.toml                             # Project dependencies and configuration
├── agent_handoff.py                           # Customer service triage and handoff workflow
├── parallel_agents.py                         # Parallel research and Map-Reduce summarization
├── supervisor_agent.py                        # Centralized supervisor orchestrating specialists
├── tool_calling_agent.py                      # Tool binding and execution graph agent
├── tool_calling_agent_with_error_handler.py   # Resilient tool execution with error handling
└── agent_communication/                       # Inter-agent communication design patterns
    ├── via_blackboard.py                      # Shared workspace draft/critique iterative loop
    └── via_message_passing.py                 # Multi-stage sequential message pipeline
```

---

## 🛠️ Key Architectural Patterns

### 1. Agent Handoffs ([`agent_handoff.py`](file:///home/dev1083/Shyam/Langchain_and_Langraph/multi_agent_system_using_lc_lg/multi_agent_system/agent_handoff.py))
- **Description**: Demonstrates routing user inquiries from a initial **Triage Agent** to domain specialists (**Sales**, **Support**, **Billing**).
- **Key Concepts**: `TypedDict` state transfer (`handoff_reason`, `context_summary`), structured output handoff decisions (`Pydantic`), and conditional graph branching.

### 2. Parallel Agents & Map-Reduce ([`parallel_agents.py`](file:///home/dev1083/Shyam/Langchain_and_Langraph/multi_agent_system_using_lc_lg/multi_agent_system/parallel_agents.py))
- **Description**:
  - **Parallel Research**: Runs Academic, Creative, and Technical research agents concurrently from `START` and synthesizes all results into a unified summary.
  - **Map-Reduce Summarizer**: Maps over a set of documents to generate short summaries in parallel, then reduces them into a final master summary.
- **Key Concepts**: Fan-out execution, state aggregation, and Map-Reduce decomposition.

### 3. Supervisor Architecture ([`supervisor_agent.py`](file:///home/dev1083/Shyam/Langchain_and_Langraph/multi_agent_system_using_lc_lg/multi_agent_system/supervisor_agent.py))
- **Description**: Orchestrates a team of worker specialists (**Researcher**, **Writer**, **Critic**) using a centralized **Supervisor Agent**.
- **Key Concepts**: Manager-worker pattern, dynamic state-based next-step routing, and task completion loop termination (`FINISH`).

### 4. Tool-Calling Agent ([`tool_calling_agent.py`](file:///home/dev1083/Shyam/Langchain_and_Langraph/multi_agent_system_using_lc_lg/multi_agent_system/tool_calling_agent.py))
- **Description**: Binds domain-specific tools (calculator, weather simulator, web search) to an LLM node using `langgraph.prebuilt.ToolNode`.
- **Key Concepts**: `model.bind_tools()`, tool call loop (`should_continue`), and message trace analysis (`ToolMessage`).

### 5. Tool Error Handling Agent ([`tool_calling_agent_with_error_handler.py`](file:///home/dev1083/Shyam/Langchain_and_Langraph/multi_agent_system_using_lc_lg/multi_agent_system/tool_calling_agent_with_error_handler.py))
- **Description**: Handles edge cases such as division by zero within tool executions without crashing the agent execution graph.
- **Key Concepts**: Defensive tool design, error message reporting to LLM, and recovery loops.

### 6. Communication via Blackboard ([`agent_communication/via_blackboard.py`](file:///home/dev1083/Shyam/Langchain_and_Langraph/multi_agent_system_using_lc_lg/multi_agent_system/agent_communication/via_blackboard.py))
- **Description**: Implements a shared state blackboard where a **Drafter** writes/revises content and a **Critic** evaluates and approves or provides revision feedback.
- **Key Concepts**: Iterative refinement, shared workspace state, and structured approval criteria.

### 7. Communication via Message Passing ([`agent_communication/via_message_passing.py`](file:///home/dev1083/Shyam/Langchain_and_Langraph/multi_agent_system_using_lc_lg/multi_agent_system/agent_communication/via_message_passing.py))
- **Description**: A sequential pipeline where agents (**Researcher** $\rightarrow$ **Fact Checker** $\rightarrow$ **Summarizer**) pass messages downstream through a shared message list.
- **Key Concepts**: Sequential pipeline graph, state message accumulation (`add_messages`).

---

## ⚡ Prerequisites & Setup

### Requirements
- **Python**: $\ge 3.11$
- **Package Manager**: [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip` / `venv`.
- **API Key**: Google Gemini API key (`GOOGLE_API_KEY`).

### Installation Steps

1. **Clone the repository and enter the directory**:
   ```bash
   cd multi_agent_system
   ```

2. **Set up virtual environment and dependencies**:
   Using `uv`:
   ```bash
   uv sync
   source .venv/bin/activate
   ```
   Or using standard `pip`:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

---

## 🚀 Running the Agent Workflows

Run any of the demonstration scripts directly:

```bash
# 1. Run Agent Handoff Demo
python agent_handoff.py

# 2. Run Parallel Agents & Map-Reduce Demo
python parallel_agents.py

# 3. Run Supervisor Architecture Demo
python supervisor_agent.py

# 4. Run Tool-Calling Agent Demo
python tool_calling_agent.py

# 5. Run Tool Error Handling Agent Demo
python tool_calling_agent_with_error_handler.py

# 6. Run Blackboard Communication Demo
python agent_communication/via_blackboard.py

# 7. Run Message Passing Communication Demo
python agent_communication/via_message_passing.py
```

---

## 📄 License

This repository is maintained for educational and demonstration purposes showcasing production-ready LangGraph agent design patterns.
