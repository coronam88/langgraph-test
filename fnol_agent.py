from dotenv import load_dotenv
from typing import Dict, List, Any, Annotated, TypedDict, Optional
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain.agents import create_agent, AgentState
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable
from langgraph.types import interrupt, Command

from uuid import uuid4
load_dotenv()


# Tools
@tool
def create_fnol(
    name: str,
    policy_number: str,
    loss_date: str,
    loss_description: str,
    loss_cause: str,
    location_of_loss: str,
    phone: Optional[str] = None,
    email: Optional[str] = None,
) -> str:
    """
    Create a FNOL from a given raw email text
    """
    return f"FNOL created with number: FNOL-2025-001234"


# Agent

# State definition - extends AgentState for future custom state fields
class State(AgentState):
    pass
    # documents: list[str]  # Example of future custom field


def get_tools():
    return [create_fnol]


# Tool and LLM definition
tools = get_tools()
llm = ChatOpenAI(model="gpt-5-nano", temperature=0.7)

# System Prompt
system_prompt = """
You are a claims intake assistant for a P&C insurer.

You receive RAW email text from insureds or agents about a new loss.
Your job:

1. Carefully extract all FNOL fields:
   - insured_name
   - policy_number (if mentioned; otherwise leave null)
   - loss_date and time
   - location of loss
   - loss_cause
   - brief_loss_description
   - contact_info (phone, email)
2. Call the `create_fnol` tool EXACTLY ONCE with a structured JSON payload.
3. Do NOT ask the user any questions; this is a backend process.
Return only what the tool returns as the final answer.

"""

# Memory/Checkpointer for conversation persistence
memory = MemorySaver()

# Create agent using create_agent function
# Note: create_agent returns a compiled graph
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
    state_schema=State,
    checkpointer=memory
)

# Initial call to the agent
@traceable(run_type="llm", name="call_agent")
def call_agent(msg: str, config: Dict) -> Dict:
    message = (
        msg
        if isinstance(msg, (HumanMessage, AIMessage, ToolMessage, SystemMessage))
        else HumanMessage(content=msg)
    )
    return agent.invoke({"messages": [message]}, config=config)

# Resume the agent after human-in-the-loop input
@traceable(run_type="llm", name="resume_agent")
def resume_agent(decision: str, config: Dict) -> Dict:
    """Resume an interrupted agent execution with a decision."""
    print(f"[AGENT] Resuming with human decision: {decision}")
    result = agent.invoke(Command(resume=decision), config=config)
    print(f"[AGENT] Resume completed")
    return result

