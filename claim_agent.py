from dotenv import load_dotenv
from typing import Dict, List, Any, Annotated, TypedDict
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
def get_fnol_details_from_fnol_number(fnol_number: str) -> str:
    """
    Get the FNOL details from a given FNOL number
    """
    fnol_text = """John Doe (insured) reported a water loss on 02/20/2025 at 09:12 AM.
    FNOL Number: FNOL-2025-001234
    Policy Number: HO3-77889900
    Email: john.doe@example.com

    Loss occurred on 02/19/2025 around 2:35 PM.

    Insured states the supply line under the upstairs bathroom sink burst
    while no one was home. Water leaked through the floor into the living room,
    causing damage to the ceiling and flooring.

    Mitigation performed:
    - Main water valve shut off
    - Emergency plumber replaced failed supply line
    - Buckets/towels used to contain water
    - Fans run overnight

    No injuries. Ceiling sagging slightly—minor hazard.
    Insured available for inspections most mornings.

    Weather: light rain but unrelated to cause.

    Insured unsure about home warranty coverage. No prior issues reported.
    Photos not yet provided."""
    
    return fnol_text

# Tools
@tool
def send_email(email: str, subject: str, body: str) -> str:
    """
    Send an email
    """

    decision = interrupt(
        {
            "action": "send_email",
            "email": email,
            "subject": subject,
            "body": body,
        }
    )

    if isinstance(decision, dict):
        if decision.get("approved"):
            subject = decision.get("subject", subject)
            body = decision.get("body", body)
            print(f"Email sent: {email} with subject {subject} and body {body}")
            return f"Email sent: {email} with subject {subject} and body {body}"
    print(f"Email to {email} not sent")
    return f"Email to {email} not sent"


@tool
def get_policy_details(policy_number: str) -> str:
    """
    Get policy information by policy number
    """
    fake_policy = {
        "policyNumber": "HO3-77889900",
        "policyType": "HO-3 Homeowners",
        "insured": {
            "insuredId": "INS-552211",
            "firstName": "John",
            "lastName": "Doe",
            "phone": "+1-555-238-9021",
            "email": "john.doe@example.com",
            "mailingAddress": {
                "street": "1245 Westbrook Ave",
                "city": "Austin",
                "state": "TX",
                "zip": "78704"
            }
        },
        "property": {
            "propertyId": "PROP-100178",
            "locationAddress": {
                "street": "1245 Westbrook Ave",
                "city": "Austin",
                "state": "TX",
                "zip": "78704"
            },
            "type": "Single Family Home",
            "yearBuilt": 1997,
            "squareFeet": 2650,
            "roofType": "Composition Shingle",
            "numStories": 2,
            "hasBasement": False,
            "constructionType": "Frame",
            "protectionClass": 4,
            "occupancy": "Owner Occupied"
        },
        "coverage": {
            "effectiveDate": "2024-06-01",
            "expirationDate": "2025-06-01",
            "deductibles": {
                "allPeril": 2500,
                "hurricaneDeductible": None,
                "windHailDeductible": None
            },
            "limits": {
                "coverageA_dwelling": 350000,
                "coverageB_otherStructures": 35000,
                "coverageC_personalProperty": 90000,
                "coverageD_lossOfUse": 35000,
                "coverageE_personalLiability": 300000,
                "coverageF_medicalPayments": 5000
            },
            "endorsements": [
                {
                    "endorsementId": "END-001",
                    "name": "Water Backup and Sump Overflow",
                    "limit": 5000
                },
                {
                    "endorsementId": "END-002",
                    "name": "Special Personal Property Coverage",
                    "limit": None
                }
            ]
        },
        "mortgagee": {
            "name": "First National Bank of Texas",
            "loanNumber": "LN-44332211",
            "address": {
                "street": "3000 Bank Plaza Blvd",
                "city": "Austin",
                "state": "TX",
                "zip": "78705"
            },
            "isPrimaryMortgagee": True
        },
        "underwriting": {
            "riskScore": 72,
            "priorClaims": 0,
            "inspectionStatus": "Completed",
            "lastInspectionDate": "2024-05-20"
        },
        "billing": {
            "paymentPlan": "Monthly",
            "annualPremium": 1675.00,
            "paymentsMade": 8,
            "nextPaymentDue": "2025-03-01"
        },
        "agentsAndContacts": {
            "agentName": "Rebecca Carson",
            "agencyName": "Carson Insurance Group",
            "agentPhone": "+1-555-782-1900",
            "agentEmail": "rebecca.carson@cigagency.com"
        },
        "status": "Active"
    }
    return fake_policy

@tool
def create_claim(
    name: str,
    policy_number: str,
    loss_date: str,
    loss_description: str,
    phone: str,
    email: str,
    relation_to_insured: str,
    property_id: str,
    loss_cause: str,
) -> str:
    """
    Create a claim from a given FNOL text

    Args:
        name: Name of claimant
        policy_number: Policy number associated with the claim
        loss_date: Date of loss
        loss_description: Description of the loss/damage
        phone: Claimant's phone number
        email: Claimant's email address
        relation_to_insured: Relationship to the insured (e.g., self, spouse, tenant)
        property_id: Identifier for the property/location
        loss_cause: Cause of the reported loss

    Returns:
        claim_id: str
    """
    fake_claim = {
        "claimId": "CLM-2025-001234",
        "policyNumber": "HO3-77889900",
        "insured": {
            "firstName": "John",
            "lastName": "Doe",
            "phone": "+1-555-238-9021",
            "email": "john.doe@example.com",
            "address": {
                "street": "1245 Westbrook Ave",
                "city": "Austin",
                "state": "TX",
                "zip": "78704"
            }
        },
        "property": {
            "propertyId": "PROP-100178",
            "address": {
                "street": "1245 Westbrook Ave",
                "city": "Austin",
                "state": "TX",
                "zip": "78704"
            },
            "type": "Single Family Home",
            "yearBuilt": 1997,
            "squareFeet": 2650,
            "roofType": "Composition Shingle",
            "numStories": 2,
            "hasBasement": False
        },
        "lossInfo": {
            "dateOfLoss": "2025-02-19T14:35:00",
            "reportedDate": "2025-02-20T09:12:00",
            "causeOfLoss": "Water Damage",
            "description": (
                "Insured reported water leaking from upstairs bathroom resulting "
                "in ceiling damage to living room."
            ),
            "mitigationSteps": [
                "Shut off main water valve",
                "Called emergency plumber",
                "Placed buckets to control leaking"
            ]
        },
        "coverage": {
            "coverageA_dwellingLimit": 350000,
            "coverageC_contentsLimit": 90000,
            "deductible": 2500,
            "policyEffective": "2024-06-01",
            "policyExpiration": "2025-06-01"
        },
        "inspection": {
            "inspectionScheduled": True,
            "inspectionDate": "2025-02-22T10:00:00",
            "inspectorName": "Sarah Mitchell"
        },
        "damageAssessment": {
            "roomsAffected": ["Living Room", "Upstairs Bathroom"],
            "estimatedRepairCost": 12840.75,
            "depreciationApplied": 950.00,
            "rcv": 12840.75,
            "acv": 11890.75
        },
        "payments": {
            "initialPaymentIssued": False,
            "amountIssued": 0.0
        },
        "status": "In Review"
    }
    return fake_claim


@tool
def get_previous_claims_by_policy_number(policy_number: str) -> str:
    """
    Get previous claims by policy number
    """
    return "There are no previous claims for this policy"

# Agent

# State definition - extends AgentState for future custom state fields
class State(AgentState):
    pass
    # documents: list[str]  # Example of future custom field


# Tool and LLM definition
def get_tools():
    return [
        get_fnol_details_from_fnol_number,
        create_claim,
        get_policy_details,
        send_email,
        get_previous_claims_by_policy_number,
    ]


tools = get_tools()
llm = ChatOpenAI(model="gpt-5-nano", temperature=0.7)

# System Prompt
system_prompt = """
        You are a claim setup assistant.

        Inputs you will be given (in the user message):
        - FNOL NUMBER
        - Policy Number
        - Brief description of the loss

        Your tasks:

        1. Call `get_fnol_details_from_fnol_number` to retrieve FNOL details.
        2. Call `get_policy_details` to retrieve policy & coverage info.
        3. Call `get_previous_claims_by_policy_number` to retrieve previous claims for the policy.
        4. Based on policy info and loss description, decide reasonable:
        - line_of_business
        - loss_party
        - initial_reserves (rough)
        - claim_type / cause_of_loss
        5. Call `create_claim` with:
        - fnol_id
        - policy (as returned by get_policy_details)
        - structured claim metadata you inferred
        6. Call `send_email` to send an email to the insured with the claim details.

        Use tools; do NOT fabricate IDs.
        Return only the final tool output (claim_id) as the answer.
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
