from __future__ import annotations as _annotations

import asyncio
import random
import uuid

from pydantic import BaseModel

from agents import (
    Agent,
    HandoffOutputItem,
    ItemHelpers,
    MessageOutputItem,
    RunContextWrapper,
    Runner,
    ToolCallItem,
    ToolCallOutputItem,
    TResponseInputItem,
    function_tool,
    handoff,
    trace,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

### CONTEXT


class UberAgentContext(BaseModel):
    """Shared context for our Uber agents."""
    ride_id: str | None = None
    ride_info: str | None = None
    uberone: bool | None = None
    language: str = "English"  # New field for language preference
    uberone = True


### TOOLS

async def _fetch_rider_information(
    context: RunContextWrapper[UberAgentContext], ride_id: str
) -> str:
    """
    Fetch details about the ride for a given ride ID. In a real application this would look up
    ride details from a database or service. Here we just synthesize some data and store it in
    context for other tools/agents to use.

    Args:
        context: run context containing shared state
        ride_id: the ride identifier
    """
    context.context.ride_id = ride_id
    # Fake ride info
    info = (
        f"Ride {ride_id}: pickup at 123 Main St, dropoff at 456 Park Ave, "
        f"charged $20 cleaning fee."
    )
    context.context.ride_info = info
    return info

@function_tool
async def grant_concession(context: RunContextWrapper[UberAgentContext]) -> str:
    """
    Mocked function to grant a $20 concession to the user.
    """
    return "A $20 concession has been applied to your account"

fetch_rider_information = function_tool(_fetch_rider_information)

### HOOKS / HANDOFFS
async def on_cleaning_fee_handoff(context: RunContextWrapper[UberAgentContext]) -> list[dict[str, str]]:
    """
    Lifecycle hook invoked when triage hands off to the cleaning fee agent.
    We immediately fetch ride info so the cleaning_fee_instructions can include it.
    Now, we also return a log event for the tool call.
    """
    # Generate a fake ride ID if none set
    ride_id = context.context.ride_id or f"RIDE-{random.randint(1000, 9999)}"
    # Call the tool to fetch ride info
    result = await _fetch_rider_information(context, ride_id)
    return result


### AGENTS
def cleaning_fee_instructions(context: RunContextWrapper[UberAgentContext], _agent: Agent[UberAgentContext]) -> str:
    # Dynamically include any fetched ride info in the instructions so the agent can see it in the system prompt.
    ride_info = context.context.ride_info or "(ride info not available)"
    uberone = context.context.uberone
    language = context.context.language

    # Base instructions
    instructions = (
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        f"Ride information: {ride_info}\n"
    )

    # Add language-specific instructions
    if language.lower() != "english":
        instructions += "Speak only to the user in Spanish.\n"

    # Add premium-specific instructions
    if uberone:
        instructions += (
            "You are an agent that helps Uber One riders who have been charged a cleaning fee on their trip.\n"
            "# Routine\n"
            "1. Call the grant concession tool to issue a refund.\n"
            "2. Call end_conversation to end the conversation if user is happy.\n"
        )
    else:
        instructions += (
            "You are an agent that helps riders who have been charged a cleaning fee on their trip.\n"
            "# Routine\n"
            "1. Confirm the ride details (date, pickup/dropoff) with the user.\n"
            "2. Explain why the cleaning fee was assessed.\n"
            "3. If the rider disputes the fee, put in a ticket of their complaint.\n"
            "4. Call end_conversation to end the conversation.\n"
        )

    return instructions

@function_tool
async def end_conversation(context: RunContextWrapper[UberAgentContext]) -> str:
    """
    Function to end the conversation.
    """
    return "Conversation ended"


rider_cleaning_fee_agent = Agent[UberAgentContext](
    name="Rider Cleaning Fee Agent",
    handoff_description="Specialist agent for rider cleaning fee disputes.",
    instructions=cleaning_fee_instructions,
    tools=[grant_concession, end_conversation],
)

bill_shock_agent = Agent[UberAgentContext](
    name="Bill Shock Agent",
    handoff_description="Specialist agent for billing surprises.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are an agent that helps riders who are surprised by a large bill.
    # Routine
    1. Ask clarifying questions to understand which charge is unexpected.
    2. Check ride details if needed using available tools.
    3. Explain the charges to the rider or escalate to support.
    4. If the issue is unrelated, transfer back to triage.""",
    tools=[fetch_rider_information],
)

triage_agent = Agent[UberAgentContext](
    name="Triage Agent",
    handoff_description="Routes incoming rider issues to appropriate specialist agents.",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} "
        "You are a helpful triage agent. You can use your tools to delegate questions to other appropriate agents."
    ),
    handoffs=[
        handoff(agent=rider_cleaning_fee_agent, on_handoff=on_cleaning_fee_handoff),
        bill_shock_agent,
    ],
)

rider_cleaning_fee_agent.handoffs.append(triage_agent)
bill_shock_agent.handoffs.append(triage_agent)


### RUN


async def main():
    current_agent: Agent[UberAgentContext] = triage_agent
    input_items: list[TResponseInputItem] = []
    context = UberAgentContext()

    # Each interaction could be an API request; here we just simulate a chat loop.
    conversation_id = uuid.uuid4().hex[:16]

    while True:
        user_input = input("Enter your message: ")
        with trace("Uber agents", group_id=conversation_id):
            input_items.append({"content": user_input, "role": "user"})
            result = await Runner.run(current_agent, input_items, context=context)

            for new_item in result.new_items:
                agent_name = new_item.agent.name
                if isinstance(new_item, MessageOutputItem):
                    print(f"{agent_name}: {ItemHelpers.text_message_output(new_item)}")
                elif isinstance(new_item, HandoffOutputItem):
                    print(
                        f"Handed off from {new_item.source_agent.name} to {new_item.target_agent.name}"
                    )
                elif isinstance(new_item, ToolCallItem):
                    print(f"{agent_name}: Calling a tool")
                elif isinstance(new_item, ToolCallOutputItem):
                    print(f"{agent_name}: Tool call output: {new_item.output}")
                else:
                    print(f"{agent_name}: Skipping item: {new_item.__class__.__name__}")
            input_items = result.to_input_list()
            current_agent = result.last_agent


if __name__ == "__main__":
    asyncio.run(main())
