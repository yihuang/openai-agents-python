# A simple FastAPI server exposing the uber_agents example as a REST API,
# plus serving a minimal React frontend.
from __future__ import annotations

import uuid
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

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
)

from examples.uber_agents.main import UberAgentContext, triage_agent


class ConversationState:
    """Tracks conversation state for a session."""
    def __init__(self) -> None:
        self.current_agent: Agent[UberAgentContext] = triage_agent
        self.input_items: list[TResponseInputItem] = []
        self.context: UberAgentContext = UberAgentContext()


app = Starlette()

# In-memory conversation store.
conversations: dict[str, ConversationState] = {}


async def init_conversation(request: Request) -> JSONResponse:
    """Create a new conversation and return its id."""
    conv_id = uuid.uuid4().hex[:16]
    conversations[conv_id] = ConversationState()
    return JSONResponse({"conversation_id": conv_id})


async def chat(request: Request) -> JSONResponse:
    data = await request.json()
    conv_id = data.get("conversation_id")
    message = data.get("message")
    if not conv_id or conv_id not in conversations:
        return JSONResponse({"detail": "Conversation not found"}, status_code=404)
    conv = conversations[conv_id]
    conv.input_items.append({"content": message, "role": "user"})

    result = await Runner.run(conv.current_agent, conv.input_items, context=conv.context)

    agent_messages: list[dict[str, str]] = []
    logs: list[dict[str, str]] = []
    # Capture logs returned by on_handoff hooks, if any

    for new_item in result.new_items:
        agent_name = new_item.agent.name
        if isinstance(new_item, MessageOutputItem):
            agent_messages.append({
                "role": agent_name,
                "content": ItemHelpers.text_message_output(new_item)
            })
        if isinstance(new_item, HandoffOutputItem):
            print(new_item)
            logs.append({
                "type": "handoff",
                "description": f"handed off to {new_item.target_agent.name}",
            })
        elif isinstance(new_item, ToolCallItem):
            tool_name = getattr(new_item.raw_item, "name", None)
            logs.append({"type": "toolcall", "description": f"calling tool {tool_name}"})
        elif isinstance(new_item, ToolCallOutputItem):
            logs.append({"type": "toolresult", "description": f"tool result: {new_item.output}"})

    conv.input_items = result.to_input_list()
    conv.current_agent = result.last_agent

    # --- Include the current agent's name and instructions (system prompt) ---
    # Some instructions are a callable (like cleaning_fee_instructions). If so, call it:
    if callable(conv.current_agent.instructions):
        system_prompt = conv.current_agent.instructions(RunContextWrapper(conv.context), conv.current_agent)
    else:
        system_prompt = conv.current_agent.instructions

    return JSONResponse({
        "agent_messages": agent_messages,
        "logs": logs,
        "context": conv.context.dict(),
        "current_agent_name": conv.current_agent.name,
        "system_prompt": system_prompt,
    })

# Serve the static frontend. We keep everything in a single HTML file for simplicity.
FRONTEND_PATH = Path(__file__).parent / "frontend" / "index.html"


async def index(request: Request) -> HTMLResponse:
    """Return the frontend HTML page."""
    if not FRONTEND_PATH.exists():
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
    return HTMLResponse(FRONTEND_PATH.read_text(encoding="utf-8"))



async def update_uberone(request: Request) -> JSONResponse:
    """
    Update the 'uberone' status in the conversation context.
    """
    data = await request.json()
    conv_id = data.get("conversation_id")
    new_uberone_value = data.get("uberone")

    if not conv_id or conv_id not in conversations:
        return JSONResponse({"detail": "Conversation not found"}, status_code=404)

    conv = conversations[conv_id]

    # new_premium_value expected to be a boolean (true/false)
    conv.context.uberone = bool(new_uberone_value)

    return JSONResponse({
        "context": conv.context.dict(),
        "uberone": conv.context.uberone,
    })


# Build the app routes.
app.routes.extend(
    [
        Route("/", index),
        Route("/init", init_conversation, methods=["POST"]),
        Route("/chat", chat, methods=["POST"]),
        Route("/update_uberone", update_uberone, methods=["POST"]),
    ]
)
