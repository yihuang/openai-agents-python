# Copyright

from __future__ import annotations

from typing import cast

import pytest
from openai.types.responses.response_input_item_param import FunctionCallOutput

from agents import (
    Agent,
    FunctionToolResult,
    RunConfig,
    RunContextWrapper,
    ToolCallOutputItem,
    ToolsToFinalOutputResult,
    UserError,
)
from agents._run_impl import RunImpl

from .test_responses import get_function_tool


def _make_function_tool_result(agent: Agent, output: str) -> FunctionToolResult:
    # Construct a FunctionToolResult with the given output using a simple function tool.
    tool = get_function_tool("dummy", return_value=output)
    raw_item: FunctionCallOutput = cast(
        FunctionCallOutput,
        {
            "call_id": "1",
            "output": output,
            "type": "function_call_output",
        },
    )
    # For this test we don't care about the specific RunItem subclass, only the output field
    run_item = ToolCallOutputItem(agent=agent, raw_item=raw_item, output=output)
    return FunctionToolResult(tool=tool, output=output, run_item=run_item)


@pytest.mark.asyncio
async def test_no_tool_results_returns_not_final_output() -> None:
    # If there are no tool results at all, tool_use_behavior should not produce a final output.
    agent = Agent(name="test")
    result = await RunImpl._check_for_final_output_from_tools(
        agent=agent,
        tool_results=[],
        context_wrapper=RunContextWrapper(context=None),
        config=RunConfig(),
    )
    assert result.is_final_output is False
    assert result.final_output is None


@pytest.mark.asyncio
async def test_run_llm_again_behavior() -> None:
    # With the default run_llm_again behavior, even with tools we still expect to keep running.
    agent = Agent(name="test", tool_use_behavior="run_llm_again")
    tool_results = [_make_function_tool_result(agent, "ignored")]
    result = await RunImpl._check_for_final_output_from_tools(
        agent=agent,
        tool_results=tool_results,
        context_wrapper=RunContextWrapper(context=None),
        config=RunConfig(),
    )
    assert result.is_final_output is False
    assert result.final_output is None


@pytest.mark.asyncio
async def test_stop_on_first_tool_behavior() -> None:
    # When tool_use_behavior is stop_on_first_tool, we should surface first tool output as final.
    agent = Agent(name="test", tool_use_behavior="stop_on_first_tool")
    tool_results = [
        _make_function_tool_result(agent, "first_tool_output"),
        _make_function_tool_result(agent, "ignored"),
    ]
    result = await RunImpl._check_for_final_output_from_tools(
        agent=agent,
        tool_results=tool_results,
        context_wrapper=RunContextWrapper(context=None),
        config=RunConfig(),
    )
    assert result.is_final_output is True
    assert result.final_output == "first_tool_output"


@pytest.mark.asyncio
async def test_custom_tool_use_behavior_sync() -> None:
    """If tool_use_behavior is a sync function, we should call it and propagate its return."""

    def behavior(
        context: RunContextWrapper, results: list[FunctionToolResult]
    ) -> ToolsToFinalOutputResult:
        assert len(results) == 3
        return ToolsToFinalOutputResult(is_final_output=True, final_output="custom")

    agent = Agent(name="test", tool_use_behavior=behavior)
    tool_results = [
        _make_function_tool_result(agent, "ignored1"),
        _make_function_tool_result(agent, "ignored2"),
        _make_function_tool_result(agent, "ignored3"),
    ]
    result = await RunImpl._check_for_final_output_from_tools(
        agent=agent,
        tool_results=tool_results,
        context_wrapper=RunContextWrapper(context=None),
        config=RunConfig(),
    )
    assert result.is_final_output is True
    assert result.final_output == "custom"


@pytest.mark.asyncio
async def test_custom_tool_use_behavior_async() -> None:
    """If tool_use_behavior is an async function, we should await it and propagate its return."""

    async def behavior(
        context: RunContextWrapper, results: list[FunctionToolResult]
    ) -> ToolsToFinalOutputResult:
        assert len(results) == 3
        return ToolsToFinalOutputResult(is_final_output=True, final_output="async_custom")

    agent = Agent(name="test", tool_use_behavior=behavior)
    tool_results = [
        _make_function_tool_result(agent, "ignored1"),
        _make_function_tool_result(agent, "ignored2"),
        _make_function_tool_result(agent, "ignored3"),
    ]
    result = await RunImpl._check_for_final_output_from_tools(
        agent=agent,
        tool_results=tool_results,
        context_wrapper=RunContextWrapper(context=None),
        config=RunConfig(),
    )
    assert result.is_final_output is True
    assert result.final_output == "async_custom"


@pytest.mark.asyncio
async def test_invalid_tool_use_behavior_raises() -> None:
    """If tool_use_behavior is invalid, we should raise a UserError."""
    agent = Agent(name="test")
    # Force an invalid value; mypy will complain, so ignore the type here.
    agent.tool_use_behavior = "bad_value"  # type: ignore[assignment]
    tool_results = [_make_function_tool_result(agent, "ignored")]
    with pytest.raises(UserError):
        await RunImpl._check_for_final_output_from_tools(
            agent=agent,
            tool_results=tool_results,
            context_wrapper=RunContextWrapper(context=None),
            config=RunConfig(),
        )
