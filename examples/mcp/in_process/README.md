# MCP In-Process Example

This example uses an in-process fastmcp server in [server.py](server.py).

Run the example via:

```
uv run python -m examples.mcp.in_process.main
```

## Details

The example uses the `MCPServerSse` class from `agents.mcp`. The server runs in a sub-process at `https://localhost:8000/sse`.

