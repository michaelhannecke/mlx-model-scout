"""MCP resource implementations: read-only context for agents and clients.

Registered in `routers/resources.py`. Resources only read; anything that changes state or calls the
Hub/LLM is a tool (`tools/`).
"""
