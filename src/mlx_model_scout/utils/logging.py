"""Process logging setup.

IMPORTANT: In stdio transport, stdout IS the MCP protocol channel. Anything
printed to stdout corrupts the JSON-RPC stream. All logging goes to stderr.
Never use print() in server code.
"""

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
