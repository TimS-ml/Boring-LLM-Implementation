"""
Tiny Transformer with Customizable Connections

Provides simple transformer implementations showcasing different connection strategies.
"""

from boring_llm.tiny.connections.base_connections import (
    ConnectionsTransformerWrapper,
    create_connections_transformer,
)

__all__ = [
    "ConnectionsTransformerWrapper",
    "create_connections_transformer",
]
