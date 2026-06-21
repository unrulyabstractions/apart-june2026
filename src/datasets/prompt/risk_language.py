"""The languages the risk grid renders prompts in.

A tiny enum rather than bare strings so the grid axis is explicit and
serialization stays stable across the framing/instruction modules.
"""

from __future__ import annotations

from enum import Enum


class RiskLanguage(Enum):
    """Supported prompt languages, value = lowercase ISO code used as the key."""

    EN = "en"
    ES = "es"
