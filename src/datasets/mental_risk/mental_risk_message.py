"""A single message posted by a MentalRiskES subject."""

from __future__ import annotations

from dataclasses import dataclass

from src.common import BaseSchema


@dataclass
class MentalRiskMessage(BaseSchema):
    """One chat/forum message from a subject's timeline.

    Mirrors the raw JSON record `{id_message, message, date}`. `id_message` is
    the corpus-assigned ordering key used to reconstruct the conversation.
    """

    id_message: int
    message: str
    date: str = ""
