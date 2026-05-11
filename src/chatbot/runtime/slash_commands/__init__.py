"""Slash command registry. Commands resolve before the model sees the message."""

from .registry import SlashCommandRegistry

__all__ = ["SlashCommandRegistry"]
