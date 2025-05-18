# maltopic package initialization

"""
MALTopic: A Multi-Agent LLM Topic Modeling Library

This package provides a framework for topic modeling using multiple LLM agents.
Users can initialize the library with their API key and model name, and choose
between different LLM providers (OpenAI, Google Gemini, Llama) for enhanced topic
modeling capabilities.
"""

from .core import MALTopic

__all__ = [
    "MALTopic",
]
