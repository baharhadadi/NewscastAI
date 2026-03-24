"""
services/mcp/hostify_schema.py
--------------------------------
Pydantic models defining the structural contract for a Hostify podcast episode.

These schemas serve two roles: (1) runtime validation of the LLM-generated JSON
to catch malformed drafts before they reach the TTS pipeline, and (2) JSON
Schema generation used as constrained-decoding hints in the LLM prompt.
"""

from typing import List, Dict
from pydantic import BaseModel, Field


class Section(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    script: str = Field(..., min_length=1)


class Episode(BaseModel):
    intro: str = Field(..., min_length=1)
    sections: List[Section] = Field(..., min_items=1, max_items=8)
    outro: str = Field(..., min_length=1)


def episode_json_schema() -> Dict:
    """
    Return a JSON Schema for Episode that can be used for constrained decoding.
    Compatible with Pydantic v2 (preferred) and v1 fallback.
    """
    try:
        # Pydantic v2
        return Episode.model_json_schema()
    except Exception:
        # Pydantic v1
        return Episode.schema()
