"""The feedback shape (handoff D13): a list of items, one per detected fault."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

TEXT_FIELDS = ("what", "why_it_matters", "cue", "drill")


class FeedbackItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fault_id: str
    what: str = Field(min_length=1)
    why_it_matters: str = Field(min_length=1)
    cue: str = Field(min_length=1)
    drill: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)


class Feedback(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[FeedbackItem]


# Written out (not Feedback.model_json_schema()) so it stays inside the JSON Schema subset
# Gemini's structured output accepts: no $defs/$ref, no additionalProperties.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fault_id": {"type": "string"},
                    **{k: {"type": "string"} for k in TEXT_FIELDS},
                    "citations": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["fault_id", *TEXT_FIELDS, "citations"],
            },
        }
    },
    "required": ["items"],
}
