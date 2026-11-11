"""The varied test set.

``REGEX_TASKS`` drive Phase 1 (finite-automaton constraints: dates, enums,
fixed formats). ``JSON_SCHEMAS`` drive the Phase 0 baseline and Phase 2
constrained JSON generation, spanning flat objects, nested objects, arrays,
enums, and optional fields — the complexity gradient the guide calls for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RegexTask:
    name: str
    pattern: str
    instruction: str


@dataclass
class JsonTask:
    name: str
    schema: dict
    instruction: str


# --------------------------------------------------------------------------- #
# Phase 1: regex / finite-automaton tasks
# --------------------------------------------------------------------------- #
REGEX_TASKS: List[RegexTask] = [
    RegexTask(
        name="iso_date",
        pattern=r"\d{4}-\d{2}-\d{2}",
        instruction="Give the date 6 November 2026 in ISO format YYYY-MM-DD. "
        "Reply with only the date.",
    ),
    RegexTask(
        name="enum_color",
        pattern=r"(red|green|blue)",
        instruction="Pick a primary additive color from red, green, or blue. "
        "Reply with only the single lowercase word.",
    ),
    RegexTask(
        name="us_phone",
        pattern=r"\(\d{3}\) \d{3}-\d{4}",
        instruction="Produce a US phone number in the format (123) 456-7890. "
        "Reply with only the number.",
    ),
    RegexTask(
        name="semver",
        pattern=r"\d+\.\d+\.\d+",
        instruction="Give a semantic version number like 2.5.1. Reply with only "
        "the version.",
    ),
    RegexTask(
        name="hex_color",
        pattern=r"#[0-9a-fA-F]{6}",
        instruction="Give a hex color code for pure red, like #RRGGBB. Reply "
        "with only the code.",
    ),
    RegexTask(
        name="yes_no",
        pattern=r"(yes|no)",
        instruction="Is 17 a prime number? Reply with only 'yes' or 'no'.",
    ),
]


# --------------------------------------------------------------------------- #
# Phase 0 / Phase 2: JSON-schema tasks, increasing in complexity
# --------------------------------------------------------------------------- #
JSON_SCHEMAS: List[JsonTask] = [
    JsonTask(
        name="flat_object",
        schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
            "additionalProperties": False,
        },
        instruction="Return JSON for a person named Ada who is 36, with keys "
        "'name' (string) and 'age' (integer).",
    ),
    JsonTask(
        name="enum_field",
        schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["active", "inactive", "pending"]},
                "id": {"type": "integer"},
            },
            "required": ["status", "id"],
            "additionalProperties": False,
        },
        instruction="Return JSON with 'status' (one of active, inactive, "
        "pending) and 'id' (integer). Use status active and id 7.",
    ),
    JsonTask(
        name="array_of_strings",
        schema={
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["tags"],
            "additionalProperties": False,
        },
        instruction="Return JSON with a 'tags' array of three string tags for a "
        "cooking blog.",
    ),
    JsonTask(
        name="optional_field",
        schema={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "subtitle": {"type": "string"},
            },
            "required": ["title"],
            "additionalProperties": False,
        },
        instruction="Return JSON with a required 'title' (string) and an "
        "optional 'subtitle' (string) for a book called Dune.",
    ),
    JsonTask(
        name="nested_object",
        schema={
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "active": {"type": "boolean"},
                    },
                    "required": ["name", "active"],
                    "additionalProperties": False,
                },
                "score": {"type": "number"},
            },
            "required": ["user", "score"],
            "additionalProperties": False,
        },
        instruction="Return JSON with a nested 'user' object (keys 'name' string, "
        "'active' boolean) and a top-level 'score' number. Use name Grace, "
        "active true, score 9.5.",
    ),
    JsonTask(
        name="array_of_objects",
        schema={
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sku": {"type": "string"},
                            "qty": {"type": "integer"},
                        },
                        "required": ["sku", "qty"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["items"],
            "additionalProperties": False,
        },
        instruction="Return JSON with an 'items' array of two objects, each with "
        "'sku' (string) and 'qty' (integer).",
    ),
]
