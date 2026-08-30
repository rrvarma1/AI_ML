"""Typed JSON rule loading and deterministic evaluation for Agent 1."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RuleDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(pattern=r"^[A-Z]+-\d{3}$")
    field: str
    operator: Literal[
        "eq", "eq_normalized", "contains_normalized",
        "lt", "lte", "gt", "gte", "contains_any"
    ]
    threshold: Any
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    message: str


class RuleCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    rules: list[RuleDefinition]


def load_rule_catalog(path: Path) -> RuleCatalog:
    if not path.is_file():
        raise FileNotFoundError(f"Rule catalog not found: {path}")
    catalog = RuleCatalog.model_validate_json(path.read_text(encoding="utf-8"))
    ids = [rule.rule_id for rule in catalog.rules]
    if len(ids) != len(set(ids)):
        raise ValueError("rules.json contains duplicate rule_id values")
    return catalog


def _normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def evaluate_value(actual: object, operator: str, threshold: object) -> bool:
    """Evaluate one present value; missing-value routing is owned by Agent 1."""
    operations = {
        "eq": lambda a, t: a == t,
        "eq_normalized": lambda a, t: _normalize(a) == _normalize(t),
        "contains_normalized": lambda a, t: f" {_normalize(t)} " in f" {_normalize(a)} ",
        "lt": lambda a, t: float(a) < float(t),
        "lte": lambda a, t: float(a) <= float(t),
        "gt": lambda a, t: float(a) > float(t),
        "gte": lambda a, t: float(a) >= float(t),
        "contains_any": lambda a, t: any(_normalize(token) in _normalize(a) for token in t),
    }
    try:
        evaluator = operations[operator]
    except KeyError as exc:
        raise ValueError(f"Unsupported rule operator: {operator}") from exc
    return bool(evaluator(actual, threshold))
