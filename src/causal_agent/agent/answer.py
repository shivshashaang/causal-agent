"""The final-answer format shared by the agent and the no-tools baseline.

An approach ends with a JSON object naming its estimate and the method behind
it. Parsing is lenient about surrounding prose and code fences, and strict
about the estimate being a finite number.
"""

import json
import math
from typing import Any

from pydantic import BaseModel, ConfigDict

from causal_agent.tools.estimators import EstimatorName

ANSWER_FORMAT = (
    "End your final message with a JSON object in exactly this form:\n"
    '{"estimate": <number>, "estimator": "<name>", '
    '"adjustment_set": [<column names>]}\n'
    "estimate is your best estimate of the average treatment effect. estimator "
    "is the method behind that estimate: one of "
    + ", ".join(f'"{name.value}"' for name in EstimatorName)
    + ". adjustment_set lists the columns that estimate adjusted for; use [] for "
    "none."
)


class FinalAnswer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    estimate: float
    estimator: EstimatorName | None
    """None when the answer names no recognized estimator."""

    adjustment_set: tuple[str, ...]


def _json_objects(text: str) -> list[dict[str, Any]]:
    """Every JSON object that can be decoded starting at a '{' in the text."""
    decoder = json.JSONDecoder()
    found = []
    for position, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            found.append(value)
    return found


def parse_final_answer(text: str) -> FinalAnswer | None:
    """Read the last JSON object with a usable `estimate`, or None."""
    for candidate in reversed(_json_objects(text)):
        estimate = candidate.get("estimate")
        if isinstance(estimate, bool) or not isinstance(estimate, int | float):
            continue
        if not math.isfinite(estimate):
            continue

        raw_estimator = candidate.get("estimator")
        estimator = (
            EstimatorName(raw_estimator)
            if raw_estimator in {name.value for name in EstimatorName}
            else None
        )
        raw_set = candidate.get("adjustment_set") or []
        if not isinstance(raw_set, list) or not all(
            isinstance(name, str) for name in raw_set
        ):
            raw_set = []
        if estimator is EstimatorName.DIFFERENCE_IN_MEANS:
            # Difference in means always uses the empty adjustment set.
            raw_set = []
        return FinalAnswer(
            estimate=float(estimate),
            estimator=estimator,
            adjustment_set=tuple(dict.fromkeys(raw_set)),
        )
    return None
