"""Tests for the final-answer format and its parser."""

import json

import pytest

from causal_agent.agent.answer import ANSWER_FORMAT, FinalAnswer, parse_final_answer
from causal_agent.tools.estimators import EstimatorName


def _answer_json(estimate, estimator, adjustment_set) -> str:
    return json.dumps(
        {"estimate": estimate, "estimator": estimator, "adjustment_set": adjustment_set}
    )


def test_format_instruction_names_every_estimator() -> None:
    for name in EstimatorName:
        assert f'"{name.value}"' in ANSWER_FORMAT
    assert '"estimate"' in ANSWER_FORMAT
    assert '"adjustment_set"' in ANSWER_FORMAT


def test_plain_json_answer_is_parsed() -> None:
    text = _answer_json(1.25, "linear_regression", ["x1", "x3"])
    assert parse_final_answer(text) == FinalAnswer(
        estimate=1.25,
        estimator=EstimatorName.LINEAR_REGRESSION,
        adjustment_set=("x1", "x3"),
    )


def test_answer_inside_prose_and_a_code_fence_is_parsed() -> None:
    text = (
        "The confounders are x1 and x3, so I adjusted for them.\n\n"
        "```json\n"
        + _answer_json(-0.5, "doubly_robust", ["x1", "x3"])
        + "\n```\nThat is my final answer."
    )
    answer = parse_final_answer(text)
    assert answer.estimate == -0.5
    assert answer.estimator is EstimatorName.DOUBLY_ROBUST


def test_the_last_usable_object_wins() -> None:
    text = (
        "First try: "
        + _answer_json(9.0, "difference_in_means", [])
        + "\nFinal: "
        + _answer_json(1.0, "linear_regression", ["x2"])
    )
    assert parse_final_answer(text).estimate == 1.0


def test_integer_estimates_are_accepted() -> None:
    answer = parse_final_answer(_answer_json(2, "doubly_robust", []))
    assert answer.estimate == 2.0


def test_unknown_estimator_becomes_none() -> None:
    answer = parse_final_answer(_answer_json(1.0, "my own guess", ["x1"]))
    assert answer.estimator is None
    assert answer.adjustment_set == ("x1",)


def test_missing_method_fields_are_tolerated() -> None:
    answer = parse_final_answer('{"estimate": 0.3}')
    assert answer == FinalAnswer(estimate=0.3, estimator=None, adjustment_set=())


def test_difference_in_means_always_has_the_empty_adjustment_set() -> None:
    answer = parse_final_answer(_answer_json(1.0, "difference_in_means", ["x1"]))
    assert answer.adjustment_set == ()


def test_duplicate_columns_are_removed() -> None:
    answer = parse_final_answer(
        _answer_json(1.0, "linear_regression", ["x1", "x1", "x2"])
    )
    assert answer.adjustment_set == ("x1", "x2")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "I could not determine the effect.",
        '{"estimator": "linear_regression"}',
        '{"estimate": "about one"}',
        '{"estimate": null}',
        '{"estimate": true}',
        '{"estimate": NaN}',
        "{estimate: 1.0}",
        "[1.0]",
    ],
)
def test_unusable_text_gives_none(text) -> None:
    assert parse_final_answer(text) is None


def test_malformed_adjustment_set_is_treated_as_empty() -> None:
    answer = parse_final_answer(_answer_json(1.0, "linear_regression", "x1"))
    assert answer.adjustment_set == ()
