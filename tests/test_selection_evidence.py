from __future__ import annotations

from src.research.selection_evidence import paired_logloss_selection_evidence


def _rows(values):
    return [
        {"fold": float(i), "logloss": float(value)}
        for i, value in enumerate(values)
    ]


def test_supported_selection_requires_effect_and_adjusted_ci():
    evidence = paired_logloss_selection_evidence(
        "challenger",
        {
            "challenger": {"logloss": 0.55, "logloss_std": 0.01, "folds": 8},
            "champion": {"logloss": 0.60, "logloss_std": 0.02, "folds": 8},
        },
        {
            "challenger": _rows([0.54, 0.55, 0.56, 0.55, 0.54, 0.55, 0.56, 0.55]),
            "champion": _rows([0.60, 0.61, 0.60, 0.62, 0.61, 0.60, 0.61, 0.60]),
        },
        trial_count=2,
        min_folds=5,
        min_relative_improvement=0.03,
    )
    assert evidence["status"] == "SUPPORTED"
    assert evidence["eligible_for_freeze"] is True
    assert evidence["relative_logloss_improvement"] >= 0.03
    assert evidence["ci_lower"] >= evidence["required_absolute_improvement"]


def test_selection_is_rejected_when_effect_is_small():
    evidence = paired_logloss_selection_evidence(
        "challenger",
        {
            "challenger": {"logloss": 0.595, "logloss_std": 0.01, "folds": 8},
            "champion": {"logloss": 0.60, "logloss_std": 0.02, "folds": 8},
        },
        {
            "challenger": _rows([0.595] * 8),
            "champion": _rows([0.600] * 8),
        },
        trial_count=2,
        min_folds=5,
        min_relative_improvement=0.03,
    )
    assert evidence["status"] == "NOT_SIGNIFICANT"
    assert evidence["eligible_for_freeze"] is False


def test_selection_is_fail_closed_with_too_few_common_folds():
    evidence = paired_logloss_selection_evidence(
        "challenger",
        {
            "challenger": {"logloss": 0.55, "folds": 4},
            "champion": {"logloss": 0.60, "folds": 4},
        },
        {
            "challenger": _rows([0.55, 0.55, 0.55, 0.55]),
            "champion": _rows([0.60, 0.60, 0.60, 0.60]),
        },
        trial_count=2,
        min_folds=5,
    )
    assert evidence["status"] == "INSUFFICIENT_EVIDENCE"
    assert evidence["eligible_for_freeze"] is False


def test_missing_comparator_is_fail_closed():
    evidence = paired_logloss_selection_evidence(
        "challenger",
        {"challenger": {"logloss": 0.55, "folds": 8}},
        {"challenger": _rows([0.55] * 8)},
        trial_count=1,
    )
    assert evidence["status"] == "INSUFFICIENT_EVIDENCE"
    assert evidence["eligible_for_freeze"] is False
