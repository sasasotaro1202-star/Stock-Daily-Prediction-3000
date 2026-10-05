from __future__ import annotations

import numpy as np
import pytest

from src.research.predictive_state import (
    REGIMES,
    aggregate_uncertainty,
    choose_information_action,
    chronological_transition_forecast,
    ensemble_disagreement,
    estimate_transition_matrix,
    heuristic_regime_posterior,
    normalized_entropy,
    scenario_probabilities,
    uncertainty_components,
)


def test_entropy_and_disagreement_are_bounded() -> None:
    entropy = normalized_entropy(
        np.asarray([[0.5, 0.5], [0.9, 0.1], [1.0, 0.0]], dtype=float)
    )
    disagreement = ensemble_disagreement(
        np.asarray([[0.50, 0.50], [0.90, 0.70], [0.20, 0.80]], dtype=float)
    )
    assert np.all((entropy >= 0.0) & (entropy <= 1.0))
    assert np.all((disagreement >= 0.0) & (disagreement <= 1.0))
    assert entropy[0] > entropy[1]


def test_transition_matrix_is_row_normalized_and_smoothed() -> None:
    matrix = estimate_transition_matrix(
        ["trend_up", "trend_up", "range", "trend_up"],
        categories=REGIMES,
        alpha=1.0,
    )
    for row in matrix.values():
        assert sum(row.values()) == pytest.approx(1.0)
    assert matrix["trend_up"]["trend_up"] > matrix["trend_up"]["range"]
    assert all(value > 0.0 for value in matrix["shock"].values())


def test_chronological_transition_forecast_excludes_current_and_future_transition() -> None:
    states = ["trend_up", "trend_up", "range", "trend_down"]
    forecasts = chronological_transition_forecast(states, categories=REGIMES, alpha=1.0)

    # At index 0 there is no prior transition, so smoothing remains uniform.
    assert forecasts[0]["history_rows"] == 0
    first = forecasts[0]["transition"]
    assert all(first[state] == pytest.approx(1.0 / len(REGIMES)) for state in REGIMES)

    # Row 2 has only the transition 0->1 in its admissible history.
    row2 = forecasts[2]["transition"]
    expected_up = 2.0 / 3.0
    expected_other = 1.0 / 6.0
    assert row2["trend_up"] == pytest.approx(expected_up)
    assert row2["range"] == pytest.approx(expected_other)
    assert row2["trend_down"] == pytest.approx(expected_other)

    # Appending a future state cannot alter any earlier row.
    earlier = chronological_transition_forecast(
        states, categories=REGIMES, alpha=1.0
    )
    later = chronological_transition_forecast(
        states + ["shock", "trend_up"], categories=REGIMES, alpha=1.0
    )
    for idx in range(len(states)):
        assert earlier[idx]["transition"] == later[idx]["transition"]


def test_heuristic_regime_posterior_is_probability_simplex() -> None:
    posterior = heuristic_regime_posterior(
        trend_score=np.asarray([1.0, 0.0, -1.0]),
        volatility_score=np.asarray([0.1, 0.2, 0.1]),
        liquidity_score=np.asarray([0.0, 0.0, 0.0]),
        event_score=np.asarray([0.0, 0.8, 0.0]),
    )
    assert posterior.shape == (3, 5)
    assert np.allclose(posterior.sum(axis=1), 1.0)
    assert posterior[0, 0] > posterior[0, 2]
    assert posterior[2, 2] > posterior[2, 0]


def test_uncertainty_components_remain_separately_auditable() -> None:
    components = uncertainty_components(
        model_disagreement=np.asarray([0.1, 0.9]),
        probability_entropy=np.asarray([0.2, 0.8]),
        data_uncertainty=np.asarray([0.1, 0.7]),
        regime_uncertainty=np.asarray([0.2, 0.6]),
        ood_score=np.asarray([0.0, 1.0]),
        volatility_uncertainty=np.asarray([0.2, 0.9]),
    )
    aggregate = aggregate_uncertainty(components)
    assert set(components) == {
        "model_disagreement",
        "probability_entropy",
        "data_uncertainty",
        "regime_uncertainty",
        "ood_score",
        "volatility_uncertainty",
    }
    assert np.allclose(aggregate, [0.13333333333333333, 4.9 / 6.0])


def test_information_policy_is_selective_and_ordered() -> None:
    actions = choose_information_action(
        aggregate_uncertainty=np.asarray([0.2, 0.6, 0.9, 0.4]),
        data_quality=np.asarray([0.9, 0.5, 0.9, 0.9]),
        ood_score=np.asarray([0.0, 0.0, 0.9, 0.0]),
        regime_stability=np.asarray([0.9, 0.9, 0.9, 0.2]),
    )
    assert actions.tolist() == ["PREDICT", "ACQUIRE_MORE", "ABSTAIN", "PREDICT"]


def test_scenario_probabilities_form_a_simplex() -> None:
    scenarios = scenario_probabilities(
        np.asarray([0.8, 0.5, 0.2]),
        shock_probability=np.asarray([0.1, 0.2, 0.9]),
        disagreement=np.asarray([0.1, 0.4, 0.8]),
    )
    assert scenarios.shape == (3, 4)
    assert np.allclose(scenarios.sum(axis=1), 1.0)
    assert scenarios[0, 0] > scenarios[0, 1]
    assert scenarios[2, 3] > scenarios[0, 3]


def test_invalid_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        ensemble_disagreement(np.asarray([[1.1, 0.2]], dtype=float))
    with pytest.raises(ValueError):
        estimate_transition_matrix(["unknown"], categories=REGIMES)
    with pytest.raises(ValueError):
        heuristic_regime_posterior(
            trend_score=np.asarray([0.0]),
            volatility_score=np.asarray([0.0, 1.0]),
        )