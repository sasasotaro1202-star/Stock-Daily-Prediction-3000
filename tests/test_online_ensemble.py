import numpy as np
import pytest

from src.research.online_ensemble import online_expert_average


def test_first_session_uses_prior_uniform_weights():
    y = np.array([1, 0, 1, 0], dtype=int)
    dates = np.array(["2026-01-02"] * 4)
    predictions = {
        "a": np.array([0.9, 0.1, 0.9, 0.1]),
        "b": np.array([0.1, 0.9, 0.1, 0.9]),
    }
    ensemble, _, history = online_expert_average(
        predictions, y, dates, learning_rate=2.0
    )
    assert np.allclose(ensemble, 0.5)
    assert len(history) == 1
    assert history[0]["weights_before"] == pytest.approx([0.5, 0.5])


def test_past_outcomes_change_only_future_session_weights():
    dates = np.array(
        ["2026-01-02"] * 2 + ["2026-01-05"] * 2
    )
    predictions = {
        "a": np.array([0.9, 0.9, 0.9, 0.9]),
        "b": np.array([0.1, 0.1, 0.1, 0.1]),
    }
    y_good_a = np.array([1, 1, 1, 1], dtype=int)
    y_bad_a = np.array([1, 1, 0, 0], dtype=int)

    out1, _, _ = online_expert_average(
        predictions, y_good_a, dates, learning_rate=2.0
    )
    out2, _, _ = online_expert_average(
        predictions, y_bad_a, dates, learning_rate=2.0
    )
    assert np.allclose(out1[:2], out2[:2])
    assert out1[2] > out2[2]


def test_input_validation():
    with pytest.raises(ValueError):
        online_expert_average({}, [0, 1], ["a", "b"], learning_rate=1.0)
    with pytest.raises(ValueError):
        online_expert_average(
            {"a": [0.5, np.nan], "b": [0.5, 0.5]},
            [0, 1],
            ["a", "b"],
            learning_rate=1.0,
        )
    with pytest.raises(ValueError):
        online_expert_average(
            {"a": [0.5, 0.5], "b": [0.5, 0.5]},
            [0, 1],
            ["a"],
            learning_rate=1.0,
        )


def test_nonnegative_learning_rate():
    y = [0, 1]
    dates = ["a", "b"]
    predictions = {"a": [0.5, 0.5], "b": [0.5, 0.5]}
    with pytest.raises(ValueError):
        online_expert_average(predictions, y, dates, learning_rate=-1.0)
