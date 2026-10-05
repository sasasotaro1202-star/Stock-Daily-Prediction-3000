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

def test_v13_scenario_row_exposes_new_research_signals() -> None:
    from src.research.ultimate_v13_extensions import _scenario_row

    row = _scenario_row(0.72, 0.10, 0.15)
    scenario = row["scenario_probability_proxy"]
    uncertainty = row["uncertainty_proxy"]
    assert set(scenario) == {
        "continuation_up",
        "continuation_down",
        "range_neutral",
        "shock",
    }
    assert sum(scenario.values()) == pytest.approx(1.0)
    assert 0.0 <= uncertainty["aggregate"] <= 1.0
    assert row["research_only"] is True