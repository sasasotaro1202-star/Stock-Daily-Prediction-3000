from src.research import cpcv_validation_audit as cpcv


def test_cpcv_groups_cover_all_positions_without_overlap():
    groups = cpcv._build_groups(60, 6)
    flat = [p for group in groups for p in group]
    assert flat == list(range(60))


def test_validate_split_rejects_purge_and_label_overlap():
    violations = cpcv.validate_split(
        train_positions={0, 10},
        test_positions={1},
        n_dates=20,
        purge=1,
        embargo=1,
    )
    assert violations
    assert any("purge_or_embargo" in item or "label_interval" in item for item in violations)


def test_validate_split_accepts_clean_split():
    violations = cpcv.validate_split(
        train_positions={0, 5, 10},
        test_positions={2},
        n_dates=20,
        purge=1,
        embargo=1,
    )
    assert not violations
