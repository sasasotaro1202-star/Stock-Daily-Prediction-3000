from scripts.normalize_price_store import input_price_partitions


def test_batch_partitions_exclude_stale_canonical_output(tmp_path):
    canonical = tmp_path / "canonical.parquet"
    batch1 = tmp_path / "batch_000.parquet"
    batch2 = tmp_path / "batch_001.parquet"
    canonical.touch()
    batch1.touch()
    batch2.touch()

    assert input_price_partitions(tmp_path) == [batch1, batch2]


def test_canonical_is_fallback_when_no_batches_exist(tmp_path):
    canonical = tmp_path / "canonical.parquet"
    canonical.touch()

    assert input_price_partitions(tmp_path) == [canonical]


def test_empty_price_directory_returns_empty(tmp_path):
    assert input_price_partitions(tmp_path) == []
