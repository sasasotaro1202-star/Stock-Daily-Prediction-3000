from src.data.external_sources import EXTERNAL_SOURCES


def main() -> None:
    ids = [s.id for s in EXTERNAL_SOURCES]
    if len(ids) != len(set(ids)):
        raise SystemExit("FAIL: duplicate external source ids")
    for source in EXTERNAL_SOURCES:
        if source.enabled_research and not source.cost.startswith("free"):
            raise SystemExit(f"FAIL: non-free source enabled: {source.id}")
        if not source.pit_mode:
            raise SystemExit(f"FAIL: missing PIT mode: {source.id}")
    print(f"external-source-catalog: {len(EXTERNAL_SOURCES)} sources validated")


if __name__ == "__main__":
    main()
