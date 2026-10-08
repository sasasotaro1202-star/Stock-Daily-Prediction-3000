from __future__ import annotations

from pathlib import Path
import yaml

WORKFLOW_DIR = Path('.github/workflows')


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: UniqueKeyLoader, node, deep: bool = False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f'duplicate YAML key: {key!r}')
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def main() -> int:
    files = sorted(WORKFLOW_DIR.glob('*.yml'))
    if not files:
        raise SystemExit('FAIL: no workflow YAML files found')
    failures = []
    for path in files:
        try:
            data = yaml.load(path.read_text(encoding='utf-8'), Loader=UniqueKeyLoader)
            if not isinstance(data, dict) or 'jobs' not in data:
                failures.append(f'{path}: missing top-level jobs mapping')
        except Exception as exc:
            failures.append(f'{path}: {type(exc).__name__}: {exc}')
    if failures:
        for failure in failures:
            print(f'workflow_yaml_invalid: {failure}')
        raise SystemExit(1)
    print(f'workflow_yaml_valid {len(files)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
