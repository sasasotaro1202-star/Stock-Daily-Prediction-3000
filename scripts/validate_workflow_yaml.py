from __future__ import annotations

from pathlib import Path
import yaml

WORKFLOW_DIR = Path('.github/workflows')


def main() -> int:
    files = sorted(WORKFLOW_DIR.glob('*.yml'))
    if not files:
        raise SystemExit('FAIL: no workflow YAML files found')
    failures = []
    for path in files:
        try:
            data = yaml.safe_load(path.read_text(encoding='utf-8'))
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
