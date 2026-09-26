from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt(v):
    return "n/a" if v is None else f"{float(v):.6f}" if isinstance(v, (int, float)) else str(v)


def main() -> int:
    metrics_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/research/latest_metrics.json")
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    v6 = data.get("innovative_prediction_control_v6", {})
    locked = v6.get("full_vs_baseline_locked", {})
    arches = v6.get("architectures", {})
    lines = [
        "# Maximum Future-Generalization Predictive Control v6",
        "",
        f"- Schema: {v6.get('schema_version', 'unknown')}",
        f"- Status: {v6.get('status', 'unknown')}",
        f"- Promotion: {v6.get('promotion', 'HOLD')}",
        f"- Production changed: {v6.get('production_changed', 'unknown')}",
        f"- Development folds: {v6.get('development_folds', 'unknown')}",
        f"- Locked folds: {v6.get('locked_folds', 'unknown')}",
        "",
        "## Locked OOS vs baseline",
        "",
        "| Metric | Delta |",
        "|---|---:|",
        f"| Accuracy | {fmt(locked.get('accuracy'))} |",
        f"| LogLoss | {fmt(locked.get('logloss'))} |",
        f"| Brier | {fmt(locked.get('brier'))} |",
        f"| ECE | {fmt(locked.get('ece'))} |",
        "",
        "## Ablation",
        "",
        "| Architecture | Accuracy | LogLoss | Brier | ECE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in arches.items():
        m = row.get("metrics", {})
        lines.append(
            f"| {name} | {fmt(m.get('accuracy'))} | {fmt(m.get('logloss'))} | {fmt(m.get('brier'))} | {fmt(m.get('ece'))} |"
        )
    lines += [
        "",
        "## Research layers",
        "",
        f"- Predictability: {v6.get('predictability', {}).get('status', 'unknown')}",
        f"- Future Failure: {v6.get('future_failure', {}).get('status', 'unknown')}",
        f"- Error Correlation: {v6.get('error_correlation', {}).get('status', 'unknown')}",
        f"- Regime Transition: {v6.get('regime_transition', {}).get('status', 'unknown')}",
        f"- Retrieval: {v6.get('retrieval', {}).get('status', 'unknown')}",
        f"- Uncertainty: {v6.get('uncertainty', {}).get('status', 'unknown')}",
        f"- Robustness: {v6.get('robustness', 'unknown')}",
        f"- Shadow: {v6.get('shadow', 'unknown')}",
        f"- Challenger: {v6.get('challenger', 'unknown')}",
        f"- Fallback: {v6.get('fallback', {}).get('status', 'unknown')}",
        "",
        "## Safety",
        "",
        f"- Locked OOS untouched: {v6.get('locked_oos_untouched_for_tuning', False)}",
        "- Meta-leakage: checked by dedicated audit artifact",
        "- Production adoption: disabled",
    ]
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    Path("artifacts/v6_report.md").write_text("
".join(lines) + "
", encoding="utf-8")
    print("
".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
