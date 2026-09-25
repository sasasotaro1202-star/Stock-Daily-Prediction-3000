from __future__ import annotations

import subprocess
from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "research-validation.yml"


def _extract_run_blocks(text: str) -> list[str]:
    lines = text.splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() != "run: |":
            i += 1
            continue
        parent_indent = len(lines[i]) - len(lines[i].lstrip(" "))
        block_indent = parent_indent + 2
        i += 1
        block: list[str] = []
        while i < len(lines):
            line = lines[i]
            if line.strip() and len(line) - len(line.lstrip(" ")) < block_indent:
                break
            block.append(line[block_indent:] if len(line) >= block_indent else "")
            i += 1
        blocks.append("\n".join(block) + "\n")
    return blocks


def test_research_validation_bash_blocks_are_syntactically_valid() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    blocks = _extract_run_blocks(text)
    assert blocks, "No run: | blocks found in research-validation workflow"

    for index, script in enumerate(blocks):
        result = subprocess.run(
            ["bash", "-n"],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"research-validation.yml run block {index} has invalid bash syntax:\n"
            f"{result.stderr}\nSCRIPT:\n{script}"
        )


def test_research_validation_status_uses_step_outcome_not_outputs() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    names = (
        "RESEARCH_STEP_OUTCOME",
        "SEC_RESEARCH_STEP_OUTCOME",
        "SEC_ABLATION_STEP_OUTCOME",
        "CPCV_STEP_OUTCOME",
    )
    bindings = {}
    for line in text.splitlines():
        stripped = line.strip()
        for name in names:
            prefix = f"{name}: "
            if stripped.startswith(prefix):
                bindings[name] = stripped[len(prefix):]
    for name in names:
        binding = bindings.get(name)
        assert binding is not None, f"missing {name} binding"
        assert ".outputs.outcome" not in binding, f"{name} must bind steps.<id>.outcome"
        assert binding.startswith("${{ steps.") and binding.endswith(".outcome }}"), (
            f"{name} must bind steps.<id>.outcome; got: {binding}"
        )
