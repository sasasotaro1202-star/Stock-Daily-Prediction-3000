from __future__ import annotations

import re
import subprocess
from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "research-validation.yml"


def _extract_run_blocks(text: str) -> list[str]:
    lines = text.splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        match = re.match(r"^(?P<indent> *)run: *\\|\\s*$", lines[i])
        if not match:
            i += 1
            continue
        block_indent = len(match.group("indent")) + 2
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
            f"research-validation.yml run block {index} has invalid bash syntax:\\n"
            f"{result.stderr}\\nSCRIPT:\\n{script}"
        )


def test_research_validation_status_uses_step_outcome_not_outputs() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    for name in ("RESEARCH_STEP_OUTCOME", "SEC_RESEARCH_STEP_OUTCOME",
                 "SEC_ABLATION_STEP_OUTCOME", "CPCV_STEP_OUTCOME"):
        prefix = r"^\\s*" + re.escape(name) + r":\\s*\\$\\{\\{steps\\."
        match = re.search(prefix + r"([^}]+)\\}\\}", text, re.MULTILINE)
        assert match, f"missing {name} binding"
        assert ".outputs.outcome" not in match.group(1), f"{name} must bind steps.<id>.outcome"
        assert match.group(1).endswith(".outcome"), f"{name} must bind steps.<id>.outcome"
