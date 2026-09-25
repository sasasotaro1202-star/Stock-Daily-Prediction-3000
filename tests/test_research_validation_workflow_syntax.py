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


def test_research_validation_has_dedicated_status_job() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "  research-status:" in text
    assert "    needs: research" in text
    assert "    if: always()" in text
    assert "python scripts/persist_research_validation_status.py" in text
    assert "GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}" in text
