"""Small Markdown report helper shared by diagnostics tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


class MarkdownReport:
    """Accumulate simple Markdown sections, paragraphs, and tables."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def heading(self, text: str, level: int = 1) -> None:
        marker = "#" * max(1, int(level))
        self.lines.extend([f"{marker} {text}", ""])

    def paragraph(self, text: str) -> None:
        self.lines.extend([str(text), ""])

    def table(self, headers: Sequence[Any], rows: Sequence[Sequence[Any]]) -> None:
        header_values = [str(value) for value in headers]
        self.lines.append("| " + " | ".join(header_values) + " |")
        self.lines.append("| " + " | ".join("---" for _ in header_values) + " |")
        for row in rows:
            self.lines.append("| " + " | ".join(self._cell(value) for value in row) + " |")
        self.lines.append("")

    def write(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines).rstrip() + "\n", encoding="utf-8")

    def _cell(self, value: Any) -> str:
        text = str(value)
        return text.replace("|", "\\|").replace("\n", "<br>")
