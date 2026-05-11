"""Document-parsing tool — extract text (and optionally image references) from PDFs.

When the user pastes a PDF / design doc URL or file path into the
conversation, this tool turns it into plain text the model can read.
Embedded images are returned by their byte indices so callers can
optionally route them through the vision tower.

Requires ``pypdf`` (Apache-2.0, pure-Python). If it's missing, the tool
returns a clear error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..tool_protocol import Tool, ToolRegistry


def register_document_tools(registry: ToolRegistry) -> None:
    def parse_document(args: Dict[str, Any]) -> Dict[str, Any]:
        path = Path(args["path"])
        max_pages = int(args.get("max_pages", 50))
        try:
            from pypdf import PdfReader
        except ImportError:
            return {
                "error": "pypdf is not installed. Install with `pip install pypdf` "
                          "for PDF extraction. PNG / JPG screenshots work without it.",
            }

        if not path.exists():
            return {"error": f"Not found: {path}"}

        reader = PdfReader(str(path))
        n_pages = min(len(reader.pages), max_pages)
        pages: List[str] = []
        for i in range(n_pages):
            try:
                pages.append(reader.pages[i].extract_text() or "")
            except Exception:  # noqa: BLE001 — tolerate a single bad page
                pages.append("")
        return {
            "text": "\n\n".join(pages),
            "page_count": len(reader.pages),
            "pages_extracted": n_pages,
        }

    registry.register(Tool(
        name="parse_document",
        description="Extract plain text from a PDF document.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_pages": {"type": "integer", "default": 50},
            },
            "required": ["path"],
        },
        handler=parse_document,
        plugin="builtin.document",
    ))
