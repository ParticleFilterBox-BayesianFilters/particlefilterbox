"""Base report class for particlefilterbox.

Provides the BaseReport container that holds sections and supports
rendering to HTML, LaTeX, and Markdown formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ReportSection:
    """A single section of a report.

    Attributes
    ----------
    title : str
        Section title.
    content : str
        Section content (text, HTML, or markdown).
    figures : list[dict]
        List of figure metadata dicts with keys 'path', 'caption', 'data'.
    tables : list[dict]
        List of table dicts with keys 'headers', 'rows', 'caption'.
    level : int
        Heading level (1=h1, 2=h2, etc.). Default is 2.
    """

    title: str
    content: str = ""
    figures: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    level: int = 2


class BaseReport:
    """Base report container.

    Manages sections and provides rendering to multiple output formats.

    Parameters
    ----------
    title : str
        Report title.
    author : str
        Report author. Default is 'particlefilterbox'.
    date : str or None
        Report date. If None, uses current date.

    Examples
    --------
    >>> report = BaseReport(title='PF Analysis')
    >>> report.add_section('Summary', 'This is the summary.')
    >>> html = report.to_html()
    >>> print(html[:50])
    """

    def __init__(
        self,
        title: str = "Particle Filter Report",
        author: str = "particlefilterbox",
        date: str | None = None,
    ) -> None:
        self.title = title
        self.author = author
        self.sections: list[ReportSection] = []

        if date is None:
            from datetime import datetime

            self.date = datetime.now().strftime("%Y-%m-%d")
        else:
            self.date = date

    def add_section(
        self,
        title: str,
        content: str = "",
        figures: list[dict[str, Any]] | None = None,
        tables: list[dict[str, Any]] | None = None,
        level: int = 2,
    ) -> None:
        """Add a section to the report.

        Parameters
        ----------
        title : str
            Section title.
        content : str
            Section content text.
        figures : list[dict] or None
            List of figure dicts with 'path' and/or 'caption'.
        tables : list[dict] or None
            List of table dicts with 'headers', 'rows', 'caption'.
        level : int
            Heading level. Default is 2.
        """
        section = ReportSection(
            title=title,
            content=content,
            figures=figures or [],
            tables=tables or [],
            level=level,
        )
        self.sections.append(section)

    def to_html(self, path: str | Path | None = None) -> str:
        """Render report to HTML.

        Parameters
        ----------
        path : str or Path or None
            If provided, write HTML to file. Otherwise, return as string.

        Returns
        -------
        str
            HTML content.
        """
        lines: list[str] = []
        lines.append("<!DOCTYPE html>")
        lines.append("<html>")
        lines.append("<head>")
        lines.append(f"  <title>{_escape_html(self.title)}</title>")
        lines.append("  <meta charset='utf-8'>")
        lines.append("  <style>")
        lines.append(
            "    body { font-family: 'Segoe UI', Tahoma, sans-serif;"
            " max-width: 900px; margin: 0 auto; padding: 20px; color: #333; }"
        )
        lines.append(
            "    h1 { color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px; }"
        )
        lines.append("    h2 { color: #A23B72; margin-top: 30px; }")
        lines.append("    h3 { color: #F18F01; }")
        lines.append("    table { border-collapse: collapse; width: 100%; margin: 15px 0; }")
        lines.append("    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        lines.append("    th { background-color: #2E86AB; color: white; }")
        lines.append("    tr:nth-child(even) { background-color: #f2f2f2; }")
        lines.append("    .figure { text-align: center; margin: 20px 0; }")
        lines.append("    .figure img { max-width: 100%; }")
        lines.append("    .figure .caption { font-style: italic; color: #666; margin-top: 5px; }")
        lines.append("    .meta { color: #666; font-size: 0.9em; margin-bottom: 20px; }")
        lines.append("  </style>")
        lines.append("</head>")
        lines.append("<body>")
        lines.append(f"  <h1>{_escape_html(self.title)}</h1>")
        lines.append(
            f"  <div class='meta'>Author: {_escape_html(self.author)}"
            f" | Date: {_escape_html(self.date)}</div>"
        )

        for section in self.sections:
            tag = f"h{section.level}"
            lines.append(f"  <{tag}>{_escape_html(section.title)}</{tag}>")

            if section.content:
                lines.append(f"  <p>{_escape_html(section.content)}</p>")

            for table in section.tables:
                lines.append("  <table>")
                caption = table.get("caption", "")
                if caption:
                    lines.append(f"    <caption>{_escape_html(caption)}</caption>")
                headers = table.get("headers", [])
                if headers:
                    lines.append("    <thead><tr>")
                    for h in headers:
                        lines.append(f"      <th>{_escape_html(str(h))}</th>")
                    lines.append("    </tr></thead>")
                rows = table.get("rows", [])
                lines.append("    <tbody>")
                for row in rows:
                    lines.append("    <tr>")
                    for cell in row:
                        lines.append(f"      <td>{_escape_html(str(cell))}</td>")
                    lines.append("    </tr>")
                lines.append("    </tbody>")
                lines.append("  </table>")

            for fig in section.figures:
                lines.append("  <div class='figure'>")
                fig_path = fig.get("path", "")
                fig_data = fig.get("data", "")
                if fig_data:
                    lines.append(f"    <img src='data:image/png;base64,{fig_data}' />")
                elif fig_path:
                    lines.append(f"    <img src='{_escape_html(str(fig_path))}' />")
                caption = fig.get("caption", "")
                if caption:
                    lines.append(f"    <div class='caption'>{_escape_html(caption)}</div>")
                lines.append("  </div>")

        lines.append("</body>")
        lines.append("</html>")

        html = "\n".join(lines)

        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html, encoding="utf-8")

        return html

    def to_latex(self, path: str | Path | None = None) -> str:
        """Render report to LaTeX.

        Parameters
        ----------
        path : str or Path or None
            If provided, write LaTeX to file.

        Returns
        -------
        str
            LaTeX content.
        """
        lines: list[str] = []
        lines.append(r"\documentclass{article}")
        lines.append(r"\usepackage[utf8]{inputenc}")
        lines.append(r"\usepackage{booktabs}")
        lines.append(r"\usepackage{graphicx}")
        lines.append(r"\usepackage{hyperref}")
        lines.append("")
        lines.append(f"\\title{{{_escape_latex(self.title)}}}")
        lines.append(f"\\author{{{_escape_latex(self.author)}}}")
        lines.append(f"\\date{{{_escape_latex(self.date)}}}")
        lines.append("")
        lines.append(r"\begin{document}")
        lines.append(r"\maketitle")
        lines.append("")

        section_cmds = {1: "section", 2: "subsection", 3: "subsubsection"}

        for section in self.sections:
            cmd = section_cmds.get(section.level, "subsection")
            lines.append(f"\\{cmd}{{{_escape_latex(section.title)}}}")
            lines.append("")

            if section.content:
                lines.append(_escape_latex(section.content))
                lines.append("")

            for table in section.tables:
                headers = table.get("headers", [])
                rows = table.get("rows", [])
                caption = table.get("caption", "")

                n_cols = len(headers) if headers else (len(rows[0]) if rows else 0)
                col_spec = "l" * n_cols

                lines.append(r"\begin{table}[h]")
                lines.append(r"\centering")
                lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
                lines.append(r"\toprule")

                if headers:
                    lines.append(" & ".join(str(h) for h in headers) + r" \\")
                    lines.append(r"\midrule")

                for row in rows:
                    lines.append(" & ".join(str(c) for c in row) + r" \\")

                lines.append(r"\bottomrule")
                lines.append(r"\end{tabular}")
                if caption:
                    lines.append(f"\\caption{{{_escape_latex(caption)}}}")
                lines.append(r"\end{table}")
                lines.append("")

            for fig in section.figures:
                fig_path = fig.get("path", "")
                caption = fig.get("caption", "")
                if fig_path:
                    lines.append(r"\begin{figure}[h]")
                    lines.append(r"\centering")
                    lines.append(f"\\includegraphics[width=0.8\\textwidth]{{{fig_path}}}")
                    if caption:
                        lines.append(f"\\caption{{{_escape_latex(caption)}}}")
                    lines.append(r"\end{figure}")
                    lines.append("")

        lines.append(r"\end{document}")

        latex = "\n".join(lines)

        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(latex, encoding="utf-8")

        return latex

    def to_markdown(self, path: str | Path | None = None) -> str:
        """Render report to Markdown.

        Parameters
        ----------
        path : str or Path or None
            If provided, write Markdown to file.

        Returns
        -------
        str
            Markdown content.
        """
        lines: list[str] = []
        lines.append(f"# {self.title}")
        lines.append("")
        lines.append(f"**Author**: {self.author}  ")
        lines.append(f"**Date**: {self.date}")
        lines.append("")

        for section in self.sections:
            prefix = "#" * section.level
            lines.append(f"{prefix} {section.title}")
            lines.append("")

            if section.content:
                lines.append(section.content)
                lines.append("")

            for table in section.tables:
                headers = table.get("headers", [])
                rows = table.get("rows", [])
                caption = table.get("caption", "")

                if caption:
                    lines.append(f"*{caption}*")
                    lines.append("")

                if headers:
                    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
                    lines.append("| " + " | ".join("---" for _ in headers) + " |")

                for row in rows:
                    lines.append("| " + " | ".join(str(c) for c in row) + " |")
                lines.append("")

            for fig in section.figures:
                fig_path = fig.get("path", "")
                caption = fig.get("caption", "")
                if fig_path:
                    lines.append(f"![{caption}]({fig_path})")
                    lines.append("")

        md = "\n".join(lines)

        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(md, encoding="utf-8")

        return md


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters."""
    special = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for char, replacement in special.items():
        text = text.replace(char, replacement)
    return text
