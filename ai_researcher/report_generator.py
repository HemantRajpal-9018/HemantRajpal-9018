"""Report generator for gap analysis results.

Produces human-readable plain-text and Markdown reports from
``AnalysisResult`` objects.
"""

from __future__ import annotations

from typing import List

from ai_researcher.gap_analyzer import AnalysisResult, GapFinding


# ---------------------------------------------------------------------------
# Severity decorators
# ---------------------------------------------------------------------------

_SEVERITY_EMOJI = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🟡",
    "low": "🟢",
}


# ---------------------------------------------------------------------------
# Plain-text report
# ---------------------------------------------------------------------------

def _wrap(text: str, width: int = 78) -> str:
    """Very simple word-wrap for plain text."""
    words = text.split()
    lines: List[str] = []
    current: List[str] = []
    length = 0
    for word in words:
        if length + len(word) + 1 > width and current:
            lines.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += len(word) + 1
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def generate_text_report(result: AnalysisResult) -> str:
    """Return a plain-text gap analysis report."""
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("  AI RESEARCHER AGENT – REASONING MODEL GAP ANALYSIS")
    lines.append("  Analysis date: 2026-03-12")
    lines.append(f"  Models analyzed: {result.model_count}")
    lines.append(
        f"  Gap categories with findings: "
        f"{sum(1 for f in result.findings)}/{len(result.category_coverage)}"
    )
    lines.append("=" * 78)
    lines.append("")

    for i, finding in enumerate(result.findings, 1):
        sev = finding.severity.upper()
        lines.append(f"  [{sev}] Finding #{i}: {finding.title}")
        lines.append("-" * 78)
        if finding.description:
            lines.append(_wrap(f"  {finding.description}"))
        lines.append("")
        lines.append(f"  Category : {finding.category}")
        lines.append(f"  Severity : {sev}")
        lines.append(f"  Affected : {', '.join(finding.affected_models)}")
        lines.append("")
        if finding.evidence:
            lines.append("  Evidence from model weakness reports:")
            for ev in finding.evidence:
                lines.append(f"    • {ev}")
            lines.append("")
        if finding.research_directions:
            lines.append("  Suggested research directions:")
            for rd in finding.research_directions:
                lines.append(f"    → {rd}")
            lines.append("")
        lines.append("")

    lines.append("=" * 78)
    lines.append("  END OF REPORT")
    lines.append("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _finding_to_markdown(finding: GapFinding, index: int) -> str:
    """Render a single finding as Markdown."""
    emoji = _SEVERITY_EMOJI.get(finding.severity, "⚪")
    parts: List[str] = []

    parts.append(
        f"### {emoji} Finding #{index}: {finding.title}"
    )
    parts.append("")
    parts.append(f"| Field | Value |")
    parts.append(f"|-------|-------|")
    parts.append(f"| **Category** | `{finding.category}` |")
    parts.append(f"| **Severity** | **{finding.severity.upper()}** |")
    parts.append(
        f"| **Affected models** | {', '.join(finding.affected_models)} |"
    )
    parts.append("")

    if finding.description:
        parts.append(finding.description)
        parts.append("")

    if finding.evidence:
        parts.append("**Evidence from model weakness reports:**")
        for ev in finding.evidence:
            parts.append(f"- {ev}")
        parts.append("")

    if finding.research_directions:
        parts.append("**Suggested research directions:**")
        for rd in finding.research_directions:
            parts.append(f"1. {rd}")
        parts.append("")

    parts.append("---")
    parts.append("")
    return "\n".join(parts)


def generate_markdown_report(result: AnalysisResult) -> str:
    """Return a Markdown-formatted gap analysis report."""
    parts: List[str] = []
    parts.append("# 🔬 AI Researcher Agent – Reasoning Model Gap Analysis")
    parts.append("")
    parts.append(f"**Analysis date:** 2026-03-12  ")
    parts.append(f"**Models analyzed:** {result.model_count}  ")
    parts.append(
        f"**Gap categories with findings:** "
        f"{sum(1 for _ in result.findings)}/{len(result.category_coverage)}"
    )
    parts.append("")
    parts.append("---")
    parts.append("")

    # Table of contents
    parts.append("## Table of Contents")
    parts.append("")
    for i, f in enumerate(result.findings, 1):
        emoji = _SEVERITY_EMOJI.get(f.severity, "⚪")
        anchor = f.title.lower().replace(" ", "-").replace("'", "")
        parts.append(
            f"{i}. {emoji} [{f.title}](#{anchor}) — **{f.severity.upper()}**"
        )
    parts.append("")
    parts.append("---")
    parts.append("")

    parts.append("## Detailed Findings")
    parts.append("")

    for i, finding in enumerate(result.findings, 1):
        parts.append(_finding_to_markdown(finding, i))

    # Summary statistics
    parts.append("## Summary Statistics")
    parts.append("")
    parts.append("| Category | Hits |")
    parts.append("|----------|------|")
    for cat, count in sorted(
        result.category_coverage.items(), key=lambda x: -x[1]
    ):
        parts.append(f"| `{cat}` | {count} |")
    parts.append("")

    return "\n".join(parts)
