"""Report generator for gap analysis results.

Produces human-readable plain-text and Markdown reports from
``AnalysisResult`` objects, trend forecasts, opportunity scores,
model comparisons, and recommendations.
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

_STATUS_EMOJI = {
    "emerging": "🌱",
    "accelerating": "🚀",
    "maturing": "📈",
}

_TIER_EMOJI = {
    "Tier 1": "🥇",
    "Tier 2": "🥈",
    "Tier 3": "🥉",
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


# ===================================================================
# NEW REPORT SECTIONS – Trends, Comparisons, Opportunities, Recs
# ===================================================================

# ---------------------------------------------------------------------------
# Trend forecast – plain text
# ---------------------------------------------------------------------------

def generate_trends_text(forecast) -> str:
    """Render trend forecast as plain text."""
    from ai_researcher.trend_forecaster import ForecastResult
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("  EMERGING TRENDS IN REASONING AI")
    lines.append("=" * 78)
    lines.append("")

    for i, trend in enumerate(forecast.trends, 1):
        lines.append(
            f"  [{trend.status.upper()}] Trend #{i}: {trend.title}"
        )
        lines.append("-" * 78)
        lines.append(_wrap(f"  {trend.description}"))
        lines.append("")
        lines.append(f"  Confidence   : {trend.confidence:.0%}")
        lines.append(f"  Time horizon : {trend.time_horizon}")
        lines.append(f"  Categories   : {', '.join(trend.affected_categories)}")
        lines.append("")
        if trend.market_implications:
            lines.append("  Market implications:")
            for mi in trend.market_implications:
                lines.append(f"    ★ {mi}")
            lines.append("")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Trend forecast – markdown
# ---------------------------------------------------------------------------

def _generate_trends_markdown(forecast) -> str:
    """Render trend forecast as Markdown."""
    parts: List[str] = []
    parts.append("## 🔮 Emerging Trends in Reasoning AI")
    parts.append("")

    for i, trend in enumerate(forecast.trends, 1):
        emoji = _STATUS_EMOJI.get(trend.status, "📊")
        parts.append(f"### {emoji} Trend #{i}: {trend.title}")
        parts.append("")
        parts.append(f"| Field | Value |")
        parts.append(f"|-------|-------|")
        parts.append(f"| **Status** | {trend.status} |")
        parts.append(f"| **Confidence** | {trend.confidence:.0%} |")
        parts.append(f"| **Time horizon** | {trend.time_horizon} |")
        parts.append(
            f"| **Affected categories** | "
            f"{', '.join(f'`{c}`' for c in trend.affected_categories)} |"
        )
        parts.append("")
        parts.append(trend.description)
        parts.append("")

        if trend.supporting_evidence:
            parts.append("**Supporting evidence:**")
            for ev in trend.supporting_evidence[:5]:
                parts.append(f"- {ev}")
            parts.append("")

        if trend.market_implications:
            parts.append("**Market implications:**")
            for mi in trend.market_implications:
                parts.append(f"- ★ {mi}")
            parts.append("")

        parts.append("---")
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Model comparison – plain text
# ---------------------------------------------------------------------------

def generate_comparison_text(comparison) -> str:
    """Render a head-to-head comparison as plain text."""
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append(
        f"  HEAD-TO-HEAD: {comparison.model_a} vs {comparison.model_b}"
    )
    lines.append("=" * 78)
    lines.append("")

    if comparison.benchmark_deltas:
        lines.append("  Benchmark comparison:")
        for d in comparison.benchmark_deltas:
            winner = (
                comparison.model_a if d.delta > 0
                else comparison.model_b if d.delta < 0
                else "Tie"
            )
            lines.append(
                f"    {d.benchmark:<25} "
                f"{d.score_a:>6.1f} vs {d.score_b:>6.1f}  "
                f"(Δ {d.delta:+.1f}, {winner})"
            )
        lines.append("")

    if comparison.shared_weaknesses:
        lines.append("  Shared weaknesses:")
        for sw in comparison.shared_weaknesses:
            lines.append(f"    ⚠ {sw}")
        lines.append("")

    lines.append(f"  Verdict: {comparison.verdict}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Opportunity scores – plain text
# ---------------------------------------------------------------------------

def generate_opportunities_text(report) -> str:
    """Render opportunity scores as plain text."""
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("  INNOVATION OPPORTUNITY SCORES")
    lines.append("=" * 78)
    lines.append("")

    for opp in report.opportunities:
        lines.append(
            f"  [{opp.priority_tier}] {opp.title}  "
            f"(composite: {opp.composite}/10)"
        )
        lines.append(
            f"    Impact={opp.impact}  Feasibility={opp.feasibility}  "
            f"Novelty={opp.novelty}  Timing={opp.market_timing}"
        )
        lines.append(f"    {_wrap(opp.rationale)}")
        lines.append("")

    if report.top_3_summary:
        lines.append("  TOP 3 OPPORTUNITIES:")
        for line in report.top_3_summary.split("\n"):
            lines.append(f"    {line}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Opportunity scores – markdown
# ---------------------------------------------------------------------------

def _generate_opportunities_markdown(report) -> str:
    """Render opportunity scores as Markdown."""
    parts: List[str] = []
    parts.append("## 💡 Innovation Opportunity Scores")
    parts.append("")
    parts.append(
        "Each gap is scored on four dimensions (0-10): "
        "**Impact**, **Feasibility**, **Novelty**, and **Market Timing**."
    )
    parts.append("")

    parts.append(
        "| Tier | Category | Composite | Impact | Feasibility "
        "| Novelty | Timing |"
    )
    parts.append(
        "|------|----------|-----------|--------|-------------|"
        "---------|--------|"
    )
    for opp in report.opportunities:
        emoji = _TIER_EMOJI.get(opp.priority_tier, "")
        parts.append(
            f"| {emoji} {opp.priority_tier} | `{opp.category}` "
            f"| **{opp.composite}** | {opp.impact} | {opp.feasibility} "
            f"| {opp.novelty} | {opp.market_timing} |"
        )
    parts.append("")

    if report.top_3_summary:
        parts.append("### 🏆 Top 3 Opportunities")
        parts.append("")
        parts.append(report.top_3_summary)
        parts.append("")

    # Detailed rationales
    parts.append("### Detailed Rationales")
    parts.append("")
    for opp in report.opportunities:
        emoji = _TIER_EMOJI.get(opp.priority_tier, "")
        parts.append(f"**{emoji} {opp.title}** ({opp.priority_tier})")
        parts.append("")
        parts.append(f"> {opp.rationale}")
        parts.append("")

    parts.append("---")
    parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Recommendations – plain text
# ---------------------------------------------------------------------------

def generate_recommendations_text(recommendations) -> str:
    """Render use-case recommendations as plain text."""
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("  BEST-FIT MODEL RECOMMENDATIONS")
    lines.append("=" * 78)
    lines.append("")

    for rec in recommendations:
        lines.append(f"  Use case: {rec.use_case}")
        lines.append(f"    ★ Recommended : {rec.recommended_model}")
        lines.append(f"    ○ Runner-up   : {rec.runner_up}")
        lines.append(f"    {rec.reason}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Recommendations – markdown
# ---------------------------------------------------------------------------

def _generate_recommendations_markdown(recommendations) -> str:
    """Render use-case recommendations as Markdown."""
    parts: List[str] = []
    parts.append("## 🎯 Best-Fit Model Recommendations")
    parts.append("")

    parts.append("| Use Case | ★ Recommended | Runner-Up |")
    parts.append("|----------|---------------|-----------|")
    for rec in recommendations:
        parts.append(
            f"| {rec.use_case} | **{rec.recommended_model}** "
            f"| {rec.runner_up} |"
        )
    parts.append("")
    parts.append("---")
    parts.append("")
    return "\n".join(parts)


# ===================================================================
# Full comprehensive reports (all modules combined)
# ===================================================================

def generate_full_text_report(
    gap_result,
    trends,
    opportunities,
    recommendations,
) -> str:
    """Combine all analysis modules into one comprehensive text report."""
    sections = [
        generate_text_report(gap_result),
        "",
        generate_trends_text(trends),
        "",
        generate_opportunities_text(opportunities),
        "",
        generate_recommendations_text(recommendations),
    ]
    return "\n".join(sections)


def generate_full_markdown_report(
    gap_result,
    trends,
    opportunities,
    recommendations,
) -> str:
    """Combine all analysis modules into one comprehensive Markdown report."""
    sections = [
        generate_markdown_report(gap_result),
        "",
        _generate_trends_markdown(trends),
        "",
        _generate_opportunities_markdown(opportunities),
        "",
        _generate_recommendations_markdown(recommendations),
    ]
    return "\n".join(sections)
