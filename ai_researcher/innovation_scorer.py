"""Innovation opportunity scorer.

Ranks each identified gap by its research/commercial opportunity value,
considering impact, feasibility, novelty, and market timing.  This
transforms a passive gap report into an actionable investment thesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ai_researcher.gap_analyzer import AnalysisResult, GapFinding, analyze_gaps
from ai_researcher.models_knowledge import ReasoningModel, get_reasoning_models


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OpportunityScore:
    """Multi-dimensional score for a single research opportunity."""
    category: str
    title: str
    impact: float          # 0-10: how much value solving this gap creates
    feasibility: float     # 0-10: how tractable the gap is to address
    novelty: float         # 0-10: how under-explored the area is
    market_timing: float   # 0-10: how well market demand aligns right now
    composite: float = 0.0  # weighted aggregate
    rationale: str = ""
    priority_tier: str = ""  # "Tier 1", "Tier 2", "Tier 3"


@dataclass
class OpportunityReport:
    """Complete opportunity-scoring output."""
    opportunities: List[OpportunityScore] = field(default_factory=list)
    top_3_summary: str = ""


# ---------------------------------------------------------------------------
# Scoring rubric – category-level heuristic parameters
# ---------------------------------------------------------------------------

_OPPORTUNITY_RUBRIC: Dict[str, Dict[str, float | str]] = {
    "faithfulness": {
        "impact": 9.0,
        "feasibility": 4.0,
        "novelty": 9.0,
        "market_timing": 8.0,
        "rationale": (
            "Verifying reasoning faithfulness is critical for trust in "
            "AI-generated conclusions.  Regulatory tailwinds (EU AI Act, "
            "US executive orders) create demand now, but the problem is "
            "technically hard."
        ),
    },
    "hallucination": {
        "impact": 9.5,
        "feasibility": 5.0,
        "novelty": 6.0,
        "market_timing": 9.0,
        "rationale": (
            "Hallucination is the #1 complaint in enterprise AI adoption. "
            "Solving it for multi-step reasoning is harder than for single "
            "answers, but the market pull is enormous."
        ),
    },
    "self_correction": {
        "impact": 8.0,
        "feasibility": 5.5,
        "novelty": 8.0,
        "market_timing": 7.0,
        "rationale": (
            "Models that can reliably detect and recover from their own "
            "errors would be a step-change in reliability.  RL-based "
            "approaches make this increasingly feasible."
        ),
    },
    "spatial_reasoning": {
        "impact": 7.0,
        "feasibility": 6.0,
        "novelty": 8.5,
        "market_timing": 6.5,
        "rationale": (
            "Spatial reasoning is under-served by current models.  "
            "Robotics, architecture, and manufacturing need it, but "
            "training data and benchmarks are scarce."
        ),
    },
    "temporal_reasoning": {
        "impact": 6.5,
        "feasibility": 6.5,
        "novelty": 9.0,
        "market_timing": 5.5,
        "rationale": (
            "Temporal reasoning is rarely benchmarked and mostly ignored "
            "in current model training.  The novelty is high but market "
            "pull is still forming."
        ),
    },
    "common_sense": {
        "impact": 7.5,
        "feasibility": 4.0,
        "novelty": 5.0,
        "market_timing": 6.0,
        "rationale": (
            "Common-sense reasoning has been a long-standing challenge. "
            "Gains here would improve every downstream application, but "
            "tractability remains low."
        ),
    },
    "mathematical_proof": {
        "impact": 7.0,
        "feasibility": 5.5,
        "novelty": 6.5,
        "market_timing": 7.0,
        "rationale": (
            "Formal mathematics and proof verification are well-funded "
            "research areas with clear benchmarks.  Competition is high."
        ),
    },
    "adversarial_robustness": {
        "impact": 8.5,
        "feasibility": 4.5,
        "novelty": 7.5,
        "market_timing": 8.5,
        "rationale": (
            "As reasoning models are deployed in production, adversarial "
            "attacks on reasoning chains become a real threat.  Security "
            "buyers are willing to pay for robustness guarantees."
        ),
    },
    "calibration": {
        "impact": 7.5,
        "feasibility": 6.0,
        "novelty": 7.5,
        "market_timing": 7.0,
        "rationale": (
            "Reliable confidence estimates are essential for human-AI "
            "teaming.  The gap widens as reasoning chains grow, creating "
            "a specific niche for calibration research."
        ),
    },
    "efficiency": {
        "impact": 9.0,
        "feasibility": 7.0,
        "novelty": 5.0,
        "market_timing": 9.5,
        "rationale": (
            "Inference cost is the #1 barrier to reasoning model adoption. "
            "Distillation and adaptive compute are tractable paths with "
            "massive commercial upside."
        ),
    },
    "multi_modal_reasoning": {
        "impact": 8.5,
        "feasibility": 5.0,
        "novelty": 8.0,
        "market_timing": 8.0,
        "rationale": (
            "Only one model (Gemini 2.5 Pro) has native multi-modal "
            "reasoning.  Demand from healthcare, security, and robotics "
            "is growing faster than supply."
        ),
    },
    "long_horizon_planning": {
        "impact": 8.0,
        "feasibility": 4.5,
        "novelty": 7.0,
        "market_timing": 7.5,
        "rationale": (
            "Autonomous agents require reliable multi-step planning.  "
            "The agentic AI trend makes this a high-priority gap to close."
        ),
    },
    "abstraction": {
        "impact": 8.5,
        "feasibility": 3.5,
        "novelty": 8.5,
        "market_timing": 6.5,
        "rationale": (
            "True abstraction and generalization remain the holy grail. "
            "ARC-AGI exposed this gap; solving it would redefine what "
            "models can do, but it's extremely difficult."
        ),
    },
    "compositionality": {
        "impact": 7.5,
        "feasibility": 5.0,
        "novelty": 7.5,
        "market_timing": 6.5,
        "rationale": (
            "Compositional generalization is key for real-world tasks that "
            "combine skills.  Modular architectures are a promising path."
        ),
    },
    "tool_use_reasoning": {
        "impact": 8.0,
        "feasibility": 6.5,
        "novelty": 7.0,
        "market_timing": 9.0,
        "rationale": (
            "The agentic AI wave makes principled tool-use reasoning "
            "immediately valuable.  This is one of the most actionable "
            "gaps with clear market demand."
        ),
    },
}

# Weights for composite score
_WEIGHTS = {
    "impact": 0.35,
    "feasibility": 0.20,
    "novelty": 0.20,
    "market_timing": 0.25,
}


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

def _compute_composite(scores: Dict[str, float]) -> float:
    """Return weighted composite score (0-10)."""
    return round(
        sum(scores.get(k, 0) * w for k, w in _WEIGHTS.items()), 2
    )


def _assign_tier(composite: float) -> str:
    """Map composite score to a priority tier."""
    if composite >= 7.5:
        return "Tier 1"
    if composite >= 6.0:
        return "Tier 2"
    return "Tier 3"


def score_opportunities(
    analysis: AnalysisResult | None = None,
    models: List[ReasoningModel] | None = None,
) -> OpportunityReport:
    """Score every gap finding by research/commercial opportunity value.

    If *analysis* is not supplied, a fresh gap analysis is run.
    """
    if analysis is None:
        if models is None:
            models = get_reasoning_models()
        analysis = analyze_gaps(models)

    opportunities: List[OpportunityScore] = []

    for finding in analysis.findings:
        rubric = _OPPORTUNITY_RUBRIC.get(finding.category)
        if rubric is None:
            continue

        dim_scores = {
            "impact": float(rubric["impact"]),
            "feasibility": float(rubric["feasibility"]),
            "novelty": float(rubric["novelty"]),
            "market_timing": float(rubric["market_timing"]),
        }
        composite = _compute_composite(dim_scores)

        opportunities.append(OpportunityScore(
            category=finding.category,
            title=finding.title,
            impact=dim_scores["impact"],
            feasibility=dim_scores["feasibility"],
            novelty=dim_scores["novelty"],
            market_timing=dim_scores["market_timing"],
            composite=composite,
            rationale=str(rubric.get("rationale", "")),
            priority_tier=_assign_tier(composite),
        ))

    # Sort by composite descending
    opportunities.sort(key=lambda o: -o.composite)

    # Build top-3 summary
    top3 = opportunities[:3]
    summary_parts = []
    for i, opp in enumerate(top3, 1):
        summary_parts.append(
            f"{i}. **{opp.title}** ({opp.priority_tier}, "
            f"composite {opp.composite}/10)"
        )
    top_3_summary = "\n".join(summary_parts) if summary_parts else ""

    return OpportunityReport(
        opportunities=opportunities,
        top_3_summary=top_3_summary,
    )
