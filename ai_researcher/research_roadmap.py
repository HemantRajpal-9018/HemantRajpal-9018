"""Research roadmap generator.

Transforms gap analysis findings and trend forecasts into a prioritized,
timeline-based research agenda with resource estimates.  This goes beyond
"gaps exist" to "here's what to build, when, and what it costs" – a key
differentiator over competitor research tools that only describe problems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ai_researcher.gap_analyzer import AnalysisResult, analyze_gaps
from ai_researcher.innovation_scorer import (
    OpportunityReport,
    score_opportunities,
)
from ai_researcher.models_knowledge import ReasoningModel, get_reasoning_models
from ai_researcher.trend_forecaster import ForecastResult, forecast_trends


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RoadmapMilestone:
    """A single milestone in a research roadmap."""
    id: str
    title: str
    description: str
    phase: str                        # "Phase 1", "Phase 2", "Phase 3"
    timeline: str                     # e.g. "0-3 months", "3-6 months"
    priority: str                     # "critical", "high", "medium"
    effort_level: str                 # "small", "medium", "large"
    prerequisites: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)
    related_gaps: List[str] = field(default_factory=list)
    related_trends: List[str] = field(default_factory=list)


@dataclass
class ResearchRoadmap:
    """Complete research roadmap with phased milestones."""
    milestones: List[RoadmapMilestone] = field(default_factory=list)
    total_phases: int = 3
    phase_summary: Dict[str, int] = field(default_factory=dict)
    executive_summary: str = ""


# ---------------------------------------------------------------------------
# Milestone templates keyed by gap category
# ---------------------------------------------------------------------------

_MILESTONE_TEMPLATES: Dict[str, Dict] = {
    "efficiency": {
        "title": "Adaptive Compute Reasoning Engine",
        "description": (
            "Build a reasoning system that dynamically allocates compute "
            "based on query difficulty – easy questions get fast, cheap "
            "answers; hard ones get deep reasoning chains."
        ),
        "phase": "Phase 1",
        "timeline": "0-3 months",
        "priority": "critical",
        "effort_level": "large",
        "deliverables": [
            "Difficulty classifier (easy/medium/hard) with >90% accuracy",
            "Adaptive routing layer between fast and deep reasoning paths",
            "Cost-per-query reduction of 40-60% on mixed workloads",
        ],
        "success_metrics": [
            "Cost reduction ≥40% vs fixed-effort reasoning",
            "No quality degradation on hard tasks (within 2% of full effort)",
            "p95 latency ≤2s for easy queries",
        ],
        "related_trends": ["adaptive_compute", "reasoning_distillation"],
    },
    "faithfulness": {
        "title": "Reasoning Chain Verification Framework",
        "description": (
            "Develop a verification layer that audits chain-of-thought "
            "traces for logical consistency, factual grounding, and "
            "faithfulness to the model's actual computation."
        ),
        "phase": "Phase 1",
        "timeline": "0-3 months",
        "priority": "critical",
        "effort_level": "large",
        "deliverables": [
            "Logical consistency checker for CoT steps",
            "Faithfulness scoring model trained on trace-vs-answer pairs",
            "Integration API for audit-as-a-service deployment",
        ],
        "success_metrics": [
            "Detect >80% of unfaithful reasoning chains on test set",
            "False positive rate <10%",
            "Process 1000 traces/minute at scale",
        ],
        "related_trends": ["reasoning_verification", "safety_alignment_reasoning"],
    },
    "hallucination": {
        "title": "Step-Level Hallucination Detection Pipeline",
        "description": (
            "Build a pipeline that checks each intermediate reasoning step "
            "against a retrieval-augmented knowledge base, flagging "
            "hallucinated facts before they propagate through the chain."
        ),
        "phase": "Phase 1",
        "timeline": "0-3 months",
        "priority": "critical",
        "effort_level": "large",
        "deliverables": [
            "Step-by-step fact checker with RAG integration",
            "Hallucination confidence score per reasoning step",
            "Alert system for high-confidence hallucination detection",
        ],
        "success_metrics": [
            "Catch >70% of multi-step hallucinations",
            "Latency overhead <500ms per reasoning chain",
            "Precision ≥85% on flagged hallucinations",
        ],
        "related_trends": ["reasoning_verification"],
    },
    "self_correction": {
        "title": "Self-Correction via Error Detection Heads",
        "description": (
            "Train lightweight error-detection heads that monitor reasoning "
            "chains in real-time and trigger backtracking when errors are "
            "detected, breaking circular reasoning loops."
        ),
        "phase": "Phase 2",
        "timeline": "3-6 months",
        "priority": "high",
        "effort_level": "large",
        "deliverables": [
            "Error detection classifier for reasoning steps",
            "Backtracking mechanism with alternative path generation",
            "Circular reasoning loop detector and breaker",
        ],
        "success_metrics": [
            "Reduce circular reasoning loops by >60%",
            "Improve answer accuracy by 5-10% on error-prone tasks",
            "Self-correction latency <1s per backtrack",
        ],
        "related_trends": ["rl_for_reasoning"],
    },
    "multi_modal_reasoning": {
        "title": "Unified Multi-Modal Chain-of-Thought",
        "description": (
            "Extend reasoning chains to natively interleave text, image, "
            "and structured data analysis within a single coherent "
            "chain-of-thought."
        ),
        "phase": "Phase 2",
        "timeline": "3-6 months",
        "priority": "high",
        "effort_level": "large",
        "deliverables": [
            "Multi-modal reasoning architecture (text + image + table)",
            "Visual chain-of-thought (VCoT) generation pipeline",
            "Cross-modal consistency verification module",
        ],
        "success_metrics": [
            "Score within 5% of text-only on text benchmarks",
            "Beat GPT-4V on visual reasoning benchmarks by >10%",
            "Handle interleaved text-image inputs end-to-end",
        ],
        "related_trends": ["multimodal_cot"],
    },
    "adversarial_robustness": {
        "title": "Adversarial Reasoning Robustness Suite",
        "description": (
            "Create a red-teaming framework specifically designed for "
            "reasoning models, with adversarial prompt templates that "
            "target chain-of-thought vulnerabilities."
        ),
        "phase": "Phase 2",
        "timeline": "3-6 months",
        "priority": "high",
        "effort_level": "medium",
        "deliverables": [
            "Adversarial prompt library (500+ reasoning-specific attacks)",
            "Automated red-teaming pipeline for reasoning chains",
            "Robustness score dashboard with trend tracking",
        ],
        "success_metrics": [
            "Identify >90% of known reasoning attack vectors",
            "Reduce successful adversarial attacks by >50% after hardening",
            "Full test run completes in <30 minutes",
        ],
        "related_trends": ["safety_alignment_reasoning"],
    },
    "long_horizon_planning": {
        "title": "Hierarchical Planning with External Memory",
        "description": (
            "Build a planning layer that decomposes long-horizon tasks "
            "into sub-goals with dependency tracking, using external "
            "memory to maintain coherence across 20+ reasoning steps."
        ),
        "phase": "Phase 3",
        "timeline": "6-12 months",
        "priority": "medium",
        "effort_level": "large",
        "deliverables": [
            "Sub-goal decomposition module",
            "External memory / scratchpad for dependency tracking",
            "Re-planning mechanism when sub-goals fail",
        ],
        "success_metrics": [
            "Handle 20+ step plans with >80% completion rate",
            "Dependency tracking accuracy >90%",
            "Performance within 10% of specialized planning systems",
        ],
        "related_trends": ["agentic_reasoning"],
    },
    "tool_use_reasoning": {
        "title": "Principled Tool Invocation Framework",
        "description": (
            "Develop a decision framework that formally reasons about "
            "when to use external tools vs. internal knowledge, with "
            "cost-benefit analysis per tool call."
        ),
        "phase": "Phase 2",
        "timeline": "3-6 months",
        "priority": "high",
        "effort_level": "medium",
        "deliverables": [
            "Tool-use decision model (invoke vs. skip classifier)",
            "Cost-benefit analyzer for each available tool",
            "Integration framework for 10+ common tool types",
        ],
        "success_metrics": [
            "Correct tool-use decisions >85% of the time",
            "Reduce unnecessary tool calls by >40%",
            "Support plug-and-play tool registration",
        ],
        "related_trends": ["agentic_reasoning"],
    },
    "calibration": {
        "title": "Per-Step Confidence Calibration Module",
        "description": (
            "Build calibration layers that produce reliable confidence "
            "estimates at each reasoning step, not just the final answer."
        ),
        "phase": "Phase 3",
        "timeline": "6-12 months",
        "priority": "medium",
        "effort_level": "medium",
        "deliverables": [
            "Step-level confidence estimator",
            "Calibration training pipeline using temperature scaling",
            "Uncertainty visualization for human-AI teaming",
        ],
        "success_metrics": [
            "Expected calibration error (ECE) <5% per step",
            "Final-answer calibration ECE <3%",
            "Human trust rating improvement ≥15% in user studies",
        ],
        "related_trends": ["adaptive_compute"],
    },
    "spatial_reasoning": {
        "title": "Spatial Reasoning Pre-training Module",
        "description": (
            "Pre-train on synthetic 3-D and geometric data to close the "
            "spatial reasoning gap that plagues all current models."
        ),
        "phase": "Phase 3",
        "timeline": "6-12 months",
        "priority": "medium",
        "effort_level": "large",
        "deliverables": [
            "Synthetic 3-D spatial reasoning dataset (100K+ tasks)",
            "Spatial-aware fine-tuning recipe for reasoning models",
            "Benchmark suite for 2-D/3-D spatial tasks",
        ],
        "success_metrics": [
            "Improve ARC-AGI-like scores by >15%",
            "Handle mental rotation tasks with >70% accuracy",
            "Generalize to unseen spatial problem types",
        ],
        "related_trends": ["multimodal_cot"],
    },
    "compositionality": {
        "title": "Compositional Skill Router",
        "description": (
            "Build a modular architecture that routes compound problems "
            "to specialized skill modules and composes their outputs."
        ),
        "phase": "Phase 3",
        "timeline": "6-12 months",
        "priority": "medium",
        "effort_level": "large",
        "deliverables": [
            "Skill decomposition classifier",
            "Modular reasoning router with skill registry",
            "Composition layer that merges sub-solutions",
        ],
        "success_metrics": [
            "Solve compound tasks 20%+ better than monolithic models",
            "Support pluggable skill modules (>5 skill types)",
            "Composition overhead <10% of total reasoning time",
        ],
        "related_trends": ["reasoning_distillation"],
    },
    "abstraction": {
        "title": "Meta-Learning for Novel Abstractions",
        "description": (
            "Apply meta-learning techniques so reasoning models can "
            "rapidly adapt to novel problem types with minimal examples."
        ),
        "phase": "Phase 3",
        "timeline": "6-12 months",
        "priority": "medium",
        "effort_level": "large",
        "deliverables": [
            "Few-shot abstract reasoning evaluation suite",
            "Meta-learning training pipeline for reasoning models",
            "Generalization benchmark with held-out problem types",
        ],
        "success_metrics": [
            "5-shot accuracy >60% on novel problem types",
            "ARC-AGI improvement >10% from meta-learning alone",
            "Adaptation time <5 minutes for new problem families",
        ],
        "related_trends": ["rl_for_reasoning"],
    },
}


# ---------------------------------------------------------------------------
# Roadmap generation
# ---------------------------------------------------------------------------

def generate_roadmap(
    analysis: AnalysisResult | None = None,
    trends: ForecastResult | None = None,
    opportunities: OpportunityReport | None = None,
    models: List[ReasoningModel] | None = None,
) -> ResearchRoadmap:
    """Generate a prioritized research roadmap from gap analysis.

    If analysis / trends / opportunities are not supplied, they are
    computed automatically.
    """
    if models is None:
        models = get_reasoning_models()
    if analysis is None:
        analysis = analyze_gaps(models)
    if trends is None:
        trends = forecast_trends(models)
    if opportunities is None:
        opportunities = score_opportunities(analysis, models)

    # Build priority map from opportunity scores
    priority_map: Dict[str, float] = {
        opp.category: opp.composite
        for opp in opportunities.opportunities
    }

    milestones: List[RoadmapMilestone] = []
    for finding in analysis.findings:
        template = _MILESTONE_TEMPLATES.get(finding.category)
        if template is None:
            continue

        ms = RoadmapMilestone(
            id=finding.category,
            title=template["title"],
            description=template["description"],
            phase=template["phase"],
            timeline=template["timeline"],
            priority=template["priority"],
            effort_level=template["effort_level"],
            deliverables=template.get("deliverables", []),
            success_metrics=template.get("success_metrics", []),
            related_gaps=[finding.category],
            related_trends=template.get("related_trends", []),
        )
        milestones.append(ms)

    # Sort: Phase 1 first, then by opportunity composite descending
    phase_order = {"Phase 1": 0, "Phase 2": 1, "Phase 3": 2}
    milestones.sort(key=lambda m: (
        phase_order.get(m.phase, 9),
        -priority_map.get(m.id, 0),
    ))

    # Phase summary
    phase_summary: Dict[str, int] = {}
    for ms in milestones:
        phase_summary[ms.phase] = phase_summary.get(ms.phase, 0) + 1

    # Executive summary
    p1_count = phase_summary.get("Phase 1", 0)
    total = len(milestones)
    executive_summary = (
        f"Research roadmap with {total} milestones across 3 phases. "
        f"Phase 1 ({p1_count} critical items) targets 0-3 months for "
        f"maximum market impact. Prioritized by innovation opportunity "
        f"scoring – highest-ROI gaps addressed first."
    )

    return ResearchRoadmap(
        milestones=milestones,
        total_phases=3,
        phase_summary=phase_summary,
        executive_summary=executive_summary,
    )
