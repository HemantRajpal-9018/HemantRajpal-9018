"""Gap analysis engine for reasoning models.

Compares models across standardised gap categories, identifies systemic
weaknesses, and produces structured findings that the report generator can
render.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ai_researcher.models_knowledge import (
    REASONING_GAP_CATEGORIES,
    ReasoningModel,
    get_reasoning_models,
)


# ---------------------------------------------------------------------------
# Data classes for analysis results
# ---------------------------------------------------------------------------

@dataclass
class GapFinding:
    """A single identified gap in reasoning model capabilities."""
    category: str
    severity: str            # "critical", "high", "medium", "low"
    title: str
    description: str
    affected_models: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    research_directions: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Complete output of a gap analysis run."""
    findings: List[GapFinding] = field(default_factory=list)
    model_count: int = 0
    category_coverage: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Keyword mapping – maps gap categories to weakness keywords
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "faithfulness": [
        "faithfulness", "opaque", "hidden", "unfaithful", "traces",
        "transparent",
    ],
    "hallucination": [
        "hallucin", "factual error", "fabricat", "confabul",
    ],
    "self_correction": [
        "self-correct", "detect", "fix own", "circular",
        "loop", "over-reason",
    ],
    "spatial_reasoning": [
        "spatial", "2-d", "3-d", "geometric", "visual puzzle",
    ],
    "temporal_reasoning": [
        "temporal", "time-based", "scheduling", "chronolog",
    ],
    "common_sense": [
        "common-sense", "common sense", "everyday", "world knowledge",
    ],
    "mathematical_proof": [
        "math", "proof", "formal", "research-level",
    ],
    "adversarial_robustness": [
        "adversarial", "robust", "jailbreak", "safety",
        "alignment", "cautious",
    ],
    "calibration": [
        "calibrat", "confidence", "overconfiden", "uncertain",
    ],
    "efficiency": [
        "cost", "expensive", "latency", "compute", "efficient",
        "inference", "budget",
    ],
    "multi_modal_reasoning": [
        "multi-modal", "multimodal", "modalities", "image",
        "video", "audio", "vision",
    ],
    "long_horizon_planning": [
        "planning", "long", "multi-step", "horizon", "dependencies",
    ],
    "abstraction": [
        "abstract", "generali", "novel", "distribution",
        "shortcut",
    ],
    "compositionality": [
        "compos", "combin", "novel tasks", "creative",
    ],
    "tool_use_reasoning": [
        "tool", "agent", "browsing", "external",
    ],
}


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _match_category(text: str, category: str) -> bool:
    """Return True if *text* contains any keyword for *category*."""
    keywords = _CATEGORY_KEYWORDS.get(category, [])
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def _severity_for_category_count(count: int, total_models: int) -> str:
    """Heuristic severity based on how many models share the gap."""
    ratio = count / max(total_models, 1)
    if ratio >= 0.7:
        return "critical"
    if ratio >= 0.5:
        return "high"
    if ratio >= 0.3:
        return "medium"
    return "low"


def _collect_category_hits(
    models: List[ReasoningModel],
) -> Dict[str, List[Tuple[str, str]]]:
    """For every gap category, collect (model_name, weakness_text) pairs."""
    hits: Dict[str, List[Tuple[str, str]]] = {
        cat: [] for cat in REASONING_GAP_CATEGORIES
    }
    for model in models:
        for weakness in model.weaknesses:
            for cat in REASONING_GAP_CATEGORIES:
                if _match_category(weakness, cat):
                    hits[cat].append((model.name, weakness))
    return hits


# ---------------------------------------------------------------------------
# Cross-model systemic gap descriptions
# ---------------------------------------------------------------------------

_SYSTEMIC_GAPS: Dict[str, Dict] = {
    "faithfulness": {
        "title": "Reasoning Chain Faithfulness Remains Unverifiable",
        "description": (
            "Most reasoning models produce chain-of-thought traces, yet there "
            "is no reliable method to verify that the visible reasoning "
            "faithfully represents the model's internal computation. OpenAI's "
            "o-series hides traces entirely; open-source models expose them "
            "but faithfulness is still unproven."
        ),
        "research_directions": [
            "Develop interpretability methods for verifying reasoning trace faithfulness",
            "Create benchmarks that measure trace-to-computation alignment",
            "Explore causal probing of reasoning tokens vs. final answers",
        ],
    },
    "hallucination": {
        "title": "Multi-Step Reasoning Amplifies Hallucination Risk",
        "description": (
            "Extended reasoning chains compound small factual errors across "
            "steps.  Models that score well on single-hop QA still produce "
            "hallucinated intermediate conclusions when reasoning depth "
            "increases, especially in scientific and medical domains."
        ),
        "research_directions": [
            "Step-level fact verification during chain-of-thought generation",
            "Retrieval-augmented reasoning to ground intermediate steps",
            "Hallucination detection fine-tuned for multi-step contexts",
        ],
    },
    "self_correction": {
        "title": "Self-Correction Is Shallow and Unreliable",
        "description": (
            "Current reasoning models can sometimes identify errors when "
            "explicitly prompted, but spontaneous self-correction during "
            "generation is rare.  Several models (DeepSeek R1, QwQ-32B) are "
            "known to enter circular reasoning loops rather than correcting "
            "course."
        ),
        "research_directions": [
            "Train explicit error-detection heads that trigger backtracking",
            "Develop RL reward signals for successful self-correction",
            "Investigate tree-of-thought search with pruning of erroneous branches",
        ],
    },
    "spatial_reasoning": {
        "title": "Spatial and Geometric Reasoning Is a Consistent Weak Spot",
        "description": (
            "Despite strong performance on symbolic math, reasoning models "
            "struggle with 2-D and 3-D spatial tasks, navigation, and "
            "geometric proofs that require mental rotation or spatial "
            "manipulation."
        ),
        "research_directions": [
            "Integrate spatial reasoning pre-training with synthetic 3-D data",
            "Combine language-based reasoning with vision encoders for spatial tasks",
            "Develop dedicated spatial reasoning benchmarks beyond ARC-AGI",
        ],
    },
    "efficiency": {
        "title": "Reasoning Compute Cost Scales Poorly",
        "description": (
            "Generating extended reasoning traces is expensive.  High-effort "
            "modes (OpenAI o3 high, Gemini thinking tokens) can cost 10-50x "
            "more than standard generation.  There is no adaptive mechanism "
            "to allocate reasoning effort proportional to task difficulty."
        ),
        "research_directions": [
            "Adaptive compute allocation – skip reasoning for easy tasks",
            "Speculative reasoning with early exit when confidence is high",
            "Distillation of reasoning into fewer, more efficient steps",
        ],
    },
    "multi_modal_reasoning": {
        "title": "Multi-Modal Reasoning Lags Behind Text-Only Performance",
        "description": (
            "Only Gemini 2.5 Pro natively reasons across modalities.  Most "
            "reasoning models are text-only or treat vision as a separate "
            "perception step disconnected from the reasoning chain.  Joint "
            "reasoning over text + image + structured data remains a gap."
        ),
        "research_directions": [
            "Unified multi-modal chain-of-thought architectures",
            "Benchmarks for interleaved text-image reasoning",
            "Cross-modal consistency verification in reasoning traces",
        ],
    },
    "calibration": {
        "title": "Confidence Calibration Degrades Under Extended Reasoning",
        "description": (
            "Reasoning models tend to become more confident as reasoning "
            "chains grow, even when the final answer is incorrect.  Extended "
            "thinking can create an illusion of certainty that makes "
            "calibration worse, not better."
        ),
        "research_directions": [
            "Calibrate per-step confidence alongside final-answer confidence",
            "Train models to express uncertainty explicitly in reasoning traces",
            "Develop inference-time calibration methods specific to CoT models",
        ],
    },
    "adversarial_robustness": {
        "title": "Reasoning Models Are Vulnerable to Adversarial Reasoning Prompts",
        "description": (
            "Adversarial prompts that embed misleading reasoning steps can "
            "cause models to follow incorrect logic paths.  The extended "
            "reasoning surface increases the attack vector compared to "
            "standard generation models."
        ),
        "research_directions": [
            "Adversarial training specifically targeting reasoning chains",
            "Formal verification of reasoning steps against logical rules",
            "Red-teaming benchmarks for reasoning model robustness",
        ],
    },
    "long_horizon_planning": {
        "title": "Long-Horizon Planning With Dependencies Is Unreliable",
        "description": (
            "Models perform well on problems requiring 3-5 reasoning steps "
            "but degrade significantly at 10+ steps.  Tasks requiring "
            "dependency management (e.g., project planning, multi-stage "
            "proofs) expose failures in maintaining coherent plans."
        ),
        "research_directions": [
            "Hierarchical reasoning with sub-goal decomposition",
            "External memory / scratchpad for tracking dependencies",
            "Evaluation benchmarks for 20+ step reasoning chains",
        ],
    },
    "compositionality": {
        "title": "Compositional Generalization Remains Weak",
        "description": (
            "Reasoning models struggle to compose individually mastered skills "
            "to solve novel compound tasks.  A model that solves algebra and "
            "geometry separately may fail when a problem requires both "
            "simultaneously."
        ),
        "research_directions": [
            "Compositional reasoning benchmarks (e.g., COGS-extended)",
            "Modular reasoning architectures with skill routing",
            "Curriculum learning that emphasizes skill composition",
        ],
    },
    "abstraction": {
        "title": "Abstraction and Out-of-Distribution Generalization Are Limited",
        "description": (
            "Models trained primarily on existing math/code datasets rely on "
            "pattern matching rather than true abstraction.  Performance "
            "drops sharply on tasks that require novel abstractions not "
            "present in training data."
        ),
        "research_directions": [
            "Program synthesis approaches to reasoning",
            "Few-shot abstract reasoning evaluations (ARC-AGI variants)",
            "Meta-learning for rapid adaptation to novel problem types",
        ],
    },
    "common_sense": {
        "title": "Common-Sense Reasoning Has Not Kept Pace With Logical Gains",
        "description": (
            "While logical and mathematical reasoning has improved "
            "dramatically, everyday common-sense reasoning (physical "
            "intuition, social norms, causal relationships) remains "
            "surprisingly fragile in reasoning models."
        ),
        "research_directions": [
            "Integrate world-model pre-training with reasoning fine-tuning",
            "Common-sense reasoning benchmarks for CoT models",
            "Grounding reasoning in embodied or simulated environments",
        ],
    },
    "temporal_reasoning": {
        "title": "Temporal Reasoning and Scheduling Are Underexplored",
        "description": (
            "Few reasoning models are explicitly evaluated on temporal tasks "
            "such as scheduling, timeline construction, or reasoning about "
            "event ordering under constraints."
        ),
        "research_directions": [
            "Temporal reasoning benchmark suites",
            "Integration of constraint satisfaction with CoT reasoning",
            "Calendar/scheduling task evaluations for reasoning agents",
        ],
    },
    "tool_use_reasoning": {
        "title": "Tool-Use Reasoning Lacks Principled Decision-Making",
        "description": (
            "While several models support tool use, the decision of *when* "
            "and *which* tool to invoke is largely heuristic.  Models often "
            "either over-rely on tools or fail to use them when beneficial."
        ),
        "research_directions": [
            "Formal decision framework for tool invocation during reasoning",
            "Benchmarks comparing tool-use reasoning across models",
            "Self-aware reasoning that estimates when internal knowledge is insufficient",
        ],
    },
    "mathematical_proof": {
        "title": "Mathematical Proof Verification Remains Fragile",
        "description": (
            "Models struggle with rigorous mathematical proofs, especially "
            "on the hardest Olympiad-tier problems.  Even top performers "
            "exhibit proof steps that are syntactically plausible but "
            "logically flawed, and verification of multi-step proofs is "
            "inconsistent."
        ),
        "research_directions": [
            "Formal proof assistant integration (Lean/Coq) for step verification",
            "Curriculum learning on progressively harder proof benchmarks",
            "Proof-sketch → formal-proof translation pipelines",
        ],
    },
}


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_gaps(
    models: List[ReasoningModel] | None = None,
) -> AnalysisResult:
    """Run a full gap analysis across the supplied (or default) model set.

    Returns an ``AnalysisResult`` with one ``GapFinding`` per gap category
    that has evidence from at least one model.
    """
    if models is None:
        models = get_reasoning_models()

    hits = _collect_category_hits(models)
    findings: List[GapFinding] = []

    for cat in REASONING_GAP_CATEGORIES:
        matched = hits[cat]
        if not matched:
            continue

        affected = sorted({name for name, _ in matched})
        evidence = sorted({text for _, text in matched})
        severity = _severity_for_category_count(len(affected), len(models))

        systemic = _SYSTEMIC_GAPS.get(cat, {})

        findings.append(
            GapFinding(
                category=cat,
                severity=severity,
                title=systemic.get("title", cat.replace("_", " ").title()),
                description=systemic.get("description", ""),
                affected_models=affected,
                evidence=evidence,
                research_directions=systemic.get("research_directions", []),
            )
        )

    # Sort by severity: critical > high > medium > low
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    findings.sort(key=lambda f: severity_order.get(f.severity, 4))

    category_coverage = {cat: len(hits[cat]) for cat in REASONING_GAP_CATEGORIES}

    return AnalysisResult(
        findings=findings,
        model_count=len(models),
        category_coverage=category_coverage,
    )
