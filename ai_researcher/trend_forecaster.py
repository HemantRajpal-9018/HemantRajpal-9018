"""Trend forecaster for reasoning AI.

Analyzes emerging trends in reasoning model research and development,
predicts upcoming capability shifts, and identifies market windows for
innovation.  This is a key differentiator — most gap-analysis tools look
backward; this module looks *forward*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ai_researcher.models_knowledge import ReasoningModel, get_reasoning_models


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Trend:
    """An observed or predicted trend in reasoning AI."""
    id: str
    title: str
    description: str
    status: str                     # "emerging", "accelerating", "maturing"
    confidence: float               # 0.0 – 1.0
    time_horizon: str               # e.g. "6-12 months", "1-2 years"
    supporting_evidence: List[str] = field(default_factory=list)
    affected_categories: List[str] = field(default_factory=list)
    market_implications: List[str] = field(default_factory=list)


@dataclass
class ForecastResult:
    """Complete output of a trend-forecasting run."""
    trends: List[Trend] = field(default_factory=list)
    model_count: int = 0
    horizon_summary: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trend catalogue – derived from model trajectory analysis
# ---------------------------------------------------------------------------

_TREND_CATALOGUE: List[Dict] = [
    # ------ emerging ------
    {
        "id": "adaptive_compute",
        "title": "Adaptive Compute Allocation for Reasoning",
        "description": (
            "Models will learn to dynamically adjust reasoning depth per "
            "query rather than using a fixed chain-of-thought budget.  "
            "OpenAI's configurable low/medium/high effort is a first step; "
            "true per-token adaptive allocation is next."
        ),
        "status": "emerging",
        "confidence": 0.85,
        "time_horizon": "6-12 months",
        "supporting_evidence": [
            "OpenAI o3 already offers low/medium/high reasoning effort",
            "Speculative decoding research enables early-exit strategies",
            "Cost pressure from enterprise customers demands efficiency",
        ],
        "affected_categories": ["efficiency", "calibration"],
        "market_implications": [
            "First mover on adaptive reasoning will dominate cost-sensitive markets",
            "Enterprise AI budgets will shift toward pay-per-reasoning-step pricing",
        ],
    },
    {
        "id": "reasoning_verification",
        "title": "Formal Verification of Reasoning Chains",
        "description": (
            "Tooling to verify whether a model's chain-of-thought is "
            "logically sound and faithful to its internal computation will "
            "become a distinct product category.  Current faithfulness is "
            "unverifiable — this is a massive opportunity."
        ),
        "status": "emerging",
        "confidence": 0.80,
        "time_horizon": "1-2 years",
        "supporting_evidence": [
            "Growing concern about 'unfaithful' reasoning in o-series models",
            "Open-source models (DeepSeek R1) expose traces but lack verification",
            "Enterprise compliance requirements demand auditable AI reasoning",
        ],
        "affected_categories": ["faithfulness", "adversarial_robustness"],
        "market_implications": [
            "Reasoning-audit-as-a-service will emerge as a compliance tool",
            "Regulated industries (finance, healthcare) will require verified chains",
        ],
    },
    {
        "id": "multimodal_cot",
        "title": "Native Multi-Modal Chain-of-Thought",
        "description": (
            "Reasoning will extend beyond text into interleaved text-image-"
            "audio-video chains.  Gemini 2.5 Pro leads, but true multi-modal "
            "reasoning — where the model reasons *about* an image mid-chain — "
            "is still nascent."
        ),
        "status": "emerging",
        "confidence": 0.78,
        "time_horizon": "6-12 months",
        "supporting_evidence": [
            "Gemini 2.5 Pro supports multi-modal input but reasoning is text-heavy",
            "Demand for visual reasoning in manufacturing, medical imaging, robotics",
            "Research on visual chain-of-thought (VCoT) is accelerating",
        ],
        "affected_categories": ["multi_modal_reasoning", "spatial_reasoning"],
        "market_implications": [
            "Medical/legal/engineering verticals will pay premium for visual reasoning",
            "Robotics and autonomous systems need spatial-reasoning breakthroughs",
        ],
    },

    # ------ accelerating ------
    {
        "id": "reasoning_distillation",
        "title": "Distillation of Reasoning into Smaller Models",
        "description": (
            "Large reasoning models will be distilled into smaller, faster "
            "variants that retain 80-90% of reasoning capability at a "
            "fraction of the cost.  DeepSeek's distilled R1 variants and "
            "Microsoft Phi-4-reasoning (14B) are early examples."
        ),
        "status": "accelerating",
        "confidence": 0.90,
        "time_horizon": "3-6 months",
        "supporting_evidence": [
            "DeepSeek R1 already has 1.5B-70B distilled variants",
            "Phi-4-reasoning achieves strong results at 14B parameters",
            "Edge deployment demands sub-10B reasoning models",
        ],
        "affected_categories": ["efficiency", "abstraction"],
        "market_implications": [
            "On-device reasoning will unlock privacy-sensitive applications",
            "Competition in small reasoning models will intensify rapidly",
        ],
    },
    {
        "id": "rl_for_reasoning",
        "title": "Reinforcement Learning as Primary Reasoning Training Signal",
        "description": (
            "RL (GRPO, PPO) is replacing supervised fine-tuning as the "
            "dominant approach for training reasoning.  DeepSeek R1, QwQ-32B, "
            "and Phi-4-reasoning all use RL.  This trend will accelerate as "
            "RL reward design matures."
        ),
        "status": "accelerating",
        "confidence": 0.92,
        "time_horizon": "3-6 months",
        "supporting_evidence": [
            "DeepSeek R1 trained with GRPO reinforcement learning",
            "QwQ-32B uses RL for reasoning capability",
            "Phi-4-reasoning relies on RL with curated reasoning data",
            "Open-source RL toolkits for reasoning are proliferating",
        ],
        "affected_categories": ["self_correction", "mathematical_proof"],
        "market_implications": [
            "RL expertise becomes a key hiring differentiator for AI labs",
            "Reward model design for reasoning is a niche consulting opportunity",
        ],
    },
    {
        "id": "agentic_reasoning",
        "title": "Reasoning Models as Autonomous Agents",
        "description": (
            "Reasoning models are increasingly used as the 'brain' of "
            "autonomous agents that plan, use tools, and execute multi-step "
            "workflows.  Claude 3.7 Sonnet and Gemini 2.5 Pro already "
            "target agentic use cases."
        ),
        "status": "accelerating",
        "confidence": 0.88,
        "time_horizon": "6-12 months",
        "supporting_evidence": [
            "Claude 3.7 Sonnet emphasizes agentic tool-use",
            "SWE-bench Verified scores track agentic coding ability",
            "TAU-bench and other agent benchmarks are gaining traction",
            "Enterprise demand for autonomous workflow agents is surging",
        ],
        "affected_categories": ["tool_use_reasoning", "long_horizon_planning"],
        "market_implications": [
            "Agent-as-a-service platforms will be a multi-billion dollar category",
            "Reasoning reliability becomes a safety-critical concern for agents",
        ],
    },

    # ------ maturing ------
    {
        "id": "open_source_reasoning",
        "title": "Open-Source Reasoning Models Reach Parity",
        "description": (
            "Open-source reasoning models are rapidly closing the gap with "
            "proprietary ones.  DeepSeek R1 matches o1; QwQ-32B competes "
            "with models 10x its size.  Within a year, open-source models "
            "will match frontier proprietary performance."
        ),
        "status": "accelerating",
        "confidence": 0.85,
        "time_horizon": "6-12 months",
        "supporting_evidence": [
            "DeepSeek R1 (open-source) is competitive with OpenAI o1",
            "QwQ-32B matches much larger proprietary models",
            "Phi-4-reasoning achieves strong results at 14B (MIT license)",
        ],
        "affected_categories": ["efficiency", "compositionality"],
        "market_implications": [
            "Proprietary moats will shift from model quality to data + tooling",
            "Enterprises will favor fine-tunable open models for domain-specific reasoning",
        ],
    },
    {
        "id": "safety_alignment_reasoning",
        "title": "Safety Alignment Tailored to Reasoning Models",
        "description": (
            "Standard RLHF-based alignment breaks down for reasoning models "
            "because the extended reasoning surface opens new attack vectors. "
            "Dedicated alignment techniques for chain-of-thought reasoning "
            "are an active research frontier."
        ),
        "status": "emerging",
        "confidence": 0.75,
        "time_horizon": "1-2 years",
        "supporting_evidence": [
            "DeepSeek R1 has limited safety alignment vs. commercial models",
            "Adversarial reasoning prompts can hijack CoT logic paths",
            "OpenAI o3 improves safety alignment but methodology is opaque",
        ],
        "affected_categories": ["adversarial_robustness", "faithfulness"],
        "market_implications": [
            "Safety-certified reasoning models will command premium pricing",
            "Regulatory frameworks will emerge requiring reasoning auditability",
        ],
    },
]


# ---------------------------------------------------------------------------
# Trend analysis logic
# ---------------------------------------------------------------------------

def _enrich_with_model_evidence(
    trends: List[Dict],
    models: List[ReasoningModel],
) -> None:
    """Augment trend evidence with model-specific data points."""
    for trend in trends:
        cats = trend.get("affected_categories", [])
        for model in models:
            for strength in model.strengths:
                s_lower = strength.lower()
                for cat in cats:
                    if cat in s_lower or any(
                        kw in s_lower
                        for kw in cat.replace("_", " ").split()
                    ):
                        evidence = f"{model.name}: {strength}"
                        if evidence not in trend["supporting_evidence"]:
                            trend["supporting_evidence"].append(evidence)
                            break


def forecast_trends(
    models: List[ReasoningModel] | None = None,
) -> ForecastResult:
    """Analyze emerging trends and produce a forecast.

    Returns a ``ForecastResult`` with trend objects sorted by confidence
    (descending).
    """
    if models is None:
        models = get_reasoning_models()

    # Deep-copy the catalogue so enrichment doesn't mutate the template
    import copy
    catalogue = copy.deepcopy(_TREND_CATALOGUE)

    _enrich_with_model_evidence(catalogue, models)

    trends = [
        Trend(
            id=t["id"],
            title=t["title"],
            description=t["description"],
            status=t["status"],
            confidence=t["confidence"],
            time_horizon=t["time_horizon"],
            supporting_evidence=t["supporting_evidence"],
            affected_categories=t["affected_categories"],
            market_implications=t["market_implications"],
        )
        for t in catalogue
    ]

    # Sort by confidence descending
    trends.sort(key=lambda t: -t.confidence)

    horizon_summary: Dict[str, int] = {}
    for t in trends:
        horizon_summary[t.time_horizon] = (
            horizon_summary.get(t.time_horizon, 0) + 1
        )

    return ForecastResult(
        trends=trends,
        model_count=len(models),
        horizon_summary=horizon_summary,
    )
