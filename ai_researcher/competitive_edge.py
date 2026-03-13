"""Competitive edge analytics for reasoning models.

Provides three capabilities that set this AI researcher apart from
competitors:

1. **Research Velocity** – measures how fast each provider is iterating
   (release cadence, benchmark improvement trajectory).
2. **Risk Assessment** – scores vendor lock-in, model deprecation, and
   API stability risks for each model/provider.
3. **Competitive Landscape Map** – identifies which models compete in
   the same segments and quantifies overlap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ai_researcher.models_knowledge import ReasoningModel, get_reasoning_models


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProviderVelocity:
    """Research velocity metrics for a single provider."""
    provider: str
    model_count: int
    latest_release: str
    release_span_months: int          # months between earliest & latest model
    avg_benchmark_score: float        # average across all benchmarks
    open_source_ratio: float          # fraction of models that are open-source
    velocity_rating: str              # "fast", "moderate", "slow"
    momentum_signals: List[str] = field(default_factory=list)


@dataclass
class VelocityReport:
    """Complete research velocity analysis."""
    providers: List[ProviderVelocity] = field(default_factory=list)
    fastest_provider: str = ""
    summary: str = ""


@dataclass
class RiskScore:
    """Risk assessment for a single model."""
    model_name: str
    provider: str
    vendor_lock_in: float       # 0-10 (10 = highest risk)
    deprecation_risk: float     # 0-10
    api_stability: float        # 0-10 (10 = most stable)
    ecosystem_maturity: float   # 0-10 (10 = most mature)
    composite_risk: float       # weighted aggregate (lower = safer)
    risk_tier: str              # "low", "medium", "high"
    mitigation_notes: List[str] = field(default_factory=list)


@dataclass
class RiskReport:
    """Complete risk assessment across all models."""
    scores: List[RiskScore] = field(default_factory=list)
    safest_model: str = ""
    riskiest_model: str = ""
    summary: str = ""


@dataclass
class SegmentOverlap:
    """Two models competing in the same segment."""
    segment: str
    model_a: str
    model_b: str
    overlap_score: float  # 0-1: how much they compete


@dataclass
class LandscapeReport:
    """Competitive landscape mapping."""
    segments: Dict[str, List[str]] = field(default_factory=dict)
    overlaps: List[SegmentOverlap] = field(default_factory=list)
    underserved_segments: List[str] = field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# Release date helpers
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    "01": 1, "02": 2, "03": 3, "04": 4, "05": 5, "06": 6,
    "07": 7, "08": 8, "09": 9, "10": 10, "11": 11, "12": 12,
}


def _parse_release_month(date_str: str) -> int:
    """Convert 'YYYY-MM' to an integer number of months since epoch."""
    parts = date_str.split("-")
    year = int(parts[0])
    month = int(parts[1]) if len(parts) > 1 else 1
    return year * 12 + month


# ---------------------------------------------------------------------------
# Research velocity
# ---------------------------------------------------------------------------

_VELOCITY_THRESHOLDS = {"fast": 3, "moderate": 2}  # model count thresholds


def _compute_provider_velocity(
    provider: str, models: List[ReasoningModel],
) -> ProviderVelocity:
    """Compute velocity metrics for one provider."""
    months = [_parse_release_month(m.release_date) for m in models]
    span = max(months) - min(months) if len(months) > 1 else 0

    all_scores = [
        bs.score for m in models for bs in m.benchmarks
    ]
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

    oss_ratio = sum(1 for m in models if m.open_source) / len(models)

    # Velocity rating heuristic
    if len(models) >= _VELOCITY_THRESHOLDS["fast"]:
        rating = "fast"
    elif len(models) >= _VELOCITY_THRESHOLDS["moderate"]:
        rating = "moderate"
    else:
        rating = "slow"

    # Momentum signals
    signals: List[str] = []
    if len(models) >= 3:
        signals.append(f"{len(models)} models released – rapid iteration")
    if oss_ratio > 0.5:
        signals.append("Majority open-source – community momentum")
    if avg_score >= 85:
        signals.append(f"High avg benchmark score ({avg_score:.1f}) – frontier quality")
    latest = max(models, key=lambda m: _parse_release_month(m.release_date))
    if _parse_release_month(latest.release_date) >= _parse_release_month("2025-02"):
        signals.append(f"Recent release ({latest.release_date}) – actively investing")

    return ProviderVelocity(
        provider=provider,
        model_count=len(models),
        latest_release=latest.release_date,
        release_span_months=span,
        avg_benchmark_score=round(avg_score, 1),
        open_source_ratio=round(oss_ratio, 2),
        velocity_rating=rating,
        momentum_signals=signals,
    )


def analyze_velocity(
    models: List[ReasoningModel] | None = None,
) -> VelocityReport:
    """Analyze research velocity across all providers."""
    if models is None:
        models = get_reasoning_models()

    by_provider: Dict[str, List[ReasoningModel]] = {}
    for m in models:
        by_provider.setdefault(m.provider, []).append(m)

    providers = [
        _compute_provider_velocity(p, ms)
        for p, ms in sorted(by_provider.items())
    ]
    providers.sort(key=lambda v: (-v.model_count, -v.avg_benchmark_score))

    fastest = providers[0].provider if providers else ""
    summary = (
        f"Analyzed {len(providers)} providers across {len(models)} models. "
        f"Fastest-iterating provider: {fastest}."
    )

    return VelocityReport(
        providers=providers,
        fastest_provider=fastest,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Risk assessment
# ---------------------------------------------------------------------------

_RISK_PROFILES: Dict[str, Dict[str, float | List[str]]] = {
    # provider-level defaults (model-specific overrides below)
    "OpenAI": {
        "vendor_lock_in": 7.0,
        "deprecation_risk": 4.0,
        "api_stability": 8.5,
        "ecosystem_maturity": 9.0,
        "mitigations": [
            "Multi-provider abstraction layer recommended",
            "Strong API stability track record",
        ],
    },
    "DeepSeek": {
        "vendor_lock_in": 2.0,
        "deprecation_risk": 5.0,
        "api_stability": 5.0,
        "ecosystem_maturity": 5.0,
        "mitigations": [
            "Open-source – self-host to avoid vendor dependency",
            "Community-maintained forks provide continuity",
        ],
    },
    "Google DeepMind": {
        "vendor_lock_in": 7.5,
        "deprecation_risk": 3.5,
        "api_stability": 8.0,
        "ecosystem_maturity": 8.5,
        "mitigations": [
            "GCP lock-in risk – consider multi-cloud strategy",
            "Google has strong backward-compat commitment for Vertex AI",
        ],
    },
    "Anthropic": {
        "vendor_lock_in": 6.5,
        "deprecation_risk": 3.0,
        "api_stability": 8.0,
        "ecosystem_maturity": 7.5,
        "mitigations": [
            "Available through AWS Bedrock and GCP – reduces single-vendor risk",
            "Strong versioning practices",
        ],
    },
    "Alibaba Qwen": {
        "vendor_lock_in": 2.0,
        "deprecation_risk": 5.5,
        "api_stability": 4.5,
        "ecosystem_maturity": 4.0,
        "mitigations": [
            "Open-source (Apache 2.0) – self-host to eliminate vendor risk",
            "Growing but less mature community ecosystem",
        ],
    },
    "xAI": {
        "vendor_lock_in": 8.0,
        "deprecation_risk": 6.0,
        "api_stability": 4.0,
        "ecosystem_maturity": 3.0,
        "mitigations": [
            "Newest entrant – limited track record",
            "API access is restricted – high switching cost if relied upon",
        ],
    },
    "Microsoft": {
        "vendor_lock_in": 3.0,
        "deprecation_risk": 3.0,
        "api_stability": 7.5,
        "ecosystem_maturity": 7.0,
        "mitigations": [
            "Open-source (MIT) – minimal vendor lock-in",
            "Azure integration available for enterprise deployment",
        ],
    },
}

_RISK_WEIGHTS = {
    "vendor_lock_in": 0.30,
    "deprecation_risk": 0.25,
    "api_stability": -0.20,  # higher stability = lower risk
    "ecosystem_maturity": -0.25,  # higher maturity = lower risk
}


def _assess_model_risk(model: ReasoningModel) -> RiskScore:
    """Compute risk scores for a single model."""
    profile = _RISK_PROFILES.get(model.provider, {
        "vendor_lock_in": 5.0,
        "deprecation_risk": 5.0,
        "api_stability": 5.0,
        "ecosystem_maturity": 5.0,
        "mitigations": ["No provider-specific risk profile available"],
    })

    vendor = float(profile["vendor_lock_in"])
    deprecation = float(profile["deprecation_risk"])
    stability = float(profile["api_stability"])
    maturity = float(profile["ecosystem_maturity"])

    # Open-source models get lower vendor lock-in
    if model.open_source:
        vendor = min(vendor, 2.5)
        deprecation = max(deprecation - 1.0, 1.0)

    composite = round(
        vendor * _RISK_WEIGHTS["vendor_lock_in"]
        + deprecation * _RISK_WEIGHTS["deprecation_risk"]
        + stability * _RISK_WEIGHTS["api_stability"]
        + maturity * _RISK_WEIGHTS["ecosystem_maturity"],
        2,
    )

    if composite >= 2.0:
        tier = "high"
    elif composite >= 0.5:
        tier = "medium"
    else:
        tier = "low"

    mitigations = list(profile.get("mitigations", []))

    return RiskScore(
        model_name=model.name,
        provider=model.provider,
        vendor_lock_in=vendor,
        deprecation_risk=deprecation,
        api_stability=stability,
        ecosystem_maturity=maturity,
        composite_risk=composite,
        risk_tier=tier,
        mitigation_notes=mitigations,
    )


def assess_risks(
    models: List[ReasoningModel] | None = None,
) -> RiskReport:
    """Run risk assessment across all models."""
    if models is None:
        models = get_reasoning_models()

    scores = [_assess_model_risk(m) for m in models]
    scores.sort(key=lambda s: s.composite_risk)

    safest = scores[0].model_name if scores else ""
    riskiest = scores[-1].model_name if scores else ""

    low_count = sum(1 for s in scores if s.risk_tier == "low")
    med_count = sum(1 for s in scores if s.risk_tier == "medium")
    high_count = sum(1 for s in scores if s.risk_tier == "high")

    summary = (
        f"Assessed {len(scores)} models: "
        f"{low_count} low-risk, {med_count} medium-risk, {high_count} high-risk. "
        f"Safest: {safest}. Riskiest: {riskiest}."
    )

    return RiskReport(
        scores=scores,
        safest_model=safest,
        riskiest_model=riskiest,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Competitive landscape mapping
# ---------------------------------------------------------------------------

_SEGMENTS = {
    "Math & Olympiad Reasoning": {
        "benchmarks": ["AIME 2025", "AIME 2024", "MATH-500"],
        "traits": ["math", "proof", "olympiad"],
    },
    "Scientific Q&A": {
        "benchmarks": ["GPQA Diamond"],
        "traits": ["science", "gpqa", "graduate"],
    },
    "Agentic Coding": {
        "benchmarks": ["SWE-bench Verified", "LiveCodeBench"],
        "traits": ["coding", "agentic", "swe"],
    },
    "Multi-Modal Reasoning": {
        "benchmarks": [],
        "traits": ["multi-modal", "image", "video", "audio"],
        "require_modalities": ["image"],
    },
    "Cost-Efficient Reasoning": {
        "benchmarks": [],
        "traits": ["cost", "efficient", "small", "fast", "distill"],
    },
    "Open-Source / Self-Hosted": {
        "benchmarks": [],
        "traits": ["open-source"],
        "require_open_source": True,
    },
    "Abstract & Novel Problem Solving": {
        "benchmarks": ["ARC-AGI"],
        "traits": ["abstract", "novel", "arc", "generali"],
    },
    "Enterprise Safety & Compliance": {
        "benchmarks": [],
        "traits": ["safety", "alignment", "compliance", "audit"],
    },
}


def _model_fits_segment(model: ReasoningModel, seg: Dict) -> bool:
    """Return True if the model competes in this segment."""
    # Check required modalities
    for mod in seg.get("require_modalities", []):
        if mod not in model.modalities:
            return False
    # Check open-source requirement
    if seg.get("require_open_source") and not model.open_source:
        return False

    # Check benchmark presence
    bm_names = {bs.benchmark.lower() for bs in model.benchmarks}
    for required_bm in seg.get("benchmarks", []):
        if required_bm.lower() in bm_names:
            return True

    # Check trait matches in strengths + description
    text = " ".join(model.strengths + [model.description]).lower()
    for trait in seg.get("traits", []):
        if trait.lower() in text:
            return True

    return False


def map_landscape(
    models: List[ReasoningModel] | None = None,
) -> LandscapeReport:
    """Map the competitive landscape across market segments."""
    if models is None:
        models = get_reasoning_models()

    segments: Dict[str, List[str]] = {}
    for seg_name, seg_def in _SEGMENTS.items():
        fitting = [m.name for m in models if _model_fits_segment(m, seg_def)]
        segments[seg_name] = fitting

    # Compute pairwise overlaps
    overlaps: List[SegmentOverlap] = []
    for seg_name, seg_models in segments.items():
        for i in range(len(seg_models)):
            for j in range(i + 1, len(seg_models)):
                overlaps.append(SegmentOverlap(
                    segment=seg_name,
                    model_a=seg_models[i],
                    model_b=seg_models[j],
                    overlap_score=1.0,
                ))

    # Underserved segments (0 or 1 models)
    underserved = [
        seg for seg, ms in segments.items() if len(ms) <= 1
    ]

    filled = sum(1 for ms in segments.values() if len(ms) >= 2)
    summary = (
        f"Mapped {len(segments)} market segments. "
        f"{filled} competitive (2+ models), "
        f"{len(underserved)} underserved."
    )

    return LandscapeReport(
        segments=segments,
        overlaps=overlaps,
        underserved_segments=underserved,
        summary=summary,
    )
