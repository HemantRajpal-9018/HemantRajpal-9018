"""Head-to-head model comparison engine.

Allows comparing any two reasoning models on benchmarks, strengths,
weaknesses, and gap exposure.  Also provides best-fit recommendations
for specific use cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ai_researcher.models_knowledge import (
    ReasoningModel,
    get_model_by_name,
    get_reasoning_models,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkDelta:
    """Score difference between two models on a shared benchmark."""
    benchmark: str
    score_a: float
    score_b: float
    delta: float           # score_a - score_b (positive = A wins)


@dataclass
class ComparisonResult:
    """Full comparison between two models."""
    model_a: str
    model_b: str
    benchmark_deltas: List[BenchmarkDelta] = field(default_factory=list)
    unique_strengths_a: List[str] = field(default_factory=list)
    unique_strengths_b: List[str] = field(default_factory=list)
    shared_weaknesses: List[str] = field(default_factory=list)
    unique_weaknesses_a: List[str] = field(default_factory=list)
    unique_weaknesses_b: List[str] = field(default_factory=list)
    verdict: str = ""


@dataclass
class UseCaseRecommendation:
    """A recommended model for a specific use case."""
    use_case: str
    recommended_model: str
    runner_up: str
    reason: str


# ---------------------------------------------------------------------------
# Use-case profiles
# ---------------------------------------------------------------------------

_USE_CASE_PROFILES: List[Dict] = [
    {
        "use_case": "Competition-level math (AIME/Olympiad)",
        "key_benchmarks": ["AIME 2025", "AIME 2024", "MATH-500"],
        "prefer_traits": ["mathematical reasoning"],
    },
    {
        "use_case": "Scientific Q&A (graduate-level)",
        "key_benchmarks": ["GPQA Diamond"],
        "prefer_traits": ["science", "GPQA"],
    },
    {
        "use_case": "Agentic coding / SWE tasks",
        "key_benchmarks": ["SWE-bench Verified", "LiveCodeBench"],
        "prefer_traits": ["coding", "agentic", "tool-use"],
    },
    {
        "use_case": "Multi-modal analysis (text + images)",
        "key_benchmarks": [],
        "prefer_traits": ["multi-modal", "image", "vision"],
        "require_modalities": ["image"],
    },
    {
        "use_case": "Cost-sensitive / high-volume inference",
        "key_benchmarks": [],
        "prefer_traits": ["cost", "efficient", "fast", "small"],
    },
    {
        "use_case": "Self-hosted / on-premise deployment",
        "key_benchmarks": [],
        "prefer_traits": ["open-source"],
        "require_open_source": True,
    },
    {
        "use_case": "Abstract / novel problem solving (ARC-like)",
        "key_benchmarks": ["ARC-AGI"],
        "prefer_traits": ["abstract", "novel", "ARC"],
    },
]


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def _compute_benchmark_deltas(
    a: ReasoningModel, b: ReasoningModel,
) -> List[BenchmarkDelta]:
    """Return score deltas for benchmarks both models share."""
    b_map = {bs.benchmark: bs.score for bs in b.benchmarks}
    deltas: List[BenchmarkDelta] = []
    for bs in a.benchmarks:
        if bs.benchmark in b_map:
            deltas.append(BenchmarkDelta(
                benchmark=bs.benchmark,
                score_a=bs.score,
                score_b=b_map[bs.benchmark],
                delta=round(bs.score - b_map[bs.benchmark], 2),
            ))
    return sorted(deltas, key=lambda d: -abs(d.delta))


def _weakness_overlap(
    a: ReasoningModel, b: ReasoningModel,
) -> Tuple[List[str], List[str], List[str]]:
    """Return (shared, unique_a, unique_b) weakness lists."""
    set_a = set(a.weaknesses)
    set_b = set(b.weaknesses)
    # Use fuzzy "word-overlap" matching for shared weaknesses
    shared: List[str] = []
    unique_a: List[str] = list(set_a)
    unique_b: List[str] = list(set_b)
    for wa in list(set_a):
        wa_words = set(wa.lower().split())
        for wb in list(set_b):
            wb_words = set(wb.lower().split())
            if len(wa_words & wb_words) >= 3:
                shared.append(f"{wa} / {wb}")
                if wa in unique_a:
                    unique_a.remove(wa)
                if wb in unique_b:
                    unique_b.remove(wb)
    return shared, sorted(unique_a), sorted(unique_b)


def _generate_verdict(
    a: ReasoningModel,
    b: ReasoningModel,
    deltas: List[BenchmarkDelta],
) -> str:
    """Generate a plain-English comparative verdict."""
    a_wins = sum(1 for d in deltas if d.delta > 0)
    b_wins = sum(1 for d in deltas if d.delta < 0)
    ties = sum(1 for d in deltas if d.delta == 0)

    parts: List[str] = []
    if deltas:
        if a_wins > b_wins:
            parts.append(
                f"{a.name} leads on {a_wins}/{len(deltas)} shared benchmarks."
            )
        elif b_wins > a_wins:
            parts.append(
                f"{b.name} leads on {b_wins}/{len(deltas)} shared benchmarks."
            )
        else:
            parts.append("Both models are evenly matched on shared benchmarks.")

    if a.open_source and not b.open_source:
        parts.append(f"{a.name} is open-source; {b.name} is proprietary.")
    elif b.open_source and not a.open_source:
        parts.append(f"{b.name} is open-source; {a.name} is proprietary.")

    a_mods = set(a.modalities)
    b_mods = set(b.modalities)
    if a_mods - b_mods:
        parts.append(
            f"{a.name} supports extra modalities: {', '.join(a_mods - b_mods)}."
        )
    if b_mods - a_mods:
        parts.append(
            f"{b.name} supports extra modalities: {', '.join(b_mods - a_mods)}."
        )

    return " ".join(parts) if parts else "Insufficient data for verdict."


def compare_models(
    name_a: str,
    name_b: str,
    models: List[ReasoningModel] | None = None,
) -> ComparisonResult:
    """Compare two reasoning models head-to-head.

    Raises ``ValueError`` if either model name is not found.
    """
    if models is None:
        models = get_reasoning_models()

    lookup = {m.name.lower(): m for m in models}
    a = lookup.get(name_a.lower())
    b = lookup.get(name_b.lower())
    if a is None:
        raise ValueError(f"Model not found: {name_a}")
    if b is None:
        raise ValueError(f"Model not found: {name_b}")

    deltas = _compute_benchmark_deltas(a, b)
    shared_w, uniq_a_w, uniq_b_w = _weakness_overlap(a, b)
    verdict = _generate_verdict(a, b, deltas)

    return ComparisonResult(
        model_a=a.name,
        model_b=b.name,
        benchmark_deltas=deltas,
        unique_strengths_a=list(a.strengths),
        unique_strengths_b=list(b.strengths),
        shared_weaknesses=shared_w,
        unique_weaknesses_a=uniq_a_w,
        unique_weaknesses_b=uniq_b_w,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Best-fit recommendations
# ---------------------------------------------------------------------------

def _score_model_for_use_case(
    model: ReasoningModel, profile: Dict,
) -> float:
    """Return a heuristic relevance score (higher = better fit)."""
    score = 0.0

    # Benchmark matches
    key_bm = {b.lower() for b in profile.get("key_benchmarks", [])}
    for bs in model.benchmarks:
        if bs.benchmark.lower() in key_bm:
            score += bs.score / 10.0  # weight by actual score

    # Trait matches in strengths
    traits = profile.get("prefer_traits", [])
    for strength in model.strengths:
        s_lower = strength.lower()
        for trait in traits:
            if trait.lower() in s_lower:
                score += 5.0
                break

    # Require open-source
    if profile.get("require_open_source") and not model.open_source:
        return -1.0

    # Require modalities
    required_mods = profile.get("require_modalities", [])
    for mod in required_mods:
        if mod not in model.modalities:
            return -1.0

    return score


def recommend_models(
    models: List[ReasoningModel] | None = None,
) -> List[UseCaseRecommendation]:
    """Return best-fit model recommendations for standard use cases."""
    if models is None:
        models = get_reasoning_models()

    recommendations: List[UseCaseRecommendation] = []
    for profile in _USE_CASE_PROFILES:
        scored: List[Tuple[float, ReasoningModel]] = []
        for m in models:
            s = _score_model_for_use_case(m, profile)
            if s >= 0:
                scored.append((s, m))
        scored.sort(key=lambda x: -x[0])
        if len(scored) >= 2:
            best = scored[0][1]
            runner = scored[1][1]
            recommendations.append(UseCaseRecommendation(
                use_case=profile["use_case"],
                recommended_model=best.name,
                runner_up=runner.name,
                reason=f"Top scorer among {len(scored)} eligible models.",
            ))
        elif len(scored) == 1:
            best = scored[0][1]
            recommendations.append(UseCaseRecommendation(
                use_case=profile["use_case"],
                recommended_model=best.name,
                runner_up="N/A",
                reason="Only eligible model.",
            ))

    return recommendations
