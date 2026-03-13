"""Knowledge base of reasoning models as of March 12, 2026.

This module catalogs the major reasoning-focused language models, their
capabilities, known limitations, and benchmark performance so that the gap
analyzer can produce a structured assessment.
"""

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkScore:
    """A single benchmark result for a model."""
    benchmark: str
    score: float          # 0-100 normalized score
    details: str = ""


@dataclass
class ReasoningModel:
    """Representation of a reasoning-focused language model."""
    name: str
    provider: str
    release_date: str                       # ISO-like, e.g. "2025-12"
    description: str
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    benchmarks: List[BenchmarkScore] = field(default_factory=list)
    open_source: bool = False
    reasoning_type: str = "chain-of-thought"  # e.g. chain-of-thought, tree-of-thought, etc.
    modalities: List[str] = field(default_factory=lambda: ["text"])


# ---------------------------------------------------------------------------
# Gap taxonomy
# ---------------------------------------------------------------------------

REASONING_GAP_CATEGORIES = [
    "faithfulness",           # reasoning chain faithfulness to internal computation
    "hallucination",          # factual errors during multi-step reasoning
    "self_correction",        # ability to detect & fix own errors
    "spatial_reasoning",      # 2-D / 3-D spatial tasks
    "temporal_reasoning",     # time-based reasoning and planning
    "common_sense",           # everyday world knowledge
    "mathematical_proof",     # formal / research-level mathematics
    "adversarial_robustness", # resilience to adversarial prompts
    "calibration",            # confidence vs. accuracy alignment
    "efficiency",             # compute / latency for reasoning
    "multi_modal_reasoning",  # reasoning across text, image, audio
    "long_horizon_planning",  # multi-step planning with dependencies
    "abstraction",            # generalizing from examples
    "compositionality",       # combining learned skills for novel tasks
    "tool_use_reasoning",     # deciding when & how to use external tools
]


# ---------------------------------------------------------------------------
# Model registry – latest reasoning models as of 2026-03-12
# ---------------------------------------------------------------------------

def get_reasoning_models() -> List[ReasoningModel]:
    """Return the catalogue of major reasoning models as of March 12, 2026."""
    return [
        # ------------------------------------------------------------------
        # OpenAI
        # ------------------------------------------------------------------
        ReasoningModel(
            name="OpenAI o3",
            provider="OpenAI",
            release_date="2025-04",
            description=(
                "Third-generation reasoning model from OpenAI. Uses extended "
                "chain-of-thought with internal verification steps. Significant "
                "improvement on ARC-AGI and GPQA benchmarks over o1."
            ),
            strengths=[
                "State-of-the-art on GPQA Diamond (~88%)",
                "Strong mathematical reasoning (AIME 2025 ≈ 96%)",
                "Improved safety alignment over o1",
                "Configurable reasoning effort (low/medium/high)",
            ],
            weaknesses=[
                "High inference cost at high-effort setting",
                "Reasoning traces are hidden from users (faithfulness opaque)",
                "Struggles with novel spatial puzzles outside training distribution",
                "Occasional hallucination in multi-hop scientific reasoning",
            ],
            benchmarks=[
                BenchmarkScore("GPQA Diamond", 87.7),
                BenchmarkScore("AIME 2025", 96.0),
                BenchmarkScore("ARC-AGI", 87.5),
                BenchmarkScore("MATH-500", 98.0),
                BenchmarkScore("SWE-bench Verified", 71.7),
            ],
            reasoning_type="chain-of-thought-with-verification",
        ),
        ReasoningModel(
            name="OpenAI o3-mini",
            provider="OpenAI",
            release_date="2025-01",
            description=(
                "Cost-efficient variant of the o3 reasoning family. Targets "
                "STEM tasks with lower latency and cost."
            ),
            strengths=[
                "Much lower cost than o3 full",
                "Competitive math/code performance",
                "Fast inference for coding tasks",
            ],
            weaknesses=[
                "Weaker on open-ended reasoning vs o3",
                "Limited multi-modal support",
                "Reduced performance on highly complex tasks",
            ],
            benchmarks=[
                BenchmarkScore("AIME 2025", 87.0),
                BenchmarkScore("MATH-500", 97.0),
                BenchmarkScore("LiveCodeBench", 78.0),
            ],
            reasoning_type="chain-of-thought",
        ),
        ReasoningModel(
            name="OpenAI o1",
            provider="OpenAI",
            release_date="2024-12",
            description=(
                "First-generation dedicated reasoning model from OpenAI. "
                "Pioneered large-scale chain-of-thought at inference time."
            ),
            strengths=[
                "Reliable multi-step math reasoning",
                "Strong coding ability",
                "Good at following complex instructions",
            ],
            weaknesses=[
                "Cannot browse or use tools natively",
                "Expensive per-token reasoning cost",
                "Sometimes over-reasons on simple tasks",
                "Occasional unfaithful reasoning chains",
            ],
            benchmarks=[
                BenchmarkScore("GPQA Diamond", 78.0),
                BenchmarkScore("MATH-500", 96.4),
                BenchmarkScore("AIME 2024", 83.3),
            ],
            reasoning_type="chain-of-thought",
        ),

        # ------------------------------------------------------------------
        # DeepSeek
        # ------------------------------------------------------------------
        ReasoningModel(
            name="DeepSeek R1",
            provider="DeepSeek",
            release_date="2025-01",
            description=(
                "Open-source reasoning model trained with reinforcement "
                "learning to produce long chain-of-thought reasoning. "
                "Competitive with OpenAI o1 on many benchmarks."
            ),
            strengths=[
                "Fully open-source (MIT license) – weights available",
                "Strong math and coding performance",
                "Transparent reasoning traces",
                "Cost-efficient inference via distilled variants",
            ],
            weaknesses=[
                "Language mixing in reasoning chains (Chinese/English)",
                "Prone to verbose and circular reasoning loops",
                "Weaker on creative / open-ended tasks",
                "Limited safety alignment vs. commercial models",
            ],
            benchmarks=[
                BenchmarkScore("AIME 2024", 79.8),
                BenchmarkScore("MATH-500", 97.3),
                BenchmarkScore("GPQA Diamond", 71.5),
                BenchmarkScore("LiveCodeBench", 65.9),
            ],
            open_source=True,
            reasoning_type="chain-of-thought-rl",
        ),

        # ------------------------------------------------------------------
        # Google DeepMind
        # ------------------------------------------------------------------
        ReasoningModel(
            name="Gemini 2.5 Pro",
            provider="Google DeepMind",
            release_date="2025-03",
            description=(
                "Google's flagship reasoning model with native multi-modal "
                "capabilities and a 1M+ token context window. Uses internal "
                "'thinking' tokens visible in API."
            ),
            strengths=[
                "Native multi-modal reasoning (text + image + video + audio)",
                "Very long context window (1M+ tokens)",
                "Strong agentic coding (SWE-bench Verified ≈ 63.8%)",
                "Good balance of reasoning and general ability",
            ],
            weaknesses=[
                "Expensive at scale with thinking tokens",
                "Occasionally inconsistent reasoning across modalities",
                "Weaker than o3 on hardest math benchmarks",
                "Tendency to over-rely on parametric knowledge vs. provided context",
            ],
            benchmarks=[
                BenchmarkScore("GPQA Diamond", 84.0),
                BenchmarkScore("MATH-500", 97.0),
                BenchmarkScore("AIME 2025", 86.0),
                BenchmarkScore("SWE-bench Verified", 63.8),
                BenchmarkScore("ARC-AGI", 72.0),
            ],
            reasoning_type="internal-thinking-tokens",
            modalities=["text", "image", "video", "audio"],
        ),
        ReasoningModel(
            name="Gemini 2.0 Flash Thinking",
            provider="Google DeepMind",
            release_date="2024-12",
            description=(
                "Experimental reasoning variant of Gemini 2.0 Flash with "
                "explicit thinking traces."
            ),
            strengths=[
                "Fast inference for a reasoning model",
                "Good cost/performance ratio",
                "Transparent thinking process",
            ],
            weaknesses=[
                "Weaker reasoning depth than full Gemini 2.5 Pro",
                "Limited multi-step planning",
                "Struggles with highly abstract problems",
            ],
            benchmarks=[
                BenchmarkScore("MATH-500", 93.0),
                BenchmarkScore("GPQA Diamond", 73.0),
            ],
            reasoning_type="chain-of-thought",
            modalities=["text", "image"],
        ),

        # ------------------------------------------------------------------
        # Anthropic
        # ------------------------------------------------------------------
        ReasoningModel(
            name="Claude 3.7 Sonnet",
            provider="Anthropic",
            release_date="2025-02",
            description=(
                "Anthropic's hybrid reasoning model with an extended thinking "
                "mode. First Claude model with explicit chain-of-thought "
                "reasoning that can be toggled on/off."
            ),
            strengths=[
                "Toggleable extended thinking mode",
                "Strong agentic tool-use capabilities",
                "Good balance of reasoning + instruction following",
                "Excellent coding performance (SWE-bench ≈ 70.3%)",
            ],
            weaknesses=[
                "Thinking tokens consume context budget",
                "Sometimes produces overly cautious reasoning",
                "Can be verbose in extended thinking mode",
                "Occasional difficulty with very long reasoning chains",
            ],
            benchmarks=[
                BenchmarkScore("MATH-500", 96.0),
                BenchmarkScore("GPQA Diamond", 80.0),
                BenchmarkScore("SWE-bench Verified", 70.3),
                BenchmarkScore("AIME 2025", 80.0),
                BenchmarkScore("TAU-bench Airline", 64.0),
            ],
            reasoning_type="extended-thinking",
        ),

        # ------------------------------------------------------------------
        # Alibaba / Qwen
        # ------------------------------------------------------------------
        ReasoningModel(
            name="QwQ-32B",
            provider="Alibaba Qwen",
            release_date="2025-03",
            description=(
                "Open-source 32B reasoning model from Alibaba's Qwen team. "
                "Trained with RL for reasoning, competitive with much larger "
                "proprietary models."
            ),
            strengths=[
                "Open-source (Apache 2.0)",
                "Strong reasoning at 32B parameter scale",
                "Good math and coding ability",
                "Agent and tool-use capabilities",
            ],
            weaknesses=[
                "Language switching in reasoning traces",
                "Circular reasoning loops on complex problems",
                "Weaker common-sense reasoning",
                "Limited multi-modal support",
            ],
            benchmarks=[
                BenchmarkScore("AIME 2024", 79.5),
                BenchmarkScore("MATH-500", 95.0),
                BenchmarkScore("LiveCodeBench", 63.4),
                BenchmarkScore("GPQA Diamond", 65.0),
            ],
            open_source=True,
            reasoning_type="chain-of-thought-rl",
        ),

        # ------------------------------------------------------------------
        # xAI
        # ------------------------------------------------------------------
        ReasoningModel(
            name="Grok 3",
            provider="xAI",
            release_date="2025-02",
            description=(
                "xAI's large reasoning model with a 'think' mode for "
                "extended chain-of-thought reasoning."
            ),
            strengths=[
                "Strong performance on math and science benchmarks",
                "Deep thinking mode for hard problems",
                "Competitive GPQA Diamond score",
            ],
            weaknesses=[
                "Closed-source with limited API access",
                "Less mature safety alignment",
                "Expensive inference",
                "Limited ecosystem and tool integrations",
            ],
            benchmarks=[
                BenchmarkScore("GPQA Diamond", 84.6),
                BenchmarkScore("MATH-500", 97.0),
                BenchmarkScore("AIME 2025", 86.0),
            ],
            reasoning_type="chain-of-thought",
        ),

        # ------------------------------------------------------------------
        # Microsoft / Phi
        # ------------------------------------------------------------------
        ReasoningModel(
            name="Phi-4-reasoning",
            provider="Microsoft",
            release_date="2025-04",
            description=(
                "A compact 14B-parameter reasoning model from Microsoft. "
                "Trained with RL on curated math/science reasoning data."
            ),
            strengths=[
                "Very small model with strong reasoning (14B params)",
                "Open-source (MIT)",
                "Efficient inference on consumer hardware",
                "Strong math performance relative to size",
            ],
            weaknesses=[
                "Limited world knowledge at 14B scale",
                "Weaker on open-ended reasoning tasks",
                "No multi-modal support",
                "Prone to reasoning shortcuts on novel problems",
            ],
            benchmarks=[
                BenchmarkScore("MATH-500", 95.0),
                BenchmarkScore("AIME 2025", 75.0),
                BenchmarkScore("GPQA Diamond", 64.0),
            ],
            open_source=True,
            reasoning_type="chain-of-thought-rl",
        ),
    ]


def get_model_by_name(name: str) -> ReasoningModel | None:
    """Look up a model by exact name (case-insensitive)."""
    for model in get_reasoning_models():
        if model.name.lower() == name.lower():
            return model
    return None


def get_models_by_provider(provider: str) -> List[ReasoningModel]:
    """Return all models from a given provider."""
    return [
        m for m in get_reasoning_models()
        if m.provider.lower() == provider.lower()
    ]
