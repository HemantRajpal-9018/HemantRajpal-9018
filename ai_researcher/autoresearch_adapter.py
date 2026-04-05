"""Adapter for karpathy/autoresearch integration.

Generates ``program.md`` experiment files from our gap analysis so that
`autoresearch <https://github.com/karpathy/autoresearch>`_ can autonomously
run targeted training experiments informed by the gaps and opportunities this
agent identifies.

Flow::

    Our agent (gap analysis + opportunity scoring)
        → autoresearch_adapter (this module)
            → program.md (autoresearch input)
                → karpathy/autoresearch runs experiments on GPU

This bridges **analytical research** (what to investigate) with
**experimental research** (actually training and measuring).

.. note::

   Running autoresearch requires an NVIDIA GPU (tested on H100).
   This adapter only *generates* the program file; you still need
   the autoresearch repo and GPU hardware to execute experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

from ai_researcher.gap_analyzer import AnalysisResult, GapFinding, analyze_gaps
from ai_researcher.innovation_scorer import (
    OpportunityReport,
    OpportunityScore,
    score_opportunities,
)
from ai_researcher.models_knowledge import ReasoningModel, get_reasoning_models


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExperimentIdea:
    """A single experiment idea derived from gap analysis."""
    id: str
    title: str
    hypothesis: str
    what_to_change: str
    success_metric: str
    priority: str               # "high", "medium", "low"
    source_gap: str             # gap category that inspired this
    estimated_impact: str       # qualitative: "large", "medium", "small"


@dataclass
class ExperimentProgram:
    """A complete autoresearch experiment program."""
    experiments: List[ExperimentIdea] = field(default_factory=list)
    preamble: str = ""
    constraints: str = ""
    generated_date: str = ""
    gap_count: int = 0
    opportunity_count: int = 0


# ---------------------------------------------------------------------------
# Gap-to-experiment mapping
# ---------------------------------------------------------------------------

_GAP_EXPERIMENTS: Dict[str, Dict] = {
    "efficiency": {
        "title": "Adaptive Compute Depth Routing",
        "hypothesis": (
            "Adding an early-exit mechanism that skips deeper transformer "
            "layers for easy tokens will reduce val_bpb while cutting "
            "average inference FLOPs by 30%+."
        ),
        "what_to_change": (
            "In train.py, add a lightweight confidence head after every "
            "other transformer block.  If confidence exceeds a threshold, "
            "skip remaining blocks for that token.  Train the confidence "
            "head jointly with the main loss."
        ),
        "success_metric": "val_bpb improves or stays within 0.01 with ≥25% fewer FLOPs",
        "estimated_impact": "large",
    },
    "faithfulness": {
        "title": "Reasoning Trace Consistency Loss",
        "hypothesis": (
            "Adding an auxiliary loss that penalizes divergence between "
            "intermediate hidden-state predictions and final output "
            "will improve reasoning faithfulness."
        ),
        "what_to_change": (
            "In train.py, add a projection from each mid-layer hidden state "
            "to vocab space.  Add KL-divergence between mid-layer and "
            "final-layer logits as an auxiliary loss term (weighted 0.1)."
        ),
        "success_metric": "val_bpb improves; mid-layer prediction accuracy increases",
        "estimated_impact": "medium",
    },
    "hallucination": {
        "title": "Factual Grounding via Retrieval Token",
        "hypothesis": (
            "Injecting a special [RETRIEVE] token that triggers a "
            "knowledge-lookup attention pattern will reduce factual "
            "hallucination in generated text."
        ),
        "what_to_change": (
            "In train.py, reserve a vocab token for [RETRIEVE].  Add a "
            "cross-attention layer that attends to a frozen embedding "
            "table of frequent n-grams from the training data."
        ),
        "success_metric": "val_bpb improves; qualitative sample inspection shows fewer hallucinations",
        "estimated_impact": "medium",
    },
    "self_correction": {
        "title": "Self-Critique Reward Head",
        "hypothesis": (
            "Training a small reward head that scores the model's own "
            "outputs and using it to reweight the loss will improve "
            "self-correction behavior."
        ),
        "what_to_change": (
            "In train.py, add a 2-layer MLP reward head on top of the "
            "final hidden state.  During training, generate a second "
            "forward pass with temperature sampling, score it, and add "
            "a policy-gradient-style loss term."
        ),
        "success_metric": "val_bpb improves; model generates self-corrections in samples",
        "estimated_impact": "medium",
    },
    "calibration": {
        "title": "Temperature-Scaled Confidence Calibration",
        "hypothesis": (
            "Learning a per-layer temperature parameter and training with "
            "a calibration loss (ECE) will produce better-calibrated "
            "output distributions."
        ),
        "what_to_change": (
            "In train.py, add a learnable scalar temperature after each "
            "attention block's logits computation.  Add expected "
            "calibration error (ECE) as an auxiliary loss (weight 0.05)."
        ),
        "success_metric": "val_bpb improves or stays equal; ECE on held-out set decreases",
        "estimated_impact": "small",
    },
    "abstraction": {
        "title": "Symbolic Abstraction Layer",
        "hypothesis": (
            "Adding a bottleneck layer that forces the model to compress "
            "representations into a smaller symbolic space will improve "
            "abstraction and generalization."
        ),
        "what_to_change": (
            "In train.py, insert a VQ-VAE style bottleneck layer at the "
            "midpoint of the transformer stack.  Use commitment loss "
            "to train the codebook jointly."
        ),
        "success_metric": "val_bpb improves on out-of-distribution evaluation set",
        "estimated_impact": "large",
    },
    "compositionality": {
        "title": "Compositional Attention Heads",
        "hypothesis": (
            "Replacing some standard attention heads with heads that "
            "explicitly compose representations via tensor product "
            "operations will improve compositional generalization."
        ),
        "what_to_change": (
            "In train.py, replace the last 2 attention heads in each "
            "layer with TPR (Tensor Product Representation) heads "
            "that bind role-filler pairs using circular convolution."
        ),
        "success_metric": "val_bpb improves; compositional reasoning tasks in samples improve",
        "estimated_impact": "medium",
    },
    "multi_modal_reasoning": {
        "title": "Multi-Scale Positional Encoding",
        "hypothesis": (
            "Using multiple positional encoding scales (character, word, "
            "sentence) simultaneously will help the model reason across "
            "different granularity levels, a precursor to multi-modal."
        ),
        "what_to_change": (
            "In train.py, replace the single RoPE frequency with three "
            "parallel RoPE encodings at different frequency scales.  "
            "Concatenate them and project back to model dimension."
        ),
        "success_metric": "val_bpb improves; longer-range dependencies captured better",
        "estimated_impact": "medium",
    },
    "long_horizon_planning": {
        "title": "Hierarchical Prediction Objective",
        "hypothesis": (
            "Adding a secondary loss that predicts at the paragraph level "
            "(not just next token) will improve long-horizon planning "
            "in the model's representations."
        ),
        "what_to_change": (
            "In train.py, add a paragraph-level prediction head that "
            "takes the hidden state at paragraph boundaries and predicts "
            "a summary embedding of the next paragraph.  Train with "
            "cosine similarity loss."
        ),
        "success_metric": "val_bpb improves; generated samples show better narrative coherence",
        "estimated_impact": "large",
    },
    "adversarial_robustness": {
        "title": "Adversarial Token Perturbation Training",
        "hypothesis": (
            "Adding adversarial perturbations to token embeddings during "
            "training (FreeLB-style) will make the model more robust "
            "to adversarial inputs."
        ),
        "what_to_change": (
            "In train.py, after computing the embedding, add a small "
            "adversarial perturbation (found via one step of gradient "
            "ascent on the loss) and train on the perturbed input."
        ),
        "success_metric": "val_bpb stays within 0.02; adversarial eval loss decreases",
        "estimated_impact": "medium",
    },
    "mathematical_proof": {
        "title": "Chain-of-Thought Data Curriculum",
        "hypothesis": (
            "Mixing in synthetic math chain-of-thought data during "
            "training will improve the model's mathematical reasoning "
            "without hurting general language modeling."
        ),
        "what_to_change": (
            "In prepare.py (data prep phase), augment the training "
            "data with synthesized math CoT sequences.  In train.py, "
            "add a curriculum schedule that increases math data ratio "
            "over training."
        ),
        "success_metric": "val_bpb improves on math-heavy eval; general val_bpb unchanged",
        "estimated_impact": "large",
    },
    "spatial_reasoning": {
        "title": "Geometric Position Embedding",
        "hypothesis": (
            "Adding 2D/3D position embeddings alongside the standard "
            "1D sequence position will help the model develop spatial "
            "reasoning representations."
        ),
        "what_to_change": (
            "In train.py, add optional 2D grid position embeddings "
            "that activate when the input contains spatial markers.  "
            "Sum them with the standard position embeddings."
        ),
        "success_metric": "val_bpb improves on spatial reasoning evaluation prompts",
        "estimated_impact": "small",
    },
    "temporal_reasoning": {
        "title": "Temporal Attention Bias",
        "hypothesis": (
            "Adding a learnable temporal decay bias to the attention "
            "scores (so recent tokens get slightly more weight) will "
            "improve temporal reasoning."
        ),
        "what_to_change": (
            "In train.py, add a learnable ALiBi-style distance bias "
            "to the attention logits.  Each head gets its own decay "
            "rate, learned during training."
        ),
        "success_metric": "val_bpb improves; temporal ordering in generated samples improves",
        "estimated_impact": "small",
    },
    "common_sense": {
        "title": "Entity-Aware Attention Mask",
        "hypothesis": (
            "Masking attention so that entity mentions always attend to "
            "their first occurrence will help the model maintain entity "
            "coherence (a common-sense proxy)."
        ),
        "what_to_change": (
            "In train.py, add a simple entity-detection heuristic "
            "(capitalized words) and an auxiliary attention mask that "
            "ensures entity tokens attend to their first mention."
        ),
        "success_metric": "val_bpb improves; entity coherence in samples improves",
        "estimated_impact": "small",
    },
    "tool_use_reasoning": {
        "title": "Action Token Prediction",
        "hypothesis": (
            "Reserving special [ACT] tokens and training the model to "
            "predict when to emit them will create a foundation for "
            "tool-use reasoning."
        ),
        "what_to_change": (
            "In train.py, reserve [ACT_START] and [ACT_END] tokens.  "
            "Insert them around structured data patterns in training "
            "data.  Add a binary classification head that predicts "
            "whether the next token should be an action token."
        ),
        "success_metric": "val_bpb improves; action token prediction F1 > 0.7",
        "estimated_impact": "medium",
    },
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _priority_from_opportunity(opp: Optional[OpportunityScore]) -> str:
    """Map opportunity tier to experiment priority."""
    if opp is None:
        return "medium"
    tier = opp.priority_tier
    if tier == "Tier 1":
        return "high"
    if tier == "Tier 2":
        return "medium"
    return "low"


def generate_experiment_program(
    gap_result: Optional[AnalysisResult] = None,
    opportunities: Optional[OpportunityReport] = None,
    models: Optional[List[ReasoningModel]] = None,
) -> ExperimentProgram:
    """Generate an autoresearch experiment program from gap analysis.

    Parameters
    ----------
    gap_result : AnalysisResult, optional
        Pre-computed gap analysis.  If *None*, runs ``analyze_gaps()``.
    opportunities : OpportunityReport, optional
        Pre-computed opportunity scores.  If *None*, runs ``score_opportunities()``.
    models : list of ReasoningModel, optional
        Models to analyze.  Defaults to the full knowledge base.

    Returns
    -------
    ExperimentProgram
        A structured program with experiment ideas, ready to be rendered
        as a ``program.md`` file.
    """
    models = models or get_reasoning_models()

    if gap_result is None:
        gap_result = analyze_gaps(models)
    if opportunities is None:
        opportunities = score_opportunities(gap_result, models)

    # Build an opportunity lookup by category
    opp_by_cat: Dict[str, OpportunityScore] = {}
    for opp in opportunities.opportunities:
        opp_by_cat[opp.category] = opp

    experiments: List[ExperimentIdea] = []
    seen_categories: set = set()

    # Generate experiments from findings, ordered by severity
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    sorted_findings = sorted(
        gap_result.findings,
        key=lambda f: severity_order.get(f.severity, 9),
    )

    for finding in sorted_findings:
        cat = finding.category
        if cat in seen_categories:
            continue
        seen_categories.add(cat)

        template = _GAP_EXPERIMENTS.get(cat)
        if template is None:
            continue

        opp = opp_by_cat.get(cat)
        priority = _priority_from_opportunity(opp)

        experiments.append(ExperimentIdea(
            id=f"exp_{cat}",
            title=template["title"],
            hypothesis=template["hypothesis"],
            what_to_change=template["what_to_change"],
            success_metric=template["success_metric"],
            priority=priority,
            source_gap=cat,
            estimated_impact=template["estimated_impact"],
        ))

    program = ExperimentProgram(
        experiments=experiments,
        preamble=_generate_preamble(len(experiments), gap_result.model_count),
        constraints=_generate_constraints(),
        generated_date=str(date.today()),
        gap_count=len(gap_result.findings),
        opportunity_count=len(opportunities.opportunities),
    )
    return program


def _generate_preamble(experiment_count: int, model_count: int) -> str:
    """Generate the preamble section for the program."""
    return (
        f"# Autoresearch Experiment Program\n"
        f"\n"
        f"Generated by AI Researcher Agent based on gap analysis of "
        f"{model_count} reasoning models.\n"
        f"\n"
        f"This program contains {experiment_count} experiment ideas "
        f"prioritized by innovation opportunity scoring.  Each experiment "
        f"targets a specific gap identified in the current reasoning model "
        f"landscape.\n"
        f"\n"
        f"## How to Use\n"
        f"\n"
        f"1. Clone karpathy/autoresearch: `git clone https://github.com/karpathy/autoresearch`\n"
        f"2. Copy this file as `program.md` in the autoresearch directory\n"
        f"3. Run: `uv run prepare.py` (one-time setup)\n"
        f"4. Start your AI agent and point it at this program.md\n"
        f"5. The agent will iterate through experiments autonomously\n"
        f"\n"
        f"**Requirements:** NVIDIA GPU (tested on H100), Python 3.10+, uv\n"
    )


def _generate_constraints() -> str:
    """Generate the constraints section."""
    return (
        "## Constraints\n"
        "\n"
        "- Each experiment must complete within the 5-minute training budget\n"
        "- Only modify train.py (prepare.py is fixed)\n"
        "- Metric: val_bpb (validation bits per byte) — lower is better\n"
        "- Always run a baseline first to establish the current val_bpb\n"
        "- Keep a log of each experiment's hypothesis, changes, and results\n"
        "- If an experiment worsens val_bpb by more than 0.05, revert it\n"
        "- Try high-priority experiments first\n"
    )


# ---------------------------------------------------------------------------
# Render to Markdown (program.md format)
# ---------------------------------------------------------------------------

def render_program_markdown(program: ExperimentProgram) -> str:
    """Render an ``ExperimentProgram`` as a Markdown ``program.md`` file.

    This output is designed to be dropped directly into a
    karpathy/autoresearch clone as ``program.md``.
    """
    lines: List[str] = []

    # Preamble
    lines.append(program.preamble)
    lines.append("")

    # Constraints
    lines.append(program.constraints)
    lines.append("")

    # Experiments
    lines.append("## Experiments\n")
    lines.append(
        f"Total: {len(program.experiments)} experiments | "
        f"Based on {program.gap_count} gap findings and "
        f"{program.opportunity_count} opportunity scores\n"
    )

    priority_order = {"high": 0, "medium": 1, "low": 2}
    sorted_exps = sorted(
        program.experiments,
        key=lambda e: priority_order.get(e.priority, 9),
    )

    for i, exp in enumerate(sorted_exps, 1):
        priority_badge = {
            "high": "🔴 HIGH",
            "medium": "🟡 MEDIUM",
            "low": "🟢 LOW",
        }.get(exp.priority, exp.priority.upper())

        lines.append(f"### Experiment {i}: {exp.title}")
        lines.append("")
        lines.append(f"**Priority:** {priority_badge} | "
                      f"**Impact:** {exp.estimated_impact} | "
                      f"**Source gap:** {exp.source_gap}")
        lines.append("")
        lines.append(f"**Hypothesis:** {exp.hypothesis}")
        lines.append("")
        lines.append(f"**What to change:** {exp.what_to_change}")
        lines.append("")
        lines.append(f"**Success metric:** {exp.success_metric}")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append(
        f"*Generated on {program.generated_date} by AI Researcher Agent. "
        f"See https://github.com/HemantRajpal-9018/HemantRajpal-9018 "
        f"for the full analysis.*"
    )
    lines.append("")

    return "\n".join(lines)


def render_program_text(program: ExperimentProgram) -> str:
    """Render an ``ExperimentProgram`` as plain text for terminal display."""
    sep = "=" * 78
    subsep = "-" * 78
    lines: List[str] = []

    lines.append(sep)
    lines.append("  AUTORESEARCH EXPERIMENT PROGRAM")
    lines.append(f"  Generated: {program.generated_date}")
    lines.append(
        f"  Based on {program.gap_count} gap findings, "
        f"{program.opportunity_count} opportunity scores"
    )
    lines.append(
        f"  Experiments: {len(program.experiments)}"
    )
    lines.append(sep)
    lines.append("")

    lines.append("  NOTE: Running experiments requires karpathy/autoresearch")
    lines.append("  and an NVIDIA GPU (tested on H100).  This program")
    lines.append("  generates the program.md input file for autoresearch.")
    lines.append("")

    priority_order = {"high": 0, "medium": 1, "low": 2}
    sorted_exps = sorted(
        program.experiments,
        key=lambda e: priority_order.get(e.priority, 9),
    )

    for i, exp in enumerate(sorted_exps, 1):
        priority_tag = f"[{exp.priority.upper()}]"
        lines.append(f"  {priority_tag} Experiment {i}: {exp.title}")
        lines.append(subsep)

        lines.append(f"    Source gap      : {exp.source_gap}")
        lines.append(f"    Est. impact     : {exp.estimated_impact}")
        lines.append(f"    Success metric  : {exp.success_metric}")
        lines.append("")
        lines.append(f"    Hypothesis:")
        # Word-wrap hypothesis
        words = exp.hypothesis.split()
        cur_line: List[str] = []
        cur_len = 0
        for w in words:
            if cur_len + len(w) + 1 > 68 and cur_line:
                lines.append("      " + " ".join(cur_line))
                cur_line = [w]
                cur_len = len(w)
            else:
                cur_line.append(w)
                cur_len += len(w) + 1
        if cur_line:
            lines.append("      " + " ".join(cur_line))

        lines.append("")
        lines.append(f"    What to change in train.py:")
        words2 = exp.what_to_change.split()
        cur2: List[str] = []
        cur2_len = 0
        for w in words2:
            if cur2_len + len(w) + 1 > 68 and cur2:
                lines.append("      " + " ".join(cur2))
                cur2 = [w]
                cur2_len = len(w)
            else:
                cur2.append(w)
                cur2_len += len(w) + 1
        if cur2:
            lines.append("      " + " ".join(cur2))

        lines.append("")
        lines.append("")

    return "\n".join(lines)
