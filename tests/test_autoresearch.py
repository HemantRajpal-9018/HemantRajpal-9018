"""Tests for the autoresearch adapter module.

Validates experiment program generation, rendering, and CLI integration.
"""

import pytest

from ai_researcher.agent import ResearcherAgent
from ai_researcher.autoresearch_adapter import (
    ExperimentIdea,
    ExperimentProgram,
    _GAP_EXPERIMENTS,
    _priority_from_opportunity,
    generate_experiment_program,
    render_program_markdown,
    render_program_text,
)
from ai_researcher.gap_analyzer import analyze_gaps
from ai_researcher.innovation_scorer import OpportunityScore, score_opportunities
from ai_researcher.models_knowledge import (
    REASONING_GAP_CATEGORIES,
    ReasoningModel,
    get_reasoning_models,
)


# =========================================================================
# Unit tests – data structures
# =========================================================================


class TestExperimentIdea:
    """Test ExperimentIdea dataclass."""

    def test_create_experiment_idea(self):
        exp = ExperimentIdea(
            id="exp_test",
            title="Test Experiment",
            hypothesis="Things will improve",
            what_to_change="Change train.py",
            success_metric="val_bpb improves",
            priority="high",
            source_gap="efficiency",
            estimated_impact="large",
        )
        assert exp.id == "exp_test"
        assert exp.priority == "high"
        assert exp.source_gap == "efficiency"


class TestExperimentProgram:
    """Test ExperimentProgram dataclass."""

    def test_create_empty_program(self):
        prog = ExperimentProgram()
        assert prog.experiments == []
        assert prog.gap_count == 0
        assert prog.opportunity_count == 0

    def test_program_with_experiments(self):
        exp = ExperimentIdea(
            id="exp_1", title="T", hypothesis="H",
            what_to_change="C", success_metric="M",
            priority="high", source_gap="efficiency",
            estimated_impact="large",
        )
        prog = ExperimentProgram(
            experiments=[exp],
            gap_count=5,
            opportunity_count=3,
        )
        assert len(prog.experiments) == 1
        assert prog.gap_count == 5


# =========================================================================
# Unit tests – priority mapping
# =========================================================================


class TestPriorityMapping:
    """Test _priority_from_opportunity helper."""

    def test_tier_1_maps_to_high(self):
        opp = OpportunityScore(
            category="test", title="Test",
            impact=9.0, feasibility=8.0, novelty=9.0,
            market_timing=8.0, composite=34.0,
            priority_tier="Tier 1",
        )
        assert _priority_from_opportunity(opp) == "high"

    def test_tier_2_maps_to_medium(self):
        opp = OpportunityScore(
            category="test", title="Test",
            impact=6.0, feasibility=5.0, novelty=6.0,
            market_timing=5.0, composite=22.0,
            priority_tier="Tier 2",
        )
        assert _priority_from_opportunity(opp) == "medium"

    def test_tier_3_maps_to_low(self):
        opp = OpportunityScore(
            category="test", title="Test",
            impact=3.0, feasibility=3.0, novelty=3.0,
            market_timing=3.0, composite=12.0,
            priority_tier="Tier 3",
        )
        assert _priority_from_opportunity(opp) == "low"

    def test_none_maps_to_medium(self):
        assert _priority_from_opportunity(None) == "medium"


# =========================================================================
# Unit tests – gap experiment templates
# =========================================================================


class TestGapExperimentTemplates:
    """Validate the experiment template catalogue."""

    def test_all_templates_have_required_keys(self):
        required = {"title", "hypothesis", "what_to_change",
                     "success_metric", "estimated_impact"}
        for cat, template in _GAP_EXPERIMENTS.items():
            assert required.issubset(template.keys()), (
                f"Template for {cat!r} missing keys: "
                f"{required - template.keys()}"
            )

    def test_templates_cover_most_gap_categories(self):
        covered = set(_GAP_EXPERIMENTS.keys())
        all_cats = set(REASONING_GAP_CATEGORIES)
        # At least 10 of 15 categories covered
        assert len(covered & all_cats) >= 10

    def test_estimated_impact_valid_values(self):
        valid = {"large", "medium", "small"}
        for cat, template in _GAP_EXPERIMENTS.items():
            assert template["estimated_impact"] in valid, (
                f"Template {cat!r} has invalid impact: "
                f"{template['estimated_impact']}"
            )


# =========================================================================
# Integration tests – program generation
# =========================================================================


class TestProgramGeneration:
    """Test the full program generation pipeline."""

    def test_generates_experiments(self):
        program = generate_experiment_program()
        assert len(program.experiments) > 0

    def test_experiments_have_valid_priorities(self):
        program = generate_experiment_program()
        valid = {"high", "medium", "low"}
        for exp in program.experiments:
            assert exp.priority in valid

    def test_experiments_have_unique_ids(self):
        program = generate_experiment_program()
        ids = [e.id for e in program.experiments]
        assert len(ids) == len(set(ids))

    def test_experiments_reference_real_gap_categories(self):
        program = generate_experiment_program()
        all_cats = set(REASONING_GAP_CATEGORIES)
        for exp in program.experiments:
            assert exp.source_gap in all_cats

    def test_preamble_mentions_model_count(self):
        program = generate_experiment_program()
        assert "10" in program.preamble  # 10 models

    def test_constraints_mention_5_minute_budget(self):
        program = generate_experiment_program()
        assert "5-minute" in program.constraints

    def test_generated_date_is_set(self):
        program = generate_experiment_program()
        assert len(program.generated_date) == 10  # YYYY-MM-DD

    def test_gap_count_matches_analysis(self):
        models = get_reasoning_models()
        gap_result = analyze_gaps(models)
        program = generate_experiment_program(gap_result=gap_result)
        assert program.gap_count == len(gap_result.findings)

    def test_idempotent_generation(self):
        p1 = generate_experiment_program()
        p2 = generate_experiment_program()
        assert len(p1.experiments) == len(p2.experiments)
        for e1, e2 in zip(p1.experiments, p2.experiments):
            assert e1.id == e2.id
            assert e1.title == e2.title


# =========================================================================
# Rendering tests – Markdown
# =========================================================================


class TestRenderMarkdown:
    """Test Markdown rendering for autoresearch program.md."""

    def test_markdown_starts_with_heading(self):
        program = generate_experiment_program()
        md = render_program_markdown(program)
        assert md.startswith("# Autoresearch Experiment Program")

    def test_markdown_contains_experiments_section(self):
        program = generate_experiment_program()
        md = render_program_markdown(program)
        assert "## Experiments" in md

    def test_markdown_contains_constraints(self):
        program = generate_experiment_program()
        md = render_program_markdown(program)
        assert "## Constraints" in md

    def test_markdown_contains_priority_badges(self):
        program = generate_experiment_program()
        md = render_program_markdown(program)
        assert "HIGH" in md or "MEDIUM" in md or "LOW" in md

    def test_markdown_contains_hypothesis(self):
        program = generate_experiment_program()
        md = render_program_markdown(program)
        assert "**Hypothesis:**" in md

    def test_markdown_mentions_gpu_requirement(self):
        program = generate_experiment_program()
        md = render_program_markdown(program)
        assert "NVIDIA GPU" in md

    def test_markdown_mentions_autoresearch_clone(self):
        program = generate_experiment_program()
        md = render_program_markdown(program)
        assert "karpathy/autoresearch" in md

    def test_markdown_footer_present(self):
        program = generate_experiment_program()
        md = render_program_markdown(program)
        assert "Generated on" in md


# =========================================================================
# Rendering tests – plain text
# =========================================================================


class TestRenderText:
    """Test plain-text rendering."""

    def test_text_has_header(self):
        program = generate_experiment_program()
        txt = render_program_text(program)
        assert "AUTORESEARCH EXPERIMENT PROGRAM" in txt

    def test_text_has_experiments(self):
        program = generate_experiment_program()
        txt = render_program_text(program)
        assert "Experiment" in txt

    def test_text_mentions_gpu(self):
        program = generate_experiment_program()
        txt = render_program_text(program)
        assert "NVIDIA GPU" in txt


# =========================================================================
# Agent integration tests
# =========================================================================


class TestAgentIntegration:
    """Test autoresearch integration via ResearcherAgent."""

    def test_agent_autoresearch_program_md(self):
        agent = ResearcherAgent()
        output = agent.autoresearch_program(fmt="md")
        assert "# Autoresearch Experiment Program" in output
        assert "## Experiments" in output

    def test_agent_autoresearch_program_text(self):
        agent = ResearcherAgent()
        output = agent.autoresearch_program(fmt="text")
        assert "AUTORESEARCH EXPERIMENT PROGRAM" in output


# =========================================================================
# CLI tests
# =========================================================================


class TestCLI:
    """Test CLI --autoresearch-program flag."""

    def test_cli_autoresearch_program(self):
        import io
        import sys
        from ai_researcher.__main__ import main

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(["--autoresearch-program"])
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "AUTORESEARCH EXPERIMENT PROGRAM" in output

    def test_cli_autoresearch_program_md(self):
        import io
        import sys
        from ai_researcher.__main__ import main

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(["--autoresearch-program", "--format", "md"])
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "# Autoresearch Experiment Program" in output

    def test_cli_autoresearch_to_file(self, tmp_path):
        from ai_researcher.__main__ import main

        outfile = tmp_path / "program.md"
        main(["--autoresearch-program", "--format", "md", "-o", str(outfile)])
        content = outfile.read_text()
        assert "## Experiments" in content
        assert "karpathy/autoresearch" in content
