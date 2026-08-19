"""Tests for the AI Researcher Agent."""

import pytest

from ai_researcher.agent import ResearcherAgent
from ai_researcher.gap_analyzer import (
    AnalysisResult,
    GapFinding,
    analyze_gaps,
    _match_category,
)
from ai_researcher.models_knowledge import (
    REASONING_GAP_CATEGORIES,
    ReasoningModel,
    get_model_by_name,
    get_models_by_provider,
    get_reasoning_models,
)
from ai_researcher.report_generator import (
    generate_markdown_report,
    generate_text_report,
)


# -----------------------------------------------------------------------
# models_knowledge tests
# -----------------------------------------------------------------------

class TestModelsKnowledge:
    def test_get_reasoning_models_returns_non_empty(self):
        models = get_reasoning_models()
        assert len(models) > 0

    def test_all_models_have_required_fields(self):
        for m in get_reasoning_models():
            assert m.name
            assert m.provider
            assert m.release_date
            assert m.description

    def test_get_model_by_name_found(self):
        model = get_model_by_name("OpenAI o3")
        assert model is not None
        assert model.provider == "OpenAI"

    def test_get_model_by_name_case_insensitive(self):
        model = get_model_by_name("deepseek r1")
        assert model is not None
        assert model.name == "DeepSeek R1"

    def test_get_model_by_name_not_found(self):
        assert get_model_by_name("NonExistentModel-999") is None

    def test_get_models_by_provider(self):
        openai_models = get_models_by_provider("OpenAI")
        assert len(openai_models) >= 2
        assert all(m.provider == "OpenAI" for m in openai_models)

    def test_gap_categories_are_strings(self):
        assert all(isinstance(c, str) for c in REASONING_GAP_CATEGORIES)

    def test_open_source_models_exist(self):
        models = get_reasoning_models()
        oss = [m for m in models if m.open_source]
        assert len(oss) >= 2

    def test_models_have_benchmarks(self):
        for m in get_reasoning_models():
            assert len(m.benchmarks) > 0, f"{m.name} has no benchmarks"


# -----------------------------------------------------------------------
# gap_analyzer tests
# -----------------------------------------------------------------------

class TestGapAnalyzer:
    def test_analyze_gaps_returns_result(self):
        result = analyze_gaps()
        assert isinstance(result, AnalysisResult)
        assert result.model_count > 0

    def test_findings_have_required_fields(self):
        result = analyze_gaps()
        for f in result.findings:
            assert f.category in REASONING_GAP_CATEGORIES
            assert f.severity in ("critical", "high", "medium", "low")
            assert f.title
            assert len(f.affected_models) > 0

    def test_findings_sorted_by_severity(self):
        result = analyze_gaps()
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        severities = [order[f.severity] for f in result.findings]
        assert severities == sorted(severities)

    def test_category_coverage_populated(self):
        result = analyze_gaps()
        assert len(result.category_coverage) == len(REASONING_GAP_CATEGORIES)

    def test_match_category_positive(self):
        assert _match_category("Expensive inference cost", "efficiency")

    def test_match_category_negative(self):
        assert not _match_category("Great performance", "hallucination")

    def test_analyze_with_custom_models(self):
        custom = [
            ReasoningModel(
                name="TestModel",
                provider="TestCo",
                release_date="2026-01",
                description="A test model.",
                weaknesses=["Expensive inference", "Hallucination risk"],
            ),
        ]
        result = analyze_gaps(custom)
        assert result.model_count == 1
        cats = {f.category for f in result.findings}
        assert "efficiency" in cats
        assert "hallucination" in cats


# -----------------------------------------------------------------------
# report_generator tests
# -----------------------------------------------------------------------

class TestReportGenerator:
    def setup_method(self):
        self.result = analyze_gaps()

    def test_text_report_contains_header(self):
        report = generate_text_report(self.result)
        assert "REASONING MODEL GAP ANALYSIS" in report

    def test_text_report_contains_findings(self):
        report = generate_text_report(self.result)
        assert "Finding #1" in report

    def test_markdown_report_contains_header(self):
        report = generate_markdown_report(self.result)
        assert "# 🔬 AI Researcher Agent" in report

    def test_markdown_report_contains_table(self):
        report = generate_markdown_report(self.result)
        assert "| Category |" in report

    def test_markdown_report_has_toc(self):
        report = generate_markdown_report(self.result)
        assert "## Table of Contents" in report

    def test_reports_not_empty(self):
        assert len(generate_text_report(self.result)) > 100
        assert len(generate_markdown_report(self.result)) > 100


# -----------------------------------------------------------------------
# agent tests
# -----------------------------------------------------------------------

class TestResearcherAgent:
    def test_run_text(self):
        agent = ResearcherAgent()
        report = agent.run(fmt="text")
        assert "REASONING MODEL GAP ANALYSIS" in report

    def test_run_markdown(self):
        agent = ResearcherAgent()
        report = agent.run(fmt="md")
        assert "# 🔬 AI Researcher Agent" in report

    def test_list_models(self):
        agent = ResearcherAgent()
        names = agent.list_models()
        assert "OpenAI o3" in names
        assert "DeepSeek R1" in names

    def test_lookup_model(self):
        agent = ResearcherAgent()
        model = agent.lookup_model("Gemini 2.5 Pro")
        assert model is not None
        assert model.provider == "Google DeepMind"

    def test_filter_by_provider(self):
        agent = ResearcherAgent()
        models = agent.filter_by_provider("Anthropic")
        assert len(models) >= 1
        assert all(m.provider == "Anthropic" for m in models)

    def test_analyze_caches_result(self):
        agent = ResearcherAgent()
        agent.analyze()
        assert agent._result is not None
        assert len(agent._result.findings) > 0
