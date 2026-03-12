"""Tests for the new differentiating modules:
- trend_forecaster
- model_comparator
- innovation_scorer
- Updated agent methods
- Updated CLI subcommands
- Updated report generator sections
"""

import pytest

from ai_researcher.agent import ResearcherAgent
from ai_researcher.gap_analyzer import analyze_gaps
from ai_researcher.innovation_scorer import (
    OpportunityReport,
    OpportunityScore,
    score_opportunities,
)
from ai_researcher.model_comparator import (
    ComparisonResult,
    UseCaseRecommendation,
    compare_models,
    recommend_models,
)
from ai_researcher.models_knowledge import ReasoningModel, get_reasoning_models
from ai_researcher.report_generator import (
    generate_comparison_text,
    generate_opportunities_text,
    generate_recommendations_text,
    generate_trends_text,
    generate_full_text_report,
    generate_full_markdown_report,
)
from ai_researcher.trend_forecaster import (
    ForecastResult,
    Trend,
    forecast_trends,
)


# -----------------------------------------------------------------------
# trend_forecaster tests
# -----------------------------------------------------------------------

class TestTrendForecaster:
    def test_forecast_returns_result(self):
        result = forecast_trends()
        assert isinstance(result, ForecastResult)
        assert result.model_count > 0

    def test_trends_not_empty(self):
        result = forecast_trends()
        assert len(result.trends) > 0

    def test_trend_fields(self):
        result = forecast_trends()
        for trend in result.trends:
            assert isinstance(trend, Trend)
            assert trend.id
            assert trend.title
            assert trend.description
            assert trend.status in ("emerging", "accelerating", "maturing")
            assert 0.0 <= trend.confidence <= 1.0
            assert trend.time_horizon

    def test_trends_sorted_by_confidence(self):
        result = forecast_trends()
        confidences = [t.confidence for t in result.trends]
        assert confidences == sorted(confidences, reverse=True)

    def test_horizon_summary(self):
        result = forecast_trends()
        assert len(result.horizon_summary) > 0
        total = sum(result.horizon_summary.values())
        assert total == len(result.trends)

    def test_trend_has_market_implications(self):
        result = forecast_trends()
        for trend in result.trends:
            assert len(trend.market_implications) > 0

    def test_forecast_with_custom_models(self):
        custom = [
            ReasoningModel(
                name="CustomModel",
                provider="TestCo",
                release_date="2026-01",
                description="A test model.",
                strengths=["Efficient inference"],
                weaknesses=["Expensive reasoning cost"],
            ),
        ]
        result = forecast_trends(custom)
        assert result.model_count == 1
        assert len(result.trends) > 0


# -----------------------------------------------------------------------
# model_comparator tests
# -----------------------------------------------------------------------

class TestModelComparator:
    def test_compare_two_models(self):
        result = compare_models("OpenAI o3", "DeepSeek R1")
        assert isinstance(result, ComparisonResult)
        assert result.model_a == "OpenAI o3"
        assert result.model_b == "DeepSeek R1"

    def test_compare_case_insensitive(self):
        result = compare_models("openai o3", "deepseek r1")
        assert result.model_a == "OpenAI o3"
        assert result.model_b == "DeepSeek R1"

    def test_benchmark_deltas_computed(self):
        result = compare_models("OpenAI o3", "DeepSeek R1")
        assert len(result.benchmark_deltas) > 0
        for d in result.benchmark_deltas:
            assert d.benchmark
            assert isinstance(d.delta, float)

    def test_verdict_not_empty(self):
        result = compare_models("OpenAI o3", "DeepSeek R1")
        assert len(result.verdict) > 0

    def test_compare_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Model not found"):
            compare_models("OpenAI o3", "NonExistent-999")

    def test_compare_same_model(self):
        result = compare_models("OpenAI o3", "OpenAI o3")
        assert result.model_a == "OpenAI o3"
        assert result.model_b == "OpenAI o3"
        # All deltas should be zero
        for d in result.benchmark_deltas:
            assert d.delta == 0.0

    def test_recommend_models(self):
        recs = recommend_models()
        assert isinstance(recs, list)
        assert len(recs) > 0
        for rec in recs:
            assert isinstance(rec, UseCaseRecommendation)
            assert rec.use_case
            assert rec.recommended_model

    def test_recommend_includes_math_use_case(self):
        recs = recommend_models()
        use_cases = [r.use_case for r in recs]
        assert any("math" in uc.lower() for uc in use_cases)

    def test_recommend_includes_coding_use_case(self):
        recs = recommend_models()
        use_cases = [r.use_case for r in recs]
        assert any("coding" in uc.lower() or "swe" in uc.lower() for uc in use_cases)


# -----------------------------------------------------------------------
# innovation_scorer tests
# -----------------------------------------------------------------------

class TestInnovationScorer:
    def test_score_opportunities_returns_report(self):
        report = score_opportunities()
        assert isinstance(report, OpportunityReport)
        assert len(report.opportunities) > 0

    def test_opportunity_fields(self):
        report = score_opportunities()
        for opp in report.opportunities:
            assert isinstance(opp, OpportunityScore)
            assert opp.category
            assert opp.title
            assert 0 <= opp.impact <= 10
            assert 0 <= opp.feasibility <= 10
            assert 0 <= opp.novelty <= 10
            assert 0 <= opp.market_timing <= 10
            assert 0 < opp.composite <= 10
            assert opp.priority_tier in ("Tier 1", "Tier 2", "Tier 3")

    def test_opportunities_sorted_by_composite(self):
        report = score_opportunities()
        composites = [o.composite for o in report.opportunities]
        assert composites == sorted(composites, reverse=True)

    def test_top_3_summary_not_empty(self):
        report = score_opportunities()
        assert len(report.top_3_summary) > 0

    def test_score_with_precomputed_analysis(self):
        analysis = analyze_gaps()
        report = score_opportunities(analysis)
        assert len(report.opportunities) > 0

    def test_tier1_opportunities_exist(self):
        report = score_opportunities()
        tier1 = [o for o in report.opportunities if o.priority_tier == "Tier 1"]
        assert len(tier1) > 0, "Expected at least one Tier 1 opportunity"


# -----------------------------------------------------------------------
# Updated report_generator tests
# -----------------------------------------------------------------------

class TestNewReportSections:
    def test_trends_text_report(self):
        forecast = forecast_trends()
        report = generate_trends_text(forecast)
        assert "EMERGING TRENDS" in report
        assert "Trend #1" in report

    def test_comparison_text_report(self):
        comp = compare_models("OpenAI o3", "DeepSeek R1")
        report = generate_comparison_text(comp)
        assert "HEAD-TO-HEAD" in report
        assert "OpenAI o3" in report
        assert "DeepSeek R1" in report

    def test_opportunities_text_report(self):
        opp = score_opportunities()
        report = generate_opportunities_text(opp)
        assert "INNOVATION OPPORTUNITY" in report
        assert "Tier" in report

    def test_recommendations_text_report(self):
        recs = recommend_models()
        report = generate_recommendations_text(recs)
        assert "BEST-FIT MODEL" in report
        assert "Recommended" in report

    def test_full_text_report(self):
        gap = analyze_gaps()
        trends = forecast_trends()
        opps = score_opportunities(gap)
        recs = recommend_models()
        report = generate_full_text_report(gap, trends, opps, recs)
        assert "GAP ANALYSIS" in report
        assert "EMERGING TRENDS" in report
        assert "INNOVATION OPPORTUNITY" in report
        assert "BEST-FIT MODEL" in report

    def test_full_markdown_report(self):
        gap = analyze_gaps()
        trends = forecast_trends()
        opps = score_opportunities(gap)
        recs = recommend_models()
        report = generate_full_markdown_report(gap, trends, opps, recs)
        assert "# 🔬 AI Researcher Agent" in report
        assert "## 🔮 Emerging Trends" in report
        assert "## 💡 Innovation Opportunity" in report
        assert "## 🎯 Best-Fit Model" in report


# -----------------------------------------------------------------------
# Updated agent integration tests
# -----------------------------------------------------------------------

class TestAgentNewCapabilities:
    def test_agent_forecast(self):
        agent = ResearcherAgent()
        result = agent.forecast()
        assert isinstance(result, ForecastResult)
        assert len(result.trends) > 0

    def test_agent_compare(self):
        agent = ResearcherAgent()
        result = agent.compare("OpenAI o3", "DeepSeek R1")
        assert isinstance(result, ComparisonResult)

    def test_agent_recommend(self):
        agent = ResearcherAgent()
        recs = agent.recommend()
        assert len(recs) > 0

    def test_agent_score_opportunities(self):
        agent = ResearcherAgent()
        report = agent.score_opportunities()
        assert isinstance(report, OpportunityReport)
        assert len(report.opportunities) > 0

    def test_agent_run_full_text(self):
        agent = ResearcherAgent()
        report = agent.run_full(fmt="text")
        assert "GAP ANALYSIS" in report
        assert "EMERGING TRENDS" in report

    def test_agent_run_full_markdown(self):
        agent = ResearcherAgent()
        report = agent.run_full(fmt="md")
        assert "# 🔬 AI Researcher Agent" in report
        assert "## 🔮 Emerging Trends" in report

    def test_agent_version_updated(self):
        import ai_researcher
        assert ai_researcher.__version__ == "3.0.0"
