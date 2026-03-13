"""Comprehensive regression test suite for the AI Researcher Agent.

Ensures stability across all modules, verifies cross-module integration,
tests edge cases, validates CLI behavior, and guards against regressions
in output structure and data integrity.
"""

import pytest
import copy

from ai_researcher import __version__
from ai_researcher.agent import ResearcherAgent
from ai_researcher.competitive_edge import (
    LandscapeReport,
    ProviderVelocity,
    RiskReport,
    RiskScore,
    SegmentOverlap,
    VelocityReport,
    analyze_velocity,
    assess_risks,
    map_landscape,
)
from ai_researcher.gap_analyzer import (
    AnalysisResult,
    GapFinding,
    _match_category,
    _severity_for_category_count,
    analyze_gaps,
)
from ai_researcher.innovation_scorer import (
    OpportunityReport,
    OpportunityScore,
    _assign_tier,
    _compute_composite,
    score_opportunities,
)
from ai_researcher.model_comparator import (
    ComparisonResult,
    UseCaseRecommendation,
    compare_models,
    recommend_models,
)
from ai_researcher.models_knowledge import (
    BenchmarkScore,
    REASONING_GAP_CATEGORIES,
    ReasoningModel,
    get_model_by_name,
    get_models_by_provider,
    get_reasoning_models,
)
from ai_researcher.report_generator import (
    generate_comparison_text,
    generate_full_markdown_report,
    generate_full_text_report,
    generate_landscape_text,
    generate_markdown_report,
    generate_opportunities_text,
    generate_recommendations_text,
    generate_risks_text,
    generate_roadmap_text,
    generate_text_report,
    generate_trends_text,
    generate_velocity_text,
)
from ai_researcher.research_roadmap import (
    ResearchRoadmap,
    RoadmapMilestone,
    generate_roadmap,
)
from ai_researcher.trend_forecaster import (
    ForecastResult,
    Trend,
    forecast_trends,
)


# ===================================================================
# REGRESSION: Version guard
# ===================================================================

class TestVersionRegression:
    def test_version_is_3_1_0(self):
        assert __version__ == "3.1.0"


# ===================================================================
# REGRESSION: Knowledge base invariants
# ===================================================================

class TestKnowledgeBaseRegression:
    """Guard against accidental changes to the knowledge base."""

    def test_minimum_model_count(self):
        """Must have at least 10 models."""
        models = get_reasoning_models()
        assert len(models) >= 10

    def test_known_model_names_present(self):
        """Key models must remain in the knowledge base."""
        names = {m.name for m in get_reasoning_models()}
        required = {
            "OpenAI o3", "OpenAI o3-mini", "OpenAI o1",
            "DeepSeek R1", "Gemini 2.5 Pro",
            "Claude 3.7 Sonnet", "QwQ-32B", "Grok 3",
            "Phi-4-reasoning",
        }
        missing = required - names
        assert not missing, f"Missing models: {missing}"

    def test_known_providers_present(self):
        providers = {m.provider for m in get_reasoning_models()}
        required = {"OpenAI", "DeepSeek", "Google DeepMind", "Anthropic"}
        assert required.issubset(providers)

    def test_gap_category_count_unchanged(self):
        """15 gap categories must remain stable."""
        assert len(REASONING_GAP_CATEGORIES) == 15

    def test_gap_categories_are_unique(self):
        assert len(REASONING_GAP_CATEGORIES) == len(set(REASONING_GAP_CATEGORIES))

    def test_all_models_have_at_least_one_benchmark(self):
        for m in get_reasoning_models():
            assert len(m.benchmarks) >= 1, f"{m.name} missing benchmarks"

    def test_all_models_have_at_least_one_weakness(self):
        for m in get_reasoning_models():
            assert len(m.weaknesses) >= 1, f"{m.name} missing weaknesses"

    def test_all_models_have_at_least_one_strength(self):
        for m in get_reasoning_models():
            assert len(m.strengths) >= 1, f"{m.name} missing strengths"

    def test_benchmark_scores_in_valid_range(self):
        for m in get_reasoning_models():
            for bs in m.benchmarks:
                assert 0 <= bs.score <= 100, (
                    f"{m.name} – {bs.benchmark}: score {bs.score} out of range"
                )

    def test_open_source_count_at_least_3(self):
        oss = [m for m in get_reasoning_models() if m.open_source]
        assert len(oss) >= 3

    def test_multimodal_model_exists(self):
        models = get_reasoning_models()
        multi = [m for m in models if len(m.modalities) > 1]
        assert len(multi) >= 1, "Need at least one multi-modal model"


# ===================================================================
# REGRESSION: Gap analysis invariants
# ===================================================================

class TestGapAnalysisRegression:
    def test_findings_count_at_least_10(self):
        result = analyze_gaps()
        assert len(result.findings) >= 10

    def test_high_or_critical_findings_exist(self):
        result = analyze_gaps()
        severe = [f for f in result.findings if f.severity in ("critical", "high")]
        assert len(severe) >= 1

    def test_all_findings_have_descriptions(self):
        result = analyze_gaps()
        for f in result.findings:
            assert f.description, f"Finding {f.category} missing description"

    def test_all_findings_have_research_directions(self):
        result = analyze_gaps()
        for f in result.findings:
            assert len(f.research_directions) >= 1, (
                f"Finding {f.category} missing research directions"
            )

    def test_severity_ordering_preserved(self):
        """Findings must be sorted: critical > high > medium > low."""
        result = analyze_gaps()
        order_map = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        prev = -1
        for f in result.findings:
            cur = order_map[f.severity]
            assert cur >= prev, "Findings not sorted by severity"
            prev = cur

    def test_category_coverage_has_all_categories(self):
        result = analyze_gaps()
        for cat in REASONING_GAP_CATEGORIES:
            assert cat in result.category_coverage

    def test_severity_heuristic_consistency(self):
        """Verify the severity heuristic produces valid severities."""
        assert _severity_for_category_count(8, 10) == "critical"
        assert _severity_for_category_count(6, 10) == "high"
        assert _severity_for_category_count(4, 10) == "medium"
        assert _severity_for_category_count(2, 10) == "low"

    def test_match_category_keywords(self):
        """Spot-check keyword matching."""
        assert _match_category("expensive cost latency", "efficiency")
        assert _match_category("hallucination risk", "hallucination")
        assert _match_category("limited multi-modal support", "multi_modal_reasoning")
        assert not _match_category("great model", "hallucination")

    def test_analyze_idempotent(self):
        """Two calls with same input → same results."""
        models = get_reasoning_models()
        r1 = analyze_gaps(models)
        r2 = analyze_gaps(models)
        assert len(r1.findings) == len(r2.findings)
        assert r1.model_count == r2.model_count
        for f1, f2 in zip(r1.findings, r2.findings):
            assert f1.category == f2.category
            assert f1.severity == f2.severity

    def test_empty_model_list_produces_empty_findings(self):
        result = analyze_gaps([])
        assert result.model_count == 0
        assert len(result.findings) == 0


# ===================================================================
# REGRESSION: Trend forecaster invariants
# ===================================================================

class TestTrendForecasterRegression:
    def test_at_least_8_trends(self):
        result = forecast_trends()
        assert len(result.trends) >= 8

    def test_all_trend_statuses_valid(self):
        result = forecast_trends()
        valid = {"emerging", "accelerating", "maturing"}
        for t in result.trends:
            assert t.status in valid

    def test_confidence_range(self):
        result = forecast_trends()
        for t in result.trends:
            assert 0.0 <= t.confidence <= 1.0

    def test_no_duplicate_trend_ids(self):
        result = forecast_trends()
        ids = [t.id for t in result.trends]
        assert len(ids) == len(set(ids))

    def test_forecast_idempotent(self):
        models = get_reasoning_models()
        r1 = forecast_trends(models)
        r2 = forecast_trends(models)
        assert len(r1.trends) == len(r2.trends)

    def test_trends_have_affected_categories(self):
        result = forecast_trends()
        for t in result.trends:
            assert len(t.affected_categories) >= 1


# ===================================================================
# REGRESSION: Model comparator invariants
# ===================================================================

class TestModelComparatorRegression:
    def test_all_models_can_be_compared_to_themselves(self):
        """Every model must compare to itself without error."""
        models = get_reasoning_models()
        for m in models:
            result = compare_models(m.name, m.name, models)
            assert result.model_a == m.name
            for d in result.benchmark_deltas:
                assert d.delta == 0.0

    def test_comparison_symmetry(self):
        """Comparing A-B and B-A should produce consistent deltas."""
        r1 = compare_models("OpenAI o3", "DeepSeek R1")
        r2 = compare_models("DeepSeek R1", "OpenAI o3")
        bm_set_1 = {d.benchmark for d in r1.benchmark_deltas}
        bm_set_2 = {d.benchmark for d in r2.benchmark_deltas}
        assert bm_set_1 == bm_set_2
        for d1 in r1.benchmark_deltas:
            for d2 in r2.benchmark_deltas:
                if d1.benchmark == d2.benchmark:
                    assert abs(d1.delta + d2.delta) < 0.01

    def test_recommendations_cover_all_use_cases(self):
        recs = recommend_models()
        assert len(recs) >= 5, "Expected at least 5 use-case recommendations"


# ===================================================================
# REGRESSION: Innovation scorer invariants
# ===================================================================

class TestInnovationScorerRegression:
    def test_all_dimensions_positive(self):
        report = score_opportunities()
        for opp in report.opportunities:
            assert opp.impact > 0
            assert opp.feasibility > 0
            assert opp.novelty > 0
            assert opp.market_timing > 0

    def test_composite_within_range(self):
        report = score_opportunities()
        for opp in report.opportunities:
            assert 0 < opp.composite <= 10

    def test_tier_assignment_consistency(self):
        assert _assign_tier(8.0) == "Tier 1"
        assert _assign_tier(7.5) == "Tier 1"
        assert _assign_tier(7.0) == "Tier 2"
        assert _assign_tier(6.0) == "Tier 2"
        assert _assign_tier(5.0) == "Tier 3"

    def test_composite_calculation(self):
        result = _compute_composite({
            "impact": 10.0,
            "feasibility": 10.0,
            "novelty": 10.0,
            "market_timing": 10.0,
        })
        assert result == 10.0


# ===================================================================
# REGRESSION: Competitive edge – research velocity
# ===================================================================

class TestVelocityRegression:
    def test_velocity_returns_report(self):
        report = analyze_velocity()
        assert isinstance(report, VelocityReport)
        assert len(report.providers) > 0

    def test_all_providers_covered(self):
        models = get_reasoning_models()
        expected_providers = {m.provider for m in models}
        report = analyze_velocity()
        actual = {pv.provider for pv in report.providers}
        assert expected_providers == actual

    def test_velocity_fields_valid(self):
        report = analyze_velocity()
        for pv in report.providers:
            assert pv.model_count >= 1
            assert pv.velocity_rating in ("fast", "moderate", "slow")
            assert 0.0 <= pv.open_source_ratio <= 1.0
            assert pv.avg_benchmark_score >= 0

    def test_fastest_provider_set(self):
        report = analyze_velocity()
        assert report.fastest_provider

    def test_velocity_with_single_model(self):
        custom = [
            ReasoningModel(
                name="Solo",
                provider="SoloCo",
                release_date="2026-01",
                description="A lone model.",
                strengths=["Fast"],
                weaknesses=["Expensive"],
                benchmarks=[BenchmarkScore("TEST", 50.0)],
            ),
        ]
        report = analyze_velocity(custom)
        assert len(report.providers) == 1
        assert report.providers[0].release_span_months == 0


# ===================================================================
# REGRESSION: Competitive edge – risk assessment
# ===================================================================

class TestRiskRegression:
    def test_risk_returns_report(self):
        report = assess_risks()
        assert isinstance(report, RiskReport)
        assert len(report.scores) > 0

    def test_all_models_assessed(self):
        models = get_reasoning_models()
        report = assess_risks()
        assert len(report.scores) == len(models)

    def test_risk_fields_valid(self):
        report = assess_risks()
        for rs in report.scores:
            assert rs.risk_tier in ("low", "medium", "high")
            assert 0 <= rs.vendor_lock_in <= 10
            assert 0 <= rs.deprecation_risk <= 10
            assert 0 <= rs.api_stability <= 10
            assert 0 <= rs.ecosystem_maturity <= 10

    def test_open_source_lower_vendor_lockin(self):
        """Open-source models should have vendor_lock_in ≤ 2.5."""
        report = assess_risks()
        for rs in report.scores:
            model = get_model_by_name(rs.model_name)
            if model and model.open_source:
                assert rs.vendor_lock_in <= 2.5

    def test_safest_and_riskiest_set(self):
        report = assess_risks()
        assert report.safest_model
        assert report.riskiest_model

    def test_risk_sorted_by_composite(self):
        report = assess_risks()
        composites = [rs.composite_risk for rs in report.scores]
        assert composites == sorted(composites)


# ===================================================================
# REGRESSION: Competitive edge – landscape mapping
# ===================================================================

class TestLandscapeRegression:
    def test_landscape_returns_report(self):
        report = map_landscape()
        assert isinstance(report, LandscapeReport)
        assert len(report.segments) > 0

    def test_at_least_8_segments(self):
        report = map_landscape()
        assert len(report.segments) >= 8

    def test_some_segments_have_models(self):
        report = map_landscape()
        filled = [seg for seg, ms in report.segments.items() if len(ms) > 0]
        assert len(filled) >= 5

    def test_overlaps_computed(self):
        report = map_landscape()
        # Any segment with 2+ models should produce overlaps
        multi = [s for s, ms in report.segments.items() if len(ms) >= 2]
        if multi:
            assert len(report.overlaps) > 0

    def test_landscape_with_empty_models(self):
        report = map_landscape([])
        assert len(report.segments) > 0
        for ms in report.segments.values():
            assert len(ms) == 0


# ===================================================================
# REGRESSION: Research roadmap
# ===================================================================

class TestRoadmapRegression:
    def test_roadmap_returns_result(self):
        roadmap = generate_roadmap()
        assert isinstance(roadmap, ResearchRoadmap)
        assert len(roadmap.milestones) > 0

    def test_milestones_have_required_fields(self):
        roadmap = generate_roadmap()
        for ms in roadmap.milestones:
            assert ms.id
            assert ms.title
            assert ms.description
            assert ms.phase in ("Phase 1", "Phase 2", "Phase 3")
            assert ms.priority in ("critical", "high", "medium")
            assert ms.effort_level in ("small", "medium", "large")

    def test_milestones_have_deliverables(self):
        roadmap = generate_roadmap()
        for ms in roadmap.milestones:
            assert len(ms.deliverables) >= 1

    def test_milestones_have_success_metrics(self):
        roadmap = generate_roadmap()
        for ms in roadmap.milestones:
            assert len(ms.success_metrics) >= 1

    def test_phase_ordering(self):
        """Phase 1 milestones should come before Phase 2, etc."""
        roadmap = generate_roadmap()
        phases = [ms.phase for ms in roadmap.milestones]
        phase_order = {"Phase 1": 0, "Phase 2": 1, "Phase 3": 2}
        prev = -1
        for p in phases:
            cur = phase_order[p]
            assert cur >= prev, "Milestones not sorted by phase"
            prev = cur

    def test_executive_summary_present(self):
        roadmap = generate_roadmap()
        assert len(roadmap.executive_summary) > 20

    def test_phase_summary_totals(self):
        roadmap = generate_roadmap()
        total = sum(roadmap.phase_summary.values())
        assert total == len(roadmap.milestones)


# ===================================================================
# REGRESSION: Report generator – all sections
# ===================================================================

class TestReportGeneratorRegression:
    """Ensure all report sections produce non-empty, structured output."""

    def setup_method(self):
        self.agent = ResearcherAgent()
        self.gap = self.agent.analyze()
        self.trends = self.agent.forecast()
        self.opps = self.agent.score_opportunities()
        self.recs = self.agent.recommend()
        self.vel = self.agent.velocity()
        self.risks = self.agent.assess_risks()
        self.land = self.agent.landscape()
        self.road = self.agent.roadmap()

    def test_text_report_not_empty(self):
        assert len(generate_text_report(self.gap)) > 500

    def test_markdown_report_not_empty(self):
        assert len(generate_markdown_report(self.gap)) > 500

    def test_velocity_text_contains_header(self):
        report = generate_velocity_text(self.vel)
        assert "RESEARCH VELOCITY" in report
        assert len(report) > 100

    def test_risks_text_contains_header(self):
        report = generate_risks_text(self.risks)
        assert "RISK ASSESSMENT" in report
        assert len(report) > 100

    def test_landscape_text_contains_header(self):
        report = generate_landscape_text(self.land)
        assert "COMPETITIVE LANDSCAPE" in report
        assert len(report) > 100

    def test_roadmap_text_contains_header(self):
        report = generate_roadmap_text(self.road)
        assert "RESEARCH ROADMAP" in report
        assert len(report) > 100

    def test_full_text_includes_all_sections(self):
        report = generate_full_text_report(
            self.gap, self.trends, self.opps, self.recs,
            self.vel, self.risks, self.land, self.road,
        )
        assert "GAP ANALYSIS" in report
        assert "EMERGING TRENDS" in report
        assert "INNOVATION OPPORTUNITY" in report
        assert "BEST-FIT MODEL" in report
        assert "RESEARCH VELOCITY" in report
        assert "RISK ASSESSMENT" in report
        assert "COMPETITIVE LANDSCAPE" in report
        assert "RESEARCH ROADMAP" in report

    def test_full_markdown_includes_all_sections(self):
        report = generate_full_markdown_report(
            self.gap, self.trends, self.opps, self.recs,
            self.vel, self.risks, self.land, self.road,
        )
        assert "# 🔬 AI Researcher Agent" in report
        assert "## 🔮 Emerging Trends" in report
        assert "## 💡 Innovation Opportunity" in report
        assert "## 🎯 Best-Fit Model" in report
        assert "## ⚡ Research Velocity" in report
        assert "## 🛡️ Risk Assessment" in report
        assert "## 🗺️ Competitive Landscape" in report
        assert "## 🗓️ Research Roadmap" in report

    def test_full_text_backward_compat_without_new_sections(self):
        """Old 4-arg signature still works (without v3 sections)."""
        report = generate_full_text_report(
            self.gap, self.trends, self.opps, self.recs,
        )
        assert "GAP ANALYSIS" in report
        assert "RESEARCH VELOCITY" not in report

    def test_full_markdown_backward_compat_without_new_sections(self):
        """Old 4-arg signature still works (without v3 sections)."""
        report = generate_full_markdown_report(
            self.gap, self.trends, self.opps, self.recs,
        )
        assert "# 🔬 AI Researcher Agent" in report
        assert "## ⚡ Research Velocity" not in report


# ===================================================================
# REGRESSION: Agent integration (full pipeline)
# ===================================================================

class TestAgentIntegrationRegression:
    def test_run_text_stable(self):
        agent = ResearcherAgent()
        report = agent.run(fmt="text")
        assert "REASONING MODEL GAP ANALYSIS" in report
        assert "Finding #1" in report

    def test_run_markdown_stable(self):
        agent = ResearcherAgent()
        report = agent.run(fmt="md")
        assert "# 🔬 AI Researcher Agent" in report

    def test_run_full_text_stable(self):
        agent = ResearcherAgent()
        report = agent.run_full(fmt="text")
        assert "GAP ANALYSIS" in report
        assert "RESEARCH VELOCITY" in report
        assert "RESEARCH ROADMAP" in report

    def test_run_full_markdown_stable(self):
        agent = ResearcherAgent()
        report = agent.run_full(fmt="md")
        assert "# 🔬 AI Researcher Agent" in report
        assert "## 🗓️ Research Roadmap" in report

    def test_agent_velocity(self):
        agent = ResearcherAgent()
        result = agent.velocity()
        assert isinstance(result, VelocityReport)
        assert len(result.providers) > 0

    def test_agent_assess_risks(self):
        agent = ResearcherAgent()
        result = agent.assess_risks()
        assert isinstance(result, RiskReport)
        assert len(result.scores) > 0

    def test_agent_landscape(self):
        agent = ResearcherAgent()
        result = agent.landscape()
        assert isinstance(result, LandscapeReport)
        assert len(result.segments) > 0

    def test_agent_roadmap(self):
        agent = ResearcherAgent()
        result = agent.roadmap()
        assert isinstance(result, ResearchRoadmap)
        assert len(result.milestones) > 0


# ===================================================================
# REGRESSION: CLI subcommand tests
# ===================================================================

class TestCLIRegression:
    """Test CLI via main() with argv injection."""

    def test_cli_default(self, capsys):
        from ai_researcher.__main__ import main
        main([])
        out = capsys.readouterr().out
        assert "REASONING MODEL GAP ANALYSIS" in out

    def test_cli_full(self, capsys):
        from ai_researcher.__main__ import main
        main(["--full"])
        out = capsys.readouterr().out
        assert "GAP ANALYSIS" in out
        assert "RESEARCH VELOCITY" in out

    def test_cli_trends(self, capsys):
        from ai_researcher.__main__ import main
        main(["--trends"])
        out = capsys.readouterr().out
        assert "EMERGING TRENDS" in out

    def test_cli_opportunities(self, capsys):
        from ai_researcher.__main__ import main
        main(["--opportunities"])
        out = capsys.readouterr().out
        assert "INNOVATION OPPORTUNITY" in out

    def test_cli_recommend(self, capsys):
        from ai_researcher.__main__ import main
        main(["--recommend"])
        out = capsys.readouterr().out
        assert "BEST-FIT MODEL" in out

    def test_cli_compare(self, capsys):
        from ai_researcher.__main__ import main
        main(["--compare", "OpenAI o3", "DeepSeek R1"])
        out = capsys.readouterr().out
        assert "HEAD-TO-HEAD" in out

    def test_cli_velocity(self, capsys):
        from ai_researcher.__main__ import main
        main(["--velocity"])
        out = capsys.readouterr().out
        assert "RESEARCH VELOCITY" in out

    def test_cli_risks(self, capsys):
        from ai_researcher.__main__ import main
        main(["--risks"])
        out = capsys.readouterr().out
        assert "RISK ASSESSMENT" in out

    def test_cli_landscape(self, capsys):
        from ai_researcher.__main__ import main
        main(["--landscape"])
        out = capsys.readouterr().out
        assert "COMPETITIVE LANDSCAPE" in out

    def test_cli_roadmap(self, capsys):
        from ai_researcher.__main__ import main
        main(["--roadmap"])
        out = capsys.readouterr().out
        assert "RESEARCH ROADMAP" in out

    def test_cli_list_models(self, capsys):
        from ai_researcher.__main__ import main
        main(["--list-models"])
        out = capsys.readouterr().out
        assert "OpenAI o3" in out
        assert "DeepSeek R1" in out

    def test_cli_markdown_format(self, capsys):
        from ai_researcher.__main__ import main
        main(["--format", "md"])
        out = capsys.readouterr().out
        assert "# 🔬 AI Researcher Agent" in out

    def test_cli_output_to_file(self, tmp_path):
        from ai_researcher.__main__ import main
        outfile = str(tmp_path / "report.txt")
        main(["--output", outfile])
        with open(outfile) as f:
            content = f.read()
        assert "REASONING MODEL GAP ANALYSIS" in content


# ===================================================================
# REGRESSION: Edge cases and error handling
# ===================================================================

class TestEdgeCases:
    def test_compare_nonexistent_model_raises(self):
        with pytest.raises(ValueError, match="Model not found"):
            compare_models("Foo", "Bar")

    def test_compare_one_nonexistent_model_raises(self):
        with pytest.raises(ValueError, match="Model not found"):
            compare_models("OpenAI o3", "DoesNotExist")

    def test_analyze_single_model(self):
        single = [
            ReasoningModel(
                name="OnlyModel",
                provider="TestCo",
                release_date="2026-01",
                description="Sole model.",
                strengths=["Fast inference"],
                weaknesses=["Expensive"],
                benchmarks=[BenchmarkScore("TEST", 80.0)],
            ),
        ]
        result = analyze_gaps(single)
        assert result.model_count == 1
        # "Expensive" matches "efficiency"
        cats = {f.category for f in result.findings}
        assert "efficiency" in cats

    def test_model_with_no_weaknesses_produces_no_findings(self):
        clean = [
            ReasoningModel(
                name="PerfectModel",
                provider="TestCo",
                release_date="2026-01",
                description="Flawless.",
                strengths=["Everything"],
                weaknesses=[],
                benchmarks=[BenchmarkScore("TEST", 100.0)],
            ),
        ]
        result = analyze_gaps(clean)
        assert len(result.findings) == 0

    def test_score_opportunities_with_no_findings(self):
        analysis = AnalysisResult(findings=[], model_count=0, category_coverage={})
        report = score_opportunities(analysis)
        assert len(report.opportunities) == 0

    def test_recommend_with_single_model(self):
        single = [
            ReasoningModel(
                name="LoneWolf",
                provider="Solo",
                release_date="2026-01",
                description="Only model.",
                strengths=["Math"],
                weaknesses=["Slow"],
                benchmarks=[BenchmarkScore("MATH-500", 95.0)],
            ),
        ]
        recs = recommend_models(single)
        assert len(recs) >= 1

    def test_velocity_with_empty_models(self):
        report = analyze_velocity([])
        assert len(report.providers) == 0

    def test_risks_with_unknown_provider(self):
        custom = [
            ReasoningModel(
                name="AlienModel",
                provider="MarsAI",
                release_date="2026-01",
                description="From Mars.",
                strengths=["Advanced"],
                weaknesses=["Unknown"],
                benchmarks=[BenchmarkScore("TEST", 60.0)],
            ),
        ]
        report = assess_risks(custom)
        assert len(report.scores) == 1
        assert report.scores[0].risk_tier in ("low", "medium", "high")


# ===================================================================
# REGRESSION: Cross-module data flow
# ===================================================================

class TestCrossModuleRegression:
    """Verify data flows correctly between modules."""

    def test_gap_findings_feed_opportunity_scorer(self):
        """Every gap finding should have a corresponding opportunity score."""
        analysis = analyze_gaps()
        report = score_opportunities(analysis)
        gap_cats = {f.category for f in analysis.findings}
        opp_cats = {o.category for o in report.opportunities}
        # Opportunity scorer should cover most gap categories
        assert len(opp_cats & gap_cats) >= len(gap_cats) * 0.7

    def test_gap_findings_feed_roadmap(self):
        """Roadmap milestones should map to gap findings."""
        roadmap = generate_roadmap()
        analysis = analyze_gaps()
        gap_cats = {f.category for f in analysis.findings}
        roadmap_gaps = set()
        for ms in roadmap.milestones:
            roadmap_gaps.update(ms.related_gaps)
        assert len(roadmap_gaps & gap_cats) >= 5

    def test_trends_feed_roadmap(self):
        """Roadmap should reference trend IDs."""
        roadmap = generate_roadmap()
        trends = forecast_trends()
        trend_ids = {t.id for t in trends.trends}
        roadmap_trends = set()
        for ms in roadmap.milestones:
            roadmap_trends.update(ms.related_trends)
        assert len(roadmap_trends & trend_ids) >= 3

    def test_agent_run_full_exercises_all_modules(self):
        """Full report should contain output from every module."""
        agent = ResearcherAgent()
        report = agent.run_full(fmt="text")
        markers = [
            "GAP ANALYSIS", "EMERGING TRENDS",
            "INNOVATION OPPORTUNITY", "BEST-FIT MODEL",
            "RESEARCH VELOCITY", "RISK ASSESSMENT",
            "COMPETITIVE LANDSCAPE", "RESEARCH ROADMAP",
        ]
        for marker in markers:
            assert marker in report, f"Missing section: {marker}"
