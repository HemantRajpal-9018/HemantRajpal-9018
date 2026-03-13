"""Main agent orchestration module.

``ResearcherAgent`` ties together the knowledge base, gap analyzer,
trend forecaster, model comparator, innovation scorer, competitive edge
analytics, research roadmap, and report generator into a single callable
interface.
"""

from __future__ import annotations

from typing import List, Optional

from ai_researcher.competitive_edge import (
    LandscapeReport,
    RiskReport,
    VelocityReport,
    analyze_velocity,
    assess_risks,
    map_landscape,
)
from ai_researcher.gap_analyzer import AnalysisResult, analyze_gaps
from ai_researcher.innovation_scorer import (
    OpportunityReport,
    score_opportunities,
)
from ai_researcher.model_comparator import (
    ComparisonResult,
    UseCaseRecommendation,
    compare_models,
    recommend_models,
)
from ai_researcher.models_knowledge import (
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
from ai_researcher.autoresearch_adapter import (
    ExperimentProgram,
    generate_experiment_program,
    render_program_markdown,
    render_program_text,
)
from ai_researcher.research_roadmap import ResearchRoadmap, generate_roadmap
from ai_researcher.sandbox_config import (
    SandboxConfig,
    detect_sandbox_config,
    render_config_markdown,
    render_config_text,
)
from ai_researcher.trend_forecaster import ForecastResult, forecast_trends


class ResearcherAgent:
    """AI Researcher Agent that finds gaps in the latest reasoning models.

    Usage::

        agent = ResearcherAgent()
        report = agent.run()               # full analysis → text report
        report = agent.run(fmt="md")       # full analysis → Markdown report
        report = agent.run_full()          # all modules → comprehensive text
        report = agent.run_full(fmt="md")  # all modules → comprehensive Markdown
    """

    def __init__(
        self,
        models: Optional[List[ReasoningModel]] = None,
    ) -> None:
        self.models = models or get_reasoning_models()
        self._result: Optional[AnalysisResult] = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def list_models(self) -> List[str]:
        """Return names of all models in the knowledge base."""
        return [m.name for m in self.models]

    def lookup_model(self, name: str) -> Optional[ReasoningModel]:
        """Look up a model by name."""
        return get_model_by_name(name)

    def filter_by_provider(self, provider: str) -> List[ReasoningModel]:
        """Return models matching *provider*."""
        return get_models_by_provider(provider)

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self) -> AnalysisResult:
        """Run gap analysis and cache the result."""
        self._result = analyze_gaps(self.models)
        return self._result

    # ------------------------------------------------------------------
    # Differentiating capabilities
    # ------------------------------------------------------------------

    def forecast(self) -> ForecastResult:
        """Run trend forecasting on the model landscape."""
        return forecast_trends(self.models)

    def compare(self, name_a: str, name_b: str) -> ComparisonResult:
        """Head-to-head comparison of two models."""
        return compare_models(name_a, name_b, self.models)

    def recommend(self) -> List[UseCaseRecommendation]:
        """Best-fit model recommendations for common use cases."""
        return recommend_models(self.models)

    def score_opportunities(self) -> OpportunityReport:
        """Score each gap by innovation opportunity value."""
        result = self.analyze()
        return score_opportunities(result, self.models)

    # ------------------------------------------------------------------
    # Competitive edge capabilities (v3)
    # ------------------------------------------------------------------

    def velocity(self) -> VelocityReport:
        """Analyze research velocity across providers."""
        return analyze_velocity(self.models)

    def assess_risks(self) -> RiskReport:
        """Run risk assessment across all models."""
        return assess_risks(self.models)

    def landscape(self) -> LandscapeReport:
        """Map the competitive landscape across market segments."""
        return map_landscape(self.models)

    def roadmap(self) -> ResearchRoadmap:
        """Generate a prioritized research roadmap."""
        gap_result = self.analyze()
        trends = self.forecast()
        opportunities = score_opportunities(gap_result, self.models)
        return generate_roadmap(gap_result, trends, opportunities, self.models)

    def autoresearch_program(self, fmt: str = "md") -> str:
        """Generate an autoresearch experiment program.

        Parameters
        ----------
        fmt : str
            ``"md"`` for Markdown (program.md), ``"text"`` for plain text.

        Returns
        -------
        str
            The rendered experiment program.
        """
        gap_result = self.analyze()
        opportunities = self.score_opportunities()
        program = generate_experiment_program(gap_result, opportunities, self.models)
        if fmt == "md":
            return render_program_markdown(program)
        return render_program_text(program)

    def sandbox_config(self, fmt: str = "text") -> str:
        """Detect and report sandbox environment configuration.

        Parameters
        ----------
        fmt : str
            ``"text"`` for plain text, ``"md"`` for Markdown.

        Returns
        -------
        str
            The rendered sandbox configuration report.
        """
        config = detect_sandbox_config()
        if fmt == "md":
            return render_config_markdown(config)
        return render_config_text(config)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def run(self, fmt: str = "text") -> str:
        """Run the gap-analysis pipeline and return a formatted report.

        Parameters
        ----------
        fmt : str
            ``"text"`` for plain-text output, ``"md"`` for Markdown.
        """
        result = self.analyze()
        if fmt == "md":
            return generate_markdown_report(result)
        return generate_text_report(result)

    def run_full(self, fmt: str = "text") -> str:
        """Run **all** analysis modules and return a comprehensive report.

        Includes: gap analysis, trend forecast, opportunity scores,
        best-fit model recommendations, research velocity, risk
        assessment, competitive landscape, and research roadmap.

        Parameters
        ----------
        fmt : str
            ``"text"`` for plain-text output, ``"md"`` for Markdown.
        """
        gap_result = self.analyze()
        trends = self.forecast()
        opportunities = self.score_opportunities()
        recommendations = self.recommend()
        vel = self.velocity()
        risks = self.assess_risks()
        land = self.landscape()
        road = self.roadmap()

        if fmt == "md":
            return generate_full_markdown_report(
                gap_result, trends, opportunities, recommendations,
                vel, risks, land, road,
            )
        return generate_full_text_report(
            gap_result, trends, opportunities, recommendations,
            vel, risks, land, road,
        )
