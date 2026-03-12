"""Main agent orchestration module.

``ResearcherAgent`` ties together the knowledge base, gap analyzer, and
report generator into a single callable interface.
"""

from __future__ import annotations

from typing import List, Optional

from ai_researcher.gap_analyzer import AnalysisResult, analyze_gaps
from ai_researcher.models_knowledge import (
    ReasoningModel,
    get_model_by_name,
    get_models_by_provider,
    get_reasoning_models,
)
from ai_researcher.report_generator import (
    generate_markdown_report,
    generate_text_report,
)


class ResearcherAgent:
    """AI Researcher Agent that finds gaps in the latest reasoning models.

    Usage::

        agent = ResearcherAgent()
        report = agent.run()          # full analysis → text report
        report = agent.run(fmt="md")  # full analysis → Markdown report
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
    # Report generation
    # ------------------------------------------------------------------

    def run(self, fmt: str = "text") -> str:
        """Run the full pipeline and return a formatted report.

        Parameters
        ----------
        fmt : str
            ``"text"`` for plain-text output, ``"md"`` for Markdown.
        """
        result = self.analyze()
        if fmt == "md":
            return generate_markdown_report(result)
        return generate_text_report(result)
