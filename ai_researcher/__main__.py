"""CLI entry-point for the AI Researcher Agent.

Run with::

    python -m ai_researcher                      # gap analysis (text)
    python -m ai_researcher --format md          # gap analysis (Markdown)
    python -m ai_researcher --full               # comprehensive report
    python -m ai_researcher --trends             # trend forecast only
    python -m ai_researcher --opportunities      # opportunity scores only
    python -m ai_researcher --compare "OpenAI o3" "DeepSeek R1"
    python -m ai_researcher --recommend          # best-fit recommendations
    python -m ai_researcher --velocity           # research velocity analysis
    python -m ai_researcher --risks              # risk assessment
    python -m ai_researcher --landscape          # competitive landscape
    python -m ai_researcher --roadmap            # research roadmap
    python -m ai_researcher --autoresearch-program  # generate autoresearch program.md
    python -m ai_researcher --sandbox-config     # show sandbox environment config
    python -m ai_researcher --colab-notebook     # Google Colab notebook for GPU/TPU
    python -m ai_researcher --list-models
"""

from __future__ import annotations

import argparse
import sys

from ai_researcher.agent import ResearcherAgent
from ai_researcher.report_generator import (
    generate_comparison_text,
    generate_landscape_text,
    generate_opportunities_text,
    generate_recommendations_text,
    generate_risks_text,
    generate_roadmap_text,
    generate_trends_text,
    generate_velocity_text,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="ai_researcher",
        description=(
            "AI Researcher Agent – find gaps in the latest reasoning models "
            "as of March 12, 2026.  Includes trend forecasting, head-to-head "
            "model comparison, innovation opportunity scoring, best-fit "
            "model recommendations, research velocity analytics, risk "
            "assessment, competitive landscape mapping, and actionable "
            "research roadmaps."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["text", "md"],
        default="text",
        help="Output format: plain text (default) or Markdown.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Write report to file instead of stdout.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all models in the knowledge base and exit.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run ALL analysis modules (gaps + trends + opportunities + recommendations + velocity + risks + landscape + roadmap).",
    )
    parser.add_argument(
        "--trends",
        action="store_true",
        help="Show emerging-trend forecast only.",
    )
    parser.add_argument(
        "--opportunities",
        action="store_true",
        help="Show innovation opportunity scores only.",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("MODEL_A", "MODEL_B"),
        help="Head-to-head comparison of two models.",
    )
    parser.add_argument(
        "--recommend",
        action="store_true",
        help="Show best-fit model recommendations for common use cases.",
    )
    parser.add_argument(
        "--velocity",
        action="store_true",
        help="Show research velocity analysis across providers.",
    )
    parser.add_argument(
        "--risks",
        action="store_true",
        help="Show risk assessment for all models.",
    )
    parser.add_argument(
        "--landscape",
        action="store_true",
        help="Show competitive landscape map across market segments.",
    )
    parser.add_argument(
        "--roadmap",
        action="store_true",
        help="Show prioritized research roadmap with milestones.",
    )
    parser.add_argument(
        "--autoresearch-program",
        action="store_true",
        help=(
            "Generate a program.md experiment file for use with "
            "karpathy/autoresearch.  Requires NVIDIA GPU to run."
        ),
    )
    parser.add_argument(
        "--sandbox-config",
        action="store_true",
        help=(
            "Show sandbox environment configuration: CPU, RAM, GPU, "
            "installed tools, and autoresearch compatibility."
        ),
    )
    parser.add_argument(
        "--colab-notebook",
        action="store_true",
        help=(
            "Generate a Google Colab notebook (.ipynb) that runs "
            "autoresearch experiments on Colab's free GPU/TPU hardware."
        ),
    )
    parser.add_argument(
        "--runtime",
        choices=["GPU", "TPU"],
        default="GPU",
        help="Target Colab runtime: GPU (default) or TPU.",
    )

    args = parser.parse_args(argv)

    agent = ResearcherAgent()

    # --- list models ---
    if args.list_models:
        for name in agent.list_models():
            print(name)
        return

    # --- single-module modes ---
    report: str | None = None

    if args.compare:
        comparison = agent.compare(args.compare[0], args.compare[1])
        report = generate_comparison_text(comparison)

    elif args.trends:
        forecast = agent.forecast()
        report = generate_trends_text(forecast)

    elif args.opportunities:
        opp = agent.score_opportunities()
        report = generate_opportunities_text(opp)

    elif args.recommend:
        recs = agent.recommend()
        report = generate_recommendations_text(recs)

    elif args.velocity:
        vel = agent.velocity()
        report = generate_velocity_text(vel)

    elif args.risks:
        risk_report = agent.assess_risks()
        report = generate_risks_text(risk_report)

    elif args.landscape:
        land = agent.landscape()
        report = generate_landscape_text(land)

    elif args.roadmap:
        road = agent.roadmap()
        report = generate_roadmap_text(road)

    elif args.autoresearch_program:
        report = agent.autoresearch_program(fmt=args.format)

    elif args.sandbox_config:
        report = agent.sandbox_config(fmt=args.format)

    elif args.colab_notebook:
        # For --colab-notebook, default to ipynb unless --format md/text
        if args.format == "md":
            report = agent.colab_notebook(fmt="md", runtime=args.runtime)
        elif args.output and args.output.endswith(".ipynb"):
            report = agent.colab_notebook(fmt="ipynb", runtime=args.runtime)
        elif args.format == "text":
            report = agent.colab_notebook(fmt="text", runtime=args.runtime)
        else:
            report = agent.colab_notebook(fmt="text", runtime=args.runtime)

    elif args.full:
        report = agent.run_full(fmt=args.format)

    else:
        report = agent.run(fmt=args.format)

    # --- output ---
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
