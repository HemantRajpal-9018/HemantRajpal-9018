"""CLI entry-point for the AI Researcher Agent.

Run with::

    python -m ai_researcher            # plain-text report to stdout
    python -m ai_researcher --format md # Markdown report to stdout
    python -m ai_researcher --output report.md --format md
"""

from __future__ import annotations

import argparse
import sys

from ai_researcher.agent import ResearcherAgent


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="ai_researcher",
        description=(
            "AI Researcher Agent – find gaps in the latest reasoning models "
            "as of March 12, 2026."
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

    args = parser.parse_args(argv)

    agent = ResearcherAgent()

    if args.list_models:
        for name in agent.list_models():
            print(name)
        return

    report = agent.run(fmt=args.format)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
