"""Tests for the GitHub-to-Google-Colab direct connection setup module.

Validates link generation, badge rendering, setup guide creation,
agent integration, and CLI behavior.
"""

import pytest

from ai_researcher.agent import ResearcherAgent
from ai_researcher.colab_setup import (
    COLAB_GITHUB_BASE,
    GITHUB_OWNER,
    GITHUB_REPO,
    NOTEBOOK_PATHS,
    ColabLink,
    ColabSetupGuide,
    build_badge_html,
    build_badge_markdown,
    build_colab_url,
    generate_colab_link,
    generate_setup_guide,
    render_setup_markdown,
    render_setup_text,
)


# =========================================================================
# Unit tests — URL and badge generation
# =========================================================================


class TestBuildColabUrl:
    """Test build_colab_url function."""

    def test_default_url_format(self):
        url = build_colab_url("notebooks/gpu_experiments.ipynb")
        assert url.startswith(COLAB_GITHUB_BASE)
        assert GITHUB_OWNER in url
        assert GITHUB_REPO in url
        assert "notebooks/gpu_experiments.ipynb" in url

    def test_includes_branch(self):
        url = build_colab_url("test.ipynb", branch="dev")
        assert "/blob/dev/" in url

    def test_custom_owner_repo(self):
        url = build_colab_url("nb.ipynb", owner="alice", repo="project")
        assert "/alice/project/" in url

    def test_url_starts_with_https(self):
        url = build_colab_url("test.ipynb")
        assert url.startswith("https://")


class TestBuildBadge:
    """Test badge generation functions."""

    def test_markdown_badge_has_image(self):
        badge = build_badge_markdown("https://example.com")
        assert "![" in badge
        assert "colab-badge.svg" in badge

    def test_markdown_badge_has_link(self):
        badge = build_badge_markdown("https://example.com/nb.ipynb")
        assert "https://example.com/nb.ipynb" in badge

    def test_markdown_badge_has_alt(self):
        badge = build_badge_markdown("https://example.com", alt="My Notebook")
        assert "My Notebook" in badge

    def test_html_badge_has_anchor(self):
        badge = build_badge_html("https://example.com")
        assert "<a href=" in badge
        assert "</a>" in badge

    def test_html_badge_has_img(self):
        badge = build_badge_html("https://example.com")
        assert "<img src=" in badge
        assert "colab-badge.svg" in badge

    def test_html_badge_has_alt(self):
        badge = build_badge_html("https://example.com", alt="My NB")
        assert 'alt="My NB"' in badge


# =========================================================================
# Unit tests — ColabLink dataclass
# =========================================================================


class TestColabLink:
    """Test ColabLink dataclass and generate_colab_link."""

    def test_generate_link(self):
        link = generate_colab_link("GPU experiments", "notebooks/gpu.ipynb")
        assert link.label == "GPU experiments"
        assert link.notebook_path == "notebooks/gpu.ipynb"
        assert COLAB_GITHUB_BASE in link.colab_url

    def test_link_has_badge_markdown(self):
        link = generate_colab_link("Test", "test.ipynb")
        assert "colab-badge.svg" in link.badge_markdown
        assert "[![" in link.badge_markdown

    def test_link_has_badge_html(self):
        link = generate_colab_link("Test", "test.ipynb")
        assert "<a href=" in link.badge_html
        assert "<img src=" in link.badge_html

    def test_link_url_contains_path(self):
        link = generate_colab_link("Test", "my/path.ipynb")
        assert "my/path.ipynb" in link.colab_url


# =========================================================================
# Integration tests — setup guide generation
# =========================================================================


class TestSetupGuide:
    """Test generate_setup_guide."""

    def test_guide_has_links(self):
        guide = generate_setup_guide()
        assert len(guide.links) > 0

    def test_guide_has_steps(self):
        guide = generate_setup_guide()
        assert len(guide.steps) >= 8

    def test_guide_links_match_notebook_paths(self):
        guide = generate_setup_guide()
        paths = {link.notebook_path for link in guide.links}
        for path in NOTEBOOK_PATHS.values():
            assert path in paths

    def test_guide_has_gpu_and_tpu(self):
        guide = generate_setup_guide()
        labels = {link.label for link in guide.links}
        assert "GPU experiments" in labels
        assert "TPU experiments" in labels

    def test_guide_owner_repo(self):
        guide = generate_setup_guide()
        assert guide.owner == GITHUB_OWNER
        assert guide.repo == GITHUB_REPO

    def test_custom_owner_repo(self):
        guide = generate_setup_guide(owner="alice", repo="project")
        assert guide.owner == "alice"
        assert guide.repo == "project"


# =========================================================================
# Rendering tests — plain text
# =========================================================================


class TestRenderText:
    """Test plain-text rendering."""

    def test_has_header(self):
        guide = generate_setup_guide()
        text = render_setup_text(guide)
        assert "GITHUB" in text
        assert "GOOGLE COLAB" in text

    def test_has_colab_urls(self):
        guide = generate_setup_guide()
        text = render_setup_text(guide)
        assert "colab.research.google.com" in text

    def test_has_steps(self):
        guide = generate_setup_guide()
        text = render_setup_text(guide)
        assert "Step 1:" in text
        assert "Step 8:" in text

    def test_has_how_it_works(self):
        guide = generate_setup_guide()
        text = render_setup_text(guide)
        assert "HOW IT WORKS" in text

    def test_has_what_you_get(self):
        guide = generate_setup_guide()
        text = render_setup_text(guide)
        assert "WHAT YOU GET" in text
        assert "NVIDIA T4" in text

    def test_mentions_no_download(self):
        guide = generate_setup_guide()
        text = render_setup_text(guide)
        assert "no download" in text.lower() or "no upload" in text.lower()


# =========================================================================
# Rendering tests — Markdown
# =========================================================================


class TestRenderMarkdown:
    """Test Markdown rendering."""

    def test_starts_with_heading(self):
        guide = generate_setup_guide()
        md = render_setup_markdown(guide)
        assert md.startswith("# ")

    def test_has_badges(self):
        guide = generate_setup_guide()
        md = render_setup_markdown(guide)
        assert "colab-badge.svg" in md

    def test_has_steps_section(self):
        guide = generate_setup_guide()
        md = render_setup_markdown(guide)
        assert "Step-by-Step" in md

    def test_has_notebook_table(self):
        guide = generate_setup_guide()
        md = render_setup_markdown(guide)
        assert "| Notebook |" in md

    def test_has_comparison_table(self):
        guide = generate_setup_guide()
        md = render_setup_markdown(guide)
        assert "This Sandbox" in md
        assert "Google Colab" in md

    def test_has_what_you_get_table(self):
        guide = generate_setup_guide()
        md = render_setup_markdown(guide)
        assert "| Feature | Details |" in md

    def test_mentions_open_in_colab(self):
        guide = generate_setup_guide()
        md = render_setup_markdown(guide)
        assert "Open in Colab" in md


# =========================================================================
# Agent integration tests
# =========================================================================


class TestAgentIntegration:
    """Test colab_link via ResearcherAgent."""

    def test_agent_colab_link_text(self):
        agent = ResearcherAgent()
        output = agent.colab_link(fmt="text")
        assert "GITHUB" in output
        assert "GOOGLE COLAB" in output
        assert "colab.research.google.com" in output

    def test_agent_colab_link_md(self):
        agent = ResearcherAgent()
        output = agent.colab_link(fmt="md")
        assert "# " in output
        assert "colab-badge.svg" in output

    def test_agent_default_is_text(self):
        agent = ResearcherAgent()
        output = agent.colab_link()
        assert "Step 1:" in output


# =========================================================================
# CLI tests
# =========================================================================


class TestCLI:
    """Test CLI --colab-link flag."""

    def test_cli_colab_link_text(self):
        import io
        import sys
        from ai_researcher.__main__ import main

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(["--colab-link"])
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "GITHUB" in output
        assert "GOOGLE COLAB" in output
        assert "colab.research.google.com" in output

    def test_cli_colab_link_md(self):
        import io
        import sys
        from ai_researcher.__main__ import main

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(["--colab-link", "--format", "md"])
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "# " in output
        assert "colab-badge.svg" in output

    def test_cli_colab_link_to_file(self, tmp_path):
        from ai_researcher.__main__ import main

        outfile = tmp_path / "setup.md"
        main(["--colab-link", "--format", "md", "-o", str(outfile)])
        content = outfile.read_text()
        assert "Open in Colab" in content
        assert len(content) > 100
