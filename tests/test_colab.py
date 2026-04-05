"""Tests for the Google Colab notebook generator module.

Validates notebook generation, rendering, runtime catalogue, and CLI integration.
"""

import json

import pytest

from ai_researcher.agent import ResearcherAgent
from ai_researcher.colab_adapter import (
    COLAB_RUNTIMES,
    ColabNotebook,
    ColabRuntime,
    generate_colab_notebook,
    render_colab_markdown,
    render_colab_text,
    render_notebook_ipynb,
)


# =========================================================================
# Unit tests — data structures
# =========================================================================


class TestColabRuntime:
    """Test ColabRuntime dataclass."""

    def test_create_runtime(self):
        rt = ColabRuntime(
            name="NVIDIA T4",
            accelerator_type="GPU",
            memory_gb=15.0,
            compute_units="2,560 CUDA cores",
            best_for="Inference",
            colab_tier="free",
        )
        assert rt.name == "NVIDIA T4"
        assert rt.accelerator_type == "GPU"
        assert rt.colab_tier == "free"

    def test_runtime_catalogue_not_empty(self):
        assert len(COLAB_RUNTIMES) >= 4

    def test_catalogue_has_gpu_and_tpu(self):
        types = {rt.accelerator_type for rt in COLAB_RUNTIMES}
        assert "GPU" in types
        assert "TPU" in types

    def test_catalogue_has_free_tier(self):
        free = [rt for rt in COLAB_RUNTIMES if rt.colab_tier == "free"]
        assert len(free) >= 2  # T4 + TPU v2

    def test_catalogue_memory_positive(self):
        for rt in COLAB_RUNTIMES:
            assert rt.memory_gb > 0


class TestColabNotebook:
    """Test ColabNotebook dataclass."""

    def test_default_notebook(self):
        nb = ColabNotebook()
        assert nb.cells == []
        assert nb.experiment_count == 0
        assert nb.runtime_type == "GPU"

    def test_notebook_with_cells(self):
        cells = [{"cell_type": "markdown", "source": ["# Test"]}]
        nb = ColabNotebook(cells=cells, experiment_count=5, runtime_type="TPU")
        assert len(nb.cells) == 1
        assert nb.experiment_count == 5
        assert nb.runtime_type == "TPU"


# =========================================================================
# Integration tests — notebook generation
# =========================================================================


class TestNotebookGeneration:
    """Test the full notebook generation pipeline."""

    def test_generates_cells(self):
        notebook = generate_colab_notebook()
        assert len(notebook.cells) > 0

    def test_has_markdown_and_code_cells(self):
        notebook = generate_colab_notebook()
        types = {c["cell_type"] for c in notebook.cells}
        assert "markdown" in types
        assert "code" in types

    def test_experiment_count_positive(self):
        notebook = generate_colab_notebook()
        assert notebook.experiment_count > 0

    def test_generated_date_set(self):
        notebook = generate_colab_notebook()
        assert len(notebook.generated_date) == 10  # YYYY-MM-DD

    def test_program_markdown_embedded(self):
        notebook = generate_colab_notebook()
        assert len(notebook.program_markdown) > 0
        assert "Experiment" in notebook.program_markdown

    def test_gpu_runtime_default(self):
        notebook = generate_colab_notebook()
        assert notebook.runtime_type == "GPU"

    def test_tpu_runtime_option(self):
        notebook = generate_colab_notebook(runtime_type="TPU")
        assert notebook.runtime_type == "TPU"

    def test_tpu_has_extra_cells(self):
        gpu_nb = generate_colab_notebook(runtime_type="GPU")
        tpu_nb = generate_colab_notebook(runtime_type="TPU")
        assert len(tpu_nb.cells) > len(gpu_nb.cells)

    def test_idempotent_generation(self):
        nb1 = generate_colab_notebook()
        nb2 = generate_colab_notebook()
        assert nb1.experiment_count == nb2.experiment_count
        assert len(nb1.cells) == len(nb2.cells)


# =========================================================================
# Rendering tests — .ipynb
# =========================================================================


class TestRenderIpynb:
    """Test .ipynb JSON rendering."""

    def test_valid_json(self):
        notebook = generate_colab_notebook()
        ipynb_str = render_notebook_ipynb(notebook)
        data = json.loads(ipynb_str)  # Should not raise
        assert isinstance(data, dict)

    def test_nbformat_version(self):
        notebook = generate_colab_notebook()
        data = json.loads(render_notebook_ipynb(notebook))
        assert data["nbformat"] == 4

    def test_has_cells(self):
        notebook = generate_colab_notebook()
        data = json.loads(render_notebook_ipynb(notebook))
        assert len(data["cells"]) > 0

    def test_has_colab_metadata(self):
        notebook = generate_colab_notebook()
        data = json.loads(render_notebook_ipynb(notebook))
        assert "colab" in data["metadata"]

    def test_gpu_accelerator_metadata(self):
        notebook = generate_colab_notebook(runtime_type="GPU")
        data = json.loads(render_notebook_ipynb(notebook))
        assert data["metadata"]["accelerator"] == "GPU"

    def test_tpu_accelerator_metadata(self):
        notebook = generate_colab_notebook(runtime_type="TPU")
        data = json.loads(render_notebook_ipynb(notebook))
        assert data["metadata"]["accelerator"] == "TPU"

    def test_cells_have_correct_types(self):
        notebook = generate_colab_notebook()
        data = json.loads(render_notebook_ipynb(notebook))
        for cell in data["cells"]:
            assert cell["cell_type"] in ("markdown", "code")


# =========================================================================
# Rendering tests — plain text
# =========================================================================


class TestRenderText:
    """Test plain-text rendering."""

    def test_text_has_header(self):
        notebook = generate_colab_notebook()
        text = render_colab_text(notebook)
        assert "GOOGLE COLAB NOTEBOOK" in text

    def test_text_has_runtime_info(self):
        notebook = generate_colab_notebook()
        text = render_colab_text(notebook)
        assert "GPU" in text

    def test_text_has_runtimes_section(self):
        notebook = generate_colab_notebook()
        text = render_colab_text(notebook)
        assert "NVIDIA T4" in text
        assert "TPU v2" in text

    def test_text_has_instructions(self):
        notebook = generate_colab_notebook()
        text = render_colab_text(notebook)
        assert "colab.research.google.com" in text

    def test_text_explains_gpu_advantage(self):
        notebook = generate_colab_notebook()
        text = render_colab_text(notebook)
        assert "CUDA" in text
        assert "No GPU/TPU" in text or "NO GPU" in text


# =========================================================================
# Rendering tests — Markdown
# =========================================================================


class TestRenderMarkdown:
    """Test Markdown rendering."""

    def test_markdown_starts_with_heading(self):
        notebook = generate_colab_notebook()
        md = render_colab_markdown(notebook)
        assert md.startswith("# ")

    def test_markdown_has_runtime_table(self):
        notebook = generate_colab_notebook()
        md = render_colab_markdown(notebook)
        assert "| Accelerator | Type |" in md

    def test_markdown_has_comparison_table(self):
        notebook = generate_colab_notebook()
        md = render_colab_markdown(notebook)
        assert "This Sandbox" in md
        assert "Google Colab" in md

    def test_markdown_has_gpu_vs_tpu_section(self):
        notebook = generate_colab_notebook()
        md = render_colab_markdown(notebook)
        assert "GPU vs TPU" in md

    def test_markdown_has_recommendation(self):
        notebook = generate_colab_notebook()
        md = render_colab_markdown(notebook)
        assert "Recommendation" in md

    def test_markdown_mentions_autoresearch(self):
        notebook = generate_colab_notebook()
        md = render_colab_markdown(notebook)
        assert "autoresearch" in md


# =========================================================================
# Agent integration tests
# =========================================================================


class TestAgentIntegration:
    """Test Colab notebook generation via ResearcherAgent."""

    def test_agent_colab_text(self):
        agent = ResearcherAgent()
        output = agent.colab_notebook(fmt="text")
        assert "GOOGLE COLAB NOTEBOOK" in output
        assert "GPU" in output

    def test_agent_colab_md(self):
        agent = ResearcherAgent()
        output = agent.colab_notebook(fmt="md")
        assert "# " in output
        assert "Google Colab" in output

    def test_agent_colab_ipynb(self):
        agent = ResearcherAgent()
        output = agent.colab_notebook(fmt="ipynb")
        data = json.loads(output)
        assert data["nbformat"] == 4
        assert len(data["cells"]) > 0

    def test_agent_colab_tpu_runtime(self):
        agent = ResearcherAgent()
        output = agent.colab_notebook(fmt="ipynb", runtime="TPU")
        data = json.loads(output)
        assert data["metadata"]["accelerator"] == "TPU"


# =========================================================================
# CLI tests
# =========================================================================


class TestCLI:
    """Test CLI --colab-notebook flag."""

    def test_cli_colab_notebook_text(self):
        import io
        import sys
        from ai_researcher.__main__ import main

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(["--colab-notebook"])
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "GOOGLE COLAB NOTEBOOK" in output
        assert "GPU" in output

    def test_cli_colab_notebook_md(self):
        import io
        import sys
        from ai_researcher.__main__ import main

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(["--colab-notebook", "--format", "md"])
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "# " in output
        assert "Google Colab" in output

    def test_cli_colab_notebook_to_ipynb_file(self, tmp_path):
        from ai_researcher.__main__ import main

        outfile = tmp_path / "experiments.ipynb"
        main(["--colab-notebook", "-o", str(outfile)])
        content = outfile.read_text()
        # Even when writing to .ipynb file, since format is 'text' by default
        # the output is text format unless we specify format
        assert len(content) > 0

    def test_cli_colab_notebook_tpu(self):
        import io
        import sys
        from ai_researcher.__main__ import main

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(["--colab-notebook", "--runtime", "TPU"])
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "TPU" in output
