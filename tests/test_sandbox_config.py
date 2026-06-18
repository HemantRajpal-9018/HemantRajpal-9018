"""Tests for the sandbox configuration inspector module.

Validates hardware detection, rendering, and CLI integration.
"""

import pytest

from ai_researcher.agent import ResearcherAgent
from ai_researcher.sandbox_config import (
    GpuInfo,
    SandboxConfig,
    _assess_autoresearch,
    _detect_cpu,
    _detect_disk,
    _detect_memory,
    _detect_os,
    _detect_tools,
    detect_sandbox_config,
    render_config_markdown,
    render_config_text,
)


# =========================================================================
# Unit tests – data structures
# =========================================================================


class TestGpuInfo:
    """Test GpuInfo dataclass."""

    def test_create_gpu_info(self):
        gpu = GpuInfo(name="NVIDIA H100", driver="535.129.03", memory_mb=81920)
        assert gpu.name == "NVIDIA H100"
        assert gpu.memory_mb == 81920
        assert gpu.is_nvidia is True

    def test_non_nvidia_gpu(self):
        gpu = GpuInfo(name="AMD Radeon RX 7900", driver="amdgpu")
        assert gpu.is_nvidia is False

    def test_nvidia_detection_from_name(self):
        gpu = GpuInfo(name="NVIDIA A100")
        assert gpu.is_nvidia is True

    def test_cuda_driver_detection(self):
        gpu = GpuInfo(name="GeForce RTX 4090", driver="cuda-12.2")
        assert gpu.is_nvidia is True


class TestSandboxConfig:
    """Test SandboxConfig dataclass."""

    def test_default_config(self):
        config = SandboxConfig()
        assert config.cpu_model == "unknown"
        assert config.cpu_cores == 0
        assert config.gpus == []
        assert config.has_nvidia_gpu is False
        assert config.can_run_autoresearch is False

    def test_config_with_gpu(self):
        gpu = GpuInfo(name="NVIDIA H100", memory_mb=81920)
        config = SandboxConfig(
            gpus=[gpu],
            has_nvidia_gpu=True,
            has_cuda=True,
            can_run_autoresearch=True,
        )
        assert len(config.gpus) == 1
        assert config.has_nvidia_gpu is True
        assert config.can_run_autoresearch is True


# =========================================================================
# Unit tests – autoresearch compatibility assessment
# =========================================================================


class TestAutoresearchAssessment:
    """Test _assess_autoresearch helper."""

    def test_no_gpu_blocks(self):
        can_run, blockers = _assess_autoresearch(
            has_nvidia=False, has_cuda=False,
            gpus=[], python_version="3.12.3",
            tools={"uv": "uv 0.5.0"},
        )
        assert can_run is False
        assert any("No NVIDIA GPU" in b for b in blockers)

    def test_nvidia_with_cuda_and_uv_passes(self):
        gpu = GpuInfo(name="NVIDIA H100", memory_mb=81920)
        can_run, blockers = _assess_autoresearch(
            has_nvidia=True, has_cuda=True,
            gpus=[gpu], python_version="3.12.3",
            tools={"uv": "uv 0.5.0"},
        )
        assert can_run is True
        assert len(blockers) == 0

    def test_old_python_blocks(self):
        gpu = GpuInfo(name="NVIDIA A100", memory_mb=40960)
        can_run, blockers = _assess_autoresearch(
            has_nvidia=True, has_cuda=True,
            gpus=[gpu], python_version="3.8.10",
            tools={"uv": "uv 0.5.0"},
        )
        assert any("3.10+" in b for b in blockers)

    def test_missing_uv_only_still_passes(self):
        """Missing uv alone is not a hard blocker (easy to install)."""
        gpu = GpuInfo(name="NVIDIA H100", memory_mb=81920)
        can_run, blockers = _assess_autoresearch(
            has_nvidia=True, has_cuda=True,
            gpus=[gpu], python_version="3.12.3",
            tools={},
        )
        assert can_run is True
        assert len(blockers) == 1
        assert "uv" in blockers[0]

    def test_small_gpu_warns(self):
        gpu = GpuInfo(name="NVIDIA GTX 1080", memory_mb=8192)
        can_run, blockers = _assess_autoresearch(
            has_nvidia=True, has_cuda=True,
            gpus=[gpu], python_version="3.12.3",
            tools={"uv": "uv 0.5.0"},
        )
        assert any("8192 MB" in b for b in blockers)

    def test_nvidia_without_cuda_warns(self):
        can_run, blockers = _assess_autoresearch(
            has_nvidia=True, has_cuda=False,
            gpus=[], python_version="3.12.3",
            tools={"uv": "uv 0.5.0"},
        )
        assert any("CUDA toolkit" in b for b in blockers)


# =========================================================================
# Integration tests – hardware detection
# =========================================================================


class TestDetection:
    """Test actual hardware detection (runs in any environment)."""

    def test_detect_cpu_returns_tuple(self):
        model, cores, threads, arch = _detect_cpu()
        assert isinstance(model, str)
        assert len(model) > 0
        assert cores > 0
        assert threads > 0
        assert arch in ("x86_64", "aarch64", "arm64", "unknown", "AMD64")

    def test_detect_memory_positive(self):
        total, available = _detect_memory()
        assert total > 0
        assert available >= 0

    def test_detect_disk_positive(self):
        total, free = _detect_disk()
        assert total > 0
        assert free >= 0

    def test_detect_os_returns_strings(self):
        os_name, os_ver, kernel, hypervisor = _detect_os()
        assert isinstance(os_name, str)
        assert isinstance(kernel, str)

    def test_detect_tools_returns_dict(self):
        tools = _detect_tools()
        assert isinstance(tools, dict)
        # Python should always be detected
        assert "python" in tools

    def test_full_detection(self):
        config = detect_sandbox_config()
        assert isinstance(config, SandboxConfig)
        assert config.cpu_cores > 0
        assert config.total_ram_gb > 0
        assert len(config.python_version) > 0


# =========================================================================
# Rendering tests – plain text
# =========================================================================


class TestRenderText:
    """Test plain-text rendering."""

    def test_text_has_header(self):
        config = detect_sandbox_config()
        text = render_config_text(config)
        assert "SANDBOX ENVIRONMENT CONFIGURATION" in text

    def test_text_has_cpu_section(self):
        config = detect_sandbox_config()
        text = render_config_text(config)
        assert "CPU" in text
        assert "Model" in text

    def test_text_has_memory_section(self):
        config = detect_sandbox_config()
        text = render_config_text(config)
        assert "MEMORY" in text
        assert "GB" in text

    def test_text_has_gpu_section(self):
        config = detect_sandbox_config()
        text = render_config_text(config)
        assert "GPU" in text

    def test_text_has_autoresearch_section(self):
        config = detect_sandbox_config()
        text = render_config_text(config)
        assert "AUTORESEARCH COMPATIBILITY" in text

    def test_text_no_gpu_environment(self):
        """In this sandbox (no GPU), should show NOT READY."""
        config = detect_sandbox_config()
        if not config.has_nvidia_gpu:
            text = render_config_text(config)
            assert "NOT READY" in text


# =========================================================================
# Rendering tests – Markdown
# =========================================================================


class TestRenderMarkdown:
    """Test Markdown rendering."""

    def test_markdown_starts_with_heading(self):
        config = detect_sandbox_config()
        md = render_config_markdown(config)
        assert md.startswith("# ")

    def test_markdown_has_table_format(self):
        config = detect_sandbox_config()
        md = render_config_markdown(config)
        assert "| Property | Value |" in md

    def test_markdown_has_gpu_section(self):
        config = detect_sandbox_config()
        md = render_config_markdown(config)
        assert "## GPU" in md

    def test_markdown_has_autoresearch_section(self):
        config = detect_sandbox_config()
        md = render_config_markdown(config)
        assert "Autoresearch Compatibility" in md

    def test_markdown_note_about_program_generation(self):
        config = detect_sandbox_config()
        md = render_config_markdown(config)
        assert "--autoresearch-program" in md


# =========================================================================
# Agent integration tests
# =========================================================================


class TestAgentIntegration:
    """Test sandbox config via ResearcherAgent."""

    def test_agent_sandbox_config_text(self):
        agent = ResearcherAgent()
        output = agent.sandbox_config(fmt="text")
        assert "SANDBOX ENVIRONMENT CONFIGURATION" in output
        assert "CPU" in output

    def test_agent_sandbox_config_md(self):
        agent = ResearcherAgent()
        output = agent.sandbox_config(fmt="md")
        assert "# " in output
        assert "## CPU" in output


# =========================================================================
# CLI tests
# =========================================================================


class TestCLI:
    """Test CLI --sandbox-config flag."""

    def test_cli_sandbox_config(self):
        import io
        import sys
        from ai_researcher.__main__ import main

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(["--sandbox-config"])
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "SANDBOX ENVIRONMENT CONFIGURATION" in output
        assert "CPU" in output
        assert "GPU" in output

    def test_cli_sandbox_config_md(self):
        import io
        import sys
        from ai_researcher.__main__ import main

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(["--sandbox-config", "--format", "md"])
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "# " in output
        assert "## CPU" in output

    def test_cli_sandbox_config_to_file(self, tmp_path):
        from ai_researcher.__main__ import main

        outfile = tmp_path / "sandbox.md"
        main(["--sandbox-config", "--format", "md", "-o", str(outfile)])
        content = outfile.read_text()
        assert "## GPU" in content
        assert "Autoresearch Compatibility" in content
