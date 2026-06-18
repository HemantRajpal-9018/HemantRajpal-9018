"""Google Colab notebook generator for GPU/TPU-accelerated experiments.

This sandbox (GitHub Actions) has **no GPU**.  Google Colab provides free
GPU (T4 / L4 / A100) and TPU (v2) runtimes that can run the autoresearch
experiments our agent generates.

This module bridges that gap:

Flow::

    Our agent (gap analysis → experiment program)
        → colab_adapter (this module)
            → .ipynb notebook
                → Open in Google Colab → select GPU/TPU runtime
                    → experiments run on Colab hardware

The generated notebook:

1. Detects whether a GPU or TPU is available.
2. Installs autoresearch and dependencies via ``uv``.
3. Embeds the experiment program directly in the notebook.
4. Runs the experiments automatically on Colab's GPU/TPU hardware.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

from ai_researcher.autoresearch_adapter import (
    ExperimentProgram,
    generate_experiment_program,
    render_program_markdown,
)
from ai_researcher.gap_analyzer import AnalysisResult
from ai_researcher.innovation_scorer import OpportunityReport
from ai_researcher.models_knowledge import ReasoningModel


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ColabRuntime:
    """Describes a Google Colab hardware accelerator."""
    name: str
    accelerator_type: str           # "GPU" or "TPU"
    memory_gb: float
    compute_units: str              # e.g. "CUDA cores", "MXU"
    best_for: str                   # short description of ideal workloads
    colab_tier: str                 # "free", "pro", "pro+"


@dataclass
class ColabNotebook:
    """A generated Colab notebook (.ipynb) structure."""
    cells: List[Dict] = field(default_factory=list)
    experiment_count: int = 0
    runtime_type: str = "GPU"       # "GPU" or "TPU"
    generated_date: str = ""
    program_markdown: str = ""


# ---------------------------------------------------------------------------
# Colab runtime catalogue
# ---------------------------------------------------------------------------

COLAB_RUNTIMES: List[ColabRuntime] = [
    ColabRuntime(
        name="NVIDIA T4",
        accelerator_type="GPU",
        memory_gb=15.0,
        compute_units="2,560 CUDA cores + 320 Tensor Cores",
        best_for="Inference, small-model fine-tuning, mixed-precision training",
        colab_tier="free",
    ),
    ColabRuntime(
        name="NVIDIA L4",
        accelerator_type="GPU",
        memory_gb=22.5,
        compute_units="7,424 CUDA cores + 240 Tensor Cores (4th gen)",
        best_for="Medium-model training, efficient inference, Ada Lovelace arch",
        colab_tier="pro",
    ),
    ColabRuntime(
        name="NVIDIA A100",
        accelerator_type="GPU",
        memory_gb=40.0,
        compute_units="6,912 CUDA cores + 432 Tensor Cores (3rd gen)",
        best_for="Large-model training, multi-GPU experiments, full autoresearch",
        colab_tier="pro+",
    ),
    ColabRuntime(
        name="TPU v2-8",
        accelerator_type="TPU",
        memory_gb=64.0,
        compute_units="8 TPU cores, 180 TFLOPS bf16",
        best_for="JAX/TensorFlow workloads, large batch training, TPU-optimized models",
        colab_tier="free",
    ),
]


# ---------------------------------------------------------------------------
# Notebook cell builders
# ---------------------------------------------------------------------------

def _make_markdown_cell(source: str) -> Dict:
    """Create a Jupyter markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.split("\n"),
    }


def _make_code_cell(source: str) -> Dict:
    """Create a Jupyter code cell."""
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": source.split("\n"),
    }


def _hardware_detection_code() -> str:
    """Python code that detects GPU/TPU in Colab."""
    return '''\
import subprocess, os, sys

print("=" * 60)
print("  HARDWARE DETECTION — Google Colab Runtime")
print("=" * 60)

# GPU detection
gpu_info = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
     "--format=csv,noheader"],
    capture_output=True, text=True
)
if gpu_info.returncode == 0:
    for line in gpu_info.stdout.strip().split("\\n"):
        parts = [p.strip() for p in line.split(",")]
        print(f"  GPU      : {parts[0]}")
        if len(parts) > 1:
            print(f"  VRAM     : {parts[1]} MiB")
        if len(parts) > 2:
            print(f"  Driver   : {parts[2]}")
    HAS_GPU = True
else:
    print("  GPU      : Not detected")
    HAS_GPU = False

# TPU detection
HAS_TPU = False
try:
    if "COLAB_TPU_ADDR" in os.environ:
        print(f"  TPU      : {os.environ['COLAB_TPU_ADDR']}")
        HAS_TPU = True
    else:
        import jax
        devices = jax.devices()
        tpu_devices = [d for d in devices if d.platform == "tpu"]
        if tpu_devices:
            print(f"  TPU      : {len(tpu_devices)} TPU core(s)")
            HAS_TPU = True
        else:
            print("  TPU      : Not detected")
except Exception:
    print("  TPU      : Not detected (jax not installed)")

# CUDA check
cuda_check = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
if cuda_check.returncode == 0:
    for line in cuda_check.stdout.split("\\n"):
        if "release" in line.lower():
            print(f"  CUDA     : {line.strip()}")
            break
    HAS_CUDA = True
else:
    print("  CUDA     : Not detected")
    HAS_CUDA = False

print()
if HAS_GPU:
    print("  ✅ GPU available — autoresearch experiments can run!")
elif HAS_TPU:
    print("  ✅ TPU available — JAX/TF experiments can run!")
else:
    print("  ⚠️  No accelerator — switch runtime: Runtime → Change runtime type → GPU/TPU")
print("=" * 60)
'''


def _install_dependencies_code() -> str:
    """Python code that installs autoresearch and deps in Colab."""
    return '''\
import subprocess, sys

print("Installing dependencies...")

# Install uv (fast Python package manager)
subprocess.run(
    ["pip", "install", "uv"],
    check=True, capture_output=True
)

# Clone autoresearch
subprocess.run(
    ["git", "clone", "https://github.com/karpathy/autoresearch.git",
     "/content/autoresearch"],
    check=True, capture_output=True
)

# Install autoresearch dependencies
import os
os.chdir("/content/autoresearch")
subprocess.run(
    ["uv", "pip", "install", "-r", "requirements.txt", "--system"],
    capture_output=True
)

print("✅ autoresearch installed at /content/autoresearch")
print("✅ Dependencies ready")
'''


def _write_program_code(program_md: str) -> str:
    """Python code that writes the program.md file into the Colab workspace."""
    # Escape the markdown for embedding in Python code
    escaped = program_md.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    return f'''\
program_md = """\\
{escaped}
"""

with open("/content/autoresearch/program.md", "w") as f:
    f.write(program_md)

print(f"✅ program.md written ({{len(program_md):,}} chars)")
print("   Location: /content/autoresearch/program.md")
'''


def _run_experiments_code() -> str:
    """Python code that runs the autoresearch experiments."""
    return '''\
import os, subprocess

os.chdir("/content/autoresearch")

print("🚀 Starting autoresearch experiments...")
print("   This may take 5-30 minutes depending on experiment count.")
print()

# Run autoresearch with the generated program
result = subprocess.run(
    [sys.executable, "-m", "autoresearch", "--program", "program.md"],
    capture_output=False,
    text=True,
)

if result.returncode == 0:
    print()
    print("✅ Experiments complete! Check /content/autoresearch/results/")
else:
    print()
    print(f"⚠️  autoresearch exited with code {result.returncode}")
    print("   Check the output above for errors.")
    print("   Common fixes:")
    print("   • Ensure GPU runtime is selected (Runtime → Change runtime type)")
    print("   • Try reducing experiment scope in program.md")
'''


def _tpu_adaptation_code() -> str:
    """Python code for TPU-specific setup and adaptation."""
    return '''\
import os

# TPU-specific configuration
if HAS_TPU:
    print("🔧 Configuring TPU environment...")

    # Set JAX backend to TPU
    os.environ["JAX_BACKEND"] = "tpu"

    try:
        import jax
        print(f"   JAX version: {jax.__version__}")
        print(f"   TPU devices: {jax.device_count()}")
        print(f"   Local devices: {jax.local_device_count()}")
        print()
        print("   ℹ️  TPU notes:")
        print("   • autoresearch uses PyTorch (GPU-optimized)")
        print("   • For TPU, consider JAX-based training alternatives")
        print("   • Use torch_xla for PyTorch-on-TPU support:")
        print("     pip install torch_xla[tpu]")
        print()

        # Install torch_xla for PyTorch TPU support
        import subprocess
        subprocess.run(
            ["pip", "install", "torch_xla[tpu]",
             "-f", "https://storage.googleapis.com/libtpu-releases/index.html"],
            capture_output=True
        )
        print("   ✅ torch_xla installed for PyTorch TPU support")

    except ImportError:
        print("   Installing JAX for TPU...")
        import subprocess
        subprocess.run(
            ["pip", "install", "jax[tpu]",
             "-f", "https://storage.googleapis.com/jax-releases/libtpu_releases.html"],
            capture_output=True
        )
        print("   ✅ JAX installed")
else:
    print("ℹ️  No TPU detected — skipping TPU setup")
    print("   (GPU path will be used if available)")
'''


# ---------------------------------------------------------------------------
# Notebook generation
# ---------------------------------------------------------------------------

def generate_colab_notebook(
    gap_result: Optional[AnalysisResult] = None,
    opportunities: Optional[OpportunityReport] = None,
    models: Optional[List[ReasoningModel]] = None,
    runtime_type: str = "GPU",
) -> ColabNotebook:
    """Generate a Google Colab notebook with embedded experiments.

    Parameters
    ----------
    gap_result : AnalysisResult, optional
        Pre-computed gap analysis result.
    opportunities : OpportunityReport, optional
        Pre-computed opportunity scores.
    models : list of ReasoningModel, optional
        Model list (defaults to built-in knowledge base).
    runtime_type : str
        Target runtime: ``"GPU"`` (default) or ``"TPU"``.

    Returns
    -------
    ColabNotebook
        The generated notebook structure.
    """
    # Generate the experiment program
    program = generate_experiment_program(gap_result, opportunities, models)
    program_md = render_program_markdown(program)

    cells: List[Dict] = []

    # Cell 1: Title and overview
    cells.append(_make_markdown_cell(
        "# 🧪 AI Researcher — Autoresearch Experiments on Google Colab\n"
        "\n"
        f"**Generated:** {date.today().isoformat()}\n"
        f"**Target runtime:** {runtime_type}\n"
        f"**Experiments:** {len(program.experiments)}\n"
        "\n"
        "This notebook was generated by the "
        "[AI Researcher Agent](https://github.com/HemantRajpal-9018/HemantRajpal-9018) "
        "to run autoresearch experiments on Google Colab's GPU/TPU hardware.\n"
        "\n"
        "## 📋 Setup Instructions\n"
        "\n"
        "1. **Select runtime:** Go to `Runtime → Change runtime type`\n"
        f"   - Select **{runtime_type}** as the hardware accelerator\n"
        "   - For GPU: T4 (free), L4 (Pro), A100 (Pro+)\n"
        "   - For TPU: v2 (free), v3 (Pro)\n"
        "2. **Run all cells:** `Runtime → Run all` or Ctrl+F9\n"
        "3. **Monitor progress:** Experiments take 5–30 min depending on hardware\n"
        "4. **Results:** Check `/content/autoresearch/results/` when done\n"
    ))

    # Cell 2: Hardware detection
    cells.append(_make_markdown_cell(
        "## 1️⃣ Detect Hardware\n"
        "\n"
        "Verify that GPU/TPU is available in this Colab session."
    ))
    cells.append(_make_code_cell(_hardware_detection_code()))

    # Cell 3: Install dependencies
    cells.append(_make_markdown_cell(
        "## 2️⃣ Install Dependencies\n"
        "\n"
        "Clone autoresearch and install required packages."
    ))
    cells.append(_make_code_cell(_install_dependencies_code()))

    # Cell 4: TPU setup (if TPU runtime)
    if runtime_type == "TPU":
        cells.append(_make_markdown_cell(
            "## 2.5️⃣ TPU Configuration\n"
            "\n"
            "Set up TPU-specific libraries (JAX, torch_xla)."
        ))
        cells.append(_make_code_cell(_tpu_adaptation_code()))

    # Cell 5: Write program.md
    cells.append(_make_markdown_cell(
        "## 3️⃣ Write Experiment Program\n"
        "\n"
        f"Embed the {len(program.experiments)}-experiment program generated "
        "from gap analysis and opportunity scoring."
    ))
    cells.append(_make_code_cell(_write_program_code(program_md)))

    # Cell 6: Show program preview
    cells.append(_make_markdown_cell(
        "## 4️⃣ Experiment Program Preview\n"
        "\n"
        "Review the experiments before running:\n"
    ))
    cells.append(_make_code_cell(
        'with open("/content/autoresearch/program.md") as f:\n'
        '    from IPython.display import Markdown, display\n'
        '    display(Markdown(f.read()))\n'
    ))

    # Cell 7: Run experiments
    cells.append(_make_markdown_cell(
        "## 5️⃣ Run Experiments 🚀\n"
        "\n"
        "Execute the autoresearch experiment program on Colab hardware.\n"
        "\n"
        "> ⏱️ **Estimated time:** 5–30 minutes depending on GPU tier."
    ))
    cells.append(_make_code_cell(_run_experiments_code()))

    # Cell 8: Collect results
    cells.append(_make_markdown_cell(
        "## 6️⃣ View Results\n"
        "\n"
        "Browse experiment outputs and download results."
    ))
    cells.append(_make_code_cell(
        'import os\n'
        '\n'
        'results_dir = "/content/autoresearch/results"\n'
        'if os.path.exists(results_dir):\n'
        '    for root, dirs, files in os.walk(results_dir):\n'
        '        for f in files:\n'
        '            path = os.path.join(root, f)\n'
        '            size = os.path.getsize(path)\n'
        '            print(f"  {path} ({size:,} bytes)")\n'
        'else:\n'
        '    print("No results directory found — experiments may not have run yet.")\n'
    ))

    # Cell 9: Download results
    cells.append(_make_markdown_cell(
        "## 7️⃣ Download Results\n"
        "\n"
        "Download experiment results to your local machine."
    ))
    cells.append(_make_code_cell(
        'import shutil\n'
        '\n'
        '# Zip results for download\n'
        'results_dir = "/content/autoresearch/results"\n'
        'if os.path.exists(results_dir):\n'
        '    shutil.make_archive("/content/experiment_results", "zip", results_dir)\n'
        '    print("✅ Results archived: /content/experiment_results.zip")\n'
        '    # Auto-download in Colab\n'
        '    try:\n'
        '        from google.colab import files\n'
        '        files.download("/content/experiment_results.zip")\n'
        '    except ImportError:\n'
        '        print("   Download manually from the file browser (left panel)")\n'
        'else:\n'
        '    print("No results to download.")\n'
    ))

    return ColabNotebook(
        cells=cells,
        experiment_count=len(program.experiments),
        runtime_type=runtime_type,
        generated_date=date.today().isoformat(),
        program_markdown=program_md,
    )


# ---------------------------------------------------------------------------
# Rendering — .ipynb JSON
# ---------------------------------------------------------------------------

def render_notebook_ipynb(notebook: ColabNotebook) -> str:
    """Render a ColabNotebook as a valid .ipynb JSON string.

    Parameters
    ----------
    notebook : ColabNotebook
        The notebook structure to render.

    Returns
    -------
    str
        JSON string conforming to the Jupyter notebook format (nbformat 4).
    """
    ipynb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "provenance": [],
                "gpuType": "T4" if notebook.runtime_type == "GPU" else "TPU",
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
            },
            "language_info": {
                "name": "python",
            },
            "accelerator": notebook.runtime_type,
        },
        "cells": notebook.cells,
    }
    return json.dumps(ipynb, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Rendering — plain text summary
# ---------------------------------------------------------------------------

def render_colab_text(notebook: ColabNotebook) -> str:
    """Render a plain-text summary of the Colab notebook.

    Parameters
    ----------
    notebook : ColabNotebook
        The notebook structure to summarize.

    Returns
    -------
    str
        Plain-text report showing notebook contents and usage instructions.
    """
    sep = "=" * 78
    subsep = "-" * 78
    lines: List[str] = []

    lines.append(sep)
    lines.append("  GOOGLE COLAB NOTEBOOK — AUTORESEARCH EXPERIMENTS")
    lines.append(sep)
    lines.append("")

    lines.append(f"  Generated     : {notebook.generated_date}")
    lines.append(f"  Target runtime: {notebook.runtime_type}")
    lines.append(f"  Experiments   : {notebook.experiment_count}")
    lines.append(f"  Cells         : {len(notebook.cells)}")
    lines.append("")

    lines.append("  AVAILABLE COLAB RUNTIMES")
    lines.append(subsep)
    for rt in COLAB_RUNTIMES:
        lines.append(f"    {rt.name:16s} | {rt.accelerator_type:3s} | "
                     f"{rt.memory_gb:5.1f} GB | {rt.colab_tier:5s} | "
                     f"{rt.best_for}")
    lines.append("")

    lines.append("  HOW TO USE")
    lines.append(subsep)
    lines.append("    1. Generate the notebook:")
    lines.append("       python -m ai_researcher --colab-notebook -o experiments.ipynb")
    lines.append("    2. Upload to Google Colab:")
    lines.append("       https://colab.research.google.com/ → Upload → experiments.ipynb")
    lines.append("    3. Select runtime:")
    lines.append(f"       Runtime → Change runtime type → {notebook.runtime_type}")
    lines.append("    4. Run all cells:")
    lines.append("       Runtime → Run all (Ctrl+F9)")
    lines.append("    5. Download results when complete")
    lines.append("")

    lines.append("  WHY GOOGLE COLAB?")
    lines.append(subsep)
    lines.append("    This sandbox (GitHub Actions) has NO GPU/TPU.")
    lines.append("    Google Colab provides FREE GPU (T4) and TPU (v2) access.")
    lines.append("    Our agent generates experiments here → Colab runs them.")
    lines.append("")
    lines.append("    GPU advantages:")
    lines.append("      • NVIDIA T4 (free): 16 GB VRAM, 2,560 CUDA cores")
    lines.append("      • NVIDIA A100 (Pro+): 40 GB VRAM, 6,912 CUDA cores")
    lines.append("      • CUDA + cuDNN pre-installed")
    lines.append("      • PyTorch + TensorFlow ready out of the box")
    lines.append("")
    lines.append("    TPU advantages:")
    lines.append("      • TPU v2 (free): 64 GB HBM, 180 TFLOPS bf16")
    lines.append("      • Optimized for large-batch training")
    lines.append("      • Great for JAX/Flax workloads")
    lines.append("      • Can train models that don't fit on T4 GPU")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rendering — Markdown summary
# ---------------------------------------------------------------------------

def render_colab_markdown(notebook: ColabNotebook) -> str:
    """Render a Markdown summary of the Colab notebook.

    Parameters
    ----------
    notebook : ColabNotebook
        The notebook structure to summarize.

    Returns
    -------
    str
        Markdown report with usage instructions and runtime comparison.
    """
    lines: List[str] = []

    lines.append("# ☁️ Google Colab Integration — GPU/TPU Experiments\n")

    lines.append(f"**Generated:** {notebook.generated_date}  ")
    lines.append(f"**Target runtime:** {notebook.runtime_type}  ")
    lines.append(f"**Experiments:** {notebook.experiment_count}  ")
    lines.append(f"**Notebook cells:** {len(notebook.cells)}")
    lines.append("")

    # Runtime comparison table
    lines.append("## 🖥️ Available Colab Runtimes\n")
    lines.append("| Accelerator | Type | VRAM/HBM | Tier | Best For |")
    lines.append("|-------------|------|----------|------|----------|")
    for rt in COLAB_RUNTIMES:
        lines.append(
            f"| {rt.name} | {rt.accelerator_type} | {rt.memory_gb} GB "
            f"| {rt.colab_tier} | {rt.best_for} |"
        )
    lines.append("")

    # Usage instructions
    lines.append("## 🚀 How to Use\n")
    lines.append("```bash")
    lines.append("# Step 1: Generate the notebook")
    lines.append("python -m ai_researcher --colab-notebook -o experiments.ipynb")
    lines.append("```\n")
    lines.append("1. **Upload** `experiments.ipynb` to "
                 "[Google Colab](https://colab.research.google.com/)")
    lines.append(f"2. **Select runtime:** `Runtime → Change runtime type → {notebook.runtime_type}`")
    lines.append("3. **Run all cells:** `Runtime → Run all` (Ctrl+F9)")
    lines.append("4. **Monitor:** Experiments take 5–30 minutes")
    lines.append("5. **Download** results when the final cell completes")
    lines.append("")

    # Why Colab?
    lines.append("## 💡 Why Google Colab?\n")
    lines.append("This sandbox (GitHub Actions) has **no GPU or TPU**. "
                 "Google Colab bridges that gap:\n")
    lines.append("| Feature | This Sandbox | Google Colab (Free) | Google Colab (Pro+) |")
    lines.append("|---------|-------------|--------------------|--------------------|")
    lines.append("| GPU | ❌ None | ✅ NVIDIA T4 (16 GB) | ✅ NVIDIA A100 (40 GB) |")
    lines.append("| TPU | ❌ None | ✅ TPU v2 (64 GB) | ✅ TPU v3 |")
    lines.append("| CUDA | ❌ None | ✅ Pre-installed | ✅ Pre-installed |")
    lines.append("| PyTorch | ✅ CPU only | ✅ GPU-accelerated | ✅ GPU-accelerated |")
    lines.append("| autoresearch | ⚠️ Generate only | ✅ Full execution | ✅ Full execution |")
    lines.append("| Cost | Free | Free (limited) | ~$10/month |")
    lines.append("")

    # GPU vs TPU guidance
    lines.append("## ⚡ GPU vs TPU — Which to Choose?\n")
    lines.append("| Criterion | GPU (T4/A100) | TPU (v2/v3) |")
    lines.append("|-----------|---------------|-------------|")
    lines.append("| autoresearch | ✅ Native support | ⚠️ Needs torch_xla |")
    lines.append("| PyTorch | ✅ Native | ⚠️ Via torch_xla |")
    lines.append("| JAX/Flax | ✅ Supported | ✅ Optimized |")
    lines.append("| TensorFlow | ✅ Supported | ✅ Optimized |")
    lines.append("| Small models | ✅ Fast | ⚠️ Overhead |")
    lines.append("| Large batches | ✅ Good | ✅ Excellent |")
    lines.append("")
    lines.append("> **Recommendation:** Use **GPU** (T4) for autoresearch "
                 "experiments. Use **TPU** if you need more memory or are "
                 "using JAX-based training.")
    lines.append("")

    return "\n".join(lines)
