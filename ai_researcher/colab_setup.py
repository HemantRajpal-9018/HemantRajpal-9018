"""GitHub-to-Google-Colab direct connection setup.

Generates one-click "Open in Colab" links, badges, and step-by-step
setup instructions so users can connect **directly** from this GitHub
repository to Google Colab — no manual uploads needed.

The standard Colab GitHub integration URL format is::

    https://colab.research.google.com/github/{owner}/{repo}/blob/{branch}/{path}

This module:

1. Builds those direct-link URLs for every notebook in the repo.
2. Generates Markdown badges (``Open in Colab`` shields).
3. Provides a step-by-step setup guide rendered as text or Markdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GITHUB_OWNER = "HemantRajpal-9018"
GITHUB_REPO = "HemantRajpal-9018"
DEFAULT_BRANCH = "main"

COLAB_GITHUB_BASE = "https://colab.research.google.com/github"

# Notebook paths committed in the repo (relative to repo root)
NOTEBOOK_PATHS: Dict[str, str] = {
    "GPU experiments": "notebooks/gpu_experiments.ipynb",
    "TPU experiments": "notebooks/tpu_experiments.ipynb",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ColabLink:
    """A direct GitHub-to-Colab connection link."""
    label: str                  # Human-readable name (e.g. "GPU experiments")
    notebook_path: str          # Path relative to repo root
    colab_url: str              # Full https://colab.research.google.com/... URL
    badge_markdown: str         # Markdown badge image + link
    badge_html: str             # HTML <a><img></a> badge


@dataclass
class ColabSetupGuide:
    """Step-by-step guide for connecting GitHub to Google Colab."""
    links: List[ColabLink] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    owner: str = GITHUB_OWNER
    repo: str = GITHUB_REPO
    branch: str = DEFAULT_BRANCH


# ---------------------------------------------------------------------------
# Link generation
# ---------------------------------------------------------------------------

def build_colab_url(
    notebook_path: str,
    owner: str = GITHUB_OWNER,
    repo: str = GITHUB_REPO,
    branch: str = DEFAULT_BRANCH,
) -> str:
    """Build a direct Google Colab URL that opens a GitHub-hosted notebook.

    Parameters
    ----------
    notebook_path : str
        Path to the ``.ipynb`` file relative to the repo root.
    owner : str
        GitHub repository owner.
    repo : str
        GitHub repository name.
    branch : str
        Git branch name (default ``main``).

    Returns
    -------
    str
        A ``https://colab.research.google.com/github/...`` URL.
    """
    return f"{COLAB_GITHUB_BASE}/{owner}/{repo}/blob/{branch}/{notebook_path}"


def build_badge_markdown(colab_url: str, alt: str = "Open In Colab") -> str:
    """Return a Markdown badge linking to the Colab URL.

    Uses the official Google Colab badge from ``colab.research.google.com``.
    """
    badge_img = "https://colab.research.google.com/assets/colab-badge.svg"
    return f"[![{alt}]({badge_img})]({colab_url})"


def build_badge_html(colab_url: str, alt: str = "Open In Colab") -> str:
    """Return an HTML badge linking to the Colab URL."""
    badge_img = "https://colab.research.google.com/assets/colab-badge.svg"
    return (
        f'<a href="{colab_url}" target="_blank">'
        f'<img src="{badge_img}" alt="{alt}"/>'
        f'</a>'
    )


def generate_colab_link(
    label: str,
    notebook_path: str,
    owner: str = GITHUB_OWNER,
    repo: str = GITHUB_REPO,
    branch: str = DEFAULT_BRANCH,
) -> ColabLink:
    """Generate a ``ColabLink`` for one notebook.

    Parameters
    ----------
    label : str
        Human-readable name (e.g. "GPU experiments").
    notebook_path : str
        Path to the ``.ipynb`` file relative to the repo root.

    Returns
    -------
    ColabLink
    """
    url = build_colab_url(notebook_path, owner, repo, branch)
    return ColabLink(
        label=label,
        notebook_path=notebook_path,
        colab_url=url,
        badge_markdown=build_badge_markdown(url, alt=f"Open {label} In Colab"),
        badge_html=build_badge_html(url, alt=f"Open {label} In Colab"),
    )


# ---------------------------------------------------------------------------
# Step-by-step guide generation
# ---------------------------------------------------------------------------

_SETUP_STEPS = [
    "Click the 'Open in Colab' badge above (or copy the link).",
    "Google Colab opens the notebook directly from GitHub — no download needed.",
    "Select your runtime: Runtime → Change runtime type → GPU (T4 free) or TPU.",
    "Click 'Runtime → Run all' (Ctrl+F9) to execute all cells.",
    "The notebook will: detect hardware → install dependencies → run experiments.",
    "Monitor progress in each cell's output (takes 5–30 minutes).",
    "When done, the final cell downloads a ZIP of your experiment results.",
    "To re-run with changes: edit cells in Colab, then 'Runtime → Run all' again.",
]


def generate_setup_guide(
    owner: str = GITHUB_OWNER,
    repo: str = GITHUB_REPO,
    branch: str = DEFAULT_BRANCH,
) -> ColabSetupGuide:
    """Generate the full setup guide with links and steps.

    Returns
    -------
    ColabSetupGuide
    """
    links = [
        generate_colab_link(label, path, owner, repo, branch)
        for label, path in NOTEBOOK_PATHS.items()
    ]
    return ColabSetupGuide(
        links=links,
        steps=list(_SETUP_STEPS),
        owner=owner,
        repo=repo,
        branch=branch,
    )


# ---------------------------------------------------------------------------
# Rendering — plain text
# ---------------------------------------------------------------------------

def render_setup_text(guide: ColabSetupGuide) -> str:
    """Render the setup guide as plain text.

    Parameters
    ----------
    guide : ColabSetupGuide

    Returns
    -------
    str
    """
    sep = "=" * 78
    subsep = "-" * 78
    lines: list[str] = []

    lines.append(sep)
    lines.append("  GITHUB → GOOGLE COLAB — DIRECT CONNECTION SETUP")
    lines.append(sep)
    lines.append("")

    lines.append("  ONE-CLICK LINKS (open directly from GitHub)")
    lines.append(subsep)
    for link in guide.links:
        lines.append(f"    {link.label}:")
        lines.append(f"      {link.colab_url}")
    lines.append("")

    lines.append("  STEP-BY-STEP SETUP")
    lines.append(subsep)
    for i, step in enumerate(guide.steps, 1):
        lines.append(f"    Step {i}: {step}")
    lines.append("")

    lines.append("  HOW IT WORKS")
    lines.append(subsep)
    lines.append("    Google Colab can open any .ipynb notebook hosted on GitHub")
    lines.append("    using the URL pattern:")
    lines.append("")
    lines.append(f"      {COLAB_GITHUB_BASE}/{{owner}}/{{repo}}/blob/{{branch}}/{{path}}")
    lines.append("")
    lines.append("    This repository has pre-built notebooks in the notebooks/ folder.")
    lines.append("    Clicking the link above opens them directly in Colab — no upload,")
    lines.append("    no download, no manual steps.  Just click and run.")
    lines.append("")

    lines.append("  WHAT YOU GET")
    lines.append(subsep)
    lines.append("    • Free NVIDIA T4 GPU (16 GB VRAM) or TPU v2 (64 GB HBM)")
    lines.append("    • CUDA + cuDNN + PyTorch pre-installed")
    lines.append("    • autoresearch experiment program embedded in the notebook")
    lines.append("    • Automatic result collection and download")
    lines.append("    • No local GPU required — everything runs in the cloud")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rendering — Markdown
# ---------------------------------------------------------------------------

def render_setup_markdown(guide: ColabSetupGuide) -> str:
    """Render the setup guide as Markdown.

    Parameters
    ----------
    guide : ColabSetupGuide

    Returns
    -------
    str
    """
    lines: list[str] = []

    lines.append("# 🔗 GitHub → Google Colab — Direct Connection Setup\n")

    lines.append("Open notebooks **directly** from this GitHub repo in Google Colab —")
    lines.append("no downloads, no uploads, just one click.\n")

    # Badges
    lines.append("## 🚀 One-Click Open in Colab\n")
    for link in guide.links:
        lines.append(f"**{link.label}:**\n")
        lines.append(link.badge_markdown)
        lines.append("")
        lines.append(f"Direct link: {link.colab_url}\n")

    # Steps
    lines.append("## 📋 Step-by-Step Setup\n")
    for i, step in enumerate(guide.steps, 1):
        lines.append(f"{i}. {step}")
    lines.append("")

    # How it works
    lines.append("## ⚙️ How It Works\n")
    lines.append("Google Colab can open **any** `.ipynb` notebook hosted on GitHub ")
    lines.append("using this URL pattern:\n")
    lines.append("```")
    lines.append(f"{COLAB_GITHUB_BASE}/{{owner}}/{{repo}}/blob/{{branch}}/{{path}}")
    lines.append("```\n")
    lines.append("This repository has pre-built notebooks in `notebooks/`:")
    lines.append("")
    lines.append("| Notebook | Runtime | Link |")
    lines.append("|----------|---------|------|")
    for link in guide.links:
        lines.append(f"| {link.label} | {'GPU' if 'gpu' in link.notebook_path else 'TPU'} | "
                     f"[Open in Colab]({link.colab_url}) |")
    lines.append("")

    lines.append("When you click **Open in Colab**, Google Colab:")
    lines.append("")
    lines.append("1. Clones the notebook directly from this GitHub repo")
    lines.append("2. Connects you to a free cloud VM with GPU or TPU")
    lines.append("3. You just press \"Run all\" — experiments start automatically")
    lines.append("4. Results are downloaded to your machine when done")
    lines.append("")

    # What you get
    lines.append("## 🎁 What You Get\n")
    lines.append("| Feature | Details |")
    lines.append("|---------|---------|")
    lines.append("| GPU | NVIDIA T4 (16 GB VRAM) — free tier |")
    lines.append("| TPU | TPU v2-8 (64 GB HBM) — free tier |")
    lines.append("| CUDA | Pre-installed (no setup needed) |")
    lines.append("| PyTorch | GPU-accelerated, pre-installed |")
    lines.append("| Experiments | 13 autoresearch experiments embedded |")
    lines.append("| Results | Auto-downloaded as ZIP when done |")
    lines.append("| Cost | Free (Colab free tier) |")
    lines.append("")

    # Comparison
    lines.append("## ⚡ This Sandbox vs Google Colab\n")
    lines.append("| Resource | This Sandbox | Google Colab (Free) |")
    lines.append("|----------|-------------|---------------------|")
    lines.append("| GPU | ❌ None | ✅ NVIDIA T4 (16 GB) |")
    lines.append("| TPU | ❌ None | ✅ TPU v2 (64 GB) |")
    lines.append("| CUDA | ❌ None | ✅ Pre-installed |")
    lines.append("| autoresearch | ⚠️ Generate only | ✅ Full execution |")
    lines.append("| Setup time | N/A | ~30 seconds (one click) |")
    lines.append("")

    return "\n".join(lines)
