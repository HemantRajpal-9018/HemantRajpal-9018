"""Sandbox environment configuration inspector.

Detects hardware capabilities (CPU, RAM, GPU), installed software,
and assesses which AI research tools can run in the current environment.
This is particularly useful for determining whether GPU-dependent tools
like karpathy/autoresearch can execute here.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GpuInfo:
    """GPU device information."""
    name: str
    driver: str = ""
    memory_mb: int = 0
    cuda_version: str = ""

    @property
    def is_nvidia(self) -> bool:
        return "nvidia" in self.name.lower() or "cuda" in self.driver.lower()


@dataclass
class SandboxConfig:
    """Complete sandbox environment configuration."""
    # CPU
    cpu_model: str = "unknown"
    cpu_cores: int = 0
    cpu_threads: int = 0
    architecture: str = ""

    # Memory
    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0

    # Disk
    total_disk_gb: float = 0.0
    free_disk_gb: float = 0.0

    # GPU
    gpus: List[GpuInfo] = field(default_factory=list)
    has_nvidia_gpu: bool = False
    has_cuda: bool = False
    cuda_version: str = ""

    # OS / Runtime
    os_name: str = ""
    os_version: str = ""
    kernel_version: str = ""
    python_version: str = ""
    hypervisor: str = ""

    # Installed tools
    installed_tools: Dict[str, str] = field(default_factory=dict)

    # Capability assessment
    can_run_autoresearch: bool = False
    autoresearch_blockers: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _run_cmd(cmd: List[str], fallback: str = "") -> str:
    """Run a shell command and return stripped stdout, or *fallback*."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else fallback
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return fallback


def _detect_cpu() -> tuple[str, int, int, str]:
    """Return (model, cores, threads, architecture)."""
    arch = platform.machine() or "unknown"

    # Try /proc/cpuinfo (Linux)
    model = "unknown"
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except (FileNotFoundError, OSError):
        model = platform.processor() or "unknown"

    cores = os.cpu_count() or 0
    # Physical cores from nproc or fallback to os.cpu_count
    threads = cores
    nproc = _run_cmd(["nproc"])
    if nproc:
        try:
            threads = int(nproc)
        except ValueError:
            pass

    return model, cores, threads, arch


def _detect_memory() -> tuple[float, float]:
    """Return (total_gb, available_gb)."""
    try:
        with open("/proc/meminfo", "r") as f:
            lines = f.readlines()
        mem = {}
        for line in lines:
            parts = line.split(":")
            if len(parts) == 2:
                key = parts[0].strip()
                val = parts[1].strip().split()[0]  # kB value
                try:
                    mem[key] = int(val)
                except ValueError:
                    pass
        total = mem.get("MemTotal", 0) / (1024 * 1024)  # kB → GB
        available = mem.get("MemAvailable", 0) / (1024 * 1024)
        return round(total, 1), round(available, 1)
    except (FileNotFoundError, OSError):
        return 0.0, 0.0


def _detect_disk() -> tuple[float, float]:
    """Return (total_gb, free_gb) for the root filesystem."""
    try:
        usage = shutil.disk_usage("/")
        return (
            round(usage.total / (1024 ** 3), 1),
            round(usage.free / (1024 ** 3), 1),
        )
    except OSError:
        return 0.0, 0.0


def _detect_gpus() -> tuple[List[GpuInfo], bool, bool, str]:
    """Detect GPU devices.  Returns (gpus, has_nvidia, has_cuda, cuda_ver)."""
    gpus: List[GpuInfo] = []
    has_nvidia = False
    has_cuda = False
    cuda_ver = ""

    # nvidia-smi check
    nvidia_out = _run_cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                           "--format=csv,noheader,nounits"])
    if nvidia_out:
        has_nvidia = True
        for line in nvidia_out.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    mem = int(float(parts[2]))
                except ValueError:
                    mem = 0
                gpus.append(GpuInfo(name=parts[0], driver=parts[1], memory_mb=mem))
            elif len(parts) >= 1:
                gpus.append(GpuInfo(name=parts[0]))

    # CUDA version
    nvcc_out = _run_cmd(["nvcc", "--version"])
    if "release" in nvcc_out.lower():
        for token in nvcc_out.split():
            if token.startswith("V") or ("." in token and any(c.isdigit() for c in token)):
                cuda_ver = token.lstrip("V").rstrip(",")
                if cuda_ver and cuda_ver[0].isdigit():
                    has_cuda = True
                    break

    # Also check /dev/nvidia* presence
    if not has_nvidia:
        try:
            has_nvidia = any(
                f.startswith("nvidia") for f in os.listdir("/dev/")
            )
        except OSError:
            pass

    return gpus, has_nvidia, has_cuda, cuda_ver


def _detect_os() -> tuple[str, str, str, str]:
    """Return (os_name, os_version, kernel, hypervisor)."""
    os_name = ""
    os_ver = ""
    try:
        with open("/etc/os-release", "r") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    os_name = line.split("=", 1)[1].strip().strip('"')
                elif line.startswith("VERSION_ID="):
                    os_ver = line.split("=", 1)[1].strip().strip('"')
    except (FileNotFoundError, OSError):
        os_name = platform.system()
        os_ver = platform.release()

    kernel = platform.release()

    # Hypervisor detection
    hypervisor = ""
    virt = _run_cmd(["systemd-detect-virt"])
    if virt:
        hypervisor = virt
    else:
        lscpu_out = _run_cmd(["lscpu"])
        for line in lscpu_out.split("\n"):
            if "Hypervisor vendor" in line:
                hypervisor = line.split(":", 1)[1].strip()
                break

    return os_name, os_ver, kernel, hypervisor


def _detect_tools() -> Dict[str, str]:
    """Detect installed development tools and their versions."""
    tools: Dict[str, str] = {}

    checks = [
        ("python", ["python3", "--version"]),
        ("pip", ["pip", "--version"]),
        ("git", ["git", "--version"]),
        ("node", ["node", "--version"]),
        ("go", ["go", "version"]),
        ("docker", ["docker", "--version"]),
        ("nvidia-smi", ["nvidia-smi", "--version"]),
        ("nvcc", ["nvcc", "--version"]),
        ("uv", ["uv", "--version"]),
    ]
    for name, cmd in checks:
        result = _run_cmd(cmd)
        if result:
            # Clean up multi-line output
            tools[name] = result.split("\n")[0].strip()

    return tools


def _assess_autoresearch(
    has_nvidia: bool,
    has_cuda: bool,
    gpus: List[GpuInfo],
    python_version: str,
    tools: Dict[str, str],
) -> tuple[bool, List[str]]:
    """Assess whether karpathy/autoresearch can run here."""
    blockers: List[str] = []

    if not has_nvidia:
        blockers.append("No NVIDIA GPU detected (autoresearch requires NVIDIA GPU, tested on H100)")

    if not has_cuda and has_nvidia:
        blockers.append("NVIDIA GPU found but CUDA toolkit not installed")

    # Check Python version (needs 3.10+)
    try:
        parts = python_version.split(".")
        major, minor = int(parts[0]), int(parts[1])
        if major < 3 or (major == 3 and minor < 10):
            blockers.append(f"Python {python_version} found; autoresearch needs 3.10+")
    except (ValueError, IndexError):
        pass

    # Check for uv (autoresearch uses uv)
    if "uv" not in tools:
        blockers.append("'uv' package manager not installed (run: curl -LsSf https://astral.sh/uv/install.sh | sh)")

    # GPU memory check (if we have GPU info)
    for gpu in gpus:
        if gpu.memory_mb > 0 and gpu.memory_mb < 16000:
            blockers.append(
                f"GPU {gpu.name} has {gpu.memory_mb} MB VRAM; "
                "autoresearch may need 40GB+ for default settings "
                "(reduce model size for smaller GPUs)"
            )

    can_run = len(blockers) == 0 or (
        len(blockers) == 1 and "uv" in blockers[0]
    )
    return can_run, blockers


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------

def detect_sandbox_config() -> SandboxConfig:
    """Detect and return the full sandbox environment configuration."""
    cpu_model, cpu_cores, cpu_threads, arch = _detect_cpu()
    total_ram, avail_ram = _detect_memory()
    total_disk, free_disk = _detect_disk()
    gpus, has_nvidia, has_cuda, cuda_ver = _detect_gpus()
    os_name, os_ver, kernel, hypervisor = _detect_os()
    tools = _detect_tools()
    py_ver = platform.python_version()

    can_run, blockers = _assess_autoresearch(
        has_nvidia, has_cuda, gpus, py_ver, tools,
    )

    return SandboxConfig(
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
        architecture=arch,
        total_ram_gb=total_ram,
        available_ram_gb=avail_ram,
        total_disk_gb=total_disk,
        free_disk_gb=free_disk,
        gpus=gpus,
        has_nvidia_gpu=has_nvidia,
        has_cuda=has_cuda,
        cuda_version=cuda_ver,
        os_name=os_name,
        os_version=os_ver,
        kernel_version=kernel,
        python_version=py_ver,
        hypervisor=hypervisor,
        installed_tools=tools,
        can_run_autoresearch=can_run,
        autoresearch_blockers=blockers,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_config_text(config: SandboxConfig) -> str:
    """Render sandbox configuration as plain text."""
    sep = "=" * 78
    subsep = "-" * 78
    lines: List[str] = []

    lines.append(sep)
    lines.append("  SANDBOX ENVIRONMENT CONFIGURATION")
    lines.append(sep)
    lines.append("")

    # CPU
    lines.append("  CPU")
    lines.append(subsep)
    lines.append(f"    Model        : {config.cpu_model}")
    lines.append(f"    Cores        : {config.cpu_cores}")
    lines.append(f"    Threads      : {config.cpu_threads}")
    lines.append(f"    Architecture : {config.architecture}")
    lines.append("")

    # Memory
    lines.append("  MEMORY")
    lines.append(subsep)
    lines.append(f"    Total RAM    : {config.total_ram_gb} GB")
    lines.append(f"    Available    : {config.available_ram_gb} GB")
    lines.append("")

    # Disk
    lines.append("  DISK")
    lines.append(subsep)
    lines.append(f"    Total        : {config.total_disk_gb} GB")
    lines.append(f"    Free         : {config.free_disk_gb} GB")
    lines.append("")

    # GPU
    lines.append("  GPU")
    lines.append(subsep)
    if config.gpus:
        for i, gpu in enumerate(config.gpus):
            lines.append(f"    GPU {i}        : {gpu.name}")
            if gpu.driver:
                lines.append(f"      Driver     : {gpu.driver}")
            if gpu.memory_mb:
                lines.append(f"      VRAM       : {gpu.memory_mb} MB")
    else:
        lines.append("    Status       : No GPU detected")
    lines.append(f"    NVIDIA       : {'Yes' if config.has_nvidia_gpu else 'No'}")
    lines.append(f"    CUDA         : {'Yes' if config.has_cuda else 'No'}"
                 + (f" ({config.cuda_version})" if config.cuda_version else ""))
    lines.append("")

    # OS
    lines.append("  OPERATING SYSTEM")
    lines.append(subsep)
    lines.append(f"    OS           : {config.os_name}")
    lines.append(f"    Version      : {config.os_version}")
    lines.append(f"    Kernel       : {config.kernel_version}")
    if config.hypervisor:
        lines.append(f"    Hypervisor   : {config.hypervisor}")
    lines.append(f"    Python       : {config.python_version}")
    lines.append("")

    # Installed tools
    lines.append("  INSTALLED TOOLS")
    lines.append(subsep)
    for name, ver in sorted(config.installed_tools.items()):
        lines.append(f"    {name:14s}: {ver}")
    lines.append("")

    # Autoresearch compatibility
    lines.append("  AUTORESEARCH COMPATIBILITY")
    lines.append(subsep)
    if config.can_run_autoresearch:
        lines.append("    Status       : READY — can run karpathy/autoresearch")
    else:
        lines.append("    Status       : NOT READY — cannot run karpathy/autoresearch")
        lines.append("    Blockers     :")
        for blocker in config.autoresearch_blockers:
            lines.append(f"      • {blocker}")
    lines.append("")
    lines.append("    NOTE: This sandbox can still GENERATE experiment programs")
    lines.append("    (--autoresearch-program) for execution on a GPU machine.")
    lines.append("")

    return "\n".join(lines)


def render_config_markdown(config: SandboxConfig) -> str:
    """Render sandbox configuration as Markdown."""
    lines: List[str] = []

    lines.append("# 🖥️ Sandbox Environment Configuration\n")

    # CPU
    lines.append("## CPU\n")
    lines.append(f"| Property | Value |")
    lines.append(f"|----------|-------|")
    lines.append(f"| Model | {config.cpu_model} |")
    lines.append(f"| Cores | {config.cpu_cores} |")
    lines.append(f"| Threads | {config.cpu_threads} |")
    lines.append(f"| Architecture | {config.architecture} |")
    lines.append("")

    # Memory
    lines.append("## Memory\n")
    lines.append(f"| Property | Value |")
    lines.append(f"|----------|-------|")
    lines.append(f"| Total RAM | {config.total_ram_gb} GB |")
    lines.append(f"| Available | {config.available_ram_gb} GB |")
    lines.append("")

    # Disk
    lines.append("## Disk\n")
    lines.append(f"| Property | Value |")
    lines.append(f"|----------|-------|")
    lines.append(f"| Total | {config.total_disk_gb} GB |")
    lines.append(f"| Free | {config.free_disk_gb} GB |")
    lines.append("")

    # GPU
    lines.append("## GPU\n")
    if config.gpus:
        lines.append(f"| GPU | Name | Driver | VRAM |")
        lines.append(f"|-----|------|--------|------|")
        for i, gpu in enumerate(config.gpus):
            lines.append(
                f"| {i} | {gpu.name} | {gpu.driver or 'N/A'} "
                f"| {gpu.memory_mb} MB |"
            )
    else:
        lines.append("**No GPU detected.**\n")

    lines.append("")
    lines.append(f"- NVIDIA GPU: {'✅ Yes' if config.has_nvidia_gpu else '❌ No'}")
    lines.append(f"- CUDA: {'✅ Yes' if config.has_cuda else '❌ No'}"
                 + (f" ({config.cuda_version})" if config.cuda_version else ""))
    lines.append("")

    # OS
    lines.append("## Operating System\n")
    lines.append(f"| Property | Value |")
    lines.append(f"|----------|-------|")
    lines.append(f"| OS | {config.os_name} |")
    lines.append(f"| Version | {config.os_version} |")
    lines.append(f"| Kernel | {config.kernel_version} |")
    if config.hypervisor:
        lines.append(f"| Hypervisor | {config.hypervisor} |")
    lines.append(f"| Python | {config.python_version} |")
    lines.append("")

    # Installed tools
    lines.append("## Installed Tools\n")
    lines.append(f"| Tool | Version |")
    lines.append(f"|------|---------|")
    for name, ver in sorted(config.installed_tools.items()):
        lines.append(f"| {name} | {ver} |")
    lines.append("")

    # Autoresearch compatibility
    lines.append("## 🧪 Autoresearch Compatibility\n")
    if config.can_run_autoresearch:
        lines.append("> ✅ **READY** — this environment can run karpathy/autoresearch\n")
    else:
        lines.append("> ❌ **NOT READY** — this environment cannot run karpathy/autoresearch\n")
        lines.append("**Blockers:**\n")
        for blocker in config.autoresearch_blockers:
            lines.append(f"- {blocker}")
        lines.append("")

    lines.append(
        "> **Note:** This sandbox can still *generate* experiment programs "
        "(`--autoresearch-program`) for execution on a GPU-equipped machine."
    )
    lines.append("")

    return "\n".join(lines)
