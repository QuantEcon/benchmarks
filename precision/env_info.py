"""
Describe the machine a benchmark ran on, without importing JAX.

A superset of the `system` block in hardware/benchmark_hardware.py, so the
two outputs can be read together.
"""
import json
import os
import platform
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

# `rocm` is ROCm's own Python package, for a machine that takes ROCm from pip, not /opt/rocm
PACKAGES = ["jax", "jaxlib", "jax-cuda13-plugin", "jax-cuda13-pjrt",
            "jax-cuda12-plugin", "jax-cuda12-pjrt", "jax-rocm7-plugin", "jax-rocm7-pjrt",
            "jax-rocm10-plugin", "jax-rocm10-pjrt", "rocm", "numpy", "scipy", "numba", "llvmlite"]
FLAG_PREFIXES = ("JAX_", "XLA_", "CUDA_", "HIP_", "ROCR_", "HSA_", "OMP_", "MKL_", "OPENBLAS_",
                 "NUMBA_")
REPO = Path(__file__).resolve().parent.parent


def _run(cmd, timeout=10):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def memory_bytes():
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def cpu():
    info = {"cpu_count": os.cpu_count(), "cpu_model": None, "cpu_mhz": None,
            "cpu_physical": None, "cpu_core_types": None}
    if sys.platform == "darwin":
        info["cpu_model"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        physical = _run(["sysctl", "-n", "hw.physicalcpu"])
        info["cpu_physical"] = int(physical) if physical else None
        # Apple Silicon splits its cores into performance levels
        levels = {}
        for level in range(4):
            name = _run(["sysctl", "-n", f"hw.perflevel{level}.name"])
            count = _run(["sysctl", "-n", f"hw.perflevel{level}.physicalcpu"])
            if not name or not count:
                break
            levels[name] = int(count)
        info["cpu_core_types"] = levels or None
        return info

    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return info
    cores, physical_id = set(), None
    for line in text.splitlines():
        key, _, value = (part.strip() for part in line.partition(":"))
        if key == "model name" and info["cpu_model"] is None:
            info["cpu_model"] = value
        elif key == "cpu MHz" and info["cpu_mhz"] is None:
            info["cpu_mhz"] = value
        elif key == "physical id":
            physical_id = value
        elif key == "core id":
            cores.add((physical_id, value))
    info["cpu_physical"] = len(cores) or None
    return info


def nvidia_gpus():
    out = _run(["nvidia-smi", "--query-gpu=index,name,pci.bus_id,memory.total,driver_version",
                "--format=csv,noheader,nounits"])
    if not out:
        return []
    cards = []
    for line in out.splitlines():
        index, name, bus, memory, driver = (f.strip() for f in line.split(","))
        cards.append({"index": int(index), "name": name, "pci_bus_id": bus,
                      "memory_total_mib": int(memory), "driver_version": driver})
    return cards


def _number(value):
    """The number in an amd-smi value: 196592, "196592 MB" or {"value": 196592, "unit": "MB"}."""
    if isinstance(value, dict):
        value = value.get("value")
    match = re.match(r"\s*(\d+)", str(value))
    return int(match.group(1)) if match else None


def amd_gpus():
    """AMD's cards from amd-smi, in the form nvidia_gpus() gives, with each card's gfx target."""
    try:
        data = json.loads(_run(["amd-smi", "static", "--json"], timeout=60) or "[]")
    except ValueError:
        return []
    # amd-smi lists the cards, or in later versions puts the list under gpu_data
    if isinstance(data, dict):
        data = data.get("gpu_data", [data])
    cards = []
    for i, card in enumerate(data if isinstance(data, list) else []):
        if not isinstance(card, dict):
            continue
        asic, bus = card.get("asic") or {}, card.get("bus") or {}
        # amd-smi's MB are MiB
        memory = _number((card.get("vram") or {}).get("size"))
        if memory is None and bus.get("bdf"):
            # The kernel driver's own count, in bytes
            total = _read(f"/sys/bus/pci/devices/{bus['bdf']}/mem_info_vram_total")
            memory = int(total) // 2**20 if total and total.isdigit() else None
        cards.append({"index": card.get("gpu", i), "name": asic.get("market_name"),
                      "pci_bus_id": bus.get("bdf"), "memory_total_mib": memory,
                      "driver_version": (card.get("driver") or {}).get("version"),
                      "gfx_target": asic.get("target_graphics_version")})
    return cards


def cuda_version():
    match = re.search(r"CUDA Version:\s*([\d.]+)", _run(["nvidia-smi"]) or "")
    return match.group(1) if match else None


def rocm_version():
    """The ROCm in /opt/rocm, or else the one amd-smi reports. ROCm from pip is in `packages`."""
    version = _read("/opt/rocm/.info/version")
    if version:
        return version
    match = re.search(r"ROCm version:\s*([\w.-]+)", _run(["amd-smi", "version"], timeout=60) or "")
    return match.group(1) if match else None


def packages():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def collect():
    cards, gpu_platform = nvidia_gpus(), "cuda"
    if not cards:
        cards, gpu_platform = amd_gpus(), "rocm"
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        **cpu(),
        "memory_bytes": memory_bytes(),
        "gpu": "; ".join(f"{c['name']}, {c['memory_total_mib']} MiB" if c["memory_total_mib"]
                         else str(c["name"]) for c in cards) or None,
        "gpus": cards,
        # The JAX platform that runs the cards: cuda for NVIDIA's, rocm for AMD's
        "gpu_platform": gpu_platform if cards else None,
        "cuda_version": cuda_version() if cards and gpu_platform == "cuda" else None,
        "rocm_version": rocm_version() if cards and gpu_platform == "rocm" else None,
        "packages": packages(),
        "flags": {k: v for k, v in sorted(os.environ.items()) if k.startswith(FLAG_PREFIXES)},
        "git_commit": _run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]),
        # Untracked files, such as earlier results, don't change the code that ran
        "git_dirty": bool(_run(["git", "-C", str(REPO), "status", "--porcelain",
                                "--untracked-files=no"])),
        "numactl": _run(["numactl", "--hardware"]),
    }


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2))
