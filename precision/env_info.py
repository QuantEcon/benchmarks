"""
Describe the machine a benchmark ran on, without importing JAX.

A superset of the `system` block in hardware/benchmark_hardware.py, so the
two outputs can be read together.
"""
import os
import platform
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

PACKAGES = ["jax", "jaxlib", "jax-cuda13-plugin", "jax-cuda13-pjrt",
            "jax-cuda12-plugin", "jax-cuda12-pjrt", "numpy", "scipy"]
FLAG_PREFIXES = ("JAX_", "XLA_", "CUDA_", "OMP_", "MKL_", "OPENBLAS_")
REPO = Path(__file__).resolve().parent.parent


def _run(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


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


def gpus():
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


def cuda_version():
    match = re.search(r"CUDA Version:\s*([\d.]+)", _run(["nvidia-smi"]) or "")
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
    cards = gpus()
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        **cpu(),
        "memory_bytes": memory_bytes(),
        "gpu": "; ".join(f"{c['name']}, {c['memory_total_mib']} MiB" for c in cards) or None,
        "gpus": cards,
        "cuda_version": cuda_version() if cards else None,
        "packages": packages(),
        "flags": {k: v for k, v in sorted(os.environ.items()) if k.startswith(FLAG_PREFIXES)},
        "git_commit": _run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(_run(["git", "-C", str(REPO), "status", "--porcelain"])),
        "numactl": _run(["numactl", "--hardware"]),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(collect(), indent=2))
