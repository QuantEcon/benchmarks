"""
Sample each CPU's clock and how busy it was, every few seconds, until the run's log ends.

    python3 clock_sampler.py <out.jsonl> <run.log> [seconds]

Each line is one sample: the UTC time, every CPU's MHz from /proc/cpuinfo (on a kernel with
aperfmperf, the clock it actually ran at since the last tick), and every CPU's busy share of the
interval since the previous sample, from /proc/stat. A serial cell's clock is the MHz of the CPU
that was busy. Started by hand on project-benchmarks-jax#47's instances, beside the suite.
"""
import json
import sys
import time
from datetime import datetime, timezone

out, log = sys.argv[1], sys.argv[2]
every = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0


def mhz():
    return [round(float(line.split(":")[1])) for line in open("/proc/cpuinfo")
            if line.startswith("cpu MHz")]


def jiffies():
    cpus = {}
    for line in open("/proc/stat"):
        name, *fields = line.split()
        if name.startswith("cpu") and name != "cpu":
            v = [int(x) for x in fields]
            idle = v[3] + v[4]  # idle and iowait
            cpus[int(name[3:])] = (sum(v[:8]) - idle, sum(v[:8]))
    return [cpus[i] for i in sorted(cpus)]


def finished():
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 200))
            return b"\nEXIT " in f.read()
    except OSError:
        return False


before = jiffies()
with open(out, "a") as f:
    while not finished():
        time.sleep(every)
        now, clocks = jiffies(), mhz()
        busy = [round((b1 - b0) / (t1 - t0), 2) if t1 > t0 else None
                for (b0, t0), (b1, t1) in zip(before, now)]
        before = now
        f.write(json.dumps({"t": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                            "mhz": clocks, "busy": busy}) + "\n")
        f.flush()
