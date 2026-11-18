"""Run-level timing, GPU-hour, memory, and counter accounting.

The profiler deliberately records allocated and reserved CUDA memory separately.
GPU-hours are allocated device-hours: elapsed wall-clock seconds multiplied by
the number of visible CUDA devices.  Stage records are JSON serializable and can
be merged across pool generation, scoring, and search processes.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import os
import resource
import time
from typing import Any, Iterator

import torch


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak() -> None:
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)


def _memory() -> tuple[int, int, int]:
    if not torch.cuda.is_available():
        return 0, 0, 0
    allocated = max(
        int(torch.cuda.max_memory_allocated(index))
        for index in range(torch.cuda.device_count())
    )
    reserved = max(
        int(torch.cuda.max_memory_reserved(index))
        for index in range(torch.cuda.device_count())
    )
    return allocated, reserved, torch.cuda.device_count()


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.  The experiments run on Linux,
    # while this branch keeps local smoke tests portable.
    return int(value * 1024 if os.uname().sysname == "Linux" else value)


@dataclass
class PhaseRecord:
    name: str
    wall_clock_seconds: float
    gpu_hours: float
    visible_gpu_count: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    peak_rss_bytes: int
    counters: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "wall_clock_seconds": self.wall_clock_seconds,
            "gpu_hours": self.gpu_hours,
            "visible_gpu_count": self.visible_gpu_count,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
            "counters": dict(self.counters),
        }


class CostProfiler:
    """Collect synchronized stage timing and resource statistics."""

    def __init__(self, *, method: str, protocol: str) -> None:
        self.method = method
        self.protocol = protocol
        self.started = time.perf_counter()
        self.phases: list[PhaseRecord] = []
        self.counters: dict[str, int] = {}
        self._peak_allocated = 0
        self._peak_reserved = 0
        self._peak_rss = 0

    def count(self, name: str, value: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + int(value)

    @contextmanager
    def phase(self, name: str, **counters: int) -> Iterator[None]:
        for key, value in counters.items():
            self.count(key, value)
        _sync()
        _reset_peak()
        started = time.perf_counter()
        try:
            yield
        finally:
            _sync()
            elapsed = time.perf_counter() - started
            allocated, reserved, visible = _memory()
            rss = _rss_bytes()
            self._peak_allocated = max(self._peak_allocated, allocated)
            self._peak_reserved = max(self._peak_reserved, reserved)
            self._peak_rss = max(self._peak_rss, rss)
            self.phases.append(
                PhaseRecord(
                    name=name,
                    wall_clock_seconds=float(elapsed),
                    gpu_hours=float(elapsed * visible / 3600.0),
                    visible_gpu_count=visible,
                    peak_allocated_bytes=allocated,
                    peak_reserved_bytes=reserved,
                    peak_rss_bytes=rss,
                    counters=dict(counters),
                )
            )

    def summary(self) -> dict[str, Any]:
        _sync()
        elapsed = time.perf_counter() - self.started
        visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
        return {
            "method": self.method,
            "protocol": self.protocol,
            "wall_clock_seconds": float(elapsed),
            "gpu_hours": float(elapsed * visible / 3600.0),
            "visible_gpu_count": visible,
            "peak_allocated_bytes": int(self._peak_allocated),
            "peak_reserved_bytes": int(self._peak_reserved),
            "peak_rss_bytes": int(self._peak_rss),
            "counters": dict(self.counters),
            "phases": [phase.to_dict() for phase in self.phases],
        }


def merge_cost_audits(*audits: dict[str, Any]) -> dict[str, Any]:
    """Merge process-level audit records without losing stage provenance."""
    phases: list[dict[str, Any]] = []
    counters: dict[str, int] = {}
    wall = 0.0
    gpu_hours = 0.0
    visible = 0
    peak_allocated = 0
    peak_reserved = 0
    peak_rss = 0
    methods: list[str] = []
    for audit in audits:
        if not audit:
            continue
        methods.append(str(audit.get("method", "unknown")))
        wall += float(audit.get("wall_clock_seconds", 0.0))
        gpu_hours += float(audit.get("gpu_hours", 0.0))
        visible = max(visible, int(audit.get("visible_gpu_count", 0)))
        peak_allocated = max(peak_allocated, int(audit.get("peak_allocated_bytes", 0)))
        peak_reserved = max(peak_reserved, int(audit.get("peak_reserved_bytes", 0)))
        peak_rss = max(peak_rss, int(audit.get("peak_rss_bytes", 0)))
        phases.extend(audit.get("phases", []))
        for key, value in audit.get("counters", {}).items():
            counters[key] = counters.get(key, 0) + int(value)
    return {
        "method": "+".join(methods),
        "wall_clock_seconds": wall,
        "gpu_hours": gpu_hours,
        "visible_gpu_count": visible,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_rss_bytes": peak_rss,
        "counters": counters,
        "phases": phases,
    }
