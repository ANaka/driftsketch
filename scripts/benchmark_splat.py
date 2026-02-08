#!/usr/bin/env python3
"""Benchmark splat renderer vs pydiffvg.

Compares forward and backward pass performance at various canvas sizes and batch sizes.
Reports wall-clock times, speedup ratios, and peak GPU memory.

Usage:
    python scripts/benchmark_splat.py [--device cuda]
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from driftsketch.splat_rendering import splat_render_beziers


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

class Timer:
    """Context manager for timing code blocks, GPU-aware."""

    def __init__(self, device: str):
        self.device = device
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        if self.device == "cuda":
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.device == "cuda":
            self.end_event.record()
            torch.cuda.synchronize()
            self.elapsed_ms = self.start_event.elapsed_time(self.end_event)
        else:
            self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0


def get_peak_memory_mb(device: str) -> float:
    """Return peak GPU memory in MB, or 0 for CPU."""
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_peak_memory(device: str) -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# pydiffvg rendering (sequential, matches existing API)
# ---------------------------------------------------------------------------

def render_pydiffvg_batch(beziers: torch.Tensor, canvas_size: int, stroke_width: float = 2.0) -> torch.Tensor:
    """Render a batch with pydiffvg (sequential loop)."""
    from driftsketch.rendering import render_beziers_differentiable

    images = []
    for i in range(beziers.shape[0]):
        img = render_beziers_differentiable(beziers[i], canvas_size=canvas_size, stroke_width=stroke_width)
        images.append(img)
    return torch.stack(images)


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def benchmark_forward(
    beziers: torch.Tensor,
    canvas_size: int,
    device: str,
    warmup: int = 3,
    repeats: int = 10,
) -> dict:
    """Benchmark forward pass for both renderers."""
    results = {}

    # --- Splat ---
    b = beziers.to(device)
    for _ in range(warmup):
        splat_render_beziers(b, canvas_size=canvas_size)
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        with Timer(device) as t:
            splat_render_beziers(b, canvas_size=canvas_size)
        times.append(t.elapsed_ms)
    results["splat_fwd_ms"] = sum(times) / len(times)

    # --- pydiffvg ---
    try:
        b_cpu = beziers.cpu()  # pydiffvg works on CPU
        for _ in range(warmup):
            render_pydiffvg_batch(b_cpu, canvas_size=canvas_size)

        times = []
        for _ in range(repeats):
            with Timer("cpu") as t:
                render_pydiffvg_batch(b_cpu, canvas_size=canvas_size)
            times.append(t.elapsed_ms)
        results["pydiffvg_fwd_ms"] = sum(times) / len(times)
    except Exception as e:
        results["pydiffvg_fwd_ms"] = None
        results["pydiffvg_error"] = str(e)

    return results


def benchmark_backward(
    beziers: torch.Tensor,
    canvas_size: int,
    device: str,
    warmup: int = 3,
    repeats: int = 10,
) -> dict:
    """Benchmark backward pass for both renderers."""
    results = {}

    # --- Splat ---
    b = beziers.to(device).detach().requires_grad_(True)

    # Warmup
    for _ in range(warmup):
        out = splat_render_beziers(b, canvas_size=canvas_size)
        loss = out.sum()
        loss.backward()
        b.grad = None
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        out = splat_render_beziers(b, canvas_size=canvas_size)
        loss = out.sum()
        if device == "cuda":
            torch.cuda.synchronize()
        with Timer(device) as t:
            loss.backward()
        times.append(t.elapsed_ms)
        b.grad = None
    results["splat_bwd_ms"] = sum(times) / len(times)

    # --- pydiffvg ---
    try:
        b_cpu = beziers.cpu().detach().requires_grad_(True)

        for _ in range(warmup):
            out = render_pydiffvg_batch(b_cpu, canvas_size=canvas_size)
            loss = out.sum()
            loss.backward()
            b_cpu.grad = None

        times = []
        for _ in range(repeats):
            out = render_pydiffvg_batch(b_cpu, canvas_size=canvas_size)
            loss = out.sum()
            with Timer("cpu") as t:
                loss.backward()
            times.append(t.elapsed_ms)
            b_cpu.grad = None
        results["pydiffvg_bwd_ms"] = sum(times) / len(times)
    except Exception as e:
        results["pydiffvg_bwd_ms"] = None
        results["pydiffvg_error"] = str(e)

    return results


def benchmark_memory(
    beziers: torch.Tensor,
    canvas_size: int,
    device: str,
) -> dict:
    """Measure peak GPU memory for a forward+backward pass."""
    results = {}

    if device != "cuda":
        results["splat_peak_mb"] = 0.0
        results["pydiffvg_peak_mb"] = 0.0
        return results

    # --- Splat ---
    gc.collect()
    torch.cuda.empty_cache()
    reset_peak_memory(device)

    b = beziers.to(device).detach().requires_grad_(True)
    out = splat_render_beziers(b, canvas_size=canvas_size)
    loss = out.sum()
    loss.backward()
    results["splat_peak_mb"] = get_peak_memory_mb(device)
    del out, loss, b

    # --- pydiffvg ---
    try:
        gc.collect()
        torch.cuda.empty_cache()
        reset_peak_memory(device)

        b_cpu = beziers.cpu().detach().requires_grad_(True)
        out = render_pydiffvg_batch(b_cpu, canvas_size=canvas_size)
        loss = out.sum()
        loss.backward()
        results["pydiffvg_peak_mb"] = get_peak_memory_mb(device)
        del out, loss, b_cpu
    except Exception:
        results["pydiffvg_peak_mb"] = None

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def format_speedup(splat_ms: float, pydiffvg_ms: float | None) -> str:
    if pydiffvg_ms is None:
        return "N/A"
    ratio = pydiffvg_ms / splat_ms if splat_ms > 0 else float("inf")
    return f"{ratio:.1f}x"


def format_ms(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark splat renderer vs pydiffvg")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=10, help="Timing iterations")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"=== Splat Renderer Benchmark ===")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    canvas_sizes = [128, 224]
    batch_sizes = [1, 16, 32, 64]

    # Header
    header = f"{'canvas':>8} {'batch':>6} | {'splat fwd':>10} {'pydiffvg fwd':>13} {'fwd speedup':>12} | {'splat bwd':>10} {'pydiffvg bwd':>13} {'bwd speedup':>12}"
    print(header)
    print("-" * len(header))

    for canvas_size in canvas_sizes:
        for batch_size in batch_sizes:
            torch.manual_seed(42)
            beziers = torch.randn(batch_size, 32, 4, 2) * 0.5

            # Forward
            fwd = benchmark_forward(beziers, canvas_size, device, warmup=args.warmup, repeats=args.repeats)

            # Backward
            bwd = benchmark_backward(beziers, canvas_size, device, warmup=args.warmup, repeats=args.repeats)

            fwd_speedup = format_speedup(fwd["splat_fwd_ms"], fwd.get("pydiffvg_fwd_ms"))
            bwd_speedup = format_speedup(bwd["splat_bwd_ms"], bwd.get("pydiffvg_bwd_ms"))

            row = (
                f"{canvas_size:>8} {batch_size:>6} | "
                f"{format_ms(fwd['splat_fwd_ms']):>10} {format_ms(fwd.get('pydiffvg_fwd_ms')):>13} {fwd_speedup:>12} | "
                f"{format_ms(bwd['splat_bwd_ms']):>10} {format_ms(bwd.get('pydiffvg_bwd_ms')):>13} {bwd_speedup:>12}"
            )
            print(row)

            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    print()
    print("Times are in milliseconds (ms), averaged over", args.repeats, "runs.")

    # Memory benchmark at largest config
    if device == "cuda":
        print("\n=== Peak GPU Memory (forward + backward) ===")
        for canvas_size in canvas_sizes:
            for batch_size in [16, 64]:
                torch.manual_seed(42)
                beziers = torch.randn(batch_size, 32, 4, 2) * 0.5
                mem = benchmark_memory(beziers, canvas_size, device)
                print(
                    f"  canvas={canvas_size}, batch={batch_size}: "
                    f"splat={mem['splat_peak_mb']:.0f} MB"
                    + (f", pydiffvg={mem['pydiffvg_peak_mb']:.0f} MB" if mem.get("pydiffvg_peak_mb") is not None else "")
                )
                gc.collect()
                torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
