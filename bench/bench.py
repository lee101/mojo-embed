"""Benchmark mojo-embed against NumPy on identical, preallocated inputs."""

from __future__ import annotations

import ctypes
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LIB = ctypes.CDLL(str(ROOT / "build" / "libmojo_embed.so"))
C_INT = ctypes.c_ssize_t


def address(array: np.ndarray) -> int:
    assert array.flags.c_contiguous
    return int(array.ctypes.data)


LIB.embed_dot_f32.argtypes = [C_INT, C_INT, C_INT]
LIB.embed_dot_f32.restype = ctypes.c_float
LIB.embed_dot_i8.argtypes = [C_INT, C_INT, C_INT]
LIB.embed_dot_i8.restype = ctypes.c_int32
LIB.embed_cosine_f32.argtypes = [C_INT, C_INT, C_INT]
LIB.embed_cosine_f32.restype = ctypes.c_float
LIB.embed_euclidean_f32.argtypes = [C_INT, C_INT, C_INT]
LIB.embed_euclidean_f32.restype = ctypes.c_float
LIB.embed_quantize.argtypes = [C_INT, C_INT, C_INT, C_INT]
LIB.embed_quantize.restype = ctypes.c_float
LIB.embed_normalize.argtypes = [C_INT, C_INT]
LIB.embed_normalize.restype = ctypes.c_float
LIB.embed_search_i8.argtypes = [
    C_INT,
    C_INT,
    C_INT,
    C_INT,
    C_INT,
    C_INT,
    C_INT,
    C_INT,
    C_INT,
    C_INT,
    ctypes.c_float,
]
LIB.embed_search_i8.restype = None


@dataclass
class Case:
    operation: str
    size: str
    mojo: Callable[[], object]
    numpy: Callable[[], object]
    iterations: int


def median_seconds(
    function: Callable[[], object], iterations: int, repeats: int = 7
) -> float:
    for _ in range(2):
        function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            function()
        samples.append((time.perf_counter_ns() - start) * 1e-9 / iterations)
    return statistics.median(samples)


def format_time(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:.0f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    return f"{seconds * 1e3:.2f} ms"


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main() -> None:
    rng = np.random.default_rng(2026)
    cases: list[Case] = []

    dim = 771
    first = rng.normal(size=dim).astype(np.float32)
    second = rng.normal(size=dim).astype(np.float32)
    first_i8 = rng.integers(-127, 128, size=dim, dtype=np.int8)
    second_i8 = rng.integers(-127, 128, size=dim, dtype=np.int8)
    first_addr = address(first)
    second_addr = address(second)
    first_i8_addr = address(first_i8)
    second_i8_addr = address(second_i8)

    mojo_dot = lambda: LIB.embed_dot_f32(first_addr, second_addr, dim)
    numpy_dot = lambda: np.dot(first, second)
    assert np.isclose(mojo_dot(), numpy_dot(), rtol=2e-6, atol=2e-6)
    cases.append(Case("Float32 dot", "771 dims", mojo_dot, numpy_dot, 10_000))

    mojo_i8_dot = lambda: LIB.embed_dot_i8(
        first_i8_addr, second_i8_addr, dim
    )

    def numpy_i8_dot() -> np.int32:
        return np.einsum(
            "i,i->", first_i8, second_i8, dtype=np.int32, optimize=False
        )

    assert mojo_i8_dot() == numpy_i8_dot()
    cases.append(
        Case("Int8 dot", "771 dims", mojo_i8_dot, numpy_i8_dot, 10_000)
    )

    mojo_cosine = lambda: LIB.embed_cosine_f32(
        first_addr, second_addr, dim
    )

    def numpy_cosine() -> np.float32:
        return np.dot(first, second) / (
            np.linalg.norm(first) * np.linalg.norm(second)
        )

    assert np.isclose(mojo_cosine(), numpy_cosine(), rtol=2e-6, atol=2e-6)
    cases.append(
        Case("Cosine similarity", "771 dims", mojo_cosine, numpy_cosine, 5_000)
    )

    mojo_euclidean = lambda: LIB.embed_euclidean_f32(
        first_addr, second_addr, dim
    )
    numpy_euclidean = lambda: np.linalg.norm(first - second)
    assert np.isclose(mojo_euclidean(), numpy_euclidean(), rtol=2e-6)
    cases.append(
        Case(
            "Euclidean distance",
            "771 dims",
            mojo_euclidean,
            numpy_euclidean,
            5_000,
        )
    )

    large_n = 1_000_003
    large = rng.normal(size=large_n).astype(np.float32)
    mojo_quantized = np.empty(large_n, dtype=np.int8)
    numpy_quantized = np.empty(large_n, dtype=np.int8)
    mojo_norm = np.empty(1, dtype=np.float32)
    numpy_scratch = np.empty(large_n, dtype=np.float32)
    large_addr = address(large)
    mojo_quantized_addr = address(mojo_quantized)
    mojo_norm_addr = address(mojo_norm)

    def mojo_quantize() -> float:
        return LIB.embed_quantize(
            large_addr,
            mojo_quantized_addr,
            mojo_norm_addr,
            large_n,
        )

    def numpy_quantize() -> tuple[np.float32, np.float32]:
        peak = np.max(np.abs(large))
        scale = peak / np.float32(127.0) if peak > 0 else np.float32(1.0)
        np.multiply(large, np.float32(1.0) / scale, out=numpy_scratch)
        np.add(
            numpy_scratch,
            np.where(large >= 0, np.float32(0.5), np.float32(-0.5)),
            out=numpy_scratch,
        )
        np.clip(numpy_scratch, -127, 127, out=numpy_scratch)
        np.copyto(numpy_quantized, numpy_scratch, casting="unsafe")
        return scale, np.linalg.norm(large)

    mojo_scale = mojo_quantize()
    numpy_scale, numpy_norm = numpy_quantize()
    assert np.isclose(mojo_scale, numpy_scale, rtol=1e-6)
    assert np.isclose(mojo_norm[0], numpy_norm, rtol=2e-5)
    assert np.array_equal(mojo_quantized, numpy_quantized)
    cases.append(
        Case(
            "Symmetric quantize",
            "1,000,003 dims",
            mojo_quantize,
            numpy_quantize,
            20,
        )
    )

    mojo_normalized = large.copy()
    numpy_normalized = large.copy()
    mojo_normalized_addr = address(mojo_normalized)
    mojo_normalize = lambda: LIB.embed_normalize(
        mojo_normalized_addr, large_n
    )

    def numpy_normalize() -> np.ndarray:
        norm = np.linalg.norm(numpy_normalized)
        np.multiply(numpy_normalized, np.float32(1.0) / norm, out=numpy_normalized)
        return numpy_normalized

    mojo_normalize()
    numpy_normalize()
    assert np.allclose(mojo_normalized, numpy_normalized, rtol=2e-5)
    cases.append(
        Case(
            "L2 normalize",
            "1,000,003 dims",
            mojo_normalize,
            numpy_normalize,
            30,
        )
    )

    count, search_dim, k = 200_000, 768, 10
    data = rng.integers(
        -127, 128, size=(count, search_dim), dtype=np.int8
    )
    factors = rng.uniform(0.001, 0.02, size=count).astype(np.float32)
    ids = np.arange(count, dtype=np.int64)
    query = rng.integers(-127, 128, size=search_dim, dtype=np.int8)
    query_factor = np.float32(0.007)

    def add_search_case(case_count: int, label: str) -> None:
        scratch = np.empty(case_count, dtype=np.float32)
        output_ids = np.empty(k, dtype=np.int64)
        output_scores = np.empty(k, dtype=np.float32)
        raw = np.empty(case_count, dtype=np.int32)
        numpy_scores = np.empty(case_count, dtype=np.float32)
        data_addr = address(data)
        factors_addr = address(factors)
        ids_addr = address(ids)
        query_addr = address(query)
        scratch_addr = address(scratch)
        output_ids_addr = address(output_ids)
        output_scores_addr = address(output_scores)
        mojo_buffers = (scratch, output_ids, output_scores)

        def mojo_search() -> np.ndarray:
            _ = mojo_buffers
            LIB.embed_search_i8(
                data_addr,
                factors_addr,
                ids_addr,
                query_addr,
                scratch_addr,
                output_ids_addr,
                output_scores_addr,
                case_count,
                search_dim,
                k,
                query_factor,
            )
            return output_ids

        def numpy_search() -> np.ndarray:
            np.einsum(
                "ij,j->i",
                data[:case_count],
                query,
                dtype=np.int32,
                optimize=False,
                out=raw,
            )
            np.multiply(raw, query_factor, out=numpy_scores)
            np.multiply(
                numpy_scores, factors[:case_count], out=numpy_scores
            )
            selected = np.argpartition(numpy_scores, -k)[-k:]
            return selected[np.argsort(numpy_scores[selected])[::-1]]

        assert np.array_equal(mojo_search(), numpy_search())
        cases.append(
            Case(
                f"Exact int8 top-10 ({label})",
                f"{case_count:,} x {search_dim}",
                mojo_search,
                numpy_search,
                1,
            )
        )

    add_search_case(10_000, "serial")
    add_search_case(200_000, "parallel")

    print(f"Machine: {cpu_name()}")
    print(
        f"Python {platform.python_version()}; NumPy {np.__version__}; "
        "warmed up; median of 7 repeats"
    )
    print()
    print("| Operation | Input size | mojo-embed | NumPy | Speedup |")
    print("|---|---:|---:|---:|---:|")
    for case in cases:
        mojo_time = median_seconds(case.mojo, case.iterations)
        numpy_time = median_seconds(case.numpy, case.iterations)
        speedup = numpy_time / mojo_time
        print(
            f"| {case.operation} | {case.size} | "
            f"{format_time(mojo_time)} | {format_time(numpy_time)} | "
            f"{speedup:.2f}x |"
        )


if __name__ == "__main__":
    main()
