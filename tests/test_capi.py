"""Correctness checks for the zero-copy C ABI."""

from __future__ import annotations

import ctypes
from pathlib import Path

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


def quantize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    output = np.empty(values.size, dtype=np.int8)
    norm = np.empty(1, dtype=np.float32)
    scale = LIB.embed_quantize(
        address(values), address(output), address(norm), values.size
    )
    return output, float(scale), float(norm[0])


def search_case(count: int, dim: int = 35) -> None:
    rng = np.random.default_rng(count)
    k = 7
    data = rng.integers(-127, 128, size=(count, dim), dtype=np.int8)
    factors = rng.uniform(0.001, 0.02, size=count).astype(np.float32)
    ids = np.arange(count, dtype=np.int64)
    query = rng.integers(-127, 128, size=dim, dtype=np.int8)
    query_factor = np.float32(0.007)
    scratch = np.empty(count, dtype=np.float32)
    output_ids = np.empty(k, dtype=np.int64)
    output_scores = np.empty(k, dtype=np.float32)

    LIB.embed_search_i8(
        address(data),
        address(factors),
        address(ids),
        address(query),
        address(scratch),
        address(output_ids),
        address(output_scores),
        count,
        dim,
        k,
        query_factor,
    )
    raw = np.einsum(
        "ij,j->i", data, query, dtype=np.int32, optimize=False
    )
    expected = raw.astype(np.float32) * query_factor * factors
    expected_ids = np.argsort(expected)[-k:][::-1]
    assert np.array_equal(output_ids, expected_ids)
    assert np.allclose(output_scores, expected[expected_ids], rtol=1e-6)


def main() -> None:
    rng = np.random.default_rng(42)
    first = rng.normal(size=131).astype(np.float32)
    second = rng.normal(size=131).astype(np.float32)
    first_i8 = rng.integers(-127, 128, size=131, dtype=np.int8)
    second_i8 = rng.integers(-127, 128, size=131, dtype=np.int8)

    assert np.isclose(
        LIB.embed_dot_f32(address(first), address(second), first.size),
        np.dot(first, second),
        rtol=2e-6,
        atol=2e-6,
    )
    assert LIB.embed_dot_i8(
        address(first_i8), address(second_i8), first_i8.size
    ) == int(
        np.dot(first_i8.astype(np.int32), second_i8.astype(np.int32))
    )
    expected_cosine = np.dot(first, second) / (
        np.linalg.norm(first) * np.linalg.norm(second)
    )
    assert np.isclose(
        LIB.embed_cosine_f32(address(first), address(second), first.size),
        expected_cosine,
        rtol=2e-6,
        atol=2e-6,
    )
    assert np.isclose(
        LIB.embed_euclidean_f32(address(first), address(second), first.size),
        np.linalg.norm(first - second),
        rtol=2e-6,
    )

    quantized, scale, norm = quantize(first)
    expected_quantized = np.clip(
        np.where(first >= 0, first / scale + 0.5, first / scale - 0.5),
        -127,
        127,
    ).astype(np.int8)
    assert np.array_equal(quantized, expected_quantized)
    assert np.isclose(norm, np.linalg.norm(first), rtol=2e-6)

    normalized = first.copy()
    old_norm = LIB.embed_normalize(address(normalized), normalized.size)
    assert np.isclose(old_norm, np.linalg.norm(first), rtol=2e-6)
    assert np.allclose(normalized, first / np.linalg.norm(first), rtol=2e-6)

    search_case(130_208, 768)
    search_case(130_209, 768)
    print("C ABI checks passed")


if __name__ == "__main__":
    main()
