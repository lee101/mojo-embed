"""Vector distance kernels.

Everything here is one loop over SIMD lanes. That is the whole trick: the
work is memory-bound, so the win comes from touching each cache line once and
keeping the accumulator in registers, not from clever mathematics.

The int8 path accumulates into int32. Two int8 values multiply to at most
16129, so a 4096-dimensional vector cannot overflow int32 - which means no
saturation checks in the inner loop.
"""

from std.math import sqrt
from std.sys.info import simd_width_of

from embed.quantize import I8_WIDTH, F32_WIDTH, QuantizedVector, _load_f32, _load_i8

# Unrolling by four independent accumulators hides FMA latency: one chain
# stalls on its own result, four interleave.
comptime UNROLL = 4


def dot_f32(a: Span[Float32, _], b: Span[Float32, _]) -> Float32:
    """Float32 dot product with four independent accumulators."""
    var n = min(len(a), len(b))
    var acc0 = SIMD[DType.float32, F32_WIDTH](0.0)
    var acc1 = SIMD[DType.float32, F32_WIDTH](0.0)
    var acc2 = SIMD[DType.float32, F32_WIDTH](0.0)
    var acc3 = SIMD[DType.float32, F32_WIDTH](0.0)

    var block = F32_WIDTH * UNROLL
    var i = 0
    while i + block <= n:
        acc0 = _load_f32(a, i).fma(_load_f32(b, i), acc0)
        acc1 = _load_f32(a, i + F32_WIDTH).fma(_load_f32(b, i + F32_WIDTH), acc1)
        acc2 = _load_f32(a, i + 2 * F32_WIDTH).fma(_load_f32(b, i + 2 * F32_WIDTH), acc2)
        acc3 = _load_f32(a, i + 3 * F32_WIDTH).fma(_load_f32(b, i + 3 * F32_WIDTH), acc3)
        i += block

    while i + F32_WIDTH <= n:
        acc0 = _load_f32(a, i).fma(_load_f32(b, i), acc0)
        i += F32_WIDTH

    var total = (acc0 + acc1 + acc2 + acc3).reduce_add()
    while i < n:
        total += a[i] * b[i]
        i += 1
    return total


def dot_i8(a: Span[Int8, _], b: Span[Int8, _]) -> Int32:
    """Int8 dot product accumulated in int32.

    32 lanes per step on AVX2. The cast to int32 before multiplying is what
    keeps the product exact; int8*int8 would wrap.
    """
    var n = min(len(a), len(b))
    var acc0 = SIMD[DType.int32, I8_WIDTH](0)
    var acc1 = SIMD[DType.int32, I8_WIDTH](0)

    var block = I8_WIDTH * 2
    var i = 0
    while i + block <= n:
        var a0 = _load_i8(a, i).cast[DType.int32]()
        var b0 = _load_i8(b, i).cast[DType.int32]()
        acc0 = acc0 + a0 * b0

        var a1 = _load_i8(a, i + I8_WIDTH).cast[DType.int32]()
        var b1 = _load_i8(b, i + I8_WIDTH).cast[DType.int32]()
        acc1 = acc1 + a1 * b1
        i += block

    while i + I8_WIDTH <= n:
        var av = _load_i8(a, i).cast[DType.int32]()
        var bv = _load_i8(b, i).cast[DType.int32]()
        acc0 = acc0 + av * bv
        i += I8_WIDTH

    var total = (acc0 + acc1).reduce_add()
    while i < n:
        total += Int32(Int(a[i])) * Int32(Int(b[i]))
        i += 1
    return total


def cosine_f32(a: Span[Float32, _], b: Span[Float32, _]) -> Float32:
    """Cosine similarity. Use `dot_f32` directly if both are already unit."""
    var dot = Float32(0.0)
    var norm_a = Float32(0.0)
    var norm_b = Float32(0.0)
    var n = min(len(a), len(b))

    var acc_d = SIMD[DType.float32, F32_WIDTH](0.0)
    var acc_a = SIMD[DType.float32, F32_WIDTH](0.0)
    var acc_b = SIMD[DType.float32, F32_WIDTH](0.0)

    var i = 0
    while i + F32_WIDTH <= n:
        var av = _load_f32(a, i)
        var bv = _load_f32(b, i)
        acc_d = av.fma(bv, acc_d)
        acc_a = av.fma(av, acc_a)
        acc_b = bv.fma(bv, acc_b)
        i += F32_WIDTH

    dot = acc_d.reduce_add()
    norm_a = acc_a.reduce_add()
    norm_b = acc_b.reduce_add()

    while i < n:
        dot += a[i] * b[i]
        norm_a += a[i] * a[i]
        norm_b += b[i] * b[i]
        i += 1

    var denom = sqrt(norm_a) * sqrt(norm_b)
    return dot / denom if denom > 0.0 else Float32(0.0)


def cosine_quantized(a: QuantizedVector, b: QuantizedVector) -> Float32:
    """Cosine between two quantised vectors.

    The scales factor out of the dot product, so this is one int8 pass plus
    two multiplies - the reason int8 storage costs nothing at query time.
    """
    var raw = Float32(Int(dot_i8(Span(a.data), Span(b.data))))
    var dot = raw * a.scale * b.scale
    var denom = a.norm * b.norm
    return dot / denom if denom > 0.0 else Float32(0.0)


def euclidean_f32(a: Span[Float32, _], b: Span[Float32, _]) -> Float32:
    var n = min(len(a), len(b))
    var acc = SIMD[DType.float32, F32_WIDTH](0.0)
    var i = 0

    while i + F32_WIDTH <= n:
        var diff = _load_f32(a, i) - _load_f32(b, i)
        acc = diff.fma(diff, acc)
        i += F32_WIDTH

    var total = acc.reduce_add()
    while i < n:
        var d = a[i] - b[i]
        total += d * d
        i += 1
    return sqrt(total)
