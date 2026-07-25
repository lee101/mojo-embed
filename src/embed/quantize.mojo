"""Int8 quantisation for embedding vectors.

Embeddings are the one place int8 is nearly free. A normalised embedding's
components cluster tightly around zero, so a symmetric per-vector scale keeps
recall within noise of float32 while cutting memory to a quarter - and memory
is what bounds vector search, not arithmetic.

Symmetric (zero-point-free) on purpose: an asymmetric scheme needs a per-pair
zero-point correction term in the dot product, which costs more than the range
it buys back on data that is already centred.
"""

from std.math import sqrt
from std.sys.info import simd_width_of

comptime F32_WIDTH = simd_width_of[DType.float32]()
comptime I8_WIDTH = simd_width_of[DType.int8]()

# int8 spans -127..127. -128 is excluded so negation is symmetric and a
# scale computed from the maximum magnitude never overflows on the low side.
comptime INT8_MAX: Float32 = 127.0


struct QuantizedVector(Movable):
    """One vector as int8 plus the scale that reconstructs it."""

    var data: List[Int8]
    var scale: Float32
    var norm: Float32

    def __init__(out self):
        self.data = List[Int8]()
        self.scale = 1.0
        self.norm = 0.0

    def __init__(out self, var data: List[Int8], scale: Float32, norm: Float32):
        self.data = data^
        self.scale = scale
        self.norm = norm

    def size(self) -> Int:
        return len(self.data)


def max_abs(values: Span[Float32, _]) -> Float32:
    """Largest magnitude, vectorised."""
    var n = len(values)
    var acc = SIMD[DType.float32, F32_WIDTH](0.0)
    var i = 0

    while i + F32_WIDTH <= n:
        var chunk = _load_f32(values, i)
        acc = max(acc, abs(chunk))
        i += F32_WIDTH

    var best = acc.reduce_max()
    while i < n:
        var v = abs(values[i])
        if v > best:
            best = v
        i += 1
    return best


def l2_norm(values: Span[Float32, _]) -> Float32:
    var n = len(values)
    var acc = SIMD[DType.float32, F32_WIDTH](0.0)
    var i = 0

    while i + F32_WIDTH <= n:
        var chunk = _load_f32(values, i)
        acc = acc + chunk * chunk
        i += F32_WIDTH

    var total = acc.reduce_add()
    while i < n:
        total += values[i] * values[i]
        i += 1
    return sqrt(total)


def quantize(values: Span[Float32, _]) -> QuantizedVector:
    """Symmetric per-vector int8 quantisation."""
    var n = len(values)
    var peak = max_abs(values)
    # An all-zero vector has no scale; 1.0 keeps the reconstruction exact.
    var scale = peak / INT8_MAX if peak > 0.0 else Float32(1.0)
    var inv = Float32(1.0) / scale

    var out = List[Int8](capacity=n)
    for i in range(n):
        var scaled = values[i] * inv
        # Round half away from zero, then clamp: a value at the boundary must
        # not wrap to the opposite sign.
        var rounded = Int(scaled + 0.5) if scaled >= 0.0 else Int(scaled - 0.5)
        if rounded > 127:
            rounded = 127
        elif rounded < -127:
            rounded = -127
        out.append(Int8(rounded))

    return QuantizedVector(out^, scale, l2_norm(values))


def dequantize(vector: QuantizedVector) -> List[Float32]:
    var out = List[Float32](capacity=vector.size())
    for i in range(vector.size()):
        out.append(Float32(Int(vector.data[i])) * vector.scale)
    return out^


def normalize(mut values: List[Float32]):
    """Scale to unit length, so a dot product is a cosine."""
    var norm = l2_norm(Span(values))
    if norm <= 0.0:
        return
    var inv = Float32(1.0) / norm
    for i in range(len(values)):
        values[i] = values[i] * inv


def _load_f32(values: Span[Float32, _], offset: Int) -> SIMD[DType.float32, F32_WIDTH]:
    var tail = values[offset:]
    return tail.unsafe_ptr().load[width=F32_WIDTH]()


def _load_i8(values: Span[Int8, _], offset: Int) -> SIMD[DType.int8, I8_WIDTH]:
    var tail = values[offset:]
    return tail.unsafe_ptr().load[width=I8_WIDTH]()
