"""Zero-copy C ABI for benchmarking and embedding in Python extensions."""

from max.algorithm import parallelize
from std.math import sqrt
from std.sys.info import num_performance_cores, simd_width_of as simdwidthof

comptime F32Ptr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime I8Ptr = UnsafePointer[Int8, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime W = simdwidthof[DType.float32]()
comptime I8W = simdwidthof[DType.int8]()
comptime PARALLEL_WORK_THRESHOLD = 100_000_000


def f32p(addr: Int) -> F32Ptr:
    return F32Ptr(unsafe_from_address=addr)


def i8p(addr: Int) -> I8Ptr:
    return I8Ptr(unsafe_from_address=addr)


def i64p(addr: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=addr)


def dot_f32_ptr(a: F32Ptr, b: F32Ptr, n: Int) -> Float32:
    var acc0 = SIMD[DType.float32, W](0.0)
    var acc1 = SIMD[DType.float32, W](0.0)
    var i = 0
    while i + 2 * W <= n:
        acc0 = a.load[width=W](i).fma(b.load[width=W](i), acc0)
        acc1 = a.load[width=W](i + W).fma(
            b.load[width=W](i + W), acc1
        )
        i += 2 * W
    while i + W <= n:
        acc0 = a.load[width=W](i).fma(b.load[width=W](i), acc0)
        i += W
    var total = (acc0 + acc1).reduce_add()
    while i < n:
        total += a[i] * b[i]
        i += 1
    return total


def dot_i8_ptr(a: I8Ptr, b: I8Ptr, n: Int) -> Int32:
    var acc0 = SIMD[DType.int32, I8W](0)
    var acc1 = SIMD[DType.int32, I8W](0)
    var i = 0
    while i + 2 * I8W <= n:
        var av0 = a.load[width=I8W](i).cast[DType.int32]()
        var bv0 = b.load[width=I8W](i).cast[DType.int32]()
        var av1 = a.load[width=I8W](i + I8W).cast[DType.int32]()
        var bv1 = b.load[width=I8W](i + I8W).cast[DType.int32]()
        acc0 += av0 * bv0
        acc1 += av1 * bv1
        i += 2 * I8W
    while i + I8W <= n:
        var av = a.load[width=I8W](i).cast[DType.int32]()
        var bv = b.load[width=I8W](i).cast[DType.int32]()
        acc0 += av * bv
        i += I8W
    var total = (acc0 + acc1).reduce_add()
    while i < n:
        total += Int32(Int(a[i])) * Int32(Int(b[i]))
        i += 1
    return total


def norm_f32_ptr(values: F32Ptr, n: Int) -> Float32:
    var acc = SIMD[DType.float32, W](0.0)
    var i = 0
    while i + W <= n:
        var chunk = values.load[width=W](i)
        acc = chunk.fma(chunk, acc)
        i += W
    var total = acc.reduce_add()
    while i < n:
        total += values[i] * values[i]
        i += 1
    return sqrt(total)


def max_abs_ptr(values: F32Ptr, n: Int) -> Float32:
    var acc = SIMD[DType.float32, W](0.0)
    var i = 0
    while i + W <= n:
        acc = max(acc, abs(values.load[width=W](i)))
        i += W
    var best = acc.reduce_max()
    while i < n:
        best = max(best, abs(values[i]))
        i += 1
    return best


@export("embed_dot_f32")
def embed_dot_f32(a_addr: Int, b_addr: Int, n: Int) abi("C") -> Float32:
    if n <= 0 or a_addr == 0 or b_addr == 0:
        return 0.0
    return dot_f32_ptr(f32p(a_addr), f32p(b_addr), n)


@export("embed_dot_i8")
def embed_dot_i8(a_addr: Int, b_addr: Int, n: Int) abi("C") -> Int32:
    if n <= 0 or a_addr == 0 or b_addr == 0:
        return 0
    return dot_i8_ptr(i8p(a_addr), i8p(b_addr), n)


@export("embed_cosine_f32")
def embed_cosine_f32(a_addr: Int, b_addr: Int, n: Int) abi("C") -> Float32:
    if n <= 0 or a_addr == 0 or b_addr == 0:
        return 0.0
    var a = f32p(a_addr)
    var b = f32p(b_addr)
    var acc_dot = SIMD[DType.float32, W](0.0)
    var acc_a = SIMD[DType.float32, W](0.0)
    var acc_b = SIMD[DType.float32, W](0.0)
    var i = 0
    while i + W <= n:
        var av = a.load[width=W](i)
        var bv = b.load[width=W](i)
        acc_dot = av.fma(bv, acc_dot)
        acc_a = av.fma(av, acc_a)
        acc_b = bv.fma(bv, acc_b)
        i += W
    var dot = acc_dot.reduce_add()
    var norm_a = acc_a.reduce_add()
    var norm_b = acc_b.reduce_add()
    while i < n:
        dot += a[i] * b[i]
        norm_a += a[i] * a[i]
        norm_b += b[i] * b[i]
        i += 1
    var denom = sqrt(norm_a) * sqrt(norm_b)
    return dot / denom if denom > 0.0 else Float32(0.0)


@export("embed_euclidean_f32")
def embed_euclidean_f32(
    a_addr: Int, b_addr: Int, n: Int
) abi("C") -> Float32:
    if n <= 0 or a_addr == 0 or b_addr == 0:
        return 0.0
    var a = f32p(a_addr)
    var b = f32p(b_addr)
    var acc = SIMD[DType.float32, W](0.0)
    var i = 0
    while i + W <= n:
        var delta = a.load[width=W](i) - b.load[width=W](i)
        acc = delta.fma(delta, acc)
        i += W
    var total = acc.reduce_add()
    while i < n:
        var delta = a[i] - b[i]
        total += delta * delta
        i += 1
    return sqrt(total)


@export("embed_quantize")
def embed_quantize(
    values_addr: Int,
    output_addr: Int,
    norm_addr: Int,
    n: Int,
) abi("C") -> Float32:
    if n <= 0 or values_addr == 0 or output_addr == 0 or norm_addr == 0:
        return 1.0
    var values = f32p(values_addr)
    var output = i8p(output_addr)
    var norm_output = f32p(norm_addr)
    var peak = max_abs_ptr(values, n)
    var scale = peak / 127.0 if peak > 0.0 else Float32(1.0)
    var inv = Float32(1.0) / scale
    var i = 0
    while i + W <= n:
        var scaled = values.load[width=W](i) * inv
        var rounded = scaled.ge(0.0).select(scaled + 0.5, scaled - 0.5)
        var integers = min(
            max(rounded.cast[DType.int32](), SIMD[DType.int32, W](-127)),
            SIMD[DType.int32, W](127),
        )
        output.store(i, integers.cast[DType.int8]())
        i += W
    while i < n:
        var scaled = values[i] * inv
        var rounded = Int(scaled + 0.5) if scaled >= 0.0 else Int(scaled - 0.5)
        output[i] = Int8(min(127, max(-127, rounded)))
        i += 1
    norm_output[0] = norm_f32_ptr(values, n)
    return scale


@export("embed_normalize")
def embed_normalize(values_addr: Int, n: Int) abi("C") -> Float32:
    if n <= 0 or values_addr == 0:
        return 0.0
    var values = f32p(values_addr)
    var norm = norm_f32_ptr(values, n)
    if norm <= 0.0:
        return norm
    var inv = Float32(1.0) / norm
    var i = 0
    while i + W <= n:
        values.store(i, values.load[width=W](i) * inv)
        i += W
    while i < n:
        values[i] *= inv
        i += 1
    return norm


def score_rows(
    data: I8Ptr,
    factors: F32Ptr,
    query: I8Ptr,
    query_factor: Float32,
    scores: F32Ptr,
    count: Int,
    dim: Int,
    allow_parallel: Bool,
):
    var workers = 1
    if allow_parallel and count * dim >= PARALLEL_WORK_THRESHOLD:
        workers = min(
            num_performance_cores(), max(1, count // 8_000)
        )

    def scan(worker: Int) { imm data, imm factors, imm query, imm query_factor, imm scores, imm count, imm dim, imm workers }:
        var start = worker * count // workers
        var end = (worker + 1) * count // workers
        for row in range(start, end):
            scores[row] = (
                Float32(Int(dot_i8_ptr(query, data + row * dim, dim)))
                * query_factor
                * factors[row]
            )

    if (
        allow_parallel
        and count * dim >= PARALLEL_WORK_THRESHOLD
        and workers > 1
    ):
        parallelize(scan, workers, workers)
    else:
        scan(0)


def search_i8_impl(
    data_addr: Int,
    factors_addr: Int,
    ids_addr: Int,
    query_addr: Int,
    scores_addr: Int,
    output_ids_addr: Int,
    output_scores_addr: Int,
    count: Int,
    dim: Int,
    k: Int,
    query_factor: Float32,
    allow_parallel: Bool,
):
    if (
        count <= 0
        or dim <= 0
        or k <= 0
        or data_addr == 0
        or factors_addr == 0
        or ids_addr == 0
        or query_addr == 0
        or scores_addr == 0
        or output_ids_addr == 0
        or output_scores_addr == 0
    ):
        return
    var data = i8p(data_addr)
    var factors = f32p(factors_addr)
    var ids = i64p(ids_addr)
    var query = i8p(query_addr)
    var scores = f32p(scores_addr)
    var output_ids = i64p(output_ids_addr)
    var output_scores = f32p(output_scores_addr)
    var wanted = min(k, count)
    score_rows(
        data,
        factors,
        query,
        query_factor,
        scores,
        count,
        dim,
        allow_parallel,
    )

    for slot in range(wanted):
        output_ids[slot] = -1
        output_scores[slot] = -1.0e30
    for row in range(count):
        var value = scores[row]
        if value <= output_scores[wanted - 1]:
            continue
        var slot = wanted - 1
        while slot > 0 and value > output_scores[slot - 1]:
            output_scores[slot] = output_scores[slot - 1]
            output_ids[slot] = output_ids[slot - 1]
            slot -= 1
        output_scores[slot] = value
        output_ids[slot] = ids[row]


@export("embed_search_i8")
def embed_search_i8(
    data_addr: Int,
    factors_addr: Int,
    ids_addr: Int,
    query_addr: Int,
    scores_addr: Int,
    output_ids_addr: Int,
    output_scores_addr: Int,
    count: Int,
    dim: Int,
    k: Int,
    query_factor: Float32,
) abi("C"):
    search_i8_impl(
        data_addr,
        factors_addr,
        ids_addr,
        query_addr,
        scores_addr,
        output_ids_addr,
        output_scores_addr,
        count,
        dim,
        k,
        query_factor,
        True,
    )


@export("embed_search_i8_serial")
def embed_search_i8_serial(
    data_addr: Int,
    factors_addr: Int,
    ids_addr: Int,
    query_addr: Int,
    scores_addr: Int,
    output_ids_addr: Int,
    output_scores_addr: Int,
    count: Int,
    dim: Int,
    k: Int,
    query_factor: Float32,
) abi("C"):
    search_i8_impl(
        data_addr,
        factors_addr,
        ids_addr,
        query_addr,
        scores_addr,
        output_ids_addr,
        output_scores_addr,
        count,
        dim,
        k,
        query_factor,
        False,
    )
