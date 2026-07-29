from std.time import perf_counter_ns

from embed.distance import dot_f32, dot_i8
from embed.index import Index
from embed.quantize import quantize


def scalar_dot(a: Span[Float32, _], b: Span[Float32, _]) -> Float32:
    var total = Float32(0.0)
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


def main() raises:
    comptime DIM = 768
    comptime N = 100000
    comptime REPS = 200

    print("mojo-embed benchmark")
    print("dim", DIM, "vectors", N)
    print("")

    var a = List[Float32](capacity=DIM)
    var b = List[Float32](capacity=DIM)
    for i in range(DIM):
        a.append(Float32((i * 37) % 101) / 50.0 - 1.0)
        b.append(Float32((i * 53) % 89) / 44.0 - 1.0)

    # scalar baseline
    var t0 = perf_counter_ns()
    var sink = Float32(0.0)
    for _ in range(REPS * 50):
        sink += scalar_dot(Span(a), Span(b))
    var t1 = perf_counter_ns()
    var scalar_ns = (t1 - t0) / (REPS * 50)

    # simd f32
    t0 = perf_counter_ns()
    for _ in range(REPS * 50):
        sink += dot_f32(Span(a), Span(b))
    t1 = perf_counter_ns()
    var simd_ns = (t1 - t0) / (REPS * 50)

    # simd int8
    var qa = quantize(Span(a))
    var qb = quantize(Span(b))
    t0 = perf_counter_ns()
    var isink = Int32(0)
    for _ in range(REPS * 50):
        isink += dot_i8(Span(qa.data), Span(qb.data))
    t1 = perf_counter_ns()
    var int8_ns = (t1 - t0) / (REPS * 50)

    print("dot product, 768 dims")
    print("  scalar f32 ", scalar_ns, "ns")
    print("  simd   f32 ", simd_ns, "ns  speedup", scalar_ns / simd_ns, "x")
    print("  simd   int8", int8_ns, "ns  speedup", scalar_ns / int8_ns, "x")
    print("")

    # index build + search
    var index = Index(DIM)
    index.reserve(N)
    t0 = perf_counter_ns()
    for row in range(N):
        var v = List[Float32](capacity=DIM)
        for d in range(DIM):
            v.append(Float32((row * 7 + d * 13) % 97) / 48.0 - 1.0)
        index.add(row, Span(v))
    t1 = perf_counter_ns()
    print("index build")
    print("  ", N, "vectors in", (t1 - t0) / 1000000, "ms")
    print("   memory", index.memory_bytes() / 1048576, "MB")
    print("   float32 would be", (N * DIM * 4) / 1048576, "MB")
    print("")

    var query = List[Float32](capacity=DIM)
    for d in range(DIM):
        query.append(Float32((999 * 7 + d * 13) % 97) / 48.0 - 1.0)

    t0 = perf_counter_ns()
    for _ in range(REPS):
        var hits = index.search(Span(query), 10)
        sink += hits[0].score
    t1 = perf_counter_ns()
    var search_us = (t1 - t0) / REPS / 1000

    print("search top-10 over", N)
    print("  ", search_us, "us per query")
    print("   qps", 1000000 / search_us)
    print("   throughput", (N * DIM) / (search_us * 1000), "M dims/s")
    print("")
    print("checksum", sink, isink)
