from std.math import abs, sqrt

from embed.distance import cosine_f32, cosine_quantized, dot_f32, dot_i8, euclidean_f32
from embed.index import Index
from embed.quantize import l2_norm, max_abs, normalize, quantize, dequantize


def check(name: StringSlice, got: Float32, want: Float32, tol: Float32 = 1e-4) -> Int:
    var delta = abs(got - want)
    if delta <= tol:
        print("  ok   ", name)
        return 0
    print("  FAIL ", name, "got", got, "want", want)
    return 1


def vec(values: List[Float32]) -> List[Float32]:
    return values.copy()


def main() raises:
    var bad = 0
    print("mojo-embed")

    # -- distance ----------------------------------------------------------
    print("distance")
    var a = List[Float32]()
    var b = List[Float32]()
    for i in range(128):
        a.append(Float32(i % 7) - 3.0)
        b.append(Float32(i % 5) - 2.0)

    var expect = Float32(0.0)
    for i in range(128):
        expect += a[i] * b[i]
    bad += check("dot_f32 vs scalar", dot_f32(Span(a), Span(b)), expect, 1e-2)

    var unit_a = a.copy()
    var unit_b = b.copy()
    normalize(unit_a)
    normalize(unit_b)
    bad += check("unit norm", l2_norm(Span(unit_a)), 1.0)
    bad += check(
        "cosine == dot of unit vectors",
        cosine_f32(Span(a), Span(b)),
        dot_f32(Span(unit_a), Span(unit_b)),
        1e-3,
    )

    var same = a.copy()
    bad += check("cosine with itself", cosine_f32(Span(a), Span(same)), 1.0, 1e-4)
    bad += check("euclidean with itself", euclidean_f32(Span(a), Span(same)), 0.0)

    # -- quantisation ------------------------------------------------------
    print("quantize")
    var q = quantize(Span(a))
    var restored = dequantize(q)
    var worst = Float32(0.0)
    for i in range(len(a)):
        var d = abs(a[i] - restored[i])
        if d > worst:
            worst = d
    # Round-trip error is bounded by half a quantisation step.
    print("  ok    round-trip worst error", worst, "step", q.scale)
    if worst > q.scale:
        print("  FAIL  round trip exceeds one step")
        bad += 1

    bad += check(
        "quantised cosine tracks float",
        cosine_quantized(quantize(Span(a)), quantize(Span(b))),
        cosine_f32(Span(a), Span(b)),
        2e-2,
    )

    var zeros = List[Float32](length=64, fill=0.0)
    var qz = quantize(Span(zeros))
    bad += check("all-zero vector is safe", qz.scale, 1.0)

    # -- index -------------------------------------------------------------
    print("index")
    var index = Index(64)
    index.reserve(200)

    for row in range(200):
        var v = List[Float32](capacity=64)
        for d in range(64):
            v.append(Float32((row * 13 + d * 7) % 23) - 11.0)
        index.add(row, Span(v))

    if index.size() != 200:
        print("  FAIL  size"); bad += 1
    else:
        print("  ok    size", index.size())

    # A stored vector must retrieve itself first.
    var probe = List[Float32](capacity=64)
    for d in range(64):
        probe.append(Float32((42 * 13 + d * 7) % 23) - 11.0)

    var hits = index.search(Span(probe), 5)
    if len(hits) != 5:
        print("  FAIL  expected 5 hits, got", len(hits)); bad += 1
    elif hits[0].id != 42:
        print("  FAIL  self-retrieval: got id", hits[0].id, "want 42"); bad += 1
    else:
        print("  ok    self-retrieval, score", hits[0].score)

    # Scores must come back in descending order.
    var ordered = True
    for i in range(1, len(hits)):
        if hits[i].score > hits[i - 1].score:
            ordered = False
    if not ordered:
        print("  FAIL  hits are not sorted"); bad += 1
    else:
        print("  ok    hits sorted descending")

    print("  ok    memory", index.memory_bytes(), "bytes for", index.size(), "x 64d")

    print("")
    if bad == 0:
        print("all checks passed")
    else:
        print(bad, "FAILED")
        raise Error("mojo-embed self-test failed")
