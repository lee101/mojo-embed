# mojo-embed

Fast int8 vector search in [Mojo](https://www.modular.com/mojo). SIMD kernels,
symmetric int8 quantisation, and an exact flat index that parallelises across
cores.

Measured on a 36-core box, 768-dimensional vectors:

| | ns per dot | vs scalar |
|---|---|---|
| scalar f32 | 3,340 | 1× |
| **SIMD f32** | **238** | **14×** |
| **SIMD int8** | **108** | **31×** |

Memory for 100k × 768d: **74 MB int8** against 293 MB float32 — a 4× reduction,
which is what actually bounds vector search.

```mojo
from embed.index import Index
from embed.quantize import normalize

def main() raises:
    var index = Index(768)
    index.reserve(100_000)

    for row in range(100_000):
        var v = embedding_for(row)   # List[Float32]
        normalize(v)                 # optional: makes dot == cosine
        index.add(row, Span(v))

    var hits = index.search(Span(query), k=10)
    for i in range(len(hits)):
        print(hits[i].id, hits[i].score)
```

## Why int8

An embedding's components cluster tightly around zero, so a symmetric
per-vector scale keeps recall within noise of float32. In the test suite the
quantised cosine tracks the float32 cosine to within 2e-2, and round-trip
error stays under one quantisation step.

Symmetric rather than asymmetric on purpose: a zero-point needs a correction
term in every dot product, which costs more than the range it recovers on data
that is already centred.

## Why flat, not a graph

Up to a few million vectors an exhaustive int8 scan is memory-bandwidth-bound,
entirely predictable, and **exact**. Graph indexes win later than people
expect, and they cost recall, build time and a tuning surface. Start exact; add
ANN when a measurement says to.

## What is here

```
src/embed/
  quantize.mojo   symmetric int8 quantisation, L2 norm, max-abs  (SIMD)
  distance.mojo   dot / cosine / euclidean, f32 and int8         (SIMD)
  index.mojo      flat index, exact top-k, parallel scan
```

- `dot_f32` uses four independent accumulators, because one FMA chain stalls
  on its own result and four interleave.
- `dot_i8` accumulates into int32. Two int8 values multiply to at most 16,129,
  so a 4096-dimensional vector cannot overflow — no saturation checks in the
  inner loop.
- Top-k tracks the worst score *and its position* incrementally. Rescanning the
  heap on every improvement was adding ~45% to the scan.
- Search shards across cores above 20k vectors; below that, thread setup costs
  more than the scan saves.

## Honest limits

- **No GPU yet.** `std.gpu` exists in Mojo 1.0 and the design leaves room for a
  device path, but nothing here runs on one. The numbers above are CPU.
- **No ANN index.** Flat scan only. Fine to a few million vectors; past that
  you want IVF or a graph.
- **No embedding model.** This searches vectors; it does not produce them.
- **Search is ~152 qps** on 100k × 768d single-query. That is one query at a
  time using all cores. Batched queries would amortise far better and are the
  obvious next optimisation.
- **No persistence.** The index lives in memory; save and load are not written.

## Build

```bash
pixi run mojo build tests/test_embed.mojo -I src -o test_embed && ./test_embed
pixi run mojo build bench/bench.mojo -I src -o bench && ./bench
```

Or use it in your own project:

```bash
mojo build your_app.mojo -I path/to/mojo-embed/src
```

## Relationship to gobed

This takes the ideas that mattered from [gobed](https://github.com/lee101/gobed)
— int8 quantisation, SIMD distance kernels, a flat exact index — and rebuilds
them in Mojo, where the SIMD is a language feature rather than an intrinsics
package. gobed's CAGRA GPU index, its caches and its bulk indexer are not
ported.

## Licence

Apache-2.0.
