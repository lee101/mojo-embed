# mojo-embed

Fast int8 vector search in [Mojo](https://www.modular.com/mojo). SIMD kernels,
symmetric int8 quantisation, and an exact flat index that parallelises across
cores.

Memory for 100k × 768d is about **74 MiB int8** against 293 MiB float32, a 4×
reduction. Memory traffic is what bounds an exhaustive vector scan.

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
  capi.mojo       zero-copy C ABI used by Python/NumPy
```

- `dot_f32` uses four independent accumulators, because one FMA chain stalls
  on its own result and four interleave.
- `dot_i8` accumulates into int32. Two int8 values multiply to at most 16,129,
  so a 4096-dimensional vector cannot overflow — no saturation checks in the
  inner loop.
- Top-k tracks the worst score *and its position* incrementally. Rescanning the
  heap on every improvement was adding ~45% to the scan.
- Quantise, dequantise, normalise, max-abs, norms, and every distance kernel
  use native-width SIMD with a scalar tail.
- `Index` shards only above 15 million dimension-products. The zero-copy C ABI
  uses a higher 100 million threshold because its serial pointer loop is
  cheaper. Both thresholds were measured on the benchmark host.
- The index stores `scale / norm` as one float per row instead of retaining
  separate scale and norm arrays.

## Benchmark

`pixi run bench` builds the shared library and runs `bench/bench.py` under
`flock /tmp/mojo-bench.lock`. Mojo and NumPy receive the same contiguous
arrays; buffers cross the C ABI by address without copies. Every case is
warmed up and reports the median of seven repeats.

Measured on an Intel Xeon E5-2697 v4, Python 3.13.14, and NumPy 2.5.1:

| Operation | Input size | mojo-embed | NumPy | Speedup |
|---|---:|---:|---:|---:|
| Float32 dot | 771 dims | 1.34 us | 1.13 us | 0.84x |
| Int8 dot | 771 dims | 1.28 us | 5.36 us | 4.19x |
| Cosine similarity | 771 dims | 1.40 us | 4.99 us | 3.56x |
| Euclidean distance | 771 dims | 1.39 us | 6.05 us | 4.34x |
| Symmetric quantize | 1,000,003 dims | 1.17 ms | 13.07 ms | 11.16x |
| L2 normalize | 1,000,003 dims | 509.73 us | 457.50 us | 0.90x |
| Exact int8 top-10 (serial) | 10,000 × 768 | 856.46 us | 5.28 ms | 6.16x |
| Exact int8 top-10 (parallel) | 200,000 × 768 | 19.47 ms | 113.26 ms | 5.82x |

The odd sizes exercise scalar SIMD tails. The exact-search baseline uses
NumPy int32 accumulation over the same int8 matrix, followed by
`argpartition`; setup and quantisation are outside both search timings.
Ratios below 1.0 mean mojo-embed is slower.

As an optimization checkpoint, the original native benchmark on the same
host went from 766 ms to 388 ms to build a 100k × 768 index, and from
4.245 ms to 3.221 ms for top-10 search. Those checkpoint timings are single
locked runs; the release comparison above is the median benchmark.

## Honest limits

- **No GPU yet.** `std.gpu` exists in Mojo 1.0 and the design leaves room for a
  device path, but nothing here runs on one. The numbers above are CPU.
- **No ANN index.** Flat scan only. Fine to a few million vectors; past that
  you want IVF or a graph.
- **No embedding model.** This searches vectors; it does not produce them.
- **No persistence.** The index lives in memory; save and load are not written.

## Build

```bash
pixi run test
pixi run bench
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

MIT.
