"""A flat int8 vector index with top-k search.

Flat, not graph-based, on purpose. Up to a few million vectors an exhaustive
int8 scan is memory-bandwidth-bound and entirely predictable, and it returns
*exact* results. Approximate graph indexes win later than people expect, and
they cost recall, build time and a tuning surface. Start exact; add ANN when a
measurement says to.

Vectors are stored in one contiguous block rather than a list of lists, so a
scan walks memory linearly and the prefetcher does its job.
"""

from std.algorithm import parallelize
from std.math import sqrt
from std.sys.info import num_performance_cores, simd_width_of

from embed.distance import dot_i8
from embed.quantize import I8_WIDTH, QuantizedVector, l2_norm, quantize

comptime DEFAULT_TOP_K = 10

# Below this many vectors, spawning threads costs more than the scan saves.
comptime PARALLEL_THRESHOLD = 20000


struct SearchHit(ImplicitlyCopyable, Movable):
    var id: Int
    var score: Float32

    def __init__(out self):
        self.id = -1
        self.score = -1.0e30

    def __init__(out self, id: Int, score: Float32):
        self.id = id
        self.score = score


struct Index(Movable):
    """Flat int8 index over fixed-dimension vectors."""

    var dim: Int
    var data: List[Int8]
    var scales: List[Float32]
    var norms: List[Float32]
    var ids: List[Int]
    var count: Int

    def __init__(out self, dim: Int):
        self.dim = dim
        self.data = List[Int8]()
        self.scales = List[Float32]()
        self.norms = List[Float32]()
        self.ids = List[Int]()
        self.count = 0

    def size(self) -> Int:
        return self.count

    def memory_bytes(self) -> Int:
        # int8 payload plus two float32 per vector. The point of the whole
        # design: 768 dims costs 776 bytes, not 3072.
        return self.count * (self.dim + 8)

    def add(mut self, id: Int, values: Span[Float32, _]) raises:
        """Quantise and append one vector."""
        if len(values) != self.dim:
            raise Error(
                "expected "
                + String(self.dim)
                + " dimensions, got "
                + String(len(values))
            )

        var q = quantize(values)
        for i in range(self.dim):
            self.data.append(q.data[i])
        self.scales.append(q.scale)
        self.norms.append(q.norm)
        self.ids.append(id)
        self.count += 1

    def add_quantized(mut self, id: Int, vector: QuantizedVector) raises:
        if vector.size() != self.dim:
            raise Error("dimension mismatch")
        for i in range(self.dim):
            self.data.append(vector.data[i])
        self.scales.append(vector.scale)
        self.norms.append(vector.norm)
        self.ids.append(id)
        self.count += 1

    def reserve(mut self, vectors: Int):
        """Pre-size the backing store; growth during a bulk load is the
        single largest avoidable cost when indexing millions of rows."""
        self.data.reserve(vectors * self.dim)
        self.scales.reserve(vectors)
        self.norms.reserve(vectors)
        self.ids.reserve(vectors)

    def _row(self, position: Int) -> Span[Int8, origin_of(self.data)]:
        var start = position * self.dim
        return Span(self.data)[start : start + self.dim]

    def search(mut self, query: Span[Float32, _], k: Int = DEFAULT_TOP_K) raises -> List[SearchHit]:
        """Exact top-k by cosine similarity."""
        var q = quantize(query)
        return self.search_quantized(q, k)

    def search_quantized(
        mut self, query: QuantizedVector, k: Int = DEFAULT_TOP_K
    ) raises -> List[SearchHit]:
        """Exact top-k. Parallel above a threshold, serial below it."""
        if query.size() != self.dim:
            raise Error("query dimension does not match the index")
        if self.count == 0:
            return List[SearchHit]()

        if self.count >= PARALLEL_THRESHOLD:
            return self._search_parallel(query, k)
        return self._search_range(query, k, 0, self.count)

    def _search_range(
        mut self, query: QuantizedVector, k: Int, start: Int, end: Int
    ) raises -> List[SearchHit]:
        """Scan [start, end) and keep the best k."""
        var wanted = min(k, end - start)
        if wanted <= 0:
            return List[SearchHit]()

        var best = List[SearchHit](capacity=wanted)
        # Track the worst score and where it sits, so a losing candidate costs
        # one comparison. Rescanning the heap on every improvement was adding
        # ~45% to the scan.
        var worst = Float32(-1.0e30)
        var worst_at = 0

        var qspan = Span(query.data)
        var query_scale = query.scale
        var query_norm = query.norm if query.norm > 0.0 else Float32(1.0)

        for position in range(start, end):
            var norm = self.norms[position]
            if norm <= 0.0:
                continue

            var raw = Float32(Int(dot_i8(qspan, self._row(position))))
            var score = (raw * query_scale * self.scales[position]) / (query_norm * norm)

            if len(best) < wanted:
                best.append(SearchHit(self.ids[position], score))
                if len(best) == wanted:
                    worst = _worst_of(best)
                    worst_at = _worst_index(best)
            elif score > worst:
                best[worst_at] = SearchHit(self.ids[position], score)
                worst_at = _worst_index(best)
                worst = best[worst_at].score

        _sort_desc(best)
        return best^

    def _search_parallel(
        mut self, query: QuantizedVector, k: Int
    ) raises -> List[SearchHit]:
        """Shard the scan across cores, then merge the shard heaps.

        Each shard writes only its own slot, so no lock is needed; the merge
        is over `shards * k` candidates, which is tiny.
        """
        var shards = min(num_performance_cores(), max(1, self.count // 8000))
        if shards <= 1:
            return self._search_range(query, k, 0, self.count)

        var per = (self.count + shards - 1) // shards
        var merged = List[SearchHit]()

        # Mojo's parallelize takes a capturing closure; results are collected
        # into a preallocated slot per shard to keep it write-disjoint.
        var partials = List[List[SearchHit]]()
        for _ in range(shards):
            partials.append(List[SearchHit]())

        @parameter
        def scan(shard: Int):
            var start = shard * per
            var end = min(start + per, self.count)
            if start >= end:
                return
            try:
                partials[shard] = self._search_range(query, k, start, end)
            except:
                pass

        parallelize[scan](shards, shards)

        for shard in range(shards):
            for i in range(len(partials[shard])):
                merged.append(partials[shard][i])

        _sort_desc(merged)
        var wanted = min(k, len(merged))
        var out = List[SearchHit](capacity=wanted)
        for i in range(wanted):
            out.append(merged[i])
        return out^


def _worst_of(hits: List[SearchHit]) -> Float32:
    var worst = Float32(1.0e30)
    for i in range(len(hits)):
        if hits[i].score < worst:
            worst = hits[i].score
    return worst


def _worst_index(hits: List[SearchHit]) -> Int:
    var index = 0
    var worst = Float32(1.0e30)
    for i in range(len(hits)):
        if hits[i].score < worst:
            worst = hits[i].score
            index = i
    return index


def _sort_desc(mut hits: List[SearchHit]):
    """Insertion sort: k is small, so this beats anything asymptotically better."""
    for i in range(1, len(hits)):
        var current = hits[i]
        var j = i - 1
        while j >= 0 and hits[j].score < current.score:
            hits[j + 1] = hits[j]
            j -= 1
        hits[j + 1] = current
