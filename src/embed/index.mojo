"""A flat int8 vector index with top-k search.

Flat, not graph-based, on purpose. Up to a few million vectors an exhaustive
int8 scan is memory-bandwidth-bound and entirely predictable, and it returns
*exact* results. Approximate graph indexes win later than people expect, and
they cost recall, build time and a tuning surface. Start exact; add ANN when a
measurement says to.

Vectors are stored in one contiguous block rather than a list of lists, so a
scan walks memory linearly and the prefetcher does its job.
"""

from max.algorithm import parallelize
from std.sys.info import num_performance_cores

from embed.distance import dot_i8
from embed.quantize import QuantizedVector, quantize

comptime DEFAULT_TOP_K = 10

# Thread only when the scan is large enough to amortise startup.
comptime PARALLEL_WORK_THRESHOLD = 15_000_000


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
    var factors: List[Float32]
    var ids: List[Int]
    var count: Int

    def __init__(out self, dim: Int):
        self.dim = dim
        self.data = List[Int8]()
        self.factors = List[Float32]()
        self.ids = List[Int]()
        self.count = 0

    def size(self) -> Int:
        return self.count

    def memory_bytes(self) -> Int:
        # Store scale / norm once because that ratio is all cosine search uses.
        return self.count * (self.dim + 4)

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
        self.data.extend(Span(q.data))
        self.factors.append(q.scale / q.norm if q.norm > 0.0 else 0.0)
        self.ids.append(id)
        self.count += 1

    def add_quantized(mut self, id: Int, vector: QuantizedVector) raises:
        if vector.size() != self.dim:
            raise Error("dimension mismatch")
        self.data.extend(Span(vector.data))
        self.factors.append(
            vector.scale / vector.norm if vector.norm > 0.0 else 0.0
        )
        self.ids.append(id)
        self.count += 1

    def reserve(mut self, vectors: Int):
        """Pre-size the backing store; growth during a bulk load is the
        single largest avoidable cost when indexing millions of rows."""
        self.data.reserve(vectors * self.dim)
        self.factors.reserve(vectors)
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

        if self.count * self.dim >= PARALLEL_WORK_THRESHOLD:
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
        var query_factor = (
            query.scale / query.norm if query.norm > 0.0 else Float32(0.0)
        )

        for position in range(start, end):
            var factor = self.factors[position]
            if factor == 0.0:
                continue

            var raw = Float32(Int(dot_i8(qspan, self._row(position))))
            var score = raw * query_factor * factor

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

        def scan(shard: Int) { mut self, imm query, imm k, imm per, mut partials }:
            var start = shard * per
            var end = min(start + per, self.count)
            if start >= end:
                return
            try:
                partials[shard] = self._search_range(query, k, start, end)
            except:
                pass

        parallelize(scan, shards, shards)

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
