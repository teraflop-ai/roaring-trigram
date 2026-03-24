import time
import pandas as pd
import roaring_trigram

PARQUET = "/home/henry/Downloads/000_00000.parquet"
INDEX = "text.idx"
TOPK = 20
NQ = 1000
QUERY_LEN = 72

df = pd.read_parquet(PARQUET, columns=["text"])
idx = roaring_trigram.TrigramIndex.load(INDEX)

docs = []
queries = []

for text in df["text"]:
    if isinstance(text, str):
        b = text.encode("utf-8", errors="ignore")
        if len(b) >= 3:
            docs.append(text)
        if len(b) >= QUERY_LEN and len(queries) < NQ:
            queries.append(b[:QUERY_LEN])

print("indexed_docs =", idx.doc_count())
print("local_docs =", len(docs))
print("query_count =", len(queries))

print("\nSANITY CHECK")
for i, q in enumerate(queries[:5]):
    hits = idx.search(q, TOPK, 12)
    print(f"\nquery[{i}] bytes={q!r}")
    try:
        print(f"query[{i}] text={q.decode('utf-8', errors='replace')!r}")
    except Exception:
        pass
    print(f"hit_count={len(hits)}")
    for rank, doc_id in enumerate(hits[:5], 1):
        if 0 <= doc_id < len(docs):
            doc_text = docs[doc_id]
            print(f"  rank={rank} doc_id={doc_id} doc_text={doc_text[:200]!r}")
        else:
            print(f"  rank={rank} doc_id={doc_id} OUT_OF_RANGE")

for max_trigrams in [3, 4, 5, 6, 8, 12]:
    for q in queries[:100]:
        idx.search(q, TOPK, max_trigrams)

    t0 = time.perf_counter()
    for q in queries:
        idx.search(q, TOPK, max_trigrams)
    t1 = time.perf_counter()

    dt = t1 - t0
    print(
        f"max_trigrams={max_trigrams} "
        f"qps={len(queries) / dt:.1f} "
        f"avg_query_ms={dt * 1000 / len(queries):.3f}"
    )