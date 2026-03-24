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

doc_id = 0
for text in df["text"]:
    if isinstance(text, str):
        b = text.encode("utf-8", errors="ignore")
        if len(b) >= 3:
            docs.append(text)
            if len(b) >= QUERY_LEN and len(queries) < NQ:
                queries.append((b[:QUERY_LEN], doc_id))
            doc_id += 1

print("indexed_docs =", idx.doc_count())
print("local_docs =", len(docs))
print("query_count =", len(queries))

for max_trigrams in [3, 4, 5, 6, 8, 12]:
    for q, _expected in queries[:100]:
        idx.search(q, TOPK, max_trigrams)

    t0 = time.perf_counter()

    hit_at_k = 0
    rank_sum = 0
    rank_count = 0

    for q, expected_doc_id in queries:
        hits = idx.search(q, TOPK, max_trigrams)

        if expected_doc_id in hits:
            hit_at_k += 1
            rank = hits.index(expected_doc_id) + 1
            rank_sum += rank
            rank_count += 1

    t1 = time.perf_counter()
    dt = t1 - t0

    recall_at_k = hit_at_k / len(queries)
    mean_rank = (rank_sum / rank_count) if rank_count else None

    print(
        f"max_trigrams={max_trigrams} "
        f"qps={len(queries) / dt:.1f} "
        f"avg_query_ms={dt * 1000 / len(queries):.3f} "
        f"recall@{TOPK}={recall_at_k:.4f} "
        f"mean_rank={mean_rank}"
    )