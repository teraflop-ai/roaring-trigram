import time
import pandas as pd
import roaring_trigram

PARQUET = "/home/henry/Downloads/000_00000.parquet"
INDEX = "text.idx"
TOPK = 20
NQ = 1000
QUERY_LEN = 24

df = pd.read_parquet(PARQUET, columns=["text"])
idx = roaring_trigram.TrigramIndex.load(INDEX)

queries = []
for text in df["text"]:
    if isinstance(text, str):
        b = text.encode("utf-8", errors="ignore")
        if len(b) >= QUERY_LEN:
            queries.append(b[:QUERY_LEN])
            if len(queries) == NQ:
                break

# Warmup
for q in queries[:100]:
    idx.search(q, TOPK)

t0 = time.perf_counter()
for q in queries:
    print(idx.search(q, TOPK))
t1 = time.perf_counter()

dt = t1 - t0
print(f"docs={len(df)}")
print(f"query_count={len(queries)}")
print(f"qps={len(queries) / dt:.1f}")
print(f"avg_query_ms={dt * 1000 / len(queries):.3f}")