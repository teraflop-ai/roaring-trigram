import time
import pandas as pd
import roaring_trigram

PARQUET = "/home/henry/Downloads/000_00000.parquet"
INDEX = "text.idx"
TOPK = 20
NQ = 1000
QUERY_LEN = 24

df = pd.read_parquet(PARQUET, columns=["text"])

# --- BUILD INDEX ---
idx = roaring_trigram.TrigramIndex()

t_build0 = time.perf_counter()

doc_id = 0
for text in df["text"]:
    if isinstance(text, str):
        b = text.encode("utf-8", errors="ignore")
        if len(b) >= 3:
            idx.add(doc_id, b)
            doc_id += 1

t_build1 = time.perf_counter()

print(f"indexed_docs={doc_id}")
print(f"build_time_sec={t_build1 - t_build0:.2f}")
print(f"docs_per_sec={doc_id / (t_build1 - t_build0):.1f}")

# Optional: optimize + save
idx.optimize()
idx.save(INDEX)

# --- PREP QUERIES ---
queries = []
for text in df["text"]:
    if isinstance(text, str):
        b = text.encode("utf-8", errors="ignore")
        if len(b) >= QUERY_LEN:
            queries.append(b[:QUERY_LEN])
            if len(queries) == NQ:
                break

# --- WARMUP ---
for q in queries[:100]:
    idx.search(q, TOPK)

# --- BENCHMARK ---
t0 = time.perf_counter()
for q in queries:
    idx.search(q, TOPK)
t1 = time.perf_counter()

dt = t1 - t0

print(f"docs={doc_id}")
print(f"query_count={len(queries)}")
print(f"qps={len(queries) / dt:.1f}")
print(f"avg_query_ms={dt * 1000 / len(queries):.3f}")