import time
import pandas as pd
import roaring_trigram

PARQUET = "/home/henry/Downloads/000_00000.parquet"
INDEX = "text.idx"
TOPK = 20
NQ = 1000
QUERY_LEN = 24

df = pd.read_parquet(PARQUET, columns=["text"])

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

idx.optimize()
idx.save(INDEX)