```
uv add roaring-trigram
```
```py
from roaring_trigram import TrigramIndexBuilder, TrigramIndex

builder = TrigramIndexBuilder()
builder.add(1, b"hello world")
builder.add(2, b"hello there")
builder.save("index.btgi")
del builder

index = TrigramIndex.load("index.btgi")
print(index.search(b"world", k=10, max_trigrams=4))
```