from catalog import get_retriever

r = get_retriever()

print("=== Java developer query ===")
results = r.search("Java developer mid-level", top_k=10)
for i, item in enumerate(results, 1):
    print(f"{i:2}. {item['name']}")

print()
print("=== Python data science query ===")
results = r.search("Python developer data science machine learning", top_k=5)
for i, item in enumerate(results, 1):
    print(f"{i:2}. {item['name']}")

print()
print("=== SQL database analyst ===")
results = r.search("SQL database analyst", top_k=5)
for i, item in enumerate(results, 1):
    print(f"{i:2}. {item['name']}")
