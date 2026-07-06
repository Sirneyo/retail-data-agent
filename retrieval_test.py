"""Test the Golden Bucket retrieval in isolation — no agent, no SQL, no chat calls."""
from agent import retrieve_trios

for q in [
    "who are our most valuable customers?",
    "what's the trend in our sales?",
    "which brands make us the most profit?",
]:
    print(f"\nQUESTION: {q}")
    for t in retrieve_trios(q, k=3):
        print(f"  -> matched Trio {t['id']}: {t['question']}")