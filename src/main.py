"""Demo: runs the 3 required test cases, or recommends for a user ID / free-text query."""
import argparse
from pathlib import Path

import pandas as pd

from recommender import Recommender

ROOT = Path(__file__).resolve().parents[1]
QUERY = "lightweight student laptop with long battery for coding"
SIMILAR_TO = "P05"  # Samsung Galaxy S24

NEW_LAPTOP = {
    "item_id": "L99", "model_name": "NovaBook Lite 14 (new)", "category": "laptop",
    "price_usd": 899,
    "spec_summary": "14 inch lightweight 1.1kg student laptop with 20 hours battery life, 16GB RAM, "
                    "fast SSD, quiet fanless design, great for coding and programming on the go",
    "highlights_tags": "lightweight;battery;student;coding",
}


def load():
    items = pd.read_csv(ROOT / "data" / "items.csv")
    ratings = pd.read_csv(ROOT / "data" / "ratings.csv")
    return items, ratings


def fmt(df):
    return df.to_string(index=False)


def run_tests():
    items, ratings = load()
    out = []

    def emit(text):
        print(text)
        out.append(text + "\n")

    rec = Recommender(items, ratings)
    emit(f"=== TEST 1: Use-case query (Hybrid) ===\nQuery: '{QUERY}'\n{fmt(rec.hybrid_query(QUERY))}\n")

    name = items.set_index("item_id").model_name[SIMILAR_TO]
    emit(f"=== TEST 2: Alternative items similar to {SIMILAR_TO} ({name}) ===\n"
         f"{fmt(rec.content_similar(SIMILAR_TO))}\n")

    items_new = pd.concat([items, pd.DataFrame([NEW_LAPTOP])], ignore_index=True)
    rec_new = Recommender(items_new, ratings)
    top = rec_new.hybrid_query(QUERY)
    emit(f"=== TEST 3: Cold-start (new laptop L99 has spec text only, 0 ratings) ===\n"
         f"Query: '{QUERY}'\n{fmt(top)}\n"
         f"L99 recommended in Top-5: {'L99' in top.item_id.values}"
         f" (rank {list(top.item_id).index('L99') + 1 if 'L99' in top.item_id.values else '-'})\n")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "sample_outputs.txt").write_text("\n".join(out), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Gadget recommender demo")
    ap.add_argument("--user", help="user ID, e.g. U01")
    ap.add_argument("--query", help="free-text query")
    args = ap.parse_args()
    if not (args.user or args.query):
        run_tests()
        return
    items, ratings = load()
    rec = Recommender(items, ratings)
    if args.query:
        print(f"Query: '{args.query}'\n{fmt(rec.hybrid_query(args.query))}\n")
    if args.user and not rec.has_user(args.user):
        known = ", ".join(sorted(rec.matrix.index))
        print(f"User '{args.user}' has no ratings (known users: {known}).\n"
              f"Showing the Popularity baseline only (cold-start user).\n")
        print(fmt(rec.popularity(k=5)))
    elif args.user:
        for title, df in [("Popularity", rec.popularity(args.user)),
                          ("Content-based", rec.content_user(args.user)),
                          ("Hybrid", rec.hybrid_user(args.user))]:
            print(f"--- {title} for {args.user} ---\n{fmt(df)}\n")


if __name__ == "__main__":
    main()
