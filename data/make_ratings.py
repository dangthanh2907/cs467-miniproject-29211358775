import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rng = random.Random(42)  # fixed seed so the data is reproducible

# user_id: (preferred tags, budget in USD, preferred category, usage phrase)
PERSONAS = {
    "U01": ({"lightweight", "battery", "student", "coding"}, 1200, "laptop", "for coding in class"),
    "U02": ({"gaming", "performance", "display"}, 1600, "laptop", "for PC gaming"),
    "U03": ({"camera", "flagship", "display"}, 1400, "smartphone", "for photography"),
    "U04": ({"budget", "student", "battery"}, 600, "laptop", "as a cheap study machine"),
    "U05": ({"business", "lightweight", "battery", "reliable"}, 1700, "laptop", "for business travel"),
    "U06": ({"creative", "display", "premium", "performance"}, 2200, "laptop", "for video editing and design"),
    "U07": ({"budget", "battery", "compact"}, 450, "smartphone", "as my daily phone"),
    "U08": ({"gaming", "performance", "battery"}, 1100, "smartphone", "for mobile gaming"),
    "U09": ({"coding", "keyboard", "reliable", "performance"}, 1700, "laptop", "for software development"),
    "U10": ({"budget", "reliable", "battery"}, 400, "smartphone", "for my parents"),
    "U11": ({"premium", "camera", "quiet", "display"}, 1700, "smartphone", "as my main device"),
    "U12": ({"camera", "compact", "battery", "lightweight"}, 1100, "smartphone", "for travel vlogs"),
    "U13": ({"office", "budget", "reliable", "keyboard"}, 700, "laptop", "for office work"),
    "U14": ({"display", "battery", "premium"}, 1300, "smartphone", "for watching videos"),
}

POS = {
    "lightweight": "very light and easy to carry",
    "battery": "long battery life is great",
    "student": "great for classes and note taking",
    "coding": "smooth and fast for coding",
    "quiet": "quiet and comfortable to use",
    "performance": "fast and smooth performance",
    "creative": "excellent for editing and design",
    "display": "the screen is bright and sharp",
    "premium": "premium and sturdy build",
    "business": "reliable for work and meetings",
    "keyboard": "the keyboard is comfortable",
    "reliable": "solid and reliable",
    "budget": "great value and worth the price",
    "office": "smooth for everyday office work",
    "gaming": "fast and smooth for games",
    "camera": "the camera is sharp and amazing",
    "flagship": "love the flagship feel",
    "compact": "compact and light in the pocket",
}
NEG = {
    "lightweight": "feels heavy and bulky",
    "battery": "the battery is weak and drains fast",
    "student": "bulky and awkward to carry around",
    "coding": "slow and laggy when compiling code",
    "quiet": "noisy and distracting to use",
    "performance": "slow and laggy under load",
    "creative": "disappointing for editing work",
    "display": "the screen is dim",
    "premium": "feels cheap and flimsy",
    "business": "feels flimsy for work",
    "keyboard": "the keyboard is poor",
    "reliable": "buggy and unreliable",
    "budget": "too expensive for what it offers",
    "office": "slow for everyday office work",
    "gaming": "gets hot and laggy in games",
    "camera": "the camera is poor in low light",
    "flagship": "disappointing for the price",
    "compact": "bulky in the pocket",
}
OPEN_GOOD = ["Really like it", "Happy with this", "Good pick", "Very satisfied"]
OPEN_MID = ["It is okay", "Decent choice", "Mixed feelings"]
OPEN_BAD = ["Regret buying this", "Disappointed", "Would not recommend"]


def build_review(rating, matched, missing, over_budget, use):
    pos = [POS[t] for t in matched]
    neg = [NEG[t] for t in missing]
    rng.shuffle(pos)
    rng.shuffle(neg)
    if rating >= 4:
        parts = " and ".join(pos[:2] or ["decent overall"])
        return f"{rng.choice(OPEN_GOOD)} {use}: {parts}"
    if rating == 3:
        good = pos[:1] or ["acceptable overall"]
        bad = neg[:1] or ["a bit slow sometimes"]
        return f"{rng.choice(OPEN_MID)} {use}: {good[0]} but {bad[0]}"
    bad = neg[:2] or ["feels average and slow"]
    if over_budget and "too expensive for what it offers" not in bad:
        bad.append("too expensive for what it offers")
    return f"{rng.choice(OPEN_BAD)} {use}: " + " and ".join(bad)


def main():
    with open(ROOT / "data" / "items.csv", encoding="utf-8", newline="") as f:
        items = list(csv.DictReader(f))
    for it in items:
        it["tags"] = set(it["highlights_tags"].split(";"))

    rows = []
    for user, (tags, budget, pref, use) in PERSONAS.items():
        pool = items[:]
        chosen = []
        for _ in range(rng.randint(15, 20)):
            w = [3 if it["category"] == pref else 1 for it in pool]
            pick = rng.choices(pool, weights=w)[0]
            chosen.append(pick)
            pool.remove(pick)
        for it in sorted(chosen, key=lambda x: x["item_id"]):
            matched = sorted(tags & it["tags"])
            missing = sorted(tags - it["tags"])
            over = float(it["price_usd"]) > budget
            raw = 2.0 + 0.9 * len(matched) - (0.8 if over else 0) + rng.gauss(0, 0.4)
            rating = max(1, min(5, round(raw)))
            rows.append({"user_id": user, "item_id": it["item_id"], "rating": rating,
                         "buyer_review": build_review(rating, matched, missing, over, use)})

    out = ROOT / "data" / "ratings.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["user_id", "item_id", "rating", "buyer_review"])
        w.writeheader()
        w.writerows(rows)

    n_items = len(items)
    print(f"{len(rows)} ratings, {len(PERSONAS)} users, {n_items} items")
    print(f"Missing entries: {1 - len(rows) / (len(PERSONAS) * n_items):.0%}")


if __name__ == "__main__":
    main()