"""Offline evaluation (Precision@5, RMSE, MAE) and chart generation."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from recommender import Recommender  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METHODS = ["Popularity", "Content-based", "Hybrid"]


def split(ratings, seed, test_frac=0.2):
    """Hold out ~20% of each user's ratings (at least 2) as the test set."""
    rng = np.random.RandomState(seed)
    test_idx = []
    for _, g in ratings.groupby("user_id"):
        n_test = max(2, round(len(g) * test_frac))
        test_idx += list(rng.choice(g.index, n_test, replace=False))
    return ratings.drop(test_idx), ratings.loc[test_idx]


def recommend(rec, method, user, k):
    if method == "Popularity":
        return rec.popularity(user, k)
    if method == "Content-based":
        return rec.content_user(user, k)
    return rec.hybrid_user(user, k)


def evaluate(items, ratings, seeds=range(20), k=5):
    prec = {m: [] for m in METHODS}
    err = {m: [] for m in METHODS}
    for seed in seeds:
        train, test = split(ratings, seed)
        rec = Recommender(items, train)
        for user, g in test.groupby("user_id"):
            relevant = set(g[g.rating >= 4].item_id)
            if relevant:
                for m in METHODS:
                    top = set(recommend(rec, m, user, k).item_id)
                    prec[m].append(len(relevant & top) / k)
            for _, row in g.iterrows():
                for m in METHODS:
                    err[m].append(row.rating - rec.predict_rating(user, row.item_id, m))
    rows = []
    for m in METHODS:
        e = np.array(err[m])
        rows.append({"Method": m, "Precision@5": np.mean(prec[m]),
                     "RMSE": np.sqrt(np.mean(e ** 2)), "MAE": np.mean(np.abs(e))})
    return pd.DataFrame(rows).set_index("Method")


def make_charts(items, ratings, table):
    RESULTS.mkdir(exist_ok=True)

    ratings.rating.value_counts().sort_index().plot(kind="bar", color="steelblue")
    plt.title("Rating distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(RESULTS / "rating_distribution.png", dpi=150)
    plt.close()

    merged = ratings.merge(items[["item_id", "category"]], on="item_id")
    merged.groupby(["category", "rating"]).size().unstack(fill_value=0).plot(
        kind="bar", stacked=True, colormap="viridis")
    plt.title("Ratings per category")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.legend(title="Rating")
    plt.tight_layout()
    plt.savefig(RESULTS / "category_distribution.png", dpi=150)
    plt.close()

    colors = ["gray", "orange", "green"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    table["Precision@5"].plot(kind="bar", ax=axes[0], color=colors)
    axes[0].set_title("Precision@5 (higher is better)")
    table[["RMSE", "MAE"]].plot(kind="bar", ax=axes[1])
    axes[1].set_title("Rating error (lower is better)")
    for ax in axes:
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(RESULTS / "method_comparison.png", dpi=150)
    plt.close()


def main():
    items = pd.read_csv(ROOT / "data" / "items.csv")
    ratings = pd.read_csv(ROOT / "data" / "ratings.csv")
    table = evaluate(items, ratings)
    print(table.round(3).to_string())
    make_charts(items, ratings, table)
    RESULTS.mkdir(exist_ok=True)
    table.round(3).to_csv(RESULTS / "evaluation_table.csv")
    print("Charts and table saved to results/")


if __name__ == "__main__":
    main()
