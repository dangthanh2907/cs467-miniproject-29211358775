"""NLP pipeline: text cleaning, TF-IDF features, query vectorization, review sentiment."""
import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

# Words that are stop words in scikit-learn but carry meaning for gadget specs.
STOP_WORDS = set(ENGLISH_STOP_WORDS) - {"thin", "full", "top", "front", "back"}

# Small synonym map so queries and item text meet in the same vocabulary.
SYNONYMS = {
    "light": "lightweight", "ultralight": "lightweight", "ultraportable": "lightweight",
    "students": "student", "programming": "coding", "programmer": "coding",
    "developer": "coding", "developers": "coding", "development": "coding",
    "cheap": "budget", "affordable": "budget", "gamer": "gaming", "games": "gaming",
    "game": "gaming", "photo": "camera", "photos": "camera", "photography": "camera",
    "batteries": "battery",
}

POSITIVE = {
    "great", "excellent", "fast", "light", "lightweight", "long", "smooth", "sharp",
    "bright", "love", "like", "good", "amazing", "comfortable", "reliable", "solid",
    "worth", "premium", "quiet", "sturdy", "responsive", "happy", "satisfied", "decent",
    "easy", "compact", "value",
}
NEGATIVE = {
    "heavy", "slow", "poor", "bad", "hot", "laggy", "dim", "expensive", "weak", "noisy",
    "cheap", "flimsy", "short", "disappointing", "disappointed", "buggy", "unreliable",
    "bulky", "loud", "drains", "awkward", "distracting", "regret",
}
NEGATORS = {"not", "no", "never", "hardly"}


def clean_text(text):
    """Lowercase, replace punctuation with spaces, collapse extra spaces."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text, remove_stop=True, normalize=True):
    """Split cleaned text into tokens; optionally drop stop words and map synonyms."""
    tokens = clean_text(text).split()
    if remove_stop:
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    if normalize:
        tokens = [SYNONYMS.get(t, t) for t in tokens]
    return tokens


def preprocess(text):
    return " ".join(tokenize(text))


def pros_cons(review):
    """Return (pros, cons) word lists found in a review; a negator flips a word's polarity."""
    text = clean_text(review).replace("low light", "lowlight").replace("drains fast", "drains")
    tokens = text.split()
    pros, cons = [], []
    for i, tok in enumerate(tokens):
        polarity = 1 if tok in POSITIVE else -1 if tok in NEGATIVE else 0
        if polarity == 0:
            continue
        if any(t in NEGATORS for t in tokens[max(0, i - 2):i]):
            polarity = -polarity
        (pros if polarity > 0 else cons).append(tok)
    return pros, cons


def sentiment_score(review):
    """Lexicon-based score in [-1, 1] = (pros - cons) / (pros + cons); 0 if no lexicon words."""
    pros, cons = pros_cons(review)
    total = len(pros) + len(cons)
    return (len(pros) - len(cons)) / total if total else 0.0


class TextModel:
    """TF-IDF model over item text. The fitted vectorizer is reused for any new text."""

    def __init__(self, items, reviews=None):
        """items: DataFrame with item_id, model_name, category, spec_summary, highlights_tags.
        reviews: optional dict {item_id: review text} appended to each item's document."""
        raw = (items.model_name + " " + items.category + " " + items.spec_summary
               + " " + items.highlights_tags.str.replace(";", " "))
        if reviews:
            raw = raw + " " + items.item_id.map(reviews).fillna("")
        self.docs = raw.map(preprocess).tolist()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.docs)
        self.terms = self.vectorizer.get_feature_names_out()

    def transform(self, text):
        """Vectorize a free-text query or new item description with the fitted vocabulary."""
        return self.vectorizer.transform([preprocess(text)])

    def keywords(self, vec, item_idx, k=3):
        """Top-k terms shared by a query/profile vector and an item (highest tf-idf overlap)."""
        q = vec.toarray().ravel() if sparse.issparse(vec) else np.asarray(vec).ravel()
        overlap = q * self.matrix[item_idx].toarray().ravel()
        top = overlap.argsort()[::-1][:k]
        return [self.terms[i] for i in top if overlap[i] > 0]


if __name__ == "__main__":
    from pathlib import Path

    import pandas as pd

    root = Path(__file__).resolve().parents[1]
    items = pd.read_csv(root / "data" / "items.csv")
    ratings = pd.read_csv(root / "data" / "ratings.csv")

    print("Cleaned :", clean_text("Lightweight, 13.4-inch laptop!!  18h battery"))
    print("Tokens  :", tokenize("A lightweight student laptop with long battery for coding"))

    model = TextModel(items)
    print("TF-IDF matrix:", model.matrix.shape, "(items x terms)")

    query = "lightweight student laptop with long battery for coding"
    q = model.transform(query)
    sims = (model.matrix @ q.T).toarray().ravel()
    for idx in sims.argsort()[::-1][:5]:
        print(f"  {items.model_name[idx]:<28} sim={sims[idx]:.3f} keywords={model.keywords(q, idx)}")

    print("Sentiment examples:")
    for rating in (5, 1):
        review = ratings[ratings.rating == rating].buyer_review.iloc[0]
        print(f"  [{rating}*] {sentiment_score(review):+.2f}  {review}")
    ratings["sentiment"] = ratings.buyer_review.map(sentiment_score)
    print("Mean sentiment by rating:")
    print(ratings.groupby("rating").sentiment.mean().round(2).to_string())
