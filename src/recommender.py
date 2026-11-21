"""Recommenders: popularity baseline, NLP content-based, and a hybrid (content + CF + sentiment)."""
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from nlp_pipeline import TextModel, pros_cons, sentiment_score

HYBRID_USER_WEIGHTS = (0.5, 0.3, 0.2)  # content, item-based CF, sentiment
HYBRID_QUERY_WEIGHTS = (0.8, 0.2)      # content, sentiment
LIKED_RATING = 4                       # reviews with rating >= 4 enrich the item text


class Recommender:
    def __init__(self, items, ratings, min_count=2):
        self.items = items.reset_index(drop=True)
        self.ratings = ratings
        self.min_count = min_count
        self.ids = self.items.item_id.tolist()
        self.pos = {iid: i for i, iid in enumerate(self.ids)}

        # item document = spec text + positive buyer reviews (only from the given ratings,
        # so a train/test split never leaks test reviews into the TF-IDF vocabulary)
        liked = ratings[ratings.rating >= LIKED_RATING]
        liked_reviews = liked.groupby("item_id").buyer_review.apply(" ".join).to_dict()
        self.text = TextModel(self.items, reviews=liked_reviews)

        # pros / cons words found in each item's reviews (lexicon based)
        self.pros_cons_by_item = {}
        for iid, group in ratings.groupby("item_id"):
            pros, cons = Counter(), Counter()
            for review in group.buyer_review:
                p, c = pros_cons(review)
                pros.update(p)
                cons.update(c)
            self.pros_cons_by_item[iid] = (pros, cons)

        # user x item rating matrix (NaN = not rated)
        self.matrix = ratings.pivot_table(index="user_id", columns="item_id",
                                          values="rating").reindex(columns=self.ids)
        self.item_mean = self.matrix.mean(axis=0)
        self.global_mean = ratings.rating.mean()

        # average review sentiment per item, mapped from [-1, 1] to [0, 1]; 0.5 = no reviews
        sent = ratings.assign(s=ratings.buyer_review.map(sentiment_score)).groupby("item_id").s.mean()
        self.sent01 = (self.items.item_id.map(sent).fillna(0.0).values + 1) / 2

        # item-item cosine similarity on user-mean-centered ratings (adjusted cosine)
        centered = self.matrix.sub(self.matrix.mean(axis=1), axis=0).fillna(0).values
        self.item_sim = cosine_similarity(centered.T)
        np.fill_diagonal(self.item_sim, 0)
        self.item_sim = np.nan_to_num(self.item_sim)  # unrated (cold-start) items -> 0

        # item-item content similarity, used for numeric rating prediction
        self.content_sim = cosine_similarity(self.text.matrix)
        np.fill_diagonal(self.content_sim, 0)

    # ---------- helpers ----------
    def seen(self, user_id):
        return set(self.ratings.loc[self.ratings.user_id == user_id, "item_id"])

    def _top(self, scores, k=5, exclude=()):
        s = pd.Series(np.asarray(scores, dtype=float), index=self.ids).dropna()
        s = s.drop([i for i in exclude if i in s.index])
        s = s.sort_values(ascending=False).head(k)
        out = self.items.set_index("item_id").loc[s.index, ["model_name", "category", "price_usd"]]
        out["score"] = s.round(3)
        return out.reset_index()

    def pros_cons_text(self, item_id, n=2):
        """Short pros/cons summary from the item's buyer reviews, e.g. '+ great, light | - heavy'."""
        if item_id not in self.pros_cons_by_item:
            return "no reviews yet"
        pros, cons = self.pros_cons_by_item[item_id]
        plus = ", ".join(w for w, _ in pros.most_common(n)) or "-"
        minus = ", ".join(w for w, _ in cons.most_common(n)) or "-"
        return f"+ {plus} | - {minus}"

    def _with_keywords(self, top, vec):
        top["keywords"] = [", ".join(self.text.keywords(vec, self.pos[i])) for i in top.item_id]
        top["pros_cons"] = [self.pros_cons_text(i) for i in top.item_id]
        return top

    def has_user(self, user_id):
        return user_id in self.matrix.index

    def _content(self, vec):
        """Cosine similarity of vec to every item, scaled so the best match = 1."""
        sims = cosine_similarity(vec, self.text.matrix).ravel()
        return sims / sims.max() if sims.max() > 0 else sims

    def user_profile(self, user_id):
        """TF-IDF profile = sum of item vectors the user liked (rating >= 4), weighted by rating - 2."""
        r = self.ratings[(self.ratings.user_id == user_id) & (self.ratings.rating >= 4)]
        if r.empty:
            return None
        w = r.rating.values - 2.0
        dense = self.text.matrix[[self.pos[i] for i in r.item_id]].toarray()
        return (w @ dense).reshape(1, -1)

    # ---------- 1. Popularity baseline ----------
    def popularity(self, user_id=None, k=5):
        """Weighted rating: (v/(v+m))*mean + (m/(v+m))*C, only for items with >= min_count ratings."""
        st = self.ratings.groupby("item_id").rating.agg(["mean", "count"])
        st = st[st["count"] >= self.min_count]
        m, c = self.min_count, self.global_mean
        score = st["count"] / (st["count"] + m) * st["mean"] + m / (st["count"] + m) * c
        exclude = self.seen(user_id) if user_id else ()
        return self._top(score.reindex(self.ids).values, k, exclude)

    # ---------- 2. NLP content-based ----------
    def content_query(self, query, k=5):
        q = self.text.transform(query)
        return self._with_keywords(self._top(self._content(q), k), q)

    def content_similar(self, item_id, k=5):
        q = self.text.matrix[self.pos[item_id]]
        return self._with_keywords(self._top(self._content(q), k, exclude=[item_id]), q)

    def content_user(self, user_id, k=5):
        prof = self.user_profile(user_id)
        if prof is None:  # user has no liked items: fall back to popularity
            return self.popularity(user_id, k)
        return self._with_keywords(self._top(self._content(prof), k, self.seen(user_id)), prof)

    # ---------- 3. Hybrid: content + item-based CF + sentiment ----------
    def cf_scores(self, user_id):
        """Item-based CF rating prediction (1-5) for every item; NaN when no similar rated item."""
        n = len(self.ids)
        if user_id not in self.matrix.index:
            return np.full(n, np.nan)
        row = self.matrix.loc[user_id].values
        rated = ~np.isnan(row)
        preds = np.full(n, np.nan)
        for j in range(n):
            sims = np.clip(self.item_sim[j, rated], 0, None)
            if sims.sum() > 0:
                preds[j] = (sims * row[rated]).sum() / sims.sum()
        return preds

    def hybrid_scores(self, user_id, w=HYBRID_USER_WEIGHTS):
        prof = self.user_profile(user_id)
        content = self._content(prof) if prof is not None else np.zeros(len(self.ids))
        cf = np.nan_to_num((self.cf_scores(user_id) - 1) / 4, nan=0.0)
        return w[0] * content + w[1] * cf + w[2] * self.sent01, prof

    def hybrid_user(self, user_id, k=5, w=HYBRID_USER_WEIGHTS):
        score, prof = self.hybrid_scores(user_id, w)
        top = self._top(score, k, self.seen(user_id))
        return self._with_keywords(top, prof) if prof is not None else top

    def hybrid_query(self, query, k=5, w=HYBRID_QUERY_WEIGHTS):
        """Free-text query: content similarity plus review sentiment (new items have neutral sentiment)."""
        q = self.text.transform(query)
        score = w[0] * self._content(q) + w[1] * self.sent01
        return self._with_keywords(self._top(score, k), q)

    # ---------- numeric rating prediction (for RMSE / MAE) ----------
    def predict_rating(self, user_id, item_id, method):
        base = self.item_mean.get(item_id)
        base = self.global_mean if pd.isna(base) else base
        if method == "Popularity":
            return base
        j = self.pos[item_id]
        row = self.matrix.loc[user_id].values if user_id in self.matrix.index else None
        if row is None:
            return base
        rated = ~np.isnan(row)

        def knn(sim_row):
            sims = np.clip(sim_row[rated], 0, None)
            return (sims * row[rated]).sum() / sims.sum() if sims.sum() > 0 else base

        content = knn(self.content_sim[j])
        if method == "Content-based":
            return content
        return 0.5 * content + 0.3 * knn(self.item_sim[j]) + 0.2 * base  # Hybrid
