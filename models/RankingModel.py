from lightgbm import LGBMRanker
import pandas as pd

class RankingModel:

    def __init__(self, artist_model, title_model):
        self.name = "Candidate Ranker"
        self.is_ranker=True
        self.artist_model = artist_model
        self.title_model = title_model
        self.ranker = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=100,
            n_jobs=-1,
            verbose=-1,
        )
        self.trained = False
        self.features = ["artist_rank", "title_rank", "has_title", "group"]

    def _generate_candidates(self, playlist_metadata, playlist_contents, track_metadata):
        artist_preds = self.artist_model.predict(playlist_metadata, playlist_contents, track_metadata, n_recs=500, g_num=None)
        artist_preds = artist_preds.rename(columns={"prediction_num": "artist_rank"})

        has_title = playlist_metadata[playlist_metadata["name"] != ""]
        if len(has_title) > 0:
            title_preds = self.title_model.predict(has_title, playlist_contents[playlist_contents["pid"].isin(has_title["pid"])], track_metadata, n_recs=500, g_num=None)
            title_preds = title_preds.rename(columns={"prediction_num": "title_rank"})
            candidates = artist_preds.merge(title_preds, on=["pid", "track_uri"], how="outer")
        else:
            candidates = artist_preds
            candidates["title_rank"] = None

        return candidates

    def _build_features(self, candidates, playlist_metadata):
        candidates = candidates.merge(playlist_metadata[["pid", "group", "name"]], on="pid")
        candidates["has_title"] = (candidates["name"] != "").astype(int)
        candidates["artist_rank"] = candidates["artist_rank"].fillna(500)
        candidates["title_rank"]  = candidates["title_rank"].fillna(500)
        return candidates.drop(columns=["name"])

    def train(self, playlist_metadata, playlist_contents, playlist_holdouts, track_metadata):
        if not self.artist_model.trained:
            self.artist_model.train(playlist_metadata, playlist_contents, playlist_holdouts, track_metadata)
        if not self.title_model.trained:
            self.title_model.train(playlist_metadata, playlist_contents, playlist_holdouts, track_metadata)

        print("Generating candidates...")
        candidates = self._generate_candidates(playlist_metadata, playlist_contents, track_metadata)
        candidates = self._build_features(candidates, playlist_metadata)

        holdout_index = playlist_holdouts.set_index(["pid", "track_uri"]).index
        candidates["label"] = candidates.set_index(["pid", "track_uri"]).index.isin(holdout_index).astype(int)

        candidates = candidates.sort_values("pid")
        group_sizes = candidates.groupby("pid", sort=False).size().values

        print(f"Training on {len(candidates):,} candidates ({candidates['label'].sum():,} positive)...")
        self.ranker.fit(candidates[self.features], candidates["label"], group=group_sizes)
        self.trained = True

    def predict(self, playlist_metadata, playlist_contents, track_metadata, n_recs, g_num):
        candidates = self._generate_candidates(playlist_metadata, playlist_contents, track_metadata)
        candidates = self._build_features(candidates, playlist_metadata)

        candidates["score"] = self.ranker.predict(candidates[self.features])
        candidates["prediction_num"] = (
            candidates.groupby("pid")["score"]
            .rank(ascending=False, method="first")
            .astype(int) - 1
        )
        return (
            candidates[candidates["prediction_num"] < n_recs][["pid", "prediction_num", "track_uri"]]
            .sort_values(["pid", "prediction_num"])
            .reset_index(drop=True)
        )