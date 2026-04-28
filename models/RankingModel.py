from lightgbm import LGBMRanker
import pandas as pd
import numpy as np

class RankingModel:

    def __init__(self, mf_model):
        self.name = "Ranking Model"
        self.is_ranker = True
        self.mf_model = mf_model
        self.ranker = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=100,
            n_jobs=1,
            verbose=-1,
        )
        self.trained  = False
        self.features = ["mf_score", "num_tracks", "has_title", "random_order"]

    def generate_candidates(self, playlist_metadata, playlist_contents, playlist_holdouts, track_metadata):
        print("Generating candidates for training...")
        candidates = self.mf_model.predict(playlist_metadata, playlist_contents, track_metadata, n_recs=500, g_num=None)

        holdouts_flagged = playlist_holdouts[["pid", "track_uri"]].assign(label=1)
        candidates = candidates.merge(holdouts_flagged, on=["pid", "track_uri"], how="left")
        candidates["label"] = candidates["label"].fillna(0).astype(np.int8)

        return candidates
    
    def build_features(self, candidates, playlist_metadata):
        candidates = candidates.merge(playlist_metadata[["pid", "num_tracks", "has_title", "random_order"]], on="pid")
        return candidates

    def train(self, playlist_metadata, playlist_contents, playlist_holdouts, track_metadata):
        if not self.mf_model.trained:
            self.mf_model.train(playlist_metadata, playlist_contents, track_metadata)

        candidates = self.generate_candidates(playlist_metadata, playlist_contents, playlist_holdouts, track_metadata)

        candidates = self.build_features(candidates, playlist_metadata)

        candidates = candidates.sort_values("pid")
        group_sizes = candidates.groupby("pid", sort=False).size().values

        print(f"Training on {len(candidates):,} candidates ({candidates['label'].sum():,} positive)...")
        self.ranker.fit(candidates[self.features], candidates["label"], group=group_sizes)
        self.trained = True

    def predict(self, playlist_metadata, playlist_contents, track_metadata, n_recs, g_num):
        candidates = self.mf_model.predict(playlist_metadata, playlist_contents, track_metadata, n_recs=500, g_num=None)
        candidates = self.build_features(candidates, playlist_metadata)

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