from lightgbm import LGBMRanker
import lightgbm as lgb
import pandas as pd
import numpy as np

# ignore a warning for logging the ndcg@500
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

class RankingModel:

    def __init__(self, mf_model):
        self.name = "Ranking Model"
        self.is_ranker = True
        self.mf_model = mf_model
        self.ranker = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=400,
            n_jobs=1,
            verbose=-1,
            eval_at=[500],
        )
        self.trained  = False
        self.features = [
            "mf_score", 
            "num_tracks", 
            "has_title", 
            "random_order", 
            "n_artist_tracks_in_playlist", 
            "n_album_tracks_in_playlist",
            "track_pop_count",
            "artist_pop_count",
            "album_pop_count"
        ]

    def generate_candidates(self, playlist_metadata, playlist_contents, playlist_holdouts, track_metadata):
        print("Generating candidates for training...")
        candidates = self.mf_model.predict(playlist_metadata, playlist_contents, track_metadata, n_recs=500, g_num=None)

        holdouts_flagged = playlist_holdouts[["pid", "track_uri"]].assign(label=1)
        candidates = candidates.merge(holdouts_flagged, on=["pid", "track_uri"], how="left")
        candidates["label"] = candidates["label"].fillna(0).astype(np.int8)

        return candidates
    
    def build_features(self, candidates, playlist_metadata, train_contents, track_metadata, feature_contents):
        # Playlist: Add group related features
        candidates = candidates.merge(playlist_metadata[["pid", "num_tracks", "has_title", "random_order"]], on="pid")
        candidates["has_title"] = candidates["has_title"].astype(bool)
        candidates["random_order"] = candidates["random_order"].astype(bool)

        # Interaction: Get number of times the artist appears in the playlist
        candidates = candidates.merge(track_metadata[["track_uri", "artist_uri", "album_uri"]], on="track_uri")
        train_contents_enriched = (
            train_contents[["pid", "track_uri"]]
            .merge(track_metadata[["track_uri", "artist_uri", "album_uri"]], on="track_uri")
        )
        feature_contents_enriched = (
            feature_contents[["pid", "track_uri"]]
            .merge(track_metadata[["track_uri", "artist_uri", "album_uri"]], on="track_uri")
        )
        artist_counts = (
            train_contents_enriched
            .groupby(["pid", "artist_uri"])
            .size()
            .reset_index(name="n_artist_tracks_in_playlist")
        )
        album_counts = (
            train_contents_enriched
            .groupby(["pid", "album_uri"])
            .size()
            .reset_index(name="n_album_tracks_in_playlist")
        )

        candidates = candidates.merge(artist_counts, on=["pid", "artist_uri"], how="left")
        candidates = candidates.merge(album_counts, on=["pid", "album_uri"], how="left")
        candidates["n_artist_tracks_in_playlist"] = candidates["n_artist_tracks_in_playlist"].fillna(0)
        candidates["n_album_tracks_in_playlist"] = candidates["n_album_tracks_in_playlist"].fillna(0)

        # Track: Get global popularity of the track
        filtered_enriched_feature_contents = feature_contents_enriched[feature_contents_enriched["track_uri"].isin(train_contents_enriched["track_uri"])]
        track_pop_count = (
            filtered_enriched_feature_contents
            .groupby("track_uri")
            .size()
            .reset_index(name="track_pop_count")
        )

        artist_pop_count = (
            filtered_enriched_feature_contents
            .groupby("artist_uri")
            .size()
            .reset_index(name="artist_pop_count")
        )

        album_pop_count = (
            filtered_enriched_feature_contents
            .groupby("album_uri")
            .size()
            .reset_index(name="album_pop_count")
        )

        candidates = candidates.merge(track_pop_count, on="track_uri", how="left")
        candidates = candidates.merge(artist_pop_count, on="artist_uri", how="left")
        candidates = candidates.merge(album_pop_count, on="album_uri", how="left")
        candidates["track_pop_count"] = candidates["track_pop_count"].fillna(0)
        candidates["artist_pop_count"] = candidates["artist_pop_count"].fillna(0)
        candidates["album_pop_count"] = candidates["album_pop_count"].fillna(0)

        return candidates

    def train(self, playlist_metadata, playlist_contents, playlist_holdouts, track_metadata, feature_playlist_metadata, feature_playlist_contents):
        if not self.mf_model.trained:
            raise RuntimeError("MF model not trained yet")

        candidates = self.generate_candidates(playlist_metadata, playlist_contents, playlist_holdouts, track_metadata)
        candidates = self.build_features(candidates, feature_playlist_metadata, playlist_contents, track_metadata, feature_playlist_contents)
        candidates = candidates.sort_values("pid")

        # Split pids into train/val to check overfitting
        pids = candidates["pid"].unique()
        val_pids = set(pd.Series(pids).sample(frac=0.1, random_state=42))
        train_mask = ~candidates["pid"].isin(val_pids)

        train_df = candidates[train_mask]
        val_df = candidates[~train_mask]

        train_groups = train_df.groupby("pid", sort=False).size().values
        val_groups = val_df.groupby("pid", sort=False).size().values

        print(f"Training on {len(train_df):,} candidates ({train_df['label'].sum():,} positive)...")
        self.ranker.fit(
            train_df[self.features], train_df["label"], group=train_groups,
            eval_set=[
                (train_df[self.features], train_df["label"]),
                (val_df[self.features], val_df["label"]),
            ],
            eval_group=[train_groups, val_groups],
            callbacks=[lgb.log_evaluation(period=10), lgb.early_stopping(25)],
        )
        self.trained = True

    def predict(self, playlist_metadata, playlist_contents, track_metadata, n_recs, g_num, feature_playlist_metadata, feature_playlist_contents):
        candidates = self.mf_model.predict(playlist_metadata, playlist_contents, track_metadata, n_recs=500, g_num=None)
        candidates = self.build_features(candidates, playlist_metadata, playlist_contents, track_metadata, feature_playlist_contents)

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