from lightgbm import LGBMClassifier
import lightgbm as lgb
import pandas as pd
import numpy as np
import duckdb as db

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

class RankingModel:

    def __init__(self, mf_model, title_model):
        self.name = "Ranking Model"
        self.is_ranker = True
        self.mf_model = mf_model
        self.title_model = title_model
        self.classifier = LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            n_estimators=400,
            n_jobs=1,
            verbose=-1,
        )
        self.trained = False
        self.features = [
            "mf_score",
            "num_samples",
            "has_title",
            "random_order",
            "n_artist_tracks_in_playlist",
            "n_album_tracks_in_playlist",
            "track_pop_count",
            "artist_pop_count",
            "album_pop_count",
            "same_artist_as_last",
            "same_album_as_last",
            "title_score"
        ]

    def generate_candidates(self, playlist_metadata, playlist_contents, track_metadata):
        print("Generating candidates...")
        candidates = self.mf_model.predict(playlist_metadata, playlist_contents, track_metadata, n_recs=500, g_num=None)

        if not playlist_contents.empty:
            # Enrich playlist contents with artist info
            enriched = playlist_contents[["pid", "track_uri"]].merge(
                track_metadata[["track_uri", "artist_uri"]], on="track_uri"
            )

            # Global track popularity
            track_popularity = (
                playlist_contents.groupby("track_uri")
                .size()
                .reset_index(name="global_pop")
            )

            # All tracks per artist ranked by popularity
            artist_track_pool = (
                track_metadata[["track_uri", "artist_uri"]]
                .merge(track_popularity, on="track_uri", how="left")
                .fillna({"global_pop": 0})
                .sort_values("global_pop", ascending=False)
            )
            # Rank each track within its artist by popularity
            artist_track_pool["artist_track_rank"] = (
                artist_track_pool.groupby("artist_uri")["global_pop"]
                .rank(ascending=False, method="first")
                .astype(int)
            )

            # Compute per-artist budget
            artist_counts = (
                enriched.groupby(["pid", "artist_uri"])
                .size()
                .reset_index(name="n_in_playlist")
            )
            playlist_sizes = enriched.groupby("pid").size().reset_index(name="playlist_len")
            artist_counts = artist_counts.merge(playlist_sizes, on="pid")
            artist_counts["artist_budget"] = (
                (artist_counts["n_in_playlist"] / artist_counts["playlist_len"] * 100)
                .round()
                .astype(int)
                .clip(lower=1)
            )

            # Cross join artists+budgets with their track pool, keep only tracks within budget
            artist_candidates = (
                artist_counts[["pid", "artist_uri", "artist_budget"]]
                .merge(artist_track_pool[["track_uri", "artist_uri", "artist_track_rank"]], on="artist_uri")
                .query("artist_track_rank <= artist_budget")
            )

            # Drop tracks already in the playlist
            existing_tracks = enriched[["pid", "track_uri"]].assign(in_playlist=True)
            artist_candidates = (
                artist_candidates
                .merge(existing_tracks, on=["pid", "track_uri"], how="left")
                .query("in_playlist.isna()")
                .drop(columns=["in_playlist"])
            )

            # Drop tracks already in MF candidates
            existing_candidates = candidates[["pid", "track_uri"]].assign(in_mf=True)
            artist_candidates = (
                artist_candidates
                .merge(existing_candidates, on=["pid", "track_uri"], how="left")
                .query("in_mf.isna()")
                .drop(columns=["in_mf"])
            )

            artist_candidates = (
                artist_candidates[["pid", "track_uri"]]
                .assign(mf_score=0.0)
            )

            candidates = pd.concat([candidates, artist_candidates], ignore_index=True)

        return candidates

    def build_features(self, candidates, playlist_metadata, playlist_contents, track_metadata, feature_contents):
        # Add playlist level data
        print("Building features...")
        if not playlist_contents.empty:
            last_tracks = db.sql("""
                SELECT c.pid, c.track_uri as last_track, tm.artist_uri as last_artist, tm.album_uri as last_album
                FROM (
                    SELECT pid, track_uri,
                        ROW_NUMBER() OVER (PARTITION BY pid ORDER BY position DESC) as rn
                    FROM playlist_contents
                ) c
                JOIN track_metadata tm
                ON c.track_uri = tm.track_uri
                WHERE rn = 1
            """).df()

            playlist_metadata = playlist_metadata.merge(last_tracks, on="pid", how="left")

        if "last_artist" not in playlist_metadata.columns:
            playlist_metadata["last_artist"] = np.nan
            playlist_metadata["last_album"] = np.nan

        candidates = candidates.merge(track_metadata[["track_uri", "artist_uri", "album_uri"]], on="track_uri")
        candidates = candidates.merge(playlist_metadata[["pid", "num_samples", "has_title", "random_order", "last_artist", "last_album"]], on="pid")
        candidates["has_title"] = candidates["has_title"].astype(bool)
        candidates["random_order"] = candidates["random_order"].astype(bool)
        candidates["same_artist_as_last"] = (candidates["artist_uri"] == candidates["last_artist"]).fillna(False)
        candidates["same_album_as_last"] = (candidates["album_uri"] == candidates["last_album"]).fillna(False)

        # Get artist and album of each track
        playlist_contents_enriched = (
            playlist_contents[["pid", "track_uri"]]
            .merge(track_metadata[["track_uri", "artist_uri", "album_uri"]], on="track_uri")
        )
        feature_contents_enriched = (
            feature_contents[["pid", "track_uri"]]
            .merge(track_metadata[["track_uri", "artist_uri", "album_uri"]], on="track_uri")
        )

        # Get counts of artist and album in playlist 
        artist_counts = (
            playlist_contents_enriched
            .groupby(["pid", "artist_uri"])
            .size()
            .reset_index(name="n_artist_tracks_in_playlist")
        )
        album_counts = (
            playlist_contents_enriched
            .groupby(["pid", "album_uri"])
            .size()
            .reset_index(name="n_album_tracks_in_playlist")
        )

        candidates = candidates.merge(artist_counts, on=["pid", "artist_uri"], how="left")
        candidates = candidates.merge(album_counts, on=["pid", "album_uri"], how="left")
        candidates["n_artist_tracks_in_playlist"] = candidates["n_artist_tracks_in_playlist"].fillna(0)
        candidates["n_album_tracks_in_playlist"] = candidates["n_album_tracks_in_playlist"].fillna(0)

        # Get popularity of each candidate at song, artist, and album level
        filtered_enriched_feature_contents = feature_contents_enriched[feature_contents_enriched["track_uri"].isin(playlist_contents_enriched["track_uri"])]
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

        # Add title based features
        candidates = candidates.merge(self.title_model.score_tracks(playlist_metadata, candidates), on=['pid', 'track_uri'])

        return candidates

    def train(self, playlist_metadata, playlist_contents, playlist_holdouts, track_metadata, feature_playlist_metadata, feature_playlist_contents):
        if not self.mf_model.trained:
            raise RuntimeError("MF model not trained yet")

        candidates = self.generate_candidates(playlist_metadata, playlist_contents, track_metadata)

        holdouts_flagged = playlist_holdouts[["pid", "track_uri"]].assign(label=1)
        candidates = candidates.merge(holdouts_flagged, on=["pid", "track_uri"], how="left")
        candidates["label"] = candidates["label"].fillna(0).astype(np.int8)

        candidates = self.build_features(candidates, playlist_metadata, playlist_contents, track_metadata, feature_playlist_contents)

        pids = candidates["pid"].unique()
        val_pids = set(pd.Series(pids).sample(frac=0.1, random_state=42))
        train_mask = ~candidates["pid"].isin(val_pids)

        train_df = candidates[train_mask]
        val_df = candidates[~train_mask]

        print(f"Training on {len(train_df):,} candidates ({train_df['label'].sum():,} positive)...")
        self.classifier.fit(
            train_df[self.features], train_df["label"],
            eval_set=[(train_df[self.features], train_df["label"]),
                      (val_df[self.features], val_df["label"])],
            callbacks=[lgb.log_evaluation(period=10), lgb.early_stopping(25)],
        )
        self.trained = True

    def predict(self, playlist_metadata, playlist_contents, track_metadata, n_recs, g_num, feature_playlist_metadata, feature_playlist_contents):
        candidates = self.generate_candidates(playlist_metadata, playlist_contents, track_metadata)
        candidates = self.build_features(candidates, playlist_metadata, playlist_contents, track_metadata, feature_playlist_contents)

        candidates["score"] = self.classifier.predict_proba(candidates[self.features])[:, 1]
        candidates["prediction_num"] = (
            candidates.groupby("pid")["score"]
            .rank(ascending=False, method="first")
            .astype(int) - 1
        )

        preds = (
            candidates[candidates["prediction_num"] < n_recs][["pid", "prediction_num", "track_uri"]]
            .sort_values(["pid", "prediction_num"])
            .reset_index(drop=True)
        )
        return preds