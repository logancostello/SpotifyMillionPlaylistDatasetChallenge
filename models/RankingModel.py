from lightgbm import LGBMClassifier
import lightgbm as lgb
import pandas as pd
import numpy as np
import duckdb as db
import random

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
        self.cooc_index = None
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
            "title_score",
            "duration_zscore",
            "playlist_duration_mean",
            "playlist_duration_std",
            "cooccurrence_score",
        ]

    def build_cooc_index(self, feature_contents, playlist_contents, k_max, max_pids_per_seed=500):
        """Build a co-occurrence index for only the seed tracks in playlist_contents"""
        if self.cooc_index is None:
            self.cooc_index = {}

        seed_uris = set(playlist_contents["track_uri"].unique())
        missing = seed_uris - self.cooc_index.keys()

        if not missing:
            print("Co-occurrence index already covers all seeds — skipping build.")
            return

        print(f"Building co-occurrence index for {len(missing):,} new seeds ({len(self.cooc_index):,} already cached)...")

        fc = feature_contents[["pid", "track_uri"]].drop_duplicates()

        seed_pids = set(fc[fc["track_uri"].isin(missing)]["pid"].unique())
        fc_relevant = fc[fc["pid"].isin(seed_pids)]

        pid_to_tracks = fc_relevant.groupby("pid")["track_uri"].apply(np.array).to_dict()

        track_to_pids = (
            fc_relevant[fc_relevant["track_uri"].isin(missing)]
            .groupby("track_uri")["pid"].apply(list).to_dict()
        )

        max_track_id = int(fc_relevant["track_uri"].max()) + 1

        total = len(missing)
        for i, seed in enumerate(missing):
            if i % 1000 == 0:
                print(f"  {i:,}/{total:,} seeds processed...")

            pids = track_to_pids.get(seed)
            if not pids:
                continue

            # Cap popular seeds to avoid excessive work
            if len(pids) > max_pids_per_seed:
                pids = random.sample(pids, max_pids_per_seed)

            all_tracks = np.concatenate([pid_to_tracks[pid] for pid in pids])

            all_tracks = all_tracks[all_tracks != seed]

            if len(all_tracks) == 0:
                continue

            counts = np.bincount(all_tracks, minlength=max_track_id)

            nonzero = np.flatnonzero(counts)
            if len(nonzero) == 0:
                continue

            nonzero_counts = counts[nonzero]
            if len(nonzero) > k_max:
                top_pos = np.argpartition(nonzero_counts, -k_max)[-k_max:]
                nonzero = nonzero[top_pos]
                nonzero_counts = nonzero_counts[top_pos]

            order = np.argsort(nonzero_counts)[::-1]
            self.cooc_index[seed] = np.column_stack([nonzero[order], nonzero_counts[order]])

        print(f"Co-occurrence index now covers {len(self.cooc_index):,} seed tracks.")

    def generate_candidates(self, playlist_metadata, playlist_contents, track_metadata, feature_contents, n_recs=1000):
        print("Generating candidates...")
        candidates = self.mf_model.predict(playlist_metadata, playlist_contents, track_metadata, n_recs=n_recs, g_num=None)
        candidates["cooccurrence_score"] = 0

        if not playlist_contents.empty:
            enriched = playlist_contents[["pid", "track_uri"]].merge(
                track_metadata[["track_uri", "artist_uri"]], on="track_uri"
            )

            track_popularity = (
                feature_contents.groupby("track_uri")
                .size()
                .reset_index(name="global_pop")
            )

            artist_track_pool = (
                track_metadata[["track_uri", "artist_uri"]]
                .merge(track_popularity, on="track_uri", how="left")
                .fillna({"global_pop": 0})
                .sort_values("global_pop", ascending=False)
            )
            artist_track_pool["artist_track_rank"] = (
                artist_track_pool.groupby("artist_uri")["global_pop"]
                .rank(ascending=False, method="first")
                .astype(int)
            )

            artist_counts = (
                enriched.groupby(["pid", "artist_uri"])
                .size()
                .reset_index(name="n_in_playlist")
            )
            playlist_sizes = enriched.groupby("pid").size().reset_index(name="playlist_len")
            artist_counts = artist_counts.merge(playlist_sizes, on="pid")
            artist_counts["artist_budget"] = (
                (artist_counts["n_in_playlist"] / artist_counts["playlist_len"] * 250)
                .round()
                .astype(int)
                .clip(lower=1)
            )

            artist_candidates = (
                artist_counts[["pid", "artist_uri", "artist_budget"]]
                .merge(artist_track_pool[["track_uri", "artist_uri", "artist_track_rank"]], on="artist_uri")
                .query("artist_track_rank <= artist_budget")
            )

            existing_tracks = enriched[["pid", "track_uri"]].assign(in_playlist=True)
            artist_candidates = (
                artist_candidates
                .merge(existing_tracks, on=["pid", "track_uri"], how="left")
                .query("in_playlist.isna()")
                .drop(columns=["in_playlist"])
            )

            existing_candidates = candidates[["pid", "track_uri"]].assign(in_mf=True)
            artist_candidates = (
                artist_candidates
                .merge(existing_candidates, on=["pid", "track_uri"], how="left")
                .query("in_mf.isna()")
                .drop(columns=["in_mf"])
            )

            artist_candidates = self.mf_model.score_candidates(
                playlist_contents,
                artist_candidates[["pid", "track_uri"]]
            )
            artist_candidates["cooccurrence_score"] = 0
            candidates = pd.concat([candidates, artist_candidates], ignore_index=True)

            candidates = self.add_cooccurrence_candidates(candidates, playlist_contents, feature_contents, 500)

        return candidates

    def add_cooccurrence_candidates(self, candidates, playlist_contents, feature_contents, n_recs):
        print("Generating co-occurrence candidates...")

        playlist_sizes = (
            playlist_contents.groupby("pid")
            .size()
            .reset_index(name="playlist_len")
        )
        CANDIDATES_PER_TRACK = 10
        playlist_sizes["total_budget"] = np.minimum(
            n_recs,
            playlist_sizes["playlist_len"] * CANDIDATES_PER_TRACK
        )
        playlist_sizes["k"] = np.maximum(
            1,
            (playlist_sizes["total_budget"] // playlist_sizes["playlist_len"]).astype(int)
        )
        k_max = int(playlist_sizes["k"].max())

        if self.cooc_index is None:
            self.build_cooc_index(feature_contents, playlist_contents, k_max)

        # Look up top-k co-occurring tracks for each (pid, seed_track)
        seed_tracks = (
            playlist_contents[["pid", "track_uri"]]
            .merge(playlist_sizes[["pid", "k"]], on="pid")
        )

        cooc_rows = []
        for row in seed_tracks.itertuples(index=False):
            entries = self.cooc_index.get(row.track_uri)
            if entries is None or len(entries) == 0:
                continue
            for cooc_uri, count in entries[:row.k]:
                cooc_rows.append((row.pid, int(cooc_uri), float(count)))

        if not cooc_rows:
            return candidates

        playlist_cooc = pd.DataFrame(cooc_rows, columns=["pid", "track_uri", "cooc_count"])

        # Aggregate: sum co-occurrence counts across all seed tracks per playlist
        playlist_cooc_agg = (
            playlist_cooc.groupby(["pid", "track_uri"])["cooc_count"]
            .sum()
            .reset_index(name="cooccurrence_score")
        )

        in_playlist = playlist_contents[["pid", "track_uri"]].assign(in_playlist=True)
        playlist_cooc_agg = (
            playlist_cooc_agg
            .merge(in_playlist, on=["pid", "track_uri"], how="left")
            .query("in_playlist.isna()")
            .drop(columns=["in_playlist"])
        )

        # Keep only the top n_recs co-occurrence candidates per playlist
        playlist_cooc_agg["_rank"] = (
            playlist_cooc_agg.groupby("pid")["cooccurrence_score"]
            .rank(ascending=False, method="first")
            .astype(int)
        )
        top_cooc = (
            playlist_cooc_agg[playlist_cooc_agg["_rank"] <= n_recs]
            .drop(columns=["_rank"])
        )

        existing = candidates[["pid", "track_uri"]].assign(in_existing=True)
        new_cooc = (
            top_cooc
            .merge(existing, on=["pid", "track_uri"], how="left")
            .query("in_existing.isna()")
            .drop(columns=["in_existing"])
        )

        if not new_cooc.empty:
            new_cooc_scored = self.mf_model.score_candidates(
                playlist_contents,
                new_cooc[["pid", "track_uri"]]
            )
            new_cooc_scored = new_cooc_scored.merge(
                new_cooc[["pid", "track_uri", "cooccurrence_score"]],
                on=["pid", "track_uri"],
                how="left"
            )
            new_cooc_scored["cooccurrence_score"] = new_cooc_scored["cooccurrence_score"].fillna(0)
            candidates = pd.concat([candidates, new_cooc_scored], ignore_index=True)

        candidates = candidates.merge(
            top_cooc[["pid", "track_uri", "cooccurrence_score"]].rename(
                columns={"cooccurrence_score": "_cooc_update"}
            ),
            on=["pid", "track_uri"],
            how="left"
        )
        candidates["cooccurrence_score"] = candidates["cooccurrence_score"].where(
            candidates["_cooc_update"].isna(), candidates["_cooc_update"]
        )
        candidates.drop(columns=["_cooc_update"], inplace=True)

        return candidates

    def build_features(self, candidates, playlist_metadata, playlist_contents, track_metadata, feature_contents):
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

        playlist_contents_enriched = (
            playlist_contents[["pid", "track_uri"]]
            .merge(track_metadata[["track_uri", "artist_uri", "album_uri", "duration_ms"]], on="track_uri")
        )
        feature_contents_enriched = (
            feature_contents[["pid", "track_uri"]]
            .merge(track_metadata[["track_uri", "artist_uri", "album_uri"]], on="track_uri")
        )

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

        playlist_duration_stats = (
            playlist_contents_enriched
            .groupby("pid")["duration_ms"]
            .agg(
                playlist_duration_mean="mean",
                playlist_duration_std="std"
            )
            .reset_index()
        )

        candidates = candidates.merge(playlist_duration_stats, on="pid", how="left")
        candidates = candidates.merge(track_metadata[["track_uri", "duration_ms"]], on="track_uri", how="left")
        candidates["duration_zscore"] = (
            (candidates["duration_ms"] - candidates["playlist_duration_mean"])
            / candidates["playlist_duration_std"]
        )
        candidates["duration_zscore"] = candidates["duration_zscore"].replace([np.inf, -np.inf], 0).fillna(0)

        # Title features
        candidates = candidates.merge(self.title_model.score_tracks(playlist_metadata, candidates), on=['pid', 'track_uri'])

        return candidates

    def train(self, playlist_metadata, playlist_contents, playlist_holdouts, track_metadata, feature_playlist_metadata, feature_playlist_contents):
        if not self.mf_model.trained:
            raise RuntimeError("MF model not trained yet")

        candidates = self.generate_candidates(playlist_metadata, playlist_contents, track_metadata, feature_playlist_contents)

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
        candidates = self.generate_candidates(playlist_metadata, playlist_contents, track_metadata, feature_playlist_contents, n_recs=n_recs)
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