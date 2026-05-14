import implicit
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pickle
import os


class MFModel:

    SAVE_DIR = "saved_models/mf_model"

    def __init__(self):
        self.name             = "Matrix Factorization Model"
        self.is_ranker        = False
        self.model            = implicit.als.AlternatingLeastSquares(
            factors=128,
            iterations=10,
            regularization=0.0025,
            num_threads=4,
            random_state=42,
        )
        self.pid_to_idx       = None
        self.track_uri_to_idx = None
        self.idx_to_track_uri = None
        self.interactions     = None
        self.trained          = False

    def _save_exists(self):
        return (
            os.path.exists(f"{self.SAVE_DIR}/model.npz") and
            os.path.exists(f"{self.SAVE_DIR}/meta.pkl")
        )

    def save(self):
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        sp.save_npz(f"{self.SAVE_DIR}/model.npz", self.interactions)
        with open(f"{self.SAVE_DIR}/meta.pkl", "wb") as f:
            pickle.dump({
                "pid_to_idx":       self.pid_to_idx,
                "track_uri_to_idx": self.track_uri_to_idx,
                "idx_to_track_uri": self.idx_to_track_uri,
                "item_factors":     self.model.item_factors,
                "user_factors":     self.model.user_factors,
            }, f)
        print(f"MF model saved to {self.SAVE_DIR}/")

    def load(self):
        self.interactions = sp.load_npz(f"{self.SAVE_DIR}/model.npz")
        with open(f"{self.SAVE_DIR}/meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self.pid_to_idx       = meta["pid_to_idx"]
        self.track_uri_to_idx = meta["track_uri_to_idx"]
        self.idx_to_track_uri = meta["idx_to_track_uri"]
        self.model.item_factors = meta["item_factors"]
        self.model.user_factors = meta["user_factors"]
        self.trained = True
        print(f"MF model loaded from {self.SAVE_DIR}/")

    def _build_interaction_matrix(self, playlist_contents):
        pids       = playlist_contents["pid"].unique()
        track_uris = playlist_contents["track_uri"].unique()

        self.pid_to_idx       = {pid: idx for idx, pid in enumerate(pids)}
        self.track_uri_to_idx = {uri: idx for idx, uri in enumerate(track_uris)}
        self.idx_to_track_uri = {idx: uri for uri, idx in self.track_uri_to_idx.items()}

        row  = playlist_contents["pid"].map(self.pid_to_idx).values
        col  = playlist_contents["track_uri"].map(self.track_uri_to_idx).values
        data = np.ones(len(playlist_contents))

        return sp.csr_matrix((data, (row, col)), shape=(len(pids), len(track_uris)))

    def train(self, playlist_metadata, playlist_contents, track_metadata):
        if self._save_exists():
            print("Save file found — loading MF model instead of retraining.")
            self.load()
            return

        self.interactions = self._build_interaction_matrix(playlist_contents)

        alpha = 50
        weighted = self.interactions.copy()
        weighted.data = 1.0 + alpha * weighted.data

        print(f"Fitting ALS on {self.interactions.shape[0]:,} playlists x {self.interactions.shape[1]:,} tracks...")
        self.model.fit(weighted)
        self.trained = True
        self.save()

    def fold_in_user(self, track_uris_in_playlist):
        indices = [
            self.track_uri_to_idx[uri]
            for uri in track_uris_in_playlist
            if uri in self.track_uri_to_idx
        ]
        if not indices:
            return None
        return self.model.item_factors[indices].mean(axis=0)

    def predict(self, playlist_metadata, playlist_contents, track_metadata, n_recs, g_num):
        pids = playlist_metadata["pid"].values

        warm_pids = [pid for pid in pids if pid in self.pid_to_idx]
        cold_pids = [pid for pid in pids if pid not in self.pid_to_idx]

        dfs = []

        if warm_pids:
            user_indices = [self.pid_to_idx[pid] for pid in warm_pids]
            all_item_ids, all_scores = self.model.recommend(
                userid=user_indices,
                user_items=self.interactions[user_indices],
                N=n_recs,
                filter_already_liked_items=True,
            )
            track_uris = np.vectorize(self.idx_to_track_uri.get)(all_item_ids.ravel())
            dfs.append(pd.DataFrame({
                "pid":            np.repeat(warm_pids, n_recs),
                "prediction_num": np.tile(np.arange(n_recs), len(warm_pids)),
                "track_uri":      track_uris,
                "mf_score":       all_scores.ravel(),
            }))

        if cold_pids:
            cold_contents = playlist_contents[playlist_contents["pid"].isin(cold_pids)]

            for pid in cold_pids:
                seed_uris = cold_contents[cold_contents["pid"] == pid]["track_uri"].tolist()
                user_vec = self.fold_in_user(seed_uris)

                if user_vec is None:
                    # Truly empty playlist — fall back to global popularity
                    global_scores = np.array(self.interactions.sum(axis=0)).flatten()
                    top_global = np.argsort(global_scores)[::-1][:n_recs]
                    top_uris = np.vectorize(self.idx_to_track_uri.get)(top_global)
                    top_scores = global_scores[top_global]
                    dfs.append(pd.DataFrame({
                        "pid":            [pid] * n_recs,
                        "prediction_num": np.arange(n_recs),
                        "track_uri":      top_uris,
                        "mf_score":       top_scores,
                    }))
                else:
                    scores = self.model.item_factors @ user_vec
                    seed_indices = {
                        self.track_uri_to_idx[uri]
                        for uri in seed_uris
                        if uri in self.track_uri_to_idx
                    }
                    scores[list(seed_indices)] = -np.inf

                    top_indices = np.argsort(scores)[::-1][:n_recs]
                    top_uris = np.vectorize(self.idx_to_track_uri.get)(top_indices)
                    dfs.append(pd.DataFrame({
                        "pid":            [pid] * n_recs,
                        "prediction_num": np.arange(n_recs),
                        "track_uri":      top_uris,
                        "mf_score":       scores[top_indices],
                    }))

        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["pid", "prediction_num", "track_uri", "mf_score"])