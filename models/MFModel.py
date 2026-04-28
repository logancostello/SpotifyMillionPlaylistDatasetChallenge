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

        print(f"Fitting ALS on {self.interactions.shape[0]:,} playlists x {self.interactions.shape[1]:,} tracks...")
        self.model.fit(self.interactions)
        self.trained = True
        self.save()

    def predict(self, playlist_metadata, playlist_contents, track_metadata, n_recs, g_num):
        existing_tracks = (
            playlist_contents.groupby("pid")["track_uri"]
            .apply(set)
            .to_dict()
        )

        global_scores = np.array(self.interactions.sum(axis=0)).flatten()
        top_global    = np.argsort(global_scores)[::-1]

        rows = []
        for pid in playlist_metadata["pid"]:
            already_in_playlist = existing_tracks.get(pid, set())
            known_indices = [
                self.track_uri_to_idx[t]
                for t in already_in_playlist
                if t in self.track_uri_to_idx
            ]

            if not known_indices or pid not in self.pid_to_idx:
                for prediction_num, idx in enumerate(top_global[:n_recs]):
                    rows.append((pid, prediction_num, self.idx_to_track_uri[idx], float(global_scores[idx])))
                continue

            user_idx = self.pid_to_idx[pid]
            item_ids, scores = self.model.recommend(
                userid=user_idx,
                user_items=self.interactions[user_idx],
                N=n_recs,
                filter_already_liked_items=True,
            )

            for prediction_num, (idx, score) in enumerate(zip(item_ids, scores)):
                rows.append((pid, prediction_num, self.idx_to_track_uri[idx], float(score)))

        return pd.DataFrame(rows, columns=["pid", "prediction_num", "track_uri", "mf_score"])