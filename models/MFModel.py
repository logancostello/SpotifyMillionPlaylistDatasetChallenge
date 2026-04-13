import implicit
import numpy as np
import pandas as pd
import scipy.sparse as sp


class MFModel:

    def __init__(self):
        self.name = "Matrix Factorization Model"
        self.is_ranker=False
        self.model = implicit.als.AlternatingLeastSquares(
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

    def _build_interaction_matrix(self, playlist_contents):
        pids       = playlist_contents["pid"].unique()
        track_uris = playlist_contents["track_uri"].unique()

        self.pid_to_idx       = {pid: idx for idx, pid in enumerate(pids)}
        self.track_uri_to_idx = {uri: idx for idx, uri in enumerate(track_uris)}
        self.idx_to_track_uri = {idx: uri for uri, idx in self.track_uri_to_idx.items()}

        row = playlist_contents["pid"].map(self.pid_to_idx).values
        col = playlist_contents["track_uri"].map(self.track_uri_to_idx).values
        data = np.ones(len(playlist_contents))

        return sp.csr_matrix((data, (row, col)), shape=(len(pids), len(track_uris)))
    

    def train(self, playlist_metadata, playlist_contents, track_metadata):
        self.interactions = self._build_interaction_matrix(playlist_contents)

        print(f"Fitting ALS on {self.interactions.shape[0]:,} playlists x {self.interactions.shape[1]:,} tracks...")
        self.model.fit(self.interactions)
        self.trained = True

    def predict(self, playlist_metadata, playlist_contents, track_metadata, n_recs, g_num):
        existing_tracks = (
            playlist_contents.groupby("pid")["track_uri"]
            .apply(set)
            .to_dict()
        )

        item_factors = self.model.item_factors
        global_scores = np.array(self.interactions.sum(axis=0)).flatten()
        top_global = np.argsort(global_scores)[::-1]

        rows = []
        for pid in playlist_metadata["pid"]:
            already_in_playlist = existing_tracks.get(pid, set())
            known_indices = [
                self.track_uri_to_idx[t]
                for t in already_in_playlist
                if t in self.track_uri_to_idx
            ]

            if not known_indices:
                for prediction_num, idx in enumerate(top_global[:n_recs]):
                    rows.append((pid, prediction_num, self.idx_to_track_uri[idx]))
                continue

            if pid not in self.pid_to_idx:
                for prediction_num, idx in enumerate(top_global[:n_recs]):
                    rows.append((pid, prediction_num, self.idx_to_track_uri[idx]))
                continue

            user_idx = self.pid_to_idx[pid]
            item_ids, _ = self.model.recommend(
                userid=user_idx,
                user_items=self.interactions[user_idx],
                N=n_recs,
                filter_already_liked_items=True,
            )

            for prediction_num, idx in enumerate(item_ids):
                rows.append((pid, prediction_num, self.idx_to_track_uri[idx]))

        return pd.DataFrame(rows, columns=["pid", "prediction_num", "track_uri"])