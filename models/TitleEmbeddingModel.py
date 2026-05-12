import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from collections import defaultdict
import faiss
import pickle
import os


class TitleEmbeddingModel:

    SAVE_DIR = "saved_models/title_embedding_model"

    def __init__(
        self,
        top_k_playlists=300,
        predict_chunk_size=500,
        approximate=False,
        n_cells=100,
        n_probe=10,
        similarity_threshold=0.15,
        weight_temperature=5,
    ):
        self.name                 = "Title Embedding Model"
        self.is_ranker            = False
        self.top_k_playlists      = top_k_playlists
        self.predict_chunk_size   = predict_chunk_size
        self.approximate          = approximate
        self.n_cells              = n_cells
        self.n_probe              = n_probe
        self.similarity_threshold = similarity_threshold
        self.weight_temperature   = weight_temperature
        self.train_pids           = None
        self.pid_to_tracks        = None
        self.global_ranking       = None
        self.index                = None
        self.trained              = False

    def _save_exists(self):
        return (
            os.path.exists(f"{self.SAVE_DIR}/train_pids.npy") and
            os.path.exists(f"{self.SAVE_DIR}/meta.pkl") and
            os.path.exists(f"{self.SAVE_DIR}/index.faiss")
        )

    def save(self):
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        np.save(f"{self.SAVE_DIR}/train_pids.npy", self.train_pids)
        faiss.write_index(self.index, f"{self.SAVE_DIR}/index.faiss")
        with open(f"{self.SAVE_DIR}/meta.pkl", "wb") as f:
            pickle.dump({
                "pid_to_tracks":  self.pid_to_tracks,
                "global_ranking": self.global_ranking,
            }, f)
        print(f"TitleEmbedding model saved to {self.SAVE_DIR}/")

    def load(self):
        self.train_pids = np.load(f"{self.SAVE_DIR}/train_pids.npy")
        self.index      = faiss.read_index(f"{self.SAVE_DIR}/index.faiss")
        if self.approximate:
            self.index.nprobe = self.n_probe
        with open(f"{self.SAVE_DIR}/meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self.pid_to_tracks  = meta["pid_to_tracks"]
        self.global_ranking = meta["global_ranking"]
        self.trained        = True
        print(f"TitleEmbedding model loaded from {self.SAVE_DIR}/")

    def train(self, playlist_metadata, playlist_contents, track_metadata):
        if self._save_exists():
            print("Save file found — loading TitleEmbedding model instead of retraining.")
            self.load()
            return

        self.pid_to_tracks = (
            playlist_contents.groupby("pid")["track_uri"]
            .apply(list)
            .to_dict()
        )

        track_freq = defaultdict(int)
        for tracks in self.pid_to_tracks.values():
            for t in tracks:
                track_freq[t] += 1
        self.global_ranking = sorted(track_freq, key=track_freq.__getitem__, reverse=True)

        train = playlist_metadata[
            playlist_metadata["pid"].isin(self.pid_to_tracks) &
            playlist_metadata["title_bert_embeddings"].notna()
        ].copy()

        self.train_pids = train["pid"].to_numpy()
        train_matrix    = normalize(
            np.stack(train["title_bert_embeddings"].to_numpy()), norm="l2"
        ).astype(np.float32)

        dim = train_matrix.shape[1]
        if self.approximate:
            quantizer  = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, self.n_cells, faiss.METRIC_INNER_PRODUCT)
            self.index.train(train_matrix)
        else:
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(train_matrix)
        print(f"Stored {len(self.train_pids):,} training playlists in faiss index (approximate={self.approximate})")
        self.trained = True
        self.save()

    def _score_tracks(self, sim_scores, neighbor_idx, already_in_playlist):
        # Apply threshold and temperature, then normalize
        pairs = [
            (weight ** self.weight_temperature, idx)
            for weight, idx in zip(sim_scores, neighbor_idx)
            if idx != -1 and weight >= self.similarity_threshold
        ]

        if not pairs:
            return {}

        total_weight = sum(w for w, _ in pairs)

        track_scores = defaultdict(float)
        for weight, idx in pairs:
            normalized_weight = weight / total_weight
            for track_uri in self.pid_to_tracks[self.train_pids[idx]]:
                if track_uri not in already_in_playlist:
                    track_scores[track_uri] += normalized_weight

        return track_scores

    def predict(self, playlist_metadata, playlist_contents, track_metadata, n_recs, g_num):
        if self.approximate:
            self.index.nprobe = self.n_probe

        existing_tracks = (
            playlist_contents.groupby("pid")["track_uri"]
            .apply(set)
            .to_dict()
        )

        pids         = playlist_metadata["pid"].to_numpy()
        query_matrix = normalize(
            np.stack(playlist_metadata["title_bert_embeddings"].to_numpy()), norm="l2"
        ).astype(np.float32)

        rows = []
        for chunk_start in range(0, len(pids), self.predict_chunk_size):
            chunk_pids    = pids[chunk_start: chunk_start + self.predict_chunk_size]
            chunk_queries = query_matrix[chunk_start: chunk_start + self.predict_chunk_size]

            k = self.top_k_playlists
            weights, top_k_idx = self.index.search(chunk_queries, k)

            for i, (pid, sim_scores, neighbor_idx) in enumerate(zip(chunk_pids, weights, top_k_idx)):
                already_in_playlist = existing_tracks.get(pid, set())

                while True:
                    track_scores = self._score_tracks(sim_scores, neighbor_idx, already_in_playlist)

                    if len(track_scores) >= n_recs or k >= len(self.train_pids):
                        break

                    k = min(k * 2, len(self.train_pids))
                    new_weights, new_idx = self.index.search(chunk_queries[i: i + 1], k)
                    sim_scores   = new_weights[0]
                    neighbor_idx = new_idx[0]

                ranked = sorted(track_scores, key=track_scores.__getitem__, reverse=True)[:n_recs]

                if len(ranked) < n_recs:
                    seen = set(ranked) | already_in_playlist
                    for track_uri in self.global_ranking:
                        if track_uri not in seen:
                            ranked.append(track_uri)
                            if len(ranked) == n_recs:
                                break

                for prediction_num, track_uri in enumerate(ranked):
                    rows.append((pid, prediction_num, track_uri))

        return pd.DataFrame(rows, columns=["pid", "prediction_num", "track_uri"])
    
    def score_tracks(self, playlist_metadata, pid_candidate_pairs):
        if isinstance(pid_candidate_pairs, pd.DataFrame):
            pairs_df = pid_candidate_pairs[["pid", "track_uri"]]
        else:
            pairs_df = pd.DataFrame(pid_candidate_pairs, columns=["pid", "track_uri"])

        # Build a set of candidates per pid for fast lookup
        pid_to_candidates = pairs_df.groupby("pid")["track_uri"].apply(set).to_dict()

        pids = playlist_metadata["pid"].to_numpy()
        query_matrix = normalize(
            np.stack(playlist_metadata["title_bert_embeddings"].to_numpy()), norm="l2"
        ).astype(np.float32)

        # Search all queries at once
        k = self.top_k_playlists
        weights, top_k_idx = self.index.search(query_matrix, k)

        rows = []
        for pid, sim_scores, neighbor_idx in zip(pids, weights, top_k_idx):
            candidates = pid_to_candidates.get(pid, set())
            if not candidates:
                continue

            # Filter to valid neighbors above threshold
            pairs = [
                (w ** self.weight_temperature, idx)
                for w, idx in zip(sim_scores, neighbor_idx)
                if idx != -1 and w >= self.similarity_threshold
            ]
            if not pairs:
                # No similar playlists found — all candidates get 0
                for track_uri in candidates:
                    rows.append((pid, track_uri, 0.0))
                continue

            total_weight = sum(w for w, _ in pairs)

            # Only accumulate scores for candidate tracks
            track_scores = defaultdict(float)
            for weight, idx in pairs:
                normalized_weight = weight / total_weight
                for track_uri in self.pid_to_tracks[self.train_pids[idx]]:
                    if track_uri in candidates:
                        track_scores[track_uri] += normalized_weight

            for track_uri in candidates:
                rows.append((pid, track_uri, track_scores.get(track_uri, 0.0)))

        return pd.DataFrame(rows, columns=["pid", "track_uri", "title_score"])