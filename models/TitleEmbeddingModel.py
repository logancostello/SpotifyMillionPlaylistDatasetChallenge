import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class TitleEmbeddingModel:

    def __init__(self, n_recommendations=500, top_k_playlists=50, predict_chunk_size=500):
        self.name = "Title Embedding Model"
        self.n_recommendations = n_recommendations
        self.top_k_playlists = top_k_playlists
        self.predict_chunk_size = predict_chunk_size
        self.train_pids = None
        self.train_matrix = None
        self.pid_to_tracks = None
        self.trained = False

    def train(self, playlist_metadata, playlist_contents, track_metadata):
        """
        Store training playlist embeddings and their track lists.
        No track-space averaging — everything stays in playlist space.
        """
        self.pid_to_tracks = (
            playlist_contents.groupby("pid")["track_uri"]
            .apply(list)
            .to_dict()
        )

        train = playlist_metadata[playlist_metadata["pid"].isin(self.pid_to_tracks)].copy()
        train = train.dropna(subset=["title_bert_embeddings"])

        self.train_pids = train["pid"].to_numpy()
        self.train_matrix = np.stack(train["title_bert_embeddings"].to_numpy())
        print(f"Stored {len(self.train_pids)} training playlists. Matrix shape: {self.train_matrix.shape}")
        self.trained = True

    def _get_recommendations_for_playlist(self, pid, playlist_scores, already_in_playlist):
        """
        Accumulate weighted track scores from top-k playlists, expanding k
        until we have enough recommendations or exhaust all training playlists.
        """
        max_k = len(self.train_pids)
        sorted_indices = np.argsort(playlist_scores)[::-1]
        k = min(self.top_k_playlists, max_k)

        while True:
            top_k_idx = sorted_indices[:k]

            track_scores = {}
            for idx in top_k_idx:
                weight = playlist_scores[idx]
                for track_uri in self.pid_to_tracks[self.train_pids[idx]]:
                    if track_uri not in already_in_playlist:
                        track_scores[track_uri] = track_scores.get(track_uri, 0.0) + weight

            if len(track_scores) >= self.n_recommendations or k >= max_k:
                break

            k = min(k * 2, max_k)

        return sorted(track_scores, key=track_scores.__getitem__, reverse=True)[:self.n_recommendations]

    def predict(self, playlist_metadata, playlist_contents, track_metadata):
        """
        For each test playlist:
          1. Find the top-k most similar training playlists by title embedding cosine similarity.
          2. Aggregate their tracks weighted by similarity score.
          3. Return the highest scoring tracks not already in the playlist,
             expanding k as needed to guarantee n_recommendations results.
        """
        existing_tracks = (
            playlist_contents.groupby("pid")["track_uri"]
            .apply(set)
            .to_dict()
        )

        pids = playlist_metadata["pid"].to_numpy()
        query_matrix = np.stack(playlist_metadata["title_bert_embeddings"].to_numpy())

        rows = []
        for chunk_start in range(0, len(pids), self.predict_chunk_size):
            chunk_pids = pids[chunk_start: chunk_start + self.predict_chunk_size]
            chunk_queries = query_matrix[chunk_start: chunk_start + self.predict_chunk_size]

            scores = cosine_similarity(chunk_queries, self.train_matrix)

            for pid, playlist_scores in zip(chunk_pids, scores):
                already_in_playlist = existing_tracks.get(pid, set())
                ranked_uris = self._get_recommendations_for_playlist(pid, playlist_scores, already_in_playlist)

                for prediction_num, track_uri in enumerate(ranked_uris):
                    rows.append((pid, prediction_num, track_uri))

        return pd.DataFrame(rows, columns=["pid", "prediction_num", "track_uri"])