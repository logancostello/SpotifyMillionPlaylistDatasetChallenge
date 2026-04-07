from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, diags
import numpy as np
import pandas as pd
import faiss


class TitleTFIDFModel:

    def __init__(self, min_playlist_count=2, max_features=500):
        self.min_playlist_count = min_playlist_count
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features, min_df=2)
        self.name = "Title TF-IDF Model"
        self.index = None
        self.track_uris = None
        self.trained = False

    def train(self, playlist_metadata, playlist_contents, track_metadata):
        # fit and transform — stays sparse
        playlist_vectors = self.vectorizer.fit_transform(playlist_metadata['name'])

        # map pid and track_uri to integer indices
        pid_to_idx = {pid: i for i, pid in enumerate(playlist_metadata['pid'])}

        # filter by min playlist count
        track_counts = playlist_contents['track_uri'].value_counts()
        valid_tracks = set(track_counts[track_counts >= self.min_playlist_count].index)
        filtered = playlist_contents[playlist_contents['track_uri'].isin(valid_tracks)]

        track_uris = filtered['track_uri'].unique()
        track_to_idx = {t: i for i, t in enumerate(track_uris)}

        # build a (n_tracks x n_playlists) membership matrix
        row = filtered['track_uri'].map(track_to_idx).values
        col = filtered['pid'].map(pid_to_idx).values
        membership = csr_matrix(
            (np.ones(len(filtered)), (row, col)),
            shape=(len(track_uris), len(playlist_metadata))
        )

        # normalize rows to get mean playlist vector per track
        row_sums = np.asarray(membership.sum(axis=1)).flatten()
        inv_counts = 1.0 / np.where(row_sums > 0, row_sums, 1)
        membership = diags(inv_counts) @ membership

        # track_vectors is (n_tracks x vocab)
        track_vectors = (membership @ playlist_vectors).toarray().astype('float32')

        # build faiss index
        faiss.normalize_L2(track_vectors)
        self.index = faiss.IndexFlatIP(track_vectors.shape[1])
        self.index.add(track_vectors)
        self.track_uris = track_uris
        self.trained = True

    def predict(self, playlist_metadata, playlist_contents, track_metadata, n_recs, g_num):
        query_vectors = self.vectorizer.transform(playlist_metadata['name']).toarray().astype('float32')
        faiss.normalize_L2(query_vectors)

        distances, indices = self.index.search(query_vectors, n_recs * 2)

        results = []
        for i, pid in enumerate(playlist_metadata['pid']):
            existing_tracks = set(
                playlist_contents.loc[playlist_contents['pid'] == pid, 'track_uri']
            )

            rank = 0
            for idx in indices[i]:
                if idx == -1:
                    continue
                track_uri = self.track_uris[idx]
                if track_uri in existing_tracks:
                    continue
                results.append({'pid': pid, 'track_uri': track_uri, 'prediction_num': rank})
                rank += 1
                if rank >= n_recs:
                    break

        return pd.DataFrame(results)