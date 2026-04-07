import pandas as pd
import numpy as np

class RandomModel:

    def __init__(self):
        self.rng = np.random.default_rng(123)
        self.name = "Random Model"
        self.trained = True

    def train(self, playlist_metadata, playlist_contents, track_metadata):
        pass

    # Technically this breaks the rule of not predicting a song already in the playlist
    # but this model is random and purely a baseline
    def predict(self, playlist_metadata, playlist_contents, track_metadata, n_recs, g_num):

        # Pair each pid to n_recs positions
        prediction_num = range(0, n_recs)
        prediction_df = pd.DataFrame(prediction_num, columns=["prediction_num"])
        prediction_df = playlist_metadata[["pid"]].merge(prediction_df, how="cross")

        # Pick random track for each prediction
        track_uris = track_metadata["track_uri"].to_numpy()
        
        out = np.empty(len(prediction_df), dtype=track_uris.dtype)

        for _, idx in prediction_df.groupby("pid").indices.items():
            out[idx] = self.rng.choice(track_uris, size=len(idx), replace=False)

        prediction_df["track_uri"] = out

        return prediction_df