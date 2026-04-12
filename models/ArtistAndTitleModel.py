import pandas as pd

GROUP_TITLE_WEIGHTS = {
    0: 0.0,
    1: 1.0,
    2: 0.75,
    3: 0.7,
    4: 0.0,
    5: 0.5,
    6: 0.0,
    7: 0.5,
    8: 0.3,
    9: 0.5,
    10: 0.3,
}

NUM_EXTRA_EACH = 1500

class ArtistAndTitleModel:
    
    def __init__(self, artist_model, title_model):
        self.name = "Artist Title Ensemble Model"
        self.artist_model = artist_model
        self.title_model = title_model
        self.trained = False
        self.results = {}

    def train(self, playlist_metadata, playlist_contents, playlist_holdouts, track_metadata):
        if not self.artist_model.trained:
            self.artist_model.train(playlist_metadata, playlist_contents, track_metadata)
        if not self.title_model.trained:
            self.title_model.train(playlist_metadata, playlist_contents, track_metadata)
        self.trained = True

    def predict(self, playlist_metadata, playlist_contents, track_metadata, n_recs, g_num):
        title_weight = GROUP_TITLE_WEIGHTS[g_num]
        artist_weight = 1 - title_weight
        k = 60

        artist_preds = self.artist_model.predict(playlist_metadata, playlist_contents, track_metadata, NUM_EXTRA_EACH, g_num)
        artist_preds["score"] = artist_weight / (k + artist_preds["prediction_num"])

        if title_weight == 0.0:
            combined = artist_preds
        else:
            title_preds = self.title_model.predict(playlist_metadata, playlist_contents, track_metadata, NUM_EXTRA_EACH, g_num)
            title_preds["score"] = title_weight / (k + title_preds["prediction_num"])
            combined = pd.concat([title_preds, artist_preds])

        fused = combined.groupby(["pid", "track_uri"])["score"].sum().reset_index()
        fused["prediction_num"] = (
            fused.groupby("pid")["score"]
            .rank(ascending=False, method="first")
            .astype(int) - 1
        )

        return (
            fused[fused["prediction_num"] < n_recs][["pid", "prediction_num", "track_uri"]]
            .sort_values(["pid", "prediction_num"])
            .reset_index(drop=True)
        )
