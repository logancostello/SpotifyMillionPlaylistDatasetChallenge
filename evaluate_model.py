import pandas as pd

from models.RandomModel import RandomModel

from evaluation_funcs import compute_r_prec, compute_clicks, compute_ndcg

playlist_metadata = pd.read_parquet("data/original/playlist_metadata.parquet")
playlist_contents = pd.read_parquet("data/original/playlist_contents.parquet")
track_metadata = pd.read_parquet("data/original/track_metadata.parquet")

test_playlist_metadata = pd.read_parquet("data/test/playlist_metadata.parquet")
test_playlist_contents = pd.read_parquet("data/test/playlist_contents.parquet")
holdout_contents = pd.read_parquet("data/test/holdout_contents.parquet")

train_playlist_metadata = playlist_metadata[~playlist_metadata["pid"].isin(test_playlist_metadata["pid"])]
train_playlist_contents = playlist_contents.merge(train_playlist_metadata[["pid"]], on="pid", how="inner")

model = RandomModel()

model.train(train_playlist_metadata, train_playlist_contents, track_metadata)

prediction_df = model.predict(test_playlist_metadata, test_playlist_contents, track_metadata)

r_prec = compute_r_prec(prediction_df, holdout_contents, test_playlist_metadata)
ndcg = compute_ndcg(prediction_df, holdout_contents, test_playlist_metadata)
clicks = compute_clicks(prediction_df, holdout_contents, test_playlist_metadata)

print(f"R-Precision: {r_prec}")
print(f"NDCG: {ndcg}")
print(f"Clicks: {clicks}")
