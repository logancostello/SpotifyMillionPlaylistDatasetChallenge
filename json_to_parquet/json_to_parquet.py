import pandas as pd
import json
import os

playlist_metadata_df = pd.DataFrame(columns=["pid", "name", "num_tracks"])
playlist_contents_df = pd.DataFrame(columns=["pid", "track_uri", "position"])
track_metadata_df = pd.DataFrame(columns=["track_uri", "album_uri", "artist_uri", "track_name", "album_name", "artist_name", "duration_ms"])
seen_tracks = set()

def process_playlist(playlist):
    playlist_metadata = {}
    playlist_metadata["pid"] = playlist["pid"]
    playlist_metadata["name"] = playlist["name"]
    playlist_metadata["num_tracks"] = playlist["num_tracks"]

    print(playlist["pid"])

    playlist_contents = []
    tracks_metadata = []

    for track in playlist["tracks"]:
        track_uri = track["track_uri"].split(":")[2]

        playlist_content = {}
        playlist_content["pid"] = playlist["pid"]
        playlist_content["track_uri"] = track_uri
        playlist_content["position"] = track["pos"]
        playlist_contents.append(playlist_content)

        if track_uri not in seen_tracks:
            seen_tracks.add(track_uri)

            track_metadata = {}
            track_metadata["track_uri"] = track_uri
            track_metadata["album_uri"] = track["album_uri"].split(":")[2]
            track_metadata["artist_uri"] = track["artist_uri"].split(":")[2]
            track_metadata["track_name"] = track["track_name"]
            track_metadata["album_name"] = track["album_name"]
            track_metadata["artist_name"] = track["artist_name"]
            track_metadata["duration_ms"] = track["duration_ms"]
            tracks_metadata.append(track_metadata)

    return playlist_metadata, playlist_contents, tracks_metadata

path = "given_files/raw/data/"
filenames = os.listdir(path)

playlist_metadata_rows = []
track_metadata_rows = []
playlist_contents_rows = []

for filename in sorted(filenames):

    if not filename.startswith("mpd.slice.") or not filename.endswith(".json"):
        continue

    f = open(path + filename)
    js = f.read()
    f.close()
    mpd_slice = json.loads(js)

    for playlist in mpd_slice["playlists"]:
        playlist_metadata, playlist_contents, tracks_metadata = process_playlist(playlist)
        playlist_metadata_rows.append(playlist_metadata)
        track_metadata_rows += tracks_metadata
        playlist_contents_rows += playlist_contents

playlist_metadata_df = pd.concat([playlist_metadata_df, pd.DataFrame(playlist_metadata_rows)], ignore_index=True)
playlist_contents_df = pd.concat([playlist_contents_df, pd.DataFrame(playlist_contents_rows)], ignore_index=True)
track_metadata_df = pd.concat([track_metadata_df, pd.DataFrame(track_metadata_rows)], ignore_index=True)

playlist_metadata_df.to_parquet("data/train/playlist_metadata.parquet")
track_metadata_df.to_parquet("data/train/track_metadata.parquet")
playlist_contents_df.to_parquet("data/train/playlist_contents.parquet")