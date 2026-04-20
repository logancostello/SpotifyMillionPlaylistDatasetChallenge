import sys
import pandas as pd
import math

NUM_TRAIN_PLAYLISTS = 5000
NUM_TEST_PLAYLISTS = 250
SEED = 123

def get_empty_embedding(playlist_metadata):
    return [0] * playlist_metadata["title_bert_embeddings"].apply(lambda x: len(x)).max()

# Title, 0 tracks
def create_test_set_1(playlist_metadata, playlist_contents, dir, n):
    test_playlists = playlist_metadata[(playlist_metadata["num_tracks"] >= 10) & (playlist_metadata["num_tracks"] <= 50)].sample(n, random_state=SEED+1)
    test_playlists["num_samples"] = 0
    test_playlists["has_title"] = True
    test_playlists["random_order"] = False
    test_playlists["num_holdouts"] = test_playlists["num_tracks"] - test_playlists["num_samples"]

    filtered_contents = test_playlists[["pid"]].merge(playlist_contents, on="pid", how="inner")
    test_contents = filtered_contents[filtered_contents["position"] < 0]
    holdout_contents = filtered_contents[filtered_contents["position"] >= 0]

    test_playlists.to_parquet(f"data/{dir}/1/playlist_metadata.parquet", index=False)
    test_contents.to_parquet(f"data/{dir}/1/playlist_contents.parquet", index=False)
    holdout_contents.to_parquet(f"data/{dir}/1/holdout_contents.parquet", index=False)

    return set(test_playlists["pid"])

# Title, 1 track, in order
def create_test_set_2(playlist_metadata, playlist_contents, dir, n):
    test_playlists = playlist_metadata[(playlist_metadata["num_tracks"] >= 10) & (playlist_metadata["num_tracks"] <= 39)].sample(n, random_state=SEED+2)
    test_playlists["num_samples"] = 1
    test_playlists["has_title"] = True
    test_playlists["random_order"] = False
    test_playlists["num_holdouts"] = test_playlists["num_tracks"] - test_playlists["num_samples"]

    filtered_contents = test_playlists[["pid"]].merge(playlist_contents, on="pid", how="inner")
    test_contents = filtered_contents[filtered_contents["position"] < 1]
    holdout_contents = filtered_contents[filtered_contents["position"] >= 1]

    test_playlists.to_parquet(f"data/{dir}/2/playlist_metadata.parquet", index=False)
    test_contents.to_parquet(f"data/{dir}/2/playlist_contents.parquet", index=False)
    holdout_contents.to_parquet(f"data/{dir}/2/holdout_contents.parquet", index=False)

    return set(test_playlists["pid"])

# Title, 5 tracks, in order
def create_test_set_3(playlist_metadata, playlist_contents, dir, n):
    first_half = playlist_metadata[(playlist_metadata["num_tracks"] >= 10) & (playlist_metadata["num_tracks"] < 50)].sample(n=math.floor(n * .360), random_state=SEED+3)
    second_half = playlist_metadata[(playlist_metadata["num_tracks"] >= 50) & (playlist_metadata["num_tracks"] <= 100)].sample(n=math.ceil(n * .640), random_state=SEED+4)
    test_playlists = pd.concat([first_half, second_half])
    test_playlists["num_samples"] = 5
    test_playlists["has_title"] = True
    test_playlists["random_order"] = False
    test_playlists["num_holdouts"] = test_playlists["num_tracks"] - test_playlists["num_samples"]

    filtered_contents = test_playlists[["pid"]].merge(playlist_contents, on="pid", how="inner")
    test_contents = filtered_contents[filtered_contents["position"] < 5]
    holdout_contents = filtered_contents[filtered_contents["position"] >= 5]

    test_playlists.to_parquet(f"data/{dir}/3/playlist_metadata.parquet", index=False)
    test_contents.to_parquet(f"data/{dir}/3/playlist_contents.parquet", index=False)
    holdout_contents.to_parquet(f"data/{dir}/3/holdout_contents.parquet", index=False)

    return set(test_playlists["pid"])

# No title, 5 tracks, in order
def create_test_set_4(playlist_metadata, playlist_content, dir, n):
    test_playlists = playlist_metadata[(playlist_metadata["num_tracks"] >= 40) & (playlist_metadata["num_tracks"] <= 100)].sample(n, random_state=SEED+5)
    test_playlists["name"] = ""
    test_playlists["num_samples"] = 5
    test_playlists["has_title"] = False
    test_playlists["random_order"] = False
    test_playlists["num_holdouts"] = test_playlists["num_tracks"] - test_playlists["num_samples"]
    empty_embedding = get_empty_embedding(test_playlists)
    test_playlists["title_bert_embeddings"] = [empty_embedding] * len(test_playlists)

    filtered_contents = test_playlists[["pid"]].merge(playlist_contents, on="pid", how="inner")
    test_contents = filtered_contents[filtered_contents["position"] < 5]
    holdout_contents = filtered_contents[filtered_contents["position"] >= 5]

    test_playlists.to_parquet(f"data/{dir}/4/playlist_metadata.parquet", index=False)
    test_contents.to_parquet(f"data/{dir}/4/playlist_contents.parquet", index=False)
    holdout_contents.to_parquet(f"data/{dir}/4/holdout_contents.parquet", index=False)

    return set(test_playlists["pid"])

# Title, 10 tracks, in order
def create_test_set_5(playlist_metadata, playlist_contents, dir, n):
    test_playlists = playlist_metadata[(playlist_metadata["num_tracks"] >= 40) & (playlist_metadata["num_tracks"] <= 100)].sample(n, random_state=SEED+6)
    test_playlists["num_samples"] = 10
    test_playlists["has_title"] = True
    test_playlists["random_order"] = False
    test_playlists["num_holdouts"] = test_playlists["num_tracks"] - test_playlists["num_samples"]

    filtered_contents = test_playlists[["pid"]].merge(playlist_contents, on="pid", how="inner")
    test_contents = filtered_contents[filtered_contents["position"] < 10]
    holdout_contents = filtered_contents[filtered_contents["position"] >= 10]

    test_playlists.to_parquet(f"data/{dir}/5/playlist_metadata.parquet", index=False)
    test_contents.to_parquet(f"data/{dir}/5/playlist_contents.parquet", index=False)
    holdout_contents.to_parquet(f"data/{dir}/5/holdout_contents.parquet", index=False)

    return set(test_playlists["pid"])

# No title, 10 tracks, in order
def create_test_set_6(playlist_metadata, playlist_contents, dir, n):
    test_playlists = playlist_metadata[(playlist_metadata["num_tracks"] >= 40) & (playlist_metadata["num_tracks"] <= 100)].sample(n, random_state=SEED+7)
    test_playlists["name"] = ""
    test_playlists["num_samples"] = 10
    test_playlists["has_title"] = False
    test_playlists["random_order"] = False
    test_playlists["num_holdouts"] = test_playlists["num_tracks"] - test_playlists["num_samples"]
    empty_embedding = get_empty_embedding(test_playlists)
    test_playlists["title_bert_embeddings"] = [empty_embedding] * len(test_playlists)

    filtered_contents = test_playlists[["pid"]].merge(playlist_contents, on="pid", how="inner")
    test_contents = filtered_contents[filtered_contents["position"] < 10]
    holdout_contents = filtered_contents[filtered_contents["position"] >= 10]

    test_playlists.to_parquet(f"data/{dir}/6/playlist_metadata.parquet", index=False)
    test_contents.to_parquet(f"data/{dir}/6/playlist_contents.parquet", index=False)
    holdout_contents.to_parquet(f"data/{dir}/6/holdout_contents.parquet", index=False)

    return set(test_playlists["pid"])

# Title, 25 tracks, in order
def create_test_set_7(playlist_metadata, playlist_contents, dir, n):
    test_playlists = playlist_metadata[playlist_metadata["num_tracks"] >= 101].sample(n, random_state=SEED+8)
    test_playlists["num_samples"] = 25
    test_playlists["has_title"] = True
    test_playlists["random_order"] = False
    test_playlists["num_holdouts"] = test_playlists["num_tracks"] - test_playlists["num_samples"]

    filtered_contents = test_playlists[["pid"]].merge(playlist_contents, on="pid", how="inner")
    test_contents = filtered_contents[filtered_contents["position"] < 25]
    holdout_contents = filtered_contents[filtered_contents["position"] >= 25]

    test_playlists.to_parquet(f"data/{dir}/7/playlist_metadata.parquet", index=False)
    test_contents.to_parquet(f"data/{dir}/7/playlist_contents.parquet", index=False)
    holdout_contents.to_parquet(f"data/{dir}/7/holdout_contents.parquet", index=False)

    return set(test_playlists["pid"])

# Title, 25 tracks, random_order
def create_test_set_8(playlist_metadata, playlist_contents, dir, n):
    test_playlists = playlist_metadata[playlist_metadata["num_tracks"] >= 101].sample(n, random_state=SEED+9)
    test_playlists["num_samples"] = 25
    test_playlists["has_title"] = True
    test_playlists["random_order"] = True
    test_playlists["num_holdouts"] = test_playlists["num_tracks"] - test_playlists["num_samples"]

    filtered_contents = test_playlists[["pid"]].merge(playlist_contents, on="pid", how="inner")
    test_contents = filtered_contents.groupby("pid").sample(n=25, random_state=SEED).sort_values(["pid", "position"])
    holdout_contents = filtered_contents[~filtered_contents.set_index(["pid", "track_uri"]).index.isin(test_contents.set_index(["pid", "track_uri"]).index)]

    test_playlists.to_parquet(f"data/{dir}/8/playlist_metadata.parquet", index=False)
    test_contents.to_parquet(f"data/{dir}/8/playlist_contents.parquet", index=False)
    holdout_contents.to_parquet(f"data/{dir}/8/holdout_contents.parquet", index=False)

    return set(test_playlists["pid"])

# Title, 100 tracks, in order
def create_test_set_9(playlist_metadata, playlist_contents, dir, n):
    test_playlists = playlist_metadata[playlist_metadata["num_tracks"] >= 150].sample(n, random_state=SEED+10)
    test_playlists["num_samples"] = 100
    test_playlists["has_title"] = True
    test_playlists["random_order"] = False
    test_playlists["num_holdouts"] = test_playlists["num_tracks"] - test_playlists["num_samples"]

    filtered_contents = test_playlists[["pid"]].merge(playlist_contents, on="pid", how="inner")
    test_contents = filtered_contents[filtered_contents["position"] < 100]
    holdout_contents = filtered_contents[filtered_contents["position"] >= 100]

    test_playlists.to_parquet(f"data/{dir}/9/playlist_metadata.parquet", index=False)
    test_contents.to_parquet(f"data/{dir}/9/playlist_contents.parquet", index=False)
    holdout_contents.to_parquet(f"data/{dir}/9/holdout_contents.parquet", index=False)

    return set(test_playlists["pid"])

# Title, 100 tracks, random_order
def create_test_set_10(playlist_metadata, playlist_contents, dir, n):
    test_playlists = playlist_metadata[playlist_metadata["num_tracks"] >= 150].sample(n, random_state=SEED+11)
    test_playlists["num_samples"] = 100
    test_playlists["has_title"] = True
    test_playlists["random_order"] = True
    test_playlists["num_holdouts"] = test_playlists["num_tracks"] - test_playlists["num_samples"]

    filtered_contents = test_playlists[["pid"]].merge(playlist_contents, on="pid", how="inner")
    test_contents = filtered_contents.groupby("pid").sample(n=100, random_state=SEED).sort_values(["pid", "position"])
    holdout_contents = filtered_contents[~filtered_contents.set_index(["pid", "track_uri"]).index.isin(test_contents.set_index(["pid", "track_uri"]).index)]

    test_playlists.to_parquet(f"data/{dir}/10/playlist_metadata.parquet", index=False)
    test_contents.to_parquet(f"data/{dir}/10/playlist_contents.parquet", index=False)
    holdout_contents.to_parquet(f"data/{dir}/10/holdout_contents.parquet", index=False)

    return set(test_playlists["pid"])

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("usage: python create_train_test_split.py [challenge group num]")
        sys.exit()

    if sys.argv[1] != "all" and int(sys.argv[1]) not in range(1, 11):
        print("must give num 1-10 or all")
        sys.exit()
        
    playlist_metadata = pd.read_parquet("data/original/playlist_metadata.parquet")
    playlist_contents = pd.read_parquet("data/original/playlist_contents.parquet")

    funcs = {
        "1": create_test_set_1,
        "2": create_test_set_2,
        "3": create_test_set_3,
        "4": create_test_set_4,
        "5": create_test_set_5,
        "6": create_test_set_6,
        "7": create_test_set_7,
        "8": create_test_set_8,
        "9": create_test_set_9,
        "10": create_test_set_10
    }

    if sys.argv[1] == "all":
        for i in range(1, 11):
            for dir, num in [("train", NUM_TRAIN_PLAYLISTS), ("test", NUM_TEST_PLAYLISTS)]:
                used_pids = funcs[str(i)](playlist_metadata, playlist_contents, dir, num)
                playlist_metadata = playlist_metadata[~playlist_metadata["pid"].isin(used_pids)]
    else:
        for dir, num in [("train", NUM_TRAIN_PLAYLISTS), ("test", NUM_TEST_PLAYLISTS)]:
            used_pids = funcs[sys.argv[1]](playlist_metadata, playlist_contents, dir, num)
            playlist_metadata = playlist_metadata[~playlist_metadata["pid"].isin(used_pids)]