import pandas as pd
import sys

from models.RandomModel import RandomModel
from models.GlobalPopularityModel import GlobalPopularityModel
from models.ArtistPopularityModel import ArtistPopularityModel
from models.TitleEmbeddingModel import TitleEmbeddingModel
from models.ArtistAndTitleModel import ArtistAndTitleModel
from models.MFModel import MFModel

from evaluation_funcs import compute_all_metrics, check_rules

if len(sys.argv) != 2 or (sys.argv[1] != "all" and int(sys.argv[1]) not in range(0, 11)):
    print("usage: python evaluate_model.py [0-10 or 'all']")
    sys.exit()

groups = list(range(11)) if sys.argv[1] == "all" else [int(sys.argv[1])]

test_sets = []
train_sets = []
for i in groups:
    test_sets.append({
        'group': i,
        'playlist_metadata': pd.read_parquet(f"data/test/{i}/playlist_metadata.parquet"),
        'playlist_contents': pd.read_parquet(f"data/test/{i}/playlist_contents.parquet"),
        'holdout_contents':  pd.read_parquet(f"data/test/{i}/holdout_contents.parquet"),
    })
    train_sets.append({
        'group': i,
        'playlist_metadata': pd.read_parquet(f"data/train/{i}/playlist_metadata.parquet"),
        'playlist_contents': pd.read_parquet(f"data/train/{i}/playlist_contents.parquet"),
        'holdout_contents':  pd.read_parquet(f"data/train/{i}/holdout_contents.parquet"),
    })

all_track_uris = pd.concat([
    *[ts['playlist_contents']['track_uri'] for ts in test_sets],
    *[ts['playlist_contents']['track_uri'] for ts in train_sets],
    *[ts['holdout_contents']['track_uri']  for ts in test_sets],
    *[ts['holdout_contents']['track_uri']  for ts in train_sets],
]).unique()

track_metadata = pd.read_parquet("data/original/track_metadata.parquet")
track_metadata = track_metadata[track_metadata["track_uri"].isin(all_track_uris)]

train_playlist_metadata = pd.concat([ts['playlist_metadata'] for ts in train_sets], ignore_index=True)
train_playlist_contents = pd.concat([ts['playlist_contents'] for ts in train_sets], ignore_index=True)
train_playlist_holdouts = pd.concat([ts['holdout_contents']  for ts in train_sets], ignore_index=True)

# ── Candidate model training data ─────────────────────────────────────────────
# All playlists: full contents for train, seed-only contents for test
original_metadata = pd.read_parquet("data/original/playlist_metadata.parquet")
original_contents = pd.read_parquet("data/original/playlist_contents.parquet")

test_pids = pd.concat([ts['playlist_metadata']['pid'] for ts in test_sets]).unique()

candidate_train_metadata = pd.concat([
    original_metadata[~original_metadata["pid"].isin(test_pids)],
    *[ts['playlist_metadata'] for ts in test_sets],
], ignore_index=True)

candidate_train_contents = pd.concat([
    original_contents[~original_contents["pid"].isin(test_pids)],
    *[ts['playlist_contents'] for ts in test_sets],   # seed-only for test playlists
], ignore_index=True)

# ── Ranker training data ───────────────────────────────────────────────────────
# Train splits (seeds + holdouts as labels) + test seeds (no holdouts)
ranker_train_metadata = pd.concat([
    train_playlist_metadata,
    *[ts['playlist_metadata'] for ts in test_sets],
], ignore_index=True)

ranker_train_contents = pd.concat([
    train_playlist_contents,
    train_playlist_holdouts,
    *[ts['playlist_contents'] for ts in test_sets],
], ignore_index=True)

all_contents = pd.concat([
    train_playlist_contents,
    train_playlist_holdouts,
    *[ts['playlist_contents'] for ts in test_sets]
], ignore_index=True)

all_metadata = pd.concat([
    train_playlist_metadata,
    *[ts['playlist_metadata'] for ts in test_sets]
], ignore_index=True)

# Seed-only contents for rankers (holdouts kept as labels)
all_seed_contents = pd.concat([
    train_playlist_contents,
    *[ts['playlist_contents'] for ts in test_sets]
], ignore_index=True)


group_names = {
    0: "No title, no tracks (baseline)",
    1: "Title only (no tracks)",
    2: "Title and first track",
    3: "Title and first 5 tracks",
    4: "First 5 tracks only",
    5: "Title and first 10 tracks",
    6: "First 10 tracks only",
    7: "Title and first 25 tracks",
    8: "Title and 25 random tracks",
    9: "Title and first 100 tracks",
    10: "Title and 100 random tracks",
}

global_pop_model = GlobalPopularityModel()
artist_pop_model = ArtistPopularityModel()
title_embedding_model = TitleEmbeddingModel()
artist_title_model = ArtistAndTitleModel(artist_pop_model, title_embedding_model)
mf_model = MFModel()

models = [
    global_pop_model,
    artist_pop_model,
    title_embedding_model,
    artist_title_model,
    mf_model
]

# Store results for CSV output
results = []

for model in models:
    print(f"\n{'='*50}")
    print(f"MODEL: {model.name}")
    print(f"{'='*50}")
    
    # Train once
    if model.is_ranker:
        model.train(ranker_train_metadata, ranker_train_contents, train_playlist_holdouts, track_metadata)
    else:
        model.train(candidate_train_metadata, candidate_train_contents, track_metadata)

    
    # Track metrics for averaging
    all_r_prec = []
    all_ndcg = []
    all_clicks = []
    
    # Dictionary to store this model's results
    model_results = {'Model': model.name}
    
    # Evaluate on each test set
    for test_set in test_sets:
        if len(test_sets) > 1:
            print(f"\n--- Group {test_set['group']}: {group_names[test_set['group']]} ---")
        
        prediction_df = model.predict(
            test_set['playlist_metadata'], 
            test_set['playlist_contents'], 
            track_metadata,
            500,
            test_set['group']
        )

        check_rules(prediction_df, test_set["playlist_contents"])
        
        r_prec, ndcg, clicks = compute_all_metrics(
            prediction_df, 
            test_set['holdout_contents'], 
            test_set['playlist_metadata'],
            track_metadata
        )
        
        if test_set['group'] > 0:
            all_r_prec.append(r_prec)
            all_ndcg.append(ndcg)
            all_clicks.append(clicks)
        
        # Store in results dictionary
        group_num = test_set['group']
        model_results[f'G{group_num}_R-Prec'] = r_prec
        model_results[f'G{group_num}_NDCG'] = ndcg
        model_results[f'G{group_num}_Clicks'] = clicks
        
        print(f"R-Precision: {r_prec:.3f}")
        print(f"NDCG: {ndcg:.4f}")
        print(f"Clicks: {clicks:.3f}")

    # Print averages if evaluating on all groups
    if len(test_sets) > 1:
        avg_r_prec = sum(all_r_prec)/len(all_r_prec)
        avg_ndcg = sum(all_ndcg)/len(all_ndcg)
        avg_clicks = sum(all_clicks)/len(all_clicks)
        
        model_results['Avg_R-Prec'] = avg_r_prec
        model_results['Avg_NDCG'] = avg_ndcg
        model_results['Avg_Clicks'] = avg_clicks
        
        print(f"\n--- Group Average ---")
        print(f"R-Precision: {avg_r_prec:.3f}")
        print(f"NDCG: {avg_ndcg:.4f}")
        print(f"Clicks: {avg_clicks:.3f}")
    
    results.append(model_results)

# Create DataFrame and save to CSV if evaluating all groups
if len(test_sets) > 1:
    results_df = pd.DataFrame(results)
    
    # Reorder columns: Model, then all G#_R-Prec, then all G#_NDCG, then all G#_Clicks, then averages
    col_order = ['Model']
    for i in range(1, 11):
        col_order.append(f'G{i}_R-Prec')
    for i in range(1, 11):
        col_order.append(f'G{i}_NDCG')
    for i in range(1, 11):
        col_order.append(f'G{i}_Clicks')
    col_order.extend(['Avg_R-Prec', 'Avg_NDCG', 'Avg_Clicks'])
    
    results_df = results_df[col_order]
    results_df.to_csv('model_evaluation_results.csv', index=False)
    print(f"\n{'='*50}")
    print("Results saved to model_evaluation_results.csv")
    print(f"{'='*50}")