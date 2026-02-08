import pandas as pd
import sys

from models.RandomModel import RandomModel
from models.GlobalPopularityModel import GlobalPopularityModel
from models.ArtistPopularityModel import ArtistPopularityModel

from evaluation_funcs import compute_all_metrics

if len(sys.argv) != 2 or (sys.argv[1] != "all" and int(sys.argv[1]) not in range(1, 11)):
    print("usage: python evaluate_model.py [1-10 or 'all']")
    sys.exit()

# Load original data once
playlist_metadata = pd.read_parquet("data/original/playlist_metadata.parquet")
playlist_contents = pd.read_parquet("data/original/playlist_contents.parquet")
track_metadata = pd.read_parquet("data/original/track_metadata.parquet")

if sys.argv[1] == "all":
    # Load all 10 test sets
    test_sets = []
    for i in range(1, 11):
        test_sets.append({
            'group': i,
            'playlist_metadata': pd.read_parquet(f"data/test/{i}/playlist_metadata.parquet"),
            'playlist_contents': pd.read_parquet(f"data/test/{i}/playlist_contents.parquet"),
            'holdout_contents': pd.read_parquet(f"data/test/{i}/holdout_contents.parquet")
        })
    
    # Get all test PIDs from all groups
    all_test_pids = pd.concat([ts['playlist_metadata']['pid'] for ts in test_sets]).unique()
    
    # Create single training set excluding all test PIDs
    train_playlist_metadata = playlist_metadata[~playlist_metadata["pid"].isin(all_test_pids)]
    train_playlist_contents = playlist_contents.merge(train_playlist_metadata[["pid"]], on="pid", how="inner")
    
else:
    # Single group evaluation
    group = sys.argv[1]
    test_sets = [{
        'group': int(group),
        'playlist_metadata': pd.read_parquet(f"data/test/{group}/playlist_metadata.parquet"),
        'playlist_contents': pd.read_parquet(f"data/test/{group}/playlist_contents.parquet"),
        'holdout_contents': pd.read_parquet(f"data/test/{group}/holdout_contents.parquet")
    }]
    
    train_playlist_metadata = playlist_metadata[~playlist_metadata["pid"].isin(test_sets[0]['playlist_metadata']["pid"])]
    train_playlist_contents = playlist_contents.merge(train_playlist_metadata[["pid"]], on="pid", how="inner")

group_names = {
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
models = [
    RandomModel(),
    GlobalPopularityModel(),
    ArtistPopularityModel()
]

# Store results for CSV output
results = []

for model in models:
    print(f"\n{'='*50}")
    print(f"MODEL: {model.name}")
    print(f"{'='*50}")
    
    # Train once
    model.train(train_playlist_metadata, train_playlist_contents, track_metadata)
    
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
            track_metadata
        )
        
        r_prec, ndcg, clicks = compute_all_metrics(
            prediction_df, 
            test_set['holdout_contents'], 
            test_set['playlist_metadata']
        )
        
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