import duckdb as db
import numpy as np

def compute_r_prec(prediction_df, holdout_df, playlist_metadata, track_metadata):

    # For each playlist, only consider top-|GT| predictions (where |GT| = num_holdouts)
    # First, join predictions with num_holdouts to know the cutoff per playlist
    eval_base = db.sql("""
        SELECT p.pid, p.track_uri, p.prediction_num, pm.num_holdouts
        FROM prediction_df p
        JOIN playlist_metadata pm ON p.pid = pm.pid
        WHERE p.prediction_num < pm.num_holdouts
    """).df()

    # |ST ∩ GT|: track-level hits in top-|GT| predictions
    track_hits = db.sql("""
        SELECT h.pid, COUNT(h.track_uri) as num_track_hits
        FROM holdout_df h
        JOIN eval_base e ON h.pid = e.pid AND h.track_uri = e.track_uri
        GROUP BY h.pid
    """).df()

    # Get artist IDs for holdout tracks (GA) and top-|GT| predicted tracks (SA)
    # then compute |SA ∩ GA| minus tracks already counted as track hits
    artist_hits = db.sql("""
        WITH holdout_artists AS (
            SELECT DISTINCT h.pid, tm.artist_uri
            FROM holdout_df h
            JOIN track_metadata tm ON h.track_uri = tm.track_uri
        ),
        predicted_artists AS (
            SELECT DISTINCT e.pid, tm.artist_uri
            FROM eval_base e
            JOIN track_metadata tm ON e.track_uri = tm.track_uri
        ),
        -- Tracks in top-|GT| that are already track hits (exclude their artists from partial)
        track_hit_artists AS (
            SELECT DISTINCT e.pid, tm.artist_uri
            FROM eval_base e
            JOIN holdout_df h ON e.pid = h.pid AND e.track_uri = h.track_uri
            JOIN track_metadata tm ON e.track_uri = tm.track_uri
        )
        SELECT pa.pid, COUNT(*) as num_artist_hits
        FROM predicted_artists pa
        JOIN holdout_artists ha ON pa.pid = ha.pid AND pa.artist_uri = ha.artist_uri
        -- Exclude artist matches that came from exact track matches
        LEFT JOIN track_hit_artists tha ON pa.pid = tha.pid AND pa.artist_uri = tha.artist_uri
        WHERE tha.artist_uri IS NULL
        GROUP BY pa.pid
    """).df()

    eval_df = playlist_metadata \
        .merge(track_hits, on="pid", how="left") \
        .merge(artist_hits, on="pid", how="left")

    eval_df["num_track_hits"] = eval_df["num_track_hits"].fillna(0)
    eval_df["num_artist_hits"] = eval_df["num_artist_hits"].fillna(0)

    eval_df["r_prec"] = (
        eval_df["num_track_hits"] + 0.25 * eval_df["num_artist_hits"]
    ) / eval_df["num_holdouts"]

    return np.mean(eval_df["r_prec"])

def idcg(R, K=500):
    return sum(1 / np.log2(i + 2) for i in range(min(R, K)))


# Track level ndcg
def compute_ndcg(prediction_df, holdout_df, playlist_metadata):
    correct_predictions = db.sql("""
        SELECT h.pid, h.track_uri, p.prediction_num
        FROM holdout_df h
        JOIN prediction_df p ON h.pid = p.pid AND h.track_uri = p.track_uri
        WHERE p.prediction_num < 500
    """).df()

    dcg_df = db.sql("""
        SELECT pid,
            SUM(1 / LOG2(prediction_num + 2)) AS dcg
        FROM correct_predictions
        GROUP BY pid
    """).df()

    eval_df = playlist_metadata.merge(dcg_df, on="pid", how="left")
    eval_df["dcg"] = eval_df["dcg"].fillna(0)
    eval_df["idcg"] = eval_df["num_holdouts"].apply(idcg)
    ndcg = np.mean(eval_df["dcg"] / eval_df["idcg"])
    return ndcg

# Track level clicks
def compute_clicks(prediction_df, holdout_df, playlist_metadata):

    loc_first_correct = db.sql("""
        SELECT h.pid, MIN(p.prediction_num) as loc_first_correct
        FROM holdout_df h
        JOIN prediction_df p ON h.pid = p.pid AND h.track_uri = p.track_uri
        WHERE p.prediction_num < 500
        GROUP BY h.pid
    """).df()

    loc_first_correct["loc_first_correct"] = loc_first_correct["loc_first_correct"] // 10
    eval_df = playlist_metadata.merge(loc_first_correct, on="pid", how="left")
    eval_df["loc_first_correct"] = eval_df["loc_first_correct"].fillna(51)
    clicks = eval_df["loc_first_correct"].mean()
    return clicks

def compute_all_metrics(prediction_df, holdout_df, playlist_metadata, track_metadata):
    r_prec = compute_r_prec(prediction_df, holdout_df, playlist_metadata, track_metadata)
    ndcg = compute_ndcg(prediction_df, holdout_df, playlist_metadata)
    clicks = compute_clicks(prediction_df, holdout_df, playlist_metadata)

    return r_prec, ndcg, clicks

def check_rules(prediction_df, playlist_contents):

    num_recs = db.sql("""
        SELECT pid, COUNT(*)
        FROM prediction_df
        GROUP BY pid
        HAVING COUNT(*) != 500
    """).df()

    if len(num_recs) > 0:
        print("WARNING: wrong number of recommendations")

    duplicates = db.sql("""
        SELECT pid, track_uri, COUNT(*) as count
        FROM prediction_df
        GROUP BY pid, track_uri
        HAVING COUNT(*) >= 2
    """).df()

    if len(duplicates) > 0:
        print("WARNING: duplicate recommendations made")

    in_playlist_already = db.sql("""
        SELECT *
        FROM prediction_df p
        JOIN playlist_contents c ON p.pid = c.pid AND p.track_uri = c.track_uri
    """).df()

    if len(in_playlist_already) > 0:
        print("WARNING: recommending tracks already in the playlist")

