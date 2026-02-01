import duckdb as db
import numpy as np

# Track level r_prec
def compute_r_prec(prediction_df, holdout_df, playlist_metadata):
    """ Return the % of holdout tracks that were in the first 500 predicted tracks """

    num_holdout_retrieved = db.sql("""
        SELECT h.pid, COUNT(h.track_uri) as num_retrieved
        FROM holdout_df h
        JOIN prediction_df p ON h.pid = p.pid AND h.track_uri = p.track_uri
        WHERE p.prediction_num < 500
        GROUP BY h.pid
    """).df()

    eval_df = playlist_metadata.merge(num_holdout_retrieved, on="pid", how="left")
    eval_df["num_retrieved"] = eval_df["num_retrieved"].fillna(0)
    r_prec = np.mean(eval_df["num_retrieved"] / eval_df["num_holdouts"])
    return r_prec

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

