import duckdb as db

class GlobalPopularityModel:

    def __init__(self):
        self.most_popular = None
        self.name = "Global Popularity Heuristic Model"

    def train(self, playlist_metadata, playlist_contents, track_metadata):
        self.most_popular = db.sql("""
            SELECT track_uri, COUNT(*) as count
            FROM playlist_contents
            GROUP BY track_uri
            ORDER BY count DESC
            LIMIT 600
        """).df()

        db.register('most_popular', self.most_popular)

    def predict(self, playlist_metadata, playlist_contents, track_metadata):
        
        result = db.sql("""
            WITH ranked_popular AS (
                SELECT 
                    pm.pid,
                    mp.track_uri,
                    ROW_NUMBER() OVER (PARTITION BY pm.pid ORDER BY mp.count DESC) as rn
                FROM playlist_metadata pm
                CROSS JOIN most_popular mp
                LEFT JOIN playlist_contents pc 
                    ON pm.pid = pc.pid AND mp.track_uri = pc.track_uri
                WHERE pc.track_uri IS NULL  -- Exclude existing tracks
            )
            SELECT pid, track_uri, rn - 1 as prediction_num
            FROM ranked_popular
            WHERE rn <= 500
            ORDER BY pid, rn
        """).df()
        
        return result