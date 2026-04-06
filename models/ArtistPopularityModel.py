import duckdb as db

class ArtistPopularityModel:

    def __init__(self):
        self.name = "Artist Popularity Model"
        self.trained = False

    def train(self, playlist_metadata, playlist_contents, track_metadata):
        track_popularity = db.sql("""
            SELECT track_uri, COUNT(*) AS popularity
            FROM playlist_contents 
            GROUP BY track_uri
            ORDER BY popularity DESC
        """)

        db.register('track_popularity', track_popularity)
        self.trained = True

    def predict(self, playlist_metadata, playlist_contents, track_metadata):
        db.register('playlist_contents', playlist_contents)
        db.register('track_metadata', track_metadata)

        result = db.sql("""
            WITH playlist_artists AS (
                SELECT pc.pid, tm.artist_uri, COUNT(*) AS num_appearances
                FROM playlist_contents pc
                JOIN track_metadata tm on pc.track_uri = tm.track_uri
                GROUP BY pc.pid, tm.artist_uri         
            ),
            candidate_tracks AS (
                SELECT 
                    pa.pid,
                    tm.track_uri,
                    pa.artist_uri,
                    pa.num_appearances
                FROM playlist_artists pa
                JOIN track_metadata tm ON pa.artist_uri = tm.artist_uri
                LEFT JOIN playlist_contents pc ON pa.pid = pc.pid AND tm.track_uri = pc.track_uri
                WHERE pc.track_uri IS NULL  -- Exclude tracks already in playlist
            ),
            ranked_artist_candidates AS (
                SELECT 
                    ct.pid,
                    ct.track_uri,
                    ROW_NUMBER() OVER (
                        PARTITION BY ct.pid 
                        ORDER BY tp.popularity DESC, ct.num_appearances DESC
                    ) as rn
                FROM candidate_tracks ct
                JOIN track_popularity tp ON ct.track_uri = tp.track_uri
            ),
            -- Get all popular tracks as fallback
            popular_fallback AS (
                SELECT 
                    pm.pid,
                    tp.track_uri,
                    tp.popularity
                FROM playlist_metadata pm
                CROSS JOIN (SELECT * FROM track_popularity LIMIT 600) tp
                LEFT JOIN playlist_contents pc ON pm.pid = pc.pid AND tp.track_uri = pc.track_uri
                WHERE pc.track_uri IS NULL  -- Exclude tracks already in playlist
            ),
            -- Combine artist-based and popular fallback
            all_candidates AS (
                SELECT pid, track_uri, rn as rank, 1 as source_priority
                FROM ranked_artist_candidates
                WHERE rn <= 500
                
                UNION ALL
                
                SELECT 
                    pid, 
                    track_uri, 
                    ROW_NUMBER() OVER (PARTITION BY pid ORDER BY popularity DESC) as rank,
                    2 as source_priority
                FROM popular_fallback
            ),
            -- Remove duplicates and re-rank
            final_ranked AS (
                SELECT 
                    pid,
                    track_uri,
                    ROW_NUMBER() OVER (
                        PARTITION BY pid 
                        ORDER BY source_priority, rank
                    ) as final_rn
                FROM (
                    SELECT pid, track_uri, 
                        MIN(source_priority) as source_priority, 
                        MIN(rank) as rank
                    FROM all_candidates
                    GROUP BY pid, track_uri  -- deduplicate here
                )
            )
            SELECT 
                pid,
                track_uri,
                final_rn - 1 as prediction_num
            FROM final_ranked
            WHERE final_rn <= 500
            ORDER BY pid, final_rn
        """).df()

        return result