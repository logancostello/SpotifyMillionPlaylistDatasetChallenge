# Spotify Million Playlist Dataset Challenge

## Introduction
I estimate that I would have placed **10th (top 6%)** in the Spotify Million Playlist Dataset Challenge. I achieved these results by building a two stage recommendation system that combines matrix factorization, BERT embeddings, and other features to recommend tracks to add to existing playlists on Spotify. 

## Challenge Overview
The Spotify Million Playlist Dataset Challenge tests participants abilities to build recommendation systems specifically designed for playlist continuation. Participants must use the titles and track contents of 1,000,000 playlists to recommend 500 tracks to add to 1,000 playlists of each of the following ten categories:
1. Title Only
2. Title and First Track
3. Title and First 5 Tracks
4. First 5 Tracks Only
5. Title and First 10 Tracks
6. First 10 Tracks Only
7. Title and First 25 Tracks
8. Title and Random 25 Tracks
9. Title and First 100 Tracks
10. Title and Random 100 Tracks

Recommendations are evaluated by three metrics: R-Precision, NDCG, and Recommended Song Clicks. R-Precision is fairly standard, but quarter credit is assigned to tracks where the artist matches. NDCG is the standard definition. Recommended Song Clicks is a custom measurement by Spotify made to determine how many pages/refreshes the user must go through in order to find the first correct recommended song. It’s calculated as the location of the first correct track floor divided by 10. 

More details on the challenge, dataset, evaluation, etc can be found [here](https://dl.acm.org/doi/epdf/10.1145/3240323.3240342). 

The challenge was originally opened in January 2018 and it ran until July 2018. It was later reopened on [aicrowd.com](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge#) and said to be indefinitely ongoing. However it seems to have been silently discontinued as the submission grader automatically fails all submissions. Only the scores and ranks of the submissions on [aicrowd.com](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge#) are available, so I use those scores to estimate where I would place in the competition. 

## Offline Evaluation
Every recommendation system needs a strong offline evaluation system. Additionally, since the challenge was effectively closed, online evaluations were unavailable to me. I wanted accurate comparisons for a fair estimation of my rank in the competition, so I spent a considerable amount of time making my offline evaluation system as robust as I could. 
	
Rather than randomly selecting playlists as holdouts, I wanted my test playlists to be from the same population of the 10 different categories. I analyzed the distribution of the number of tracks for each category and tried to learn the distribution such that I could sample from the million playlists and get a similar distribution. My analysis found that 8 categories were random samples with min and max bounds, 1 category was a random sample with min and max bounds where the max bound was greatly surpassed in 3/1000 cases (I treated these as outliers and lowered the max bound accordingly), and 1 category was 2 random samples with their own mins and maxes where one sample was weighted ~2x more than the other (~333 fit into one sample while the other ~666 fit into the other). In hindsight I should have used statistical methods to measure the probability that my distributions and the challenge distributions were the same. I did evaluate this qualitatively by visually comparing the distributions across many random samples. 

Using the distributions described above, I selected and withheld playlists from the original million to use as test data. I removed tracks from each playlist such that they fit their selected category (A playlist with 125 tracks in the “25 Random Tracks” category had 100 tracks randomly withheld). These tracks served as my ground truth and were never exposed to the model at any point. 

To be consistent with the challenge, I always recommend 500 tracks to each playlist, ensuring there are no duplicates and no tracks already in the playlist. I evaluate these recommendations using the evaluation metrics described in the challenge description. 

The challenge description notes that a baseline popularity model achieves a R-Prec of 0.0458, NDCG of 0.0993, and Clicks of 13.217. My implementation of the same baseline achieves a R-Prec of 0.0422, NDCG of 0.0815, and Clicks of 18.313, indicating that my offline test set is likely different from the ground truth challenge set. I tried many times to fix this but never could. Unfortunately my estimated challenge rank is likely a bit incorrect due to this difference, however one could argue that the estimated rank is conservative since my scores are worse than they should be. Despite this, the main goal of my participation in the challenge was not to win, but to learn about and build a modern recommendation system.

## Final Model
My final model consists of two distinct parts: candidate generation and candidate ranking. The candidate generation stage is in charge of picking tracks that are likely to be added to the playlist, so the goal is to have a high recall. The candidate ranking stage is in charge of ordering the candidates such that the first recommendations are the most likely to be added to the playlist. The two stages are necessary as it is unrealistic to rank all 2.2 million tracks in the dataset per playlist. This model is used for the nine groups that have tracks in the playlist. For the title only group, I use title embeddings to find tracks in similar playlists.

### Candidate Generation
Candidates are generated via three different processes, then unioned together to form the final candidate pool. The primary way of generating candidates is via matrix factorization. A matrix factorization model is trained on playlist-track interaction data, resulting in a latent space that contains both playlists and tracks. The 1000 most similar (ie. closest) tracks are selected as candidates for each playlist. Matrix factorization struggles with smaller playlists and niche playlists. To help compensate, 250 tracks by artists already in the playlists are selected as candidates. Additionally, the most co-occurring tracks are selected as candidates. The number of co-occurring tracks selected starts at 10 candidates per seed track and decreases to 5 candidates as the length of the playlist grows, capping at 500 total co-occurring candidates. In total, a maximum of 1750 candidates are selected to be ranked. 

### Candidate Ranking
The candidate ranking model is a gradient boosted decision tree that predicts whether or not a candidate will be added to a playlist. It incorporates many new features that the candidate generators don’t have, allowing it to better rank the candidates. The features include:
- The matrix factorization score and the co-occurrence score
- Group information: Playlist length, whether it has a title, and whether it was randomly sampled
- The popularity of the track, the artist, and the album
- The number of artists and albums in the playlist
- Whether or not the candidate is the same artist/album as the last track
- The mean and standard deviation of track length, along with the z score for each candidate in terms of length
- A title score

This model is trained until the loss on a separate validation set does not decrease for 25 iterations. 

### Title Model
As previously mentioned, predictions on the Title Only group are made via a different model. This model computes Sentence-BERT embeddings for each title. At prediction time, the K most similar playlists are found and their most common tracks are chosen as recommendations. Recommendations are ordered based on a weighted count where the weight is the dot product between the title embeddings. This is the same process used for computing the title score used in the ranking model. 

## Final Results
My final model has an R-Precision of 0.2173 (3rd), NDCG of 0.3426 (14th), and Recommended Song Clicks of 2.510 (19th). With these results, I estimate that I would place 10th (top 6%) in the Spotify Million Playlist Dataset Challenge (when only comparing to submissions on [aicrowd.com](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge#)). These results came from making predictions on 10,000 playlists (1,000 per group, like the original challenge) randomly selected with a seed that was never used for training or fine-tuning. 

The following visualizations give a per group, per model, per metric breakdown of my performance. The raw scores as well as the leaderboard can be found [here](https://docs.google.com/spreadsheets/d/1BC7mftCrDw7rwXr35a2v-jedTljqb0tVR0EBUvXg3bg/edit?gid=1374574207#gid=1374574207). 

<img width="804" height="605" alt="Model R-Precision By Group" src="https://github.com/user-attachments/assets/79d4d6db-44ba-4f03-8e99-4bed9680fe75" />
<img width="804" height="605" alt="Model NDCG By Group" src="https://github.com/user-attachments/assets/8a11db15-8f8a-47d1-9d22-650846fd5bbe" />
<img width="804" height="605" alt="Model Recommended Song Clicks By Group" src="https://github.com/user-attachments/assets/12a4edcd-b19c-4ca8-b530-b7ccb6c2a990" />

## Future Work
While I feel that my work is very thorough, there is still more that could be done. More features could be created, hyperparameters could be further fine-tuned, and many optimizations could be made to improve computational constraints. However, the biggest area for improvement is using the sequence information. My models completely ignore the order of the given tracks, which is potentially leaving a lot of signal unused. Given how powerful transformers are, it’s likely that they would be able to make good track recommendations. If I continue this project in the future, that would be my primary focus. Finally, I would also focus on directly optimizing on NDCG and clicks during the ranking stage, since I fall behind in these two metrics compared to R-Precision.

