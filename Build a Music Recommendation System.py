import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the dataset
data = pd.read_csv('music_data.csv')

# Collaborative Filtering using Surprise Library
reader = Reader(rating_scale=(1, 5))
data_surprise = Dataset.load_from_df(data[['user_id', 'track_id', 'rating']], reader)
trainset, testset = train_test_split(data_surprise, test_size=0.2)

# Use SVD for collaborative filtering
model = SVD()
model.fit(trainset)

# Make predictions
predictions = model.test(testset)
accuracy.rmse(predictions)

# Function to get top N recommendations for a user
def get_top_n_recommendations(predictions, n=5):
    # Map the predictions to each user
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if not uid in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))

    # Sort the predictions for each user and retrieve the top N
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

# Get top 5 recommendations for each user
top_n_recommendations = get_top_n_recommendations(predictions, n=5)
print("Collaborative Filtering Recommendations:")
for uid, user_ratings in top_n_recommendations.items():
    print(f"User  {uid}: {[f'Track ID: {iid}, Estimated Rating: {est:.2f}' for (iid, est) in user_ratings]}")

# Content-Based Filtering
# Create a TF-IDF Vectorizer for track features
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(data['genre'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend tracks based on content similarity
def content_based_recommendations(track_id, n=5):
    idx = data[data['track_id'] == track_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]  # Exclude the first one (itself)
    track_indices = [i[0] for i in sim_scores]
    return data['track_name'].iloc[track_indices]

# Example: Get content-based recommendations for a specific track
track_id_example = 101
recommended_tracks = content_based_recommendations(track_id_example)
print(f"\nContent-Based Recommendations for Track ID {track_id_example}: {recommended_tracks.tolist()}")