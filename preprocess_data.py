# preprocess_data.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import ast # For safely evaluating string-formatted lists/dicts

def get_director(crew_data):
    """Extracts the director's name from the crew data."""
    try:
        for member in ast.literal_eval(crew_data):
            if member['job'] == 'Director':
                # Remove spaces from name for better tagging
                return member['name'].replace(' ', '')
    except (ValueError, SyntaxError):
        pass # Ignore errors in malformed data
    return ''

def get_top_cast(cast_data, top_n=3):
    """Extracts the top N cast members' names."""
    try:
        cast = ast.literal_eval(cast_data)
        return [member['name'].replace(' ', '') for member in cast[:top_n]]
    except (ValueError, SyntaxError):
        pass # Ignore errors
    return []

def preprocess_data(credits_path):
    """
    Loads and preprocesses data from a single credits CSV file to create
    artifacts for the recommendation engine.
    """
    print("Step 1: Loading data from single CSV...")
    try:
        df = pd.read_csv(credits_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure '{credits_path}' is in the correct path.")
        return

    print("Step 2: Feature engineering and cleaning...")
    # Handle missing values
    df.dropna(subset=['cast', 'crew', 'title'], inplace=True)

    # Extract features from JSON-like columns
    df['director'] = df['crew'].apply(get_director)
    df['cast'] = df['cast'].apply(get_top_cast)

    # Create the 'tags' - the combination of features for vectorization
    # We use title, top 3 cast, and the director
    df['tags'] = df['title'].apply(lambda x: x.split()) + \
                 df['cast'] + \
                 df['director'].apply(lambda x: [x]) # Ensure director is in a list

    # Join all parts of the tag into a single string
    df['tags'] = df['tags'].apply(lambda x: " ".join(x).lower())

    # Final DataFrame for vectorization, ensuring unique titles
    final_df = df[['movie_id', 'title', 'tags']].copy()
    final_df.drop_duplicates(subset=['title'], inplace=True)

    print(f"Step 3: Vectorizing text data with TF-IDF... (Number of movies: {len(final_df)})")
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    vectors = tfidf.fit_transform(final_df['tags']).toarray()

    print("Step 4: Calculating Cosine Similarity Matrix...")
    cosine_sim_matrix = cosine_similarity(vectors)

    print("Step 5: Saving artifacts...")
    # Save the similarity matrix
    np.save('cosine_sim_matrix.npy', cosine_sim_matrix)

    # Reset index of the final DataFrame to ensure it's 0-based and aligns with the matrix
    final_df.reset_index(drop=True, inplace=True)

    # Save the list of movie titles for the dropdown menu
    joblib.dump(final_df['title'].values, 'movies_list.pkl')

    # Create and save the mapping from movie titles to their index in the matrix
    title_to_indices = pd.Series(final_df.index, index=final_df['title'])
    joblib.dump(title_to_indices, 'title_to_indices.joblib')

    print("\nPreprocessing complete! Artifacts for single-file recommender saved successfully.")

if __name__ == '__main__':
    # Define the path to your single CSV file
    credits_file = 'tmdb_5000_credits.csv'

    if not os.path.exists(credits_file):
        print(f"Error: Make sure '{credits_file}' is in the same directory as this script.")
    else:
        preprocess_data(credits_file)