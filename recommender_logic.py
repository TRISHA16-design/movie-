# recommender_logic.py

import numpy as np
import joblib
import os
import pandas as pd

# --- Global Variables to hold loaded models/data ---
cosine_sim_matrix = None
title_to_indices = None
movie_titles_list = None
artifacts_loaded = False
artifacts_error = None

def load_artifacts():
    """Loads the precomputed artifacts."""
    global cosine_sim_matrix, title_to_indices, movie_titles_list, artifacts_loaded, artifacts_error

    if artifacts_loaded or artifacts_error:
        return

    print("Attempting to load precomputed artifacts for recommender...")
    try:
        # Define paths to artifact files
        cosine_sim_matrix_path = 'cosine_sim_matrix.npy'
        title_to_indices_path = 'title_to_indices.joblib'
        movies_list_path = 'movies_list.pkl'

        # Check if all files exist
        if not all(os.path.exists(p) for p in [cosine_sim_matrix_path, title_to_indices_path, movies_list_path]):
            artifacts_error = "One or more artifact files are missing. Please run preprocess_data.py first."
            print(f"Error: {artifacts_error}")
            return

        # Load the artifacts
        cosine_sim_matrix = np.load(cosine_sim_matrix_path)
        title_to_indices = joblib.load(title_to_indices_path)
        movie_titles_list = joblib.load(movies_list_path)
        artifacts_loaded = True
        print("Artifacts loaded successfully.")

    except Exception as e:
        artifacts_error = f"An error occurred while loading artifacts: {e}"
        print(f"Error: {artifacts_error}")

def recommend(movie_title, num_recommendations=10):
    """Recommends movies similar to the given movie title."""
    if not artifacts_loaded:
        return ["Error: Artifacts not loaded. Please check the logs."]

    try:
        # Get the index of the movie that matches the title
        movie_index = title_to_indices[movie_title]
    except KeyError:
        return [f"Movie with title '{movie_title}' not found in the dataset."]

    # Get the pairwise similarity scores and sort them
    sim_scores = list(enumerate(cosine_sim_matrix[movie_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the N most similar movies (skipping the first one, which is the movie itself)
    sim_scores = sim_scores[1:num_recommendations + 1]
    movie_indices = [i[0] for i in sim_scores]

    # Return the titles of the most similar movies
    return [movie_titles_list[i] for i in movie_indices]

# Load artifacts when the module is imported
load_artifacts()