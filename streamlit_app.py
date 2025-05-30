# streamlit_app.py

import streamlit as st
import recommender_logic

# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Main Application UI ---
st.title("🎬 Content-Based Movie Recommender")
st.markdown(
    "Discover new movies based on your favorites! "
    "Select a movie from the dropdown and get instant recommendations."
)
st.markdown("---")

if recommender_logic.artifacts_error:
    st.error(f"Failed to load recommender model artifacts: {recommender_logic.artifacts_error}")
    st.warning("Please run `preprocess_data.py` and ensure the artifact files are in the same directory.")
else:
    st.header("✨ Get Your Movie Recommendations")

    # Use a selectbox for movie selection
    available_movies = sorted(list(recommender_logic.movie_titles_list))
    selected_movie_title = st.selectbox(
        'Choose a movie you like:',
        options=available_movies
    )

    num_recs = st.slider("How many recommendations would you like?", 5, 20, 10)

    if st.button('Get Recommendations', type="primary"):
        if selected_movie_title:
            with st.spinner(f"Finding movies similar to '{selected_movie_title}'..."):
                recommendations = recommender_logic.recommend(selected_movie_title, num_recommendations=num_recs)

            if recommendations and not recommendations[0].startswith("Error:"):
                st.success(f"Here are {len(recommendations)} movies you might also like:")
                cols = st.columns(2)
                for i, rec_title in enumerate(recommendations):
                    with cols[i % 2]:
                        st.markdown(f"**{i+1}.** {rec_title}")
            else:
                st.error(recommendations[0])


# --- Sidebar Information ---
st.sidebar.header("About This App")
st.sidebar.info(
    "This Movie Recommender uses a content-based filtering approach, "
    "built using a single dataset (`tmdb_5000_credits.csv`)."
)
st.sidebar.markdown("---")
st.sidebar.subheader("How It Works")
st.sidebar.markdown(
    """
    1.  **Data Processing**: Movie data is loaded. Features are extracted from the `title`, `cast`, and `crew` columns.
    2.  **Vectorization**: These features are combined and converted into numerical vectors.
    3.  **Cosine Similarity**: The similarity between all movies is calculated and stored.
    4.  **Recommendation**: When you select a movie, the app finds others with the highest similarity scores.
    """
)