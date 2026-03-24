import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_resource
def load_recommender():
    csv_path = os.path.join("data", "IMDB-Movie-Dataset(2023-1951).csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    movies = pd.read_csv(csv_path)

    # Normalize column names
    movies.columns = [c.strip().lower() for c in movies.columns]

    # Rename columns to standard names
    rename_map = {
        "movie_name": "title",
        "name": "title",
        "title": "title",
        "movie id": "movie_id",
        "movie_id": "movie_id",
        "year": "year",
        "genre": "genre",
        "genres": "genre",
        "overview": "overview",
        "synopsis": "overview",
        "description": "overview",
        "director": "director",
        "cast": "cast",
        "actors": "cast"
    }

    movies = movies.rename(columns=rename_map)

    # Check required columns
    required = ["title", "genre", "overview"]
    for col in required:
        if col not in movies.columns:
            raise ValueError(
                f"Required column '{col}' missing from CSV.\nFound columns: {movies.columns.tolist()}"
            )

    # Keep only useful columns
    keep = [c for c in ["movie_id", "title", "year", "genre", "overview", "director", "cast"] if c in movies.columns]
    movies = movies[keep]

    # Fill missing text
    for col in ["genre", "overview", "director", "cast"]:
        if col in movies.columns:
            movies[col] = movies[col].fillna("")

    # Make soup
    def col_text(col):
        return movies[col].astype(str) if col in movies.columns else ""

    movies["soup"] = (
        col_text("genre") + " " +
        col_text("overview") + " " +
        col_text("director") + " " +
        col_text("cast")
    )

    # Vectorizer
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["soup"])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    movies = movies.reset_index(drop=True)

    indices = pd.Series(
        movies.index,
        index=movies["title"].astype(str).str.lower()
    ).drop_duplicates()

    return movies, cosine_sim, indices


def get_recommendations(title, movies, cosine_sim, indices, n=10):
    title_key = str(title).lower().strip()

    if title_key not in indices:
        return pd.DataFrame(columns=["title", "year", "genre", "director"])

    idx = indices[title_key]

    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:n + 1]

    movie_indices = [i[0] for i in scores]

    cols = [c for c in ["title", "year", "genre", "director"] if c in movies.columns]
    return movies.loc[movie_indices, cols].reset_index(drop=True)


# =========================
#       STREAMLIT UI
# =========================

st.set_page_config(page_title="Bollywood Recommender", page_icon="🎬")

st.title("🎬 Bollywood Movie Recommendation System")
st.write(
    """
    Select any Bollywood movie and get similar movie recommendations  
    based on Genre, Overview, Director & Cast.
    """
)

with st.spinner("Loading AI model and dataset..."):
    movies, cosine_sim, indices = load_recommender()

st.success("Model loaded successfully! 🎉")

all_titles = sorted(movies["title"].dropna().unique().tolist())
selected_title = st.selectbox("Choose a movie:", all_titles)

n_recs = st.slider("Number of recommendations:", 5, 20, 10)

if st.button("Recommend"):
    results = get_recommendations(selected_title, movies, cosine_sim, indices, n_recs)

    if results.empty:
        st.warning("No recommendations found.")
    else:
        st.subheader(f"Movies similar to: {selected_title}")
        st.table(results)
