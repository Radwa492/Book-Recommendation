import streamlit as st
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
from PIL import Image

# Set NLTK data directory
nltk_data_dir = "/content/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download necessary NLTK resources
nltk.download("punkt", download_dir=nltk_data_dir)
nltk.download("stopwords", download_dir=nltk_data_dir)
nltk.download("wordnet", download_dir=nltk_data_dir)

# Load the datasets
books_df = pd.read_csv("/content/drive/MyDrive/Copy_of_cleaned_books_data.csv")
ratings_df = pd.read_csv("/content/drive/MyDrive/ratings.csv")
genre_df = pd.read_csv("/content/drive/MyDrive/unique_genres.csv")
author_df = pd.read_csv("/content/drive/MyDrive/unique_authors.csv")

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()


# Preprocess functions and recommendation logic
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


def preprocess_genres(genres):
    return genres.lower().replace(",", " ").replace(" ", "_")


@st.cache_data
def preprocess_books(books_df):
    books_df["description"] = books_df["description"].fillna("")
    books_df["genres"] = books_df["genres"].fillna("")
    books_df["authors"] = books_df["authors"].fillna("")

    books_df["description"] = books_df["description"].apply(preprocess_text)
    books_df["genres"] = books_df["genres"].apply(preprocess_genres)
    books_df["content"] = (
        books_df["description"] + " " + books_df["genres"] + " " + books_df["authors"]
    )

    return books_df


def popularity_recommendations(
    books_df, ratings_df, num_recommendations, metric="average_rating"
):
    """
    Recommend books based on popularity.

    Parameters:
    - books_df (DataFrame): DataFrame containing book details.
    - ratings_df (DataFrame): DataFrame containing user ratings.
    - num_recommendations (int): Number of books to recommend.
    - metric (str): Metric to determine popularity ('average_rating', 'ratings_count', 'weighted_score').

    Returns:
    - DataFrame: Recommended books sorted by the chosen metric.
    """
    if metric == "average_rating":
        popular_books = (
            ratings_df.groupby("book_id")
            .agg({"rating": "mean"})
            .rename(columns={"rating": "average_rating"})
        )
        popular_books = popular_books.merge(books_df, on="book_id").sort_values(
            "average_rating", ascending=False
        )

    elif metric == "ratings_count":
        popular_books = (
            ratings_df.groupby("book_id")
            .agg({"rating": "count"})
            .rename(columns={"rating": "ratings_count"})
        )
        popular_books = popular_books.merge(books_df, on="book_id").sort_values(
            "ratings_count", ascending=False
        )

    elif metric == "weighted_score":
        C = ratings_df["rating"].mean()
        m = ratings_df["book_id"].value_counts().quantile(0.9)
        q_books = ratings_df.groupby("book_id").agg(
            average_rating=("rating", "mean"), ratings_count=("rating", "count")
        )
        q_books = q_books[q_books["ratings_count"] >= m]
        q_books["weighted_score"] = (
            q_books["average_rating"] * q_books["ratings_count"] + C * m
        ) / (q_books["ratings_count"] + m)
        popular_books = q_books.merge(books_df, on="book_id").sort_values(
            "weighted_score", ascending=False
        )

    else:
        raise ValueError(
            "Metric not recognized. Choose from 'average_rating', 'ratings_count', 'weighted_score'"
        )

    # Clean up column names
    popular_books.columns = popular_books.columns.str.replace(
        "_x", "", regex=True
    ).str.replace("_y", "", regex=True)
    return popular_books.head(num_recommendations)


@st.cache_data
def content_based_model(books_df):
    books_df = preprocess_books(books_df)
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(books_df["content"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(
        books_df.index, index=books_df["title"].str.lower()
    ).drop_duplicates()
    return cosine_sim, indices


def content_based(selected_book_title, books_df):
    books_df = preprocess_books(books_df)
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(books_df["content"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(
        books_df.index, index=books_df["title"].str.lower()
    ).drop_duplicates()

    idx = indices[selected_book_title.lower()]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    book_indices = [i[0] for i in sim_scores]
    return books_df.iloc[book_indices], [i[1] for i in sim_scores]


@st.cache_data
def train_svd_model(ratings_df):
    reader = Reader(
        rating_scale=(ratings_df["rating"].min(), ratings_df["rating"].max())
    )
    data = Dataset.load_from_df(ratings_df[["user_id", "book_id", "rating"]], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    svd_model = SVD(random_state=42)
    svd_model.fit(trainset)
    return svd_model


def hybrid_recommendations(
    user_id, books_df, ratings_df, svd_model, num_recommendations=20
):
    all_book_ids = books_df["book_id"].unique()
    rated_books = ratings_df[ratings_df["user_id"] == user_id]["book_id"].tolist()
    books_to_predict = [book for book in all_book_ids if book not in rated_books]

    predictions = [svd_model.predict(user_id, book_id) for book_id in books_to_predict]

    pred_df = pd.DataFrame(
        {
            "book_id": books_to_predict,
            "predicted_rating": [pred.est for pred in predictions],
        }
    )

    pred_df = pred_df.sort_values("predicted_rating", ascending=False)
    top_collab_recommendations = pred_df.head(num_recommendations)

    combined_recommendations = top_collab_recommendations.merge(books_df, on="book_id")
    combined_recommendations = combined_recommendations[
        ["book_id", "predicted_rating", "title", "authors", "image_url"]
    ]
    return combined_recommendations


# Load models and data
books_df = preprocess_books(books_df)
cosine_sim, indices = content_based_model(books_df)
svd_model = train_svd_model(ratings_df)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to", ["Home", "Book Search", "User Recommendations"], index=0
)

# Render content based on page
# Home page - updated with popularity recommendations
if page == "Home":
    st.title("Top Books Based on Popularity")

    # Get popular books based on 'weighted_score'
    top_books = popularity_recommendations(
        books_df, ratings_df, num_recommendations=25, metric="weighted_score"
    )

    st.header("Trending Books")

    # Display the books in rows of 5
    for i in range(0, len(top_books), 5):
        cols = st.columns(5)
        for j, col in enumerate(cols):
            if i + j < len(top_books):
                row = top_books.iloc[i + j]
                with col:
                    st.image(row["image_url"], width=100)
                    st.write(f"{row['title']}")
                    st.write(f"By: {row['authors']}")


elif page == "Book Search":
    st.title("Search for a Book (Content-based Recommendations)")

    # Dropdown for selecting a book title
    book_titles = books_df["title"].tolist()
    selected_book = st.selectbox("Select a Book:", book_titles)

    if selected_book:
        st.write(f"You selected: *{selected_book}*")
        selected_book_info = books_df[books_df["title"] == selected_book].iloc[0]

        # Display selected book information
        st.image(selected_book_info["image_url"], width=150)
        st.write(f"*Description:* {selected_book_info['description']}")
        st.write(f"*Genres:* {selected_book_info['genres']}")
        st.write(f"*Authors:* {selected_book_info['authors']}")

        # Button to get content-based recommendations
        if st.button("Get Content-based Recommendations"):
            recommendations, scores = content_based(selected_book, books_df)

            # Check if recommendations are returned
            if not recommendations.empty:
                st.write("### Content-based Recommendations:")

                # Display the books in rows of 5
                for i in range(0, len(recommendations), 5):
                    cols = st.columns(5)
                    for j, col in enumerate(cols):
                        if i + j < len(recommendations):
                            row = recommendations.iloc[i + j]
                            with col:
                                st.image(row["image_url"], width=100)
                                st.write(f"{row['title']}")
                                st.write(f"By: {row['authors']}")
                                st.write(f"Similarity Score: {scores[i + j]:.2f}")
            else:
                st.write("No recommendations found.")


elif page == "User Recommendations":
    st.title("User Recommendations (Hybrid Collaborative Filtering)")

    # Input for user ID
    user_id = st.number_input("Enter your User ID:", min_value=1)

    # Get hybrid recommendations when button is clicked
    if st.button("Get Hybrid Recommendations"):
        recommendations = hybrid_recommendations(
            user_id, books_df, ratings_df, svd_model
        )

        # Check if recommendations are available
        if not recommendations.empty:
            st.write("### Recommendations:")

            # Display the books in rows of 5
            for i in range(0, len(recommendations), 5):
                cols = st.columns(5)
                for j, col in enumerate(cols):
                    if i + j < len(recommendations):
                        row = recommendations.iloc[i + j]
                        with col:
                            st.image(row["image_url"], width=100)
                            st.write(f"{row['title']}")
                            st.write(f"By: {row['authors']}")
        else:
            st.write("No recommendations found for this user.")
