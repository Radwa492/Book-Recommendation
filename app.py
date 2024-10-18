
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

import joblib
import gdown

def download_file(file_id, output):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=False)
    return output

rating_file ="1BBWxMFF60ZVUfIU8bMuDNU1d93wWMXyi"
download_file(rating_file,"ratings.csv")
# book_file = "1-2Yh_mVjhKt9P7d8DVE06gZlOpgFmL5i"
# download_file(book_file,"books.csv")
# ------------------------------
# Load DataFrames
# ------------------------------
@st.cache_data()
def load_data():
    books_df = pd.read_csv(
        "cleaned_books_data.csv"
    )  # Contains 'book_id', 'title', 'authors', 'genres', 'description', 'small_image_url'
    ratings_df = pd.read_csv(
        "ratings.csv"
    )  # Contains 'user_id', 'book_id', 'rating'
    return books_df, ratings_df


books_df, ratings_df = load_data()


# ------------------------------
# Popularity-Based Recommendations
# ------------------------------
def popularity_recommendations(
    books_df, ratings_df, num_recommendations=10, metric="average_rating"
):
    if metric == "average_rating":
        popular_books = (
            ratings_df.groupby("book_id")
            .agg({"rating": "mean"})
            .rename(columns={"rating": "average_rating"})
        )
        popular_books = popular_books.merge(
            books_df, on="book_id", suffixes=("", "_books")
        ).sort_values("average_rating", ascending=False)

    elif metric == "ratings_count":
        popular_books = (
            ratings_df.groupby("book_id")
            .agg({"rating": "count"})
            .rename(columns={"rating": "ratings_count"})
        )
        popular_books = popular_books.merge(
            books_df, on="book_id", suffixes=("", "_books")
        ).sort_values("ratings_count", ascending=False)

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
        popular_books = q_books.merge(
            books_df, on="book_id", suffixes=("", "_books")
        ).sort_values("weighted_score", ascending=False)

    else:
        raise ValueError(
            "Metric not recognized. Choose from 'average_rating', 'ratings_count', 'weighted_score'"
        )
    popular_books.columns = popular_books.columns.str.replace(
        "_x", "", regex=True
    ).str.replace("_y", "", regex=True)
    popular_books = popular_books.loc[:, ~popular_books.columns.duplicated()]

    return popular_books.head(num_recommendations)


# ------------------------------
# Content-Based Filtering
# ------------------------------
@st.cache(allow_output_mutation=True)
def build_content_model(books_df):
    books_df["description"] = books_df["description"].fillna("")
    books_df["genres"] = books_df["genres"].fillna("")
    books_df["authors"] = books_df["authors"].fillna("")
    books_df["content"] = (
        books_df["description"] + " " + books_df["genres"] + " " + books_df["authors"]
    )
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(books_df["content"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(books_df.index, index=books_df["title"]).drop_duplicates()
    return cosine_sim, indices


cosine_sim, indices = build_content_model(books_df)


def get_content_recommendations(
    title, books_df, cosine_sim, indices, num_recommendations=5
):
    if title not in indices:
        st.error(f"Book titled '{title}' not found in the database.")
        return pd.DataFrame()

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : num_recommendations + 1]
    book_indices = [i[0] for i in sim_scores]
    return books_df[["book_id", "title", "authors", "small_image_url"]].iloc[book_indices]


# ------------------------------
# Collaborative Filtering
# ------------------------------

# Create a user-item matrix (train_matrix)




# def collaborative_filtering_simple(user_id, ratings_df, books_df, num_recommendations=5):
#     # Get the books rated by the user
#     user_ratings = ratings_df[ratings_df["user_id"] == user_id]
    
#     if user_ratings.empty:
#         st.error("This user has not rated any books.")
#         return pd.DataFrame(columns=["book_id", "title", "authors", "small_image_url"])

#     # Find similar users based on ratings
#     similar_users = ratings_df[ratings_df["book_id"].isin(user_ratings["book_id"])]
#     recommended_books = similar_users.groupby("book_id").agg({"rating": "mean"}).reset_index()
    
#     # Filter out books already rated by the user
#     recommended_books = recommended_books[~recommended_books["book_id"].isin(user_ratings["book_id"])]
    
#     # Get top recommendations
#     top_recommendations = recommended_books.sort_values(by="rating", ascending=False).head(num_recommendations)

#     # Merge with books_df to get book details
#     recommendations = books_df[books_df["book_id"].isin(top_recommendations["book_id"])]
#     return recommendations[["book_id", "title", "authors", "small_image_url"]]

from sklearn.neighbors import NearestNeighbors

def collaborative_filtering_simple(user_id, ratings_df, books_df, num_recommendations=5):
    user_ratings = ratings_df[ratings_df["user_id"] == user_id]
    
    if user_ratings.empty:
        print(f"No ratings found for User ID {user_id}")
        return pd.DataFrame()

    # Get the book IDs the user has rated
    book_ids = user_ratings["book_id"].values.reshape(-1, 1)
    
    if len(book_ids) == 0:
        print("No book IDs found for this user")
        return pd.DataFrame()

    # Fit a NearestNeighbors model on book IDs (or more sophisticated data if available)
    model_knn = NearestNeighbors(metric='cosine', algorithm='auto')
    model_knn.fit(book_ids)
    
    # Find the nearest neighbors
    distances, indices = model_knn.kneighbors(book_ids, n_neighbors=num_recommendations)
    
    # Get the recommended books from the indices
    recommended_books = books_df.iloc[indices.flatten()]
    
    return recommended_books[["book_id", "title", "authors", "small_image_url"]]





# ------------------------------
# Streamlit App Layout
# ------------------------------
st.title("📚 Book Recommendation System")

st.sidebar.title("Recommendation Methods")
recommendation_method = st.sidebar.selectbox(
    "Choose a recommendation method:",
    ("Popularity-Based", "Content-Based", "Collaborative Filtering"),
)

# Popularity-Based Recommendations
if recommendation_method == "Popularity-Based":
    st.header("📈 Popularity-Based Recommendations")
    metric = st.selectbox(
        "Choose a popularity metric:",
        ("average_rating", "ratings_count", "weighted_score"),
    )
    num_recommend = st.slider("Number of recommendations:", 1, 20, 10)
    if st.button("Show Recommendations"):
        top_books = popularity_recommendations(
            books_df, ratings_df, num_recommendations=num_recommend, metric=metric
        )
        for index, row in top_books.iterrows():
            st.image(row["small_image_url"], width=100)
            st.write(f"**{row['title']}** by {row['authors']}")
            st.write("---")

# Content-Based Recommendations
elif recommendation_method == "Content-Based":
    st.header("🔍 Content-Based Recommendations")
    book_title = st.selectbox("Select a book you like:", books_df["title"].values)
    num_recommend = st.slider("Number of recommendations:", 1, 20, 5)
    if st.button("Show Recommendations"):
        recommended_books = get_content_recommendations(
            book_title, books_df, cosine_sim, indices, num_recommendations=num_recommend
        )
        if not recommended_books.empty:
            for index, row in recommended_books.iterrows():
                st.image(row["small_image_url"], width=100)
                st.write(f"**{row['title']}** by {row['authors']}")
                st.write("---")
        else:
            st.write("No recommendations found.")

# Collaborative Filtering Recommendations
elif recommendation_method == "Collaborative Filtering":
    st.header("👥 Collaborative Filtering Recommendations")
    user_id = st.number_input("Enter your User ID:", min_value=1, step=1)
    user_id = int(user_id)
    num_recommend = st.slider("Number of recommendations:", 1, 20, 5)
    num_recommend = int(num_recommend)
    if st.button("Show Recommendations"):
        if user_id not in ratings_df["user_id"].unique():
            st.error("User ID not found. Please enter a valid User ID.")
        else:
            recommended_books = collaborative_filtering_simple(user_id=user_id, ratings_df=ratings_df, books_df=books_df, num_recommendations=num_recommend)
            if not recommended_books.empty:
                for index, row in recommended_books.iterrows():
                    st.image(row["small_image_url"], width=100)
                    st.write(f"**{row['title']}** by {row['authors']}")
                    st.write("---")
            else:
                st.write("No recommendations found.")

st.markdown("---")
st.markdown("Developed by [Your Name](https://yourwebsite.com)")

