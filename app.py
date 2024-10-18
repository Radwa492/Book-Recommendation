
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.decomposition import TruncatedSVD

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



svd_model = joblib.load('svd_model.h5')


def recommend_collaborative(
    user_id, ratings_df, books_df, train_matrix, svd_model, num_recommendations=5
):
    # All book IDs in the dataset
    all_book_ids = books_df["book_id"].unique()

    # Books already rated by the user
    rated_books = ratings_df[ratings_df["user_id"] == user_id]["book_id"].tolist()

    # Books the user hasn't rated yet
    books_to_predict = [book_id for book_id in all_book_ids if book_id not in rated_books]

    # Ensure the user exists in the training data, otherwise we need to handle new users
    if user_id not in train_matrix.index:
        print(f"User {user_id} not found in the training set.")
        return pd.DataFrame(columns=["book_id", "title", "authors", "small_image_url"])

    # Get the latent features of the user (U)
    user_index = train_matrix.index.get_loc(user_id)  # Get index of the user in the train matrix
    user_features = svd_model.transform(train_matrix)[user_index, :]

    # Get the Vt matrix (item latent features)
    Vt = svd_model.components_

    # Predict the ratings for all books the user hasn't rated
    predictions = []
    for book_id in books_to_predict:
        if book_id in train_matrix.columns:
            book_index = train_matrix.columns.get_loc(book_id)
            predicted_rating = np.dot(user_features, Vt[:, book_index])
            predictions.append((book_id, predicted_rating))

    # Create a DataFrame for the predictions
    pred_df = pd.DataFrame(predictions, columns=["book_id", "predicted_rating"])

    # Sort the predictions by rating in descending order and get the top recommendations
    pred_df = pred_df.sort_values("predicted_rating", ascending=False)
    top_recommendations = pred_df.head(num_recommendations)

    # Merge with books_df to get book details
    recommended_books = top_recommendations.merge(books_df, on="book_id")

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
    num_recommend = st.slider("Number of recommendations:", 1, 20, 5)
    if st.button("Show Recommendations"):
        if user_id not in ratings_df["user_id"].unique():
            st.error("User ID not found. Please enter a valid User ID.")
        else:
            recommended_books = recommend_collaborative(
                user_id,
                ratings_df,
                books_df,
                svd_model,
                num_recommendations=num_recommend,
            )
            if not recommended_books.empty:
                for index, row in recommended_books.iterrows():
                    st.image(row["small_image_url"], width=100)
                    st.write(f"**{row['title']}** by {row['authors']}")
                    st.write("---")
            else:
                st.write("No recommendations found.")

st.markdown("---")
st.markdown("Developed by [Your Name](https://yourwebsite.com)")

