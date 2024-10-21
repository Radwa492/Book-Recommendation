# app.py
import streamlit as st
import pandas as pd
from PIL import Image
from model import popularity_recommendations, content_based, hybrid_recommendations, train_svd_model, content_based_model

@st.cache
def load_data():
    books_df = pd.read_csv("https://raw.githubusercontent.com/Radwa492/Book-Recommendation/refs/heads/master/cleaned_books_data.csv")
    ratings_df = pd.read_csv("https://raw.githubusercontent.com/Radwa492/Book-Recommendation/refs/heads/master/ratings.csv")
    return books_df, ratings_df

books_df, ratings_df = load_data()

# Load models and data
cosine_sim, indices = content_based_model(books_df)
svd_model = train_svd_model(ratings_df)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Book Search", "User Recommendations"],
    index=0
)

# Render content based on page
if page == "Home":
    st.title("Top Books Based on Popularity")
    
    # Get popular books based on 'weighted_score'
    top_books = popularity_recommendations(books_df, ratings_df, num_recommendations=25, metric='weighted_score')

    st.header("Trending Books")
    
    # Display the books in rows of 5
    for i in range(0, len(top_books), 5):
        cols = st.columns(5)
        for j, col in enumerate(cols):
            if i + j < len(top_books):
                row = top_books.iloc[i + j]
                with col:
                    st.image(row['image_url'], width=100)
                    st.write(f"{row['title']}")
                    st.write(f"By: {row['authors']}")

elif page == "Book Search":
    st.title("Search for a Book (Content-based Recommendations)")
    
    # Dropdown for selecting a book title
    book_titles = books_df['title'].tolist()
    selected_book = st.selectbox("Select a Book:", book_titles)
    
    if selected_book:
        st.write(f"You selected: *{selected_book}*")
        selected_book_info = books_df[books_df['title'] == selected_book].iloc[0]
        
        # Display selected book information
        st.image(selected_book_info['image_url'], width=150)
        st.write(f"*Description:* {selected_book_info['description']}")
        st.write(f"*Genres:* {selected_book_info['genres']}")
        st.write(f"*Authors:* {selected_book_info['authors']}")
        
        # Button to get content-based recommendations
        if st.button("Recommend Similar Books"):
            recommendations, scores = content_based(selected_book, books_df)
            st.write("### Recommended Books:")
            for i, (index, row) in enumerate(recommendations.iterrows()):
                st.write(f"{i+1}. {row['title']} by {row['authors']}")

elif page == "User Recommendations":
    st.title("User-based Recommendations (Collaborative Filtering)")
    
    # Input user_id
    user_id = st.number_input("Enter User ID", min_value=1, step=1)
    
    if user_id:
        st.write(f"Getting recommendations for User ID: {user_id}")
        
        # Get hybrid recommendations for the user
        hybrid_recommendations_list = hybrid_recommendations(user_id, books_df, ratings_df, svd_model, num_recommendations=10)
        
        # Display recommendations
        st.write("### Recommended Books:")
        for i, row in hybrid_recommendations_list.iterrows():
            st.image(row['image_url'], width=100)
            st.write(f"{i+1}. {row['title']} by {row['authors']}")
