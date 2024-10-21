import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import gdown

# Set NLTK data directory
nltk_data_dir = './nltk_data'  # Use a local directory
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download necessary NLTK resources
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocess functions
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def preprocess_genres(genres):
    return genres.lower().replace(',', ' ').replace(' ', '_')

def preprocess_books(books_df):
    books_df['description'] = books_df['description'].fillna('')
    books_df['genres'] = books_df['genres'].fillna('')
    books_df['authors'] = books_df['authors'].fillna('')
    
    books_df['description'] = books_df['description'].apply(preprocess_text)
    books_df['genres'] = books_df['genres'].apply(preprocess_genres)
    books_df['content'] = books_df['description'] + ' ' + books_df['genres'] + ' ' + books_df['authors']
    
    return books_df

def content_based_model(books_df):
    books_df = preprocess_books(books_df)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(books_df['content'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(books_df.index, index=books_df['title'].str.lower()).drop_duplicates()
    return cosine_sim, indices

def train_knn_model(ratings_df):
    # Create a pivot table for ratings
    pivot_table = ratings_df.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(pivot_table.values.T)  # Transpose to have books as rows
    return knn_model, pivot_table

def hybrid_recommendations(user_id, books_df, ratings_df, knn_model, pivot_table, selected_book_title=None, num_recommendations=10):
    # Content-based recommendations
    cosine_sim, indices = content_based_model(books_df)
    
    # Get collaborative filtering recommendations
    all_book_ids = books_df['book_id'].unique()
    rated_books = ratings_df[ratings_df['user_id'] == user_id]['book_id'].tolist()
    
    # Find the books the user hasn't rated yet
    books_to_predict = [book for book in all_book_ids if book not in rated_books]
    user_ratings = pivot_table.loc[user_id].values.reshape(1, -1)
    
    # Find the nearest neighbors (books) for the user's ratings
    distances, indices_collab = knn_model.kneighbors(user_ratings, n_neighbors=num_recommendations + 1)  # +1 to exclude the book itself
    
    # Get recommended book IDs from collaborative filtering
    recommended_book_ids_collab = pivot_table.columns[indices_collab.flatten()[1:]].tolist()  # Skip the first index (the book itself)
    
    # Get content-based recommendations for a specific book if provided
    if selected_book_title:
        idx = indices[selected_book_title.lower()]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        book_indices_content = [i[0] for i in sim_scores[1:num_recommendations + 1]]  # Skip the first (itself)
        
        # Combine both sets of recommendations
        recommended_books_content = books_df.iloc[book_indices_content]
        recommended_books_collab = books_df[books_df['book_id'].isin(recommended_book_ids_collab)]
        
        # Merge content and collaborative recommendations
        recommended_books = pd.concat([recommended_books_content, recommended_books_collab]).drop_duplicates()
    else:
        recommended_books = books_df[books_df['book_id'].isin(recommended_book_ids_collab)]
    
    # Return only relevant columns
    return recommended_books[['book_id', 'title', 'authors', 'image_url']]

def content_based(selected_book_title, books_df, cosine_sim, indices, num_recommendations=10):
    """Get content-based recommendations for a given book."""
    idx = indices[selected_book_title.lower()]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    book_indices_content = [i[0] for i in sim_scores[1:num_recommendations + 1]]  # Skip the first (itself)
    return books_df.iloc[book_indices_content], [i[1] for i in sim_scores[1:num_recommendations + 1]]
