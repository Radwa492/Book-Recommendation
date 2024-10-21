import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Define a simple list of stopwords
stop_words = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", 
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", 
    "at", "by", "for", "with", "about", "against", "between", "into", "through", 
    "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", 
    "once", "here", "there", "when", "where", "why", "how", "all", "any", 
    "both", "each", "few", "more", "most", "other", "some", "such", "no", 
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", 
    "s", "t", "can", "will", "just", "don", "should", "now"
])

# Preprocess functions
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = re.findall(r'\w+', text)  # Simple tokenization
    tokens = [word for word in tokens if word not in stop_words]
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

def popularity_recommendations(books_df, ratings_df, num_recommendations, metric='average_rating'):
    if metric == 'average_rating':
        popular_books = ratings_df.groupby('book_id').agg({'rating': 'mean'}).rename(columns={'rating': 'average_rating'})
        popular_books = popular_books.merge(books_df, on='book_id').sort_values('average_rating', ascending=False)
    elif metric == 'ratings_count':
        popular_books = ratings_df.groupby('book_id').agg({'rating': 'count'}).rename(columns={'rating': 'ratings_count'})
        popular_books = popular_books.merge(books_df, on='book_id').sort_values('ratings_count', ascending=False)
    elif metric == 'weighted_score':
        C = ratings_df['rating'].mean()
        m = ratings_df['book_id'].value_counts().quantile(0.9)
        q_books = ratings_df.groupby('book_id').agg(average_rating=('rating', 'mean'), ratings_count=('rating', 'count'))
        q_books = q_books[q_books['ratings_count'] >= m]
        q_books['weighted_score'] = (q_books['average_rating'] * q_books['ratings_count'] + C * m) / (q_books['ratings_count'] + m)
        popular_books = q_books.merge(books_df, on='book_id').sort_values('weighted_score', ascending=False)
    else:
        raise ValueError("Metric not recognized. Choose from 'average_rating', 'ratings_count', 'weighted_score'")
    popular_books.columns = popular_books.columns.str.replace('_x', '', regex=True).str.replace('_y', '', regex=True)
    return popular_books.head(num_recommendations)

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
