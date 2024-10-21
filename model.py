# model.py
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

# Set NLTK data directory
nltk_data_dir = '/content/nltk_data'
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

def content_based(selected_book_title, books_df):
    books_df = preprocess_books(books_df)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(books_df['content'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(books_df.index, index=books_df['title'].str.lower()).drop_duplicates()

    idx = indices[selected_book_title.lower()]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    book_indices = [i[0] for i in sim_scores]
    return books_df.iloc[book_indices], [i[1] for i in sim_scores]

def train_svd_model(ratings_df):
    reader = Reader(rating_scale=(ratings_df['rating'].min(), ratings_df['rating'].max()))
    data = Dataset.load_from_df(ratings_df[['user_id', 'book_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    svd_model = SVD(random_state=42)
    svd_model.fit(trainset)
    return svd_model

def hybrid_recommendations(user_id, books_df, ratings_df, svd_model, num_recommendations=20):
    all_book_ids = books_df['book_id'].unique()
    rated_books = ratings_df[ratings_df['user_id'] == user_id]['book_id'].tolist()
    books_to_predict = [book for book in all_book_ids if book not in rated_books]

    predictions = [svd_model.predict(user_id, book_id) for book_id in books_to_predict]
    
    pred_df = pd.DataFrame({
        'book_id': books_to_predict,
        'predicted_rating': [pred.est for pred in predictions]
    })

    pred_df = pred_df.sort_values('predicted_rating', ascending=False)
    top_collab_recommendations = pred_df.head(num_recommendations)
    
    combined_recommendations = top_collab_recommendations.merge(books_df, on='book_id')
    combined_recommendations = combined_recommendations[['book_id', 'predicted_rating', 'title', 'authors', 'image_url']]
    return combined_recommendations
