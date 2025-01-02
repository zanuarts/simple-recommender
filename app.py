from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Create Flask app
app = Flask(__name__)

url = 'http://files.grouplens.org/datasets/movielens/ml-100k/u.item'
col_names = ['movieId', 'movieTitle', 'releaseDate', 'videoReleaseDate', 'IMDbURL', 
                'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                'Crime', 'Documentary', 'Drama', 'Fantasy', 'FilmNoir', 'Horror', 
                'Musical', 'Mystery', 'Romance', 'SciFi', 'Thriller', 'War', 'Western']

    # Load data into DataFrame
movie_data = pd.read_csv(url, sep='|', encoding='latin-1', names=col_names, header=None)
movie_data = movie_data[['movieId', 'movieTitle'] + col_names[5:]]  # Keep only relevant columns

    # Create genre matrix (Movies x Genres)
genre_matrix = movie_data.drop(columns=['movieId', 'movieTitle']).copy()
# Compute Cosine Similarity between movies
cosine_sim = cosine_similarity(genre_matrix)

# Function to get similar movies based on movie title
def get_similar_movies(movie_title, cosine_sim=cosine_sim):
    try:
        idx = movie_data.index[movie_data['movieTitle'] == movie_title].tolist()[0]
    except IndexError:
        return ["Movie title not found. Please try another title."]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:11]]  # Get top 10 similar movies
    recommended_movies = movie_data['movieTitle'].iloc[movie_indices]
    return recommended_movies.tolist()

# Flask route to handle home page and recommendations
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        movie_title = request.form['movieTitle']
        recommendations = get_similar_movies(movie_title)
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
