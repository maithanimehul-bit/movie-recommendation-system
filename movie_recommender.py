# Movie Recommendation System
# Content-Based Recommendation using TF-IDF and Cosine Similarity

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import ast
from difflib import get_close_matches


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

movies = pd.read_csv("data/tmdb_5000_movies.csv")

print("Dataset loaded successfully!")
print("Number of movies:", len(movies))


# --------------------------------------------------
# 2. SELECT IMPORTANT COLUMNS
# --------------------------------------------------

movies = movies[["title", "genres", "keywords", "overview"]]

# Remove movies with missing values
movies = movies.dropna()

print("\nColumns being used:")
print(movies.columns)


# --------------------------------------------------
# 3. CONVERT GENRES AND KEYWORDS
# --------------------------------------------------
# The genres and keywords are stored like this:
#
# [{'id': 28, 'name': 'Action'},
#  {'id': 12, 'name': 'Adventure'}]
#
# We only want the names.
# Example:
# "Action Adventure"


def extract_names(text):
    """
    Convert the genre/keyword string into
    a simple space-separated string.
    """

    try:
        data = ast.literal_eval(text)

        names = []

        for item in data:
            names.append(item["name"])

        return " ".join(names)

    except:
        return ""


movies["genres"] = movies["genres"].apply(extract_names)
movies["keywords"] = movies["keywords"].apply(extract_names)


# --------------------------------------------------
# 4. CREATE A COMBINED FEATURE
# --------------------------------------------------

movies["combined_features"] = (
    movies["genres"] + " " +
    movies["keywords"] + " " +
    movies["overview"]
)


# Convert everything to lowercase
movies["combined_features"] = movies["combined_features"].str.lower()


# --------------------------------------------------
# 5. TF-IDF
# --------------------------------------------------

tfidf = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(movies["combined_features"])

print("\nTF-IDF matrix created!")
print("Matrix shape:", tfidf_matrix.shape)


# --------------------------------------------------
# 6. CREATE MOVIE INDEX
# --------------------------------------------------

# Convert movie title into lowercase for searching

movies["title_lower"] = movies["title"].str.lower()

movie_indices = pd.Series(
    movies.index,
    index=movies["title_lower"]
).drop_duplicates()


# --------------------------------------------------
# 7. RECOMMENDATION FUNCTION
# --------------------------------------------------

def recommend_movies(movie_name, number_of_recommendations=10):

    movie_name = movie_name.lower().strip()

    # Check whether exact movie exists
    if movie_name in movie_indices:

        movie_index = movie_indices[movie_name]

    else:

        # Try to find a similar title
        possible_matches = get_close_matches(
            movie_name,
            movie_indices.index,
            n=1,
            cutoff=0.5
        )

        if not possible_matches:
            print("\nMovie not found.")
            return

        movie_index = movie_indices[possible_matches[0]]

        print(
            f"\nDid you mean: "
            f"{movies.loc[movie_index, 'title']}?"
        )


    # --------------------------------------------------
    # 8. CALCULATE COSINE SIMILARITY
    # --------------------------------------------------

    similarity_scores = linear_kernel(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()


    # --------------------------------------------------
    # 9. SORT MOVIES BY SIMILARITY
    # --------------------------------------------------

    similar_movies = similarity_scores.argsort()[::-1]


    # --------------------------------------------------
    # 10. DISPLAY RECOMMENDATIONS
    # --------------------------------------------------

    print("\nRecommended Movies:")
    print("-" * 40)

    count = 0

    for index in similar_movies:

        # Skip the movie itself
        if index == movie_index:
            continue

        print(movies.iloc[index]["title"])

        count += 1

        if count >= number_of_recommendations:
            break


# --------------------------------------------------
# 11. RUN THE PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    print("\n🎬 Movie Recommendation System")
    print("--------------------------------")

    movie = input(
        "\nEnter the name of a movie: "
    )

    recommend_movies(movie, 10)