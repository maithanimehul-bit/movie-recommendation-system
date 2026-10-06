# 🎬 Movie Recommendation System

A beginner-friendly Machine Learning project that recommends movies based on how similar their genres, keywords, and descriptions are.

## 📌 Project Overview

This project uses a **content-based movie recommendation system**.

The user enters the name of a movie, and the system finds other movies with similar content.

For example, if you enter:

```text
Iron Man
```

the system may recommend movies such as other superhero, action, or science-fiction movies.

## 🧠 How It Works

The recommendation process follows these steps:

1. Load the movie dataset using **Pandas**.
2. Clean and prepare the movie information.
3. Extract movie **genres** and **keywords**.
4. Combine genres, keywords, and the movie overview into one text feature.
5. Convert the text into numerical features using **TF-IDF Vectorization**.
6. Calculate similarity between movies using **Cosine Similarity**.
7. Display the movies with the highest similarity scores.

### Simple Workflow

```text
Movie Dataset
      ↓
Data Cleaning
      ↓
Feature Creation
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity
      ↓
Similar Movies
      ↓
Recommendations 🎬
```

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity

## 📂 Project Structure

```text
movie-recommendation-system/
│
├── data/
│   └── tmdb_5000_movies.csv
│
├── movie_recommender.py
├── README.md
├── requirements.txt
└── .gitignore
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/maithanmehul-bit/movie-recommendation-system.git
```

### 2. Open the project folder

```bash
cd movie-recommendation-system
```

### 3. Install the required Python libraries

```bash
pip install -r requirements.txt
```

## ▶️ How to Run

Run the Python program:

```bash
python movie_recommender.py
```

The program will ask:

```text
Enter the name of a movie:
```

Enter a movie name, for example:

```text
Iron Man
```

The program will then display a list of recommended movies.

## 📊 Machine Learning Concepts Learned

This project demonstrates several important concepts:

- Data preprocessing
- Feature engineering
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Cosine Similarity
- Content-Based Recommendation Systems

## 🚀 Future Improvements

Possible improvements for future versions:

- Add a graphical user interface
- Add movie posters
- Add movie ratings
- Improve recommendation quality
- Create a web application using Streamlit
- Add more recommendation features

## 👨‍💻 Author

**Mehul Maithani**

GitHub: https://github.com/maithanmehul-bit
