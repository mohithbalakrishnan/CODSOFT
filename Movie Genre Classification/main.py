import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Generate Movie Dataset
# -----------------------------

data = {
    "Plot": [
        "A superhero saves the world from aliens",
        "Two college students fall in love",
        "A detective investigates a murder mystery",
        "Ghosts haunt an old mansion",
        "Scientists discover a new planet",
        "Friends go on a funny road trip",
        "A soldier fights in a dangerous war",
        "A magician protects a magical kingdom",
        "A hacker stops a cyber attack",
        "A family enjoys a vacation",
        "A zombie virus spreads across the city",
        "A prince falls in love with a princess",
        "Police chase dangerous criminals",
        "Astronauts explore another galaxy",
        "Children enter a magical forest",
        "A musician struggles to achieve success",
        "Doctors fight against a deadly virus",
        "A thief plans a bank robbery",
        "A haunted doll attacks a family",
        "An athlete prepares for the Olympics",
        "A robot becomes self-aware",
        "A group survives on a deserted island",
        "A serial killer targets innocent people",
        "A comedian makes everyone laugh",
        "A team travels through time",
        "An undercover officer catches mafia members",
        "A dragon attacks a fantasy kingdom",
        "A poor boy becomes a famous singer",
        "A family survives a natural disaster",
        "A scientist invents a time machine"
    ],
    "Genre": [
        "Action",
        "Romance",
        "Thriller",
        "Horror",
        "Sci-Fi",
        "Comedy",
        "War",
        "Fantasy",
        "Action",
        "Comedy",
        "Horror",
        "Romance",
        "Crime",
        "Sci-Fi",
        "Fantasy",
        "Drama",
        "Drama",
        "Crime",
        "Horror",
        "Sports",
        "Sci-Fi",
        "Adventure",
        "Thriller",
        "Comedy",
        "Sci-Fi",
        "Crime",
        "Fantasy",
        "Drama",
        "Adventure",
        "Sci-Fi"
    ]
}

df = pd.DataFrame(data)

print("Movie Dataset")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nGenre Distribution")
print(df["Genre"].value_counts())

# -----------------------------
# Features & Target
# -----------------------------

X = df["Plot"]
y = df["Genre"]

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Build Pipeline
# -----------------------------

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", LogisticRegression(max_iter=1000))
])

# -----------------------------
# Train Model
# -----------------------------

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# -----------------------------
# Prediction
# -----------------------------

y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------

print("\nAccuracy Score")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))
# -----------------------------
# Save Model
# -----------------------------

joblib.dump(model, "movie_genre_model.pkl")

print("\nModel Saved Successfully!")

# -----------------------------
# User Prediction
# -----------------------------

print("\n===== Movie Genre Prediction =====")

plot = input("Enter Movie Plot: ")

prediction = model.predict([plot])

print("\nPredicted Genre:", prediction[0])

print("\nProject Completed Successfully!")


