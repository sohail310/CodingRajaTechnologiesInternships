import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import random

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Sample social media posts dataset
random.seed(42)
positive_posts = ["I love this product!", "Such a great day!", "Amazing experience!"]
negative_posts = ["This is terrible.", "I hate waiting.", "Not impressed."]
neutral_posts = ["Just another day.", "Neutral comment.", "No strong feelings."]

all_posts = positive_posts + negative_posts + neutral_posts
labels = ["positive"] * len(positive_posts) + ["negative"] * len(negative_posts) + ["neutral"] * len(neutral_posts)

# Preprocess and tokenize the text data
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return " ".join(words)

processed_posts = [preprocess_text(post) for post in all_posts]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(processed_posts, labels, test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
predictions = classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, predictions))
