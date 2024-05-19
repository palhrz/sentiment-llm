from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from app import preprocess

# Dummy training data
texts = ["I love this!", "This is awful.", "It's okay, not great.", "Fantastic product!", "Terrible experience."]
labels = ["positive", "negative", "neutral", "positive", "negative"]

# Preprocess texts
preprocessed_texts = [preprocess(text) for text in texts]

# Convert texts to feature vectors
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(preprocessed_texts)

# Train the SVM classifier
classifier = SVC(kernel='linear')
classifier.fit(X_train, labels)
