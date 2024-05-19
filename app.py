from flask import Flask, request, jsonify, render_template
import re
from openai import OpenAI
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os
load_dotenv()
app = Flask(__name__)

client = OpenAI( api_key=os.getenv('OPENAI_API_KEY') )
def preprocess(text):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

import json
def get_sentiment(text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": 'You are a helpful analyzer. Provide sentiment scores: positive, neutral, and negative for each of following words. Example text: "I love this!", Result: "neutral positive neutral'},
            {"role": "user", "content": text},
        ]
    )
    
    print('>>>',response.choices[0].message.content)
    sentiment = response.choices[0].message.content
    # data_dict = json.loads(sentiment)
    
    # positive_score = data_dict['positive']
    # neutral_score = data_dict['neutral']
    # negative_score = data_dict['negative']
    positive_score = sentiment.count('positive')
    neutral_score = sentiment.count('neutral')
    negative_score = sentiment.count('negative')

    return {'positive': positive_score, 'neutral': neutral_score, 'negative': negative_score}

#Training
train_data = pd.read_csv('df_train90.csv')
test_data = pd.read_csv('df_test10.csv')
train_data['Sentiment'] = train_data['Sentiment'].apply(preprocess)
test_data['Sentiment'] = test_data['Sentiment'].apply(preprocess)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data['Sentiment'])
X_test = vectorizer.transform(test_data['Sentiment'])

y_train = train_data['Label']
y_test = test_data['Label']

# Train the SVM classifier
classifier = SVC(kernel='linear', probability=True)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test) 

# Evaluate the classifier
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


def classify(text, scores):
    # input_data = [[scores['positive'], scores['neutral'], scores['negative']]]
    # print('input',input_data)
    # vectorizer = CountVectorizer()
    # vectorizer.fit(input_data)
    
    # # Transform input data
    input_vector = vectorizer.fit_transform(text)
    
    # # Initialize and fit classifier
    # classifier = SVC(kernel='linear')
    # target_labels = ['positive', 'neutral', 'negative']
    # classifier.fit(input_vector, target_labels)
    # prediction = classifier.predict(input_vector)
    # input_data = np.array([[scores['positive'], scores['neutral'], scores['negative']]])
    prediction = classifier.predict(input_vector)
    pred = classifier.predict_proba(input_vector)
    print(prediction, pred)
    return prediction[0]



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    user_input = request.form['text']
    preprocessed_text = preprocess(user_input)
    print(preprocessed_text)
    sentiment_scores = get_sentiment(preprocessed_text)
    print(sentiment_scores)
    trainFeature, testFeature, label, testLabel = train_test_split(preprocessed_text, sentiment_scores, test_size=0.3, random_state=0)
    sentiment = classify(testFeature, testLabel)
    return jsonify({'input': user_input, 'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)