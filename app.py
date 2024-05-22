from flask import Flask, request, jsonify, render_template
import re
from openai import OpenAI
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
import nltk
from nltk.corpus import stopwords
from dotenv import load_dotenv
import os
import json

load_dotenv()
app = Flask(__name__)
client = OpenAI( api_key=os.getenv('OPENAI_API_KEY'))

def preprocess(text):
    # Remove punctuation, special char
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

def get_sentiment(text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", 
                          "content": 'You are a sentiment analysis tool. Analyze the sentiment of the following text and provide the sentiment scores in the format: "negative", [0.9,0.05,0.05] /"neutral",[0.2,0.6,0.2] /"positive,[0.1,0.2,0.7]". Ensure the scores accurately reflect overall text and follow their order [negative, neutral, positive].'},

            #  "content": 'You are a sentiment analysis tool. Analyze the sentiment of the following text and provide the sentiment scores in the format: "negative"/"neutral"/"positive", [<negative_score>, <neutral_score>, <positive_score>]. Ensure the scores accurately reflect each words and total all of them equals to 1.'},
            {"role": "user", "content": text},
        ]
    )
    
    print('>>>',response.choices[0].message.content)
    sentiment = response.choices[0].message.content
    sentiment_label = sentiment.split(",")[0].strip()

    # Extracting and converting the scores to a list of floats
    scores_str = re.search(r'\[(.*?)\]', sentiment).group(1)
    scores_list = list(map(float, scores_str.split(',')))
    # print(negative_match,'+',neutral_match,'+',positive_match)
    # return sentiment
    return sentiment_label, scores_list

#Training
# train_data = pd.read_csv('df_train90.csv')
# test_data = pd.read_csv('df_test10.csv')
data = pd.read_csv('chat_dataset.csv')
# train_data['Sentiment'] = train_data['Sentiment'].apply(preprocess)
# test_data['Sentiment'] = test_data['Sentiment'].apply(preprocess)
data['message'] = data['message'].apply(preprocess)
X = data['message']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# vectorizer = CountVectorizer()
# X_train = vectorizer.fit_transform(train_data['Sentiment'])
# X_test = vectorizer.transform(test_data['Sentiment'])
# X_train = (train_data['Sentiment'])
# X_test = (test_data['Sentiment'])
# y_train = train_data['Label']
# y_test = test_data['Label']

# Basic SVM setup
# classifier = SVC(kernel='linear', probability=True)
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test) 
# Evaluate the classifier
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# SVM with auto tuning
en_stopwords = set(stopwords.words("english"))
vectorizer = CountVectorizer(
    analyzer = 'word',
    lowercase = True,
    ngram_range=(1, 1),
    stop_words = en_stopwords)
kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
pipeline_svm = make_pipeline(vectorizer, 
                            SVC(probability=True, kernel="linear", class_weight="balanced"))

grid_svm = GridSearchCV(pipeline_svm,
                    param_grid = {'svc__C': [0.01, 0.1, 1]}, 
                    cv = kfolds,
                    scoring="accuracy",
                    verbose=1,   
                    n_jobs=1) 

grid_svm.fit(X_train, y_train)
grid_svm.score(X_test, y_test)
# print(grid_svm.best_params_)
# print(grid_svm.best_score_)

def classify(text, scores):
    #direct svm
    # text = ["I hate you stupid"]
    # text = vectorizer.transform(text)
    # prediction = classifier.predict(text)
    # pred = classifier.predict_proba(text)

    #grid svm 
    res = grid_svm.predict([text])
    res2 = grid_svm.predict_proba([text])
    # print('Res', res)
    # print('Res2',res2[0])
    # print(prediction)
    # print(pred)
    # print(grid_svm.score(res, y_test)))
    # return prediction[0]
    return res[0], res2[0]



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    user_input = request.form['text']
    preprocessed_text = preprocess(user_input)
    print(preprocessed_text)
    sentimentLLM, scoreLLM = get_sentiment(preprocessed_text)
    print('Sentiment LLM : ', sentimentLLM, scoreLLM)
    label, scores = classify(preprocessed_text, preprocessed_text)
    print('Sentiment ML : ', label, scores)

    combine = [(openai + ml) / 2  for openai, ml in zip(scoreLLM, scores)]
    
    # Determine the final sentiment based on combined scores
    lab = ["negative", "neutral", "positive"]
    final = lab[np.argmax(combine)]
    print('Final Sentiment : ', final)
    print('Combined Scores : ', combine)
    return jsonify({'input': user_input, 'sentiment': final, 'scores': combine})

if __name__ == '__main__':
    app.run(debug=True)