from flask import Flask, request, jsonify, render_template
import re

app = Flask(__name__)

def preprocess(text):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text


def get_sentiment(text):
    # TODO: implement sentiment analysis
    return {'positive': 0, 'neutral': 0, 'negative': 0}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # llm request and sentiment
    return

if __name__ == '__main__':
    app.run(debug=True)