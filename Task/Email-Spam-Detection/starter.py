from collections import defaultdict

import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

emails = pd.read_csv("emails.csv")

def process_email(text):
    text = text.lower()
    return list(text.split())

emails["words"] = emails["text"].apply(process_email)

spam = emails[emails['spam'] == 1]
normal = emails[emails['spam'] == 0]

total_emails = len(emails)
total_spam = len(spam)
total_normal = len(normal)

p_spam = total_spam / total_emails
p_normal = total_normal / total_emails

spam_word_freq = defaultdict(int)
normal_word_freq = defaultdict(int)

for words in spam["words"]:
    for word in words:
        spam_word_freq[word] += 1

for words in normal["words"]:
    for word in words:
        normal_word_freq[word] += 1

def predict_naive_bayes(word):
    p_word_given_spam = (spam_word_freq[word] + 1) / (total_spam + len(spam_word_freq))
    p_word_given_normal = (normal_word_freq[word] + 1) / (total_normal + len(normal_word_freq))
    p_spam_given_word = (p_word_given_spam * p_spam) / ((p_word_given_spam * p_spam) + (p_word_given_normal * p_normal))
    return p_spam_given_word

def calculate_posterior(email_text):
    words = process_email(email_text)
    spam_prob = 1
    normal_prob = 1
    
    for word in words:
        spam_prob *= (spam_word_freq[word] + 1) / (total_spam + len(spam_word_freq))
        normal_prob *= (normal_word_freq[word] + 1) / (total_normal + len(normal_word_freq))
    
    spam_prob *= p_spam
    normal_prob *= p_normal
    
    if spam_prob > normal_prob:
        return "spam"
    else:
        return "normal"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_word', methods=['GET'])
def predict_word():
    sample_word = request.args.get('word')
    probability = predict_naive_bayes(sample_word)
    return jsonify({"word": sample_word, "probability": probability})

@app.route('/predict_email', methods=['POST'])
def predict_email():
    sample_email = request.json.get('email')
    posterior = calculate_posterior(sample_email)
    return jsonify({"Result": posterior})

if __name__ == '__main__':
    app.run(debug=True)
