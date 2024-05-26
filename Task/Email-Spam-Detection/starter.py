import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

emails = pd.read_csv("emails.csv")

def process_email(text):
    text = text.lower()
    return list(set(text.split()))

emails["words"] = emails["text"].apply(process_email)

total_emails_count = len(emails)
spam_emails_count = len(emails[emails["spam"] == 1])
normal_emails_count = len(emails[emails["spam"] == 0])
prior_spam_prob = spam_emails_count / total_emails_count
prior_normal_prob = normal_emails_count / total_emails_count

spam_emails = emails[emails["spam"] == 1]
normal_emails = emails[emails["spam"] == 0]
spam_words = {}
normal_words = {}

for email in spam_emails["words"]:
    for word in email:
        if word in spam_words:
            spam_words[word] += 1
        else:
            spam_words[word] = 1

for email in normal_emails["words"]:
    for word in email:
        if word in normal_words:
            normal_words[word] += 1
        else:
            normal_words[word] = 1

def calculate_posteriors(word):
    likelihood_spam = (spam_words.get(word, 0) + 1) / (total_spam_words_count + vsize)
    likelihood_normal = (normal_words.get(word, 0) + 1) / (total_normal_words_count + vsize)
    return likelihood_spam, likelihood_normal

total_spam_words_count = sum(spam_words.values())
total_normal_words_count = sum(normal_words.values())
vsize = len(set(spam_words.keys()).union(set(normal_words.keys())))

def predict_naive_bayes(email):
    email_words = process_email(email)
    spam_score = np.log(prior_spam_prob)
    normal_score = np.log(prior_normal_prob)

    for word in email_words:
        likelihood_spam, likelihood_normal = calculate_posteriors(word)
        spam_score += np.log(likelihood_spam)
        normal_score += np.log(likelihood_normal)

    if spam_score > normal_score:
        return "spam"
    else:
        return "normal"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_email', methods=['POST'])
def predict_email():
    data = request.json
    email = data.get("email", "")
    result = predict_naive_bayes(email)
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
