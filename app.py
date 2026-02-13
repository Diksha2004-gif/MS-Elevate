from flask import Flask, render_template, request, jsonify
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load dataset
with open("intents.json") as file:
    data = json.load(file)

# Prepare training data
sentences = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern.lower())
        labels.append(intent["tag"])

# Train ML model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

model = LogisticRegression(max_iter=200)
model.fit(X, labels)

# Predict intent
def predict_intent(message):
    message = message.lower()
    X_test = vectorizer.transform([message])
    return model.predict(X_test)[0]

# Get response
def get_response(intent):
    for i in data["intents"]:
        if i["tag"] == intent:
            return random.choice(i["responses"])
    return "Sorry, I didn't understand. Please contact support."

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_message = request.form["msg"]
    intent = predict_intent(user_message)
    reply = get_response(intent)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)