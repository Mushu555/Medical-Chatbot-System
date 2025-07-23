# import json 
# import numpy as np
# import re
# import pickle
# import nltk
# from tensorflow import keras
# from flask import Flask, request
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# app = Flask(__name__)

# def preprocess_text(text):
#     stop_words = set(stopwords.words('english'))
#     lemmatizer = WordNetLemmatizer()
#     text = re.sub('[^a-zA-Z]', ' ', text)
#     words = nltk.word_tokenize(text)
#     words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
#     return ' '.join(words)

# # Load resources once
# with open("intents.json") as file:
#     intents = json.load(file)

# model = keras.models.load_model('./chat_model')
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)
# with open('label_encoder.pickle', 'rb') as enc:
#     lbl_encoder = pickle.load(enc)

# @app.route('/chat', methods=['POST'])
# def chat():
#     inp = request.json['input']
#     preprocessed = preprocess_text(inp)
#     sequence = tokenizer.texts_to_sequences([preprocessed])
#     padded = keras.preprocessing.sequence.pad_sequences(sequence, truncating='post', maxlen=200)
#     prediction = model.predict(padded)
#     tag = lbl_encoder.inverse_transform([np.argmax(prediction)])[0]
#     confidence = np.max(prediction)

#     for intent in intents['intents']:
#         if intent['tags'][0] == tag:
#             return {'response': intent['answer'], 'score': str(confidence)}

#     return {'response': "Sorry, I didn't understand that.", 'score': str(confidence)}

# if __name__ == "__main__":
#     app.run(host="127.0.0.1", port=8090)
import json 
import numpy as np
import re
import pickle
import nltk
from tensorflow import keras
import random
from flask import Flask, jsonify, request
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
    return ' '.join(words)

# Load resources once
with open("intents.json") as file:
    intents = json.load(file)

model = keras.models.load_model('./chat_model')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

@app.route('/')
def home():
    return "Medical Chatbot API is running. Use the /chat endpoint with POST requests."

@app.route('/chat', methods=['POST'])
def chat():
    inp = request.json['input']
    preprocessed = preprocess_text(inp)
    sequence = tokenizer.texts_to_sequences([preprocessed])
    padded = keras.preprocessing.sequence.pad_sequences(sequence, truncating='post', maxlen=200)
    prediction = model.predict(padded)
    tag = lbl_encoder.inverse_transform([np.argmax(prediction)])[0]
    confidence = np.max(prediction)

    for intent in intents['intents']:
        if 'tags' in intent and intent['tags'] and intent['tags'][0] == tag:
            if 'responses' in intent:
                response_text = random.choice(intent['responses'])
            else:
                response_text = "Sorry, I don't have a response for that yet."

    return jsonify({
        "response": response_text,
        "score": str(confidence) # type: ignore
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8090)

