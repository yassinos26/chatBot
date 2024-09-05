from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
import json
import random
import string
from googletrans import Translator
import pyttsx3
import speech_recognition as sr

# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Load the model
model = load_model('bot.keras')
# Initialize Tokenizer and LabelEncoder
tokenizer = Tokenizer(num_words=2000)
Lencoder = LabelEncoder()
# Load intents (assuming the JSON file is available)
with open('chatbot.json', encoding="utf8") as file:
    intents = json.load(file)
# Prepare responses dictionary
responses = {}
patterns = []
tags = []
for intent in intents['intents']:
    responses[intent['tag']] = intent['responses']
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])
# Fit Tokenizer and LabelEncoder
tokenizer.fit_on_texts(patterns)
Lencoder.fit(tags)
# Get input shape from model
input_shape = model.input_shape[1]
# Initialize text-to-speech engine
engine = pyttsx3.init()
# Function to process input and get response
def get_response(input_text):
    translator = Translator()
    translation = translator.translate(input_text, dest='en')
    language_code = translation.src
    input_text = translation.text

    input_text = [letters.lower() for letters in input_text if letters not in string.punctuation]
    input_text = ''.join(input_text)
    texts_p = [input_text]

    input_text = tokenizer.texts_to_sequences(texts_p)
    input_text = np.array(input_text).reshape(-1)
    input_text = pad_sequences([input_text], input_shape)

    output = model.predict(input_text)
    output = output.argmax()
    response_tag = Lencoder.inverse_transform([output])[0]
    response_text = random.choice(responses[response_tag])

    translation = translator.translate(response_text, language_code)
    return translation.text
# Route for text-based chat
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = get_response(user_input)
    return jsonify({"response": response})
# Route for voice-based chat
@app.route('/voice_chat', methods=['POST'])
def voice_chat():
    recognizer = sr.Recognizer()
    audio_data = request.files['audio']
    with sr.AudioFile(audio_data) as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.record(source)
        try:
            input_text = recognizer.recognize_google(audio)
            response = get_response(input_text)
            engine.say(response)
            engine.runAndWait()
            return jsonify({"response": response})
        except sr.UnknownValueError:
            return jsonify({"response": "Sorry, I did not understand that."})
        except sr.RequestError as e:
            return jsonify({"response": f"Could not request results; {e}"})
if __name__ == '__main__':
    app.run(debug=True)