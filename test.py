from tensorflow.keras.models import load_model
import numpy as np
import json
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
import string
import random
from googletrans import Translator
import speech_recognition as sr
import pyttsx3

# Load the model
model = load_model('bot.keras')

# Initialize the recognizer and text-to-speech engine
recognizer = sr.Recognizer()

engine = pyttsx3.init()

# Load intents (assuming the JSON file is available)
with open('chatbot.json', encoding="utf8") as file:
    intents = json.load(file)

# Prepare responses dictionary and tokenizer/label encoder
responses = {}
patterns = []
tags = []

for intent in intents['intents']:
    responses[intent['tag']] = intent['responses']
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Initialize and fit Tokenizer and LabelEncoder
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(patterns)

Lencoder = LabelEncoder()
Lencoder.fit(tags)

# Get input shape from model
input_shape = model.input_shape[1]

# Chat loop
while True:
    try:
        # Choose between voice or text input
        input_type = input("Enter 't' for text or 'v' for voice: ").lower()

        if input_type == 't':
            prediction_input = input('You: ')
        elif input_type == 'v':
            with sr.Microphone() as source:
                print("Listening...")
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)
                prediction_input = recognizer.recognize_google(audio)
                print(f"You: {prediction_input}")

        # Translate input if necessary
        translator = Translator()
        translation = translator.translate(prediction_input, dest='en')
        Language_code = translation.src
        prediction_input = translation.text

        # Removing punctuation and converting to lowercase
        prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
        prediction_input = ''.join(prediction_input)
        texts_p = [prediction_input]

        # Tokenizing and padding
        prediction_input = tokenizer.texts_to_sequences(texts_p)
        prediction_input = np.array(prediction_input).reshape(-1)
        prediction_input = pad_sequences([prediction_input], input_shape)

        # Getting output from model
        output = model.predict(prediction_input)
        output = output.argmax()
        # Finding the right tag and predicting
        response_tag = Lencoder.inverse_transform([output])[0]
        translation = translator.translate(random.choice(responses[response_tag]), Language_code)
        print(f"Going Compta: {translation.text}")
        # Speak the response (optional)
        engine.say(translation.text)
        engine.runAndWait()
        if response_tag == "goodbye":
            break
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
