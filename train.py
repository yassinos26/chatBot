import numpy as np
import pandas as pd
import json
import string
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
from googletrans import Translator
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# Télécharger les ressources nécessaires de NLTK
nltk.download('stopwords')
nltk.download('wordnet')

# Initialisation des objets NLTK
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Instancier un traducteur
translator = Translator()  # Initialisation du traducteur Google

# Charger et parser les données du fichier JSON
with open('chatbot.json', encoding="utf8") as data_file:
    intents = json.load(data_file)

# Initialisation des listes et dictionnaires pour les patterns, tags, et réponses
patterns = []
tags = []
responses = {}

# Extraire les patterns, tags et responses du fichier JSON
for intent in intents['intents']:
    responses[intent['tag']] = intent['responses']
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Fonction de nettoyage de texte (normalisation)
def clean_text(text):
    text = text.lower()  # Conversion en minuscules
    text = ''.join([char for char in text if char not in string.punctuation])  # Suppression de la ponctuation
    words = text.split()  # Tokenisation en mots
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatisation et suppression des stopwords
    return ' '.join(words)


# Fonction pour augmenter une phrase en remplaçant certains mots par leurs synonymes
def augment_sentence(sentence):
    words = sentence.split()
    new_sentence = []
    for word in words:
        synonyms = get_synonyms(word)
        if synonyms:
            new_sentence.append(random.choice(list(synonyms)))  # Choisir un synonyme aléatoire
        else:
            new_sentence.append(word)  # Si aucun synonyme trouvé, garder le mot original
    return ' '.join(new_sentence)

# Fonction pour obtenir les synonymes d'un mot
def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return set(synonyms)

# Fonction pour traduire du texte
def translate_text(text, dest_lang='en'):
    try:
        translation = translator.translate(text, dest=dest_lang)
        return translation.text
    except Exception as e:
        print(f"Erreur de traduction: {e}")
        return text  # Retourne le texte original en cas d'erreur

# Créer un DataFrame pandas pour organiser les données
data_df = pd.DataFrame({'patterns': patterns, 'tags': tags})

# Nettoyage et augmentation des patterns
data_df['patterns'] = data_df['patterns'].apply(clean_text)  # Nettoyage
data_df['patterns'] = data_df['patterns'].apply(augment_sentence)  # Augmentation avec des synonymes
# data_df['patterns'] = data_df['patterns'].apply(get_synonyms)  # Obtenir les synonymes d'un mot
# data_df['patterns'] = data_df['patterns'].apply(translate_text)  # Traduire du texte

# Tokenisation et padding des séquences
tokenizer = Tokenizer(num_words=2000)  # Limiter à 2000 mots les plus fréquents
tokenizer.fit_on_texts(data_df['patterns'])  # Adapter le tokenizer aux données
train_sequences = tokenizer.texts_to_sequences(data_df['patterns'])  # Transformer les patterns en séquences d'index
train_pad = pad_sequences(train_sequences)  # Padding pour obtenir des séquences de longueur égale

# Encodage des étiquettes (tags)
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(data_df['tags'])

# Définir les dimensions d'entrée et de sortie pour le modèle
input_shape = train_pad.shape[1]  # Longueur des séquences en entrée
vocabulary_size = len(tokenizer.word_index) + 1  # Taille du vocabulaire
output_length = len(set(train_labels))  # Nombre de classes (tags)

# Construction du modèle LSTM
text_input = Input(shape=(input_shape,))
text_embedding = Embedding(input_dim=vocabulary_size, output_dim=200)(text_input)
text_lstm = LSTM(200, return_sequences=False)(text_embedding)
dropout = Dropout(0.5)(text_lstm)
output = Dense(output_length, activation='softmax')(dropout)

# Compilation du modèle
model = Model(inputs=text_input, outputs=output)
model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(train_pad, train_labels, epochs=300, validation_split=0.2)  # Entraînement avec validation

# Sauvegarder le modèle
model.save('bot.keras')

# Fonction de traduction sécurisée avec gestion des erreurs
def safe_translate(translator, text, dest_lang='en'):
    try:
        translation = translator.translate(text, dest=dest_lang)
        return translation.text, translation.src  # Retourner le texte traduit et la langue source
    except Exception as e:
        print(f"Erreur de traduction: {e}")
        return text, 'en'  # Retourner le texte original en cas d'erreur

# Boucle de chat interactive
def chat():
    while True:
        # Entrée utilisateur
        user_input = input("You: ")

        # Traduire l'entrée utilisateur vers l'anglais
        translated_input, source_language = safe_translate(translator, user_input, dest_lang='en')

        # Nettoyer et tokeniser l'entrée traduite
        prediction_input_processed = clean_text(translated_input)
        prediction_input_seq = tokenizer.texts_to_sequences([prediction_input_processed])
        prediction_input_pad = pad_sequences(prediction_input_seq, maxlen=input_shape)

        # Prédire l'étiquette de la séquence
        output = model.predict(prediction_input_pad)
        predicted_tag = label_encoder.inverse_transform([np.argmax(output)])[0]

        # Sélectionner une réponse aléatoire en fonction de l'étiquette prédite
        response = random.choice(responses[predicted_tag])

        # Traduire la réponse dans la langue d'origine de l'utilisateur
        translated_response, _ = safe_translate(translator, response, dest_lang=source_language)

        # Afficher la réponse
        print(f"Bot: {translated_response}")

        # Quitter si le tag est "goodbye"
        if translated_response == "goodbye":
            break

# Démarrer la session de chat
chat()