from Setup import *

# regex
import re, uuid, requests
from pydantic import BaseModel
from datetime import datetime
import numpy as np


# for question difficulty estimation
import textstat
import inflect
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the encoder
# label_encoder = LabelEncoder()
import pandas as pd

import unicodedata
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from textstat import flesch_kincaid_grade, syllable_count
import torch


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
words_to_keep = {"how", "what", "where", "why", "when"}
stop_words = stop_words - words_to_keep


def clean_text(text):

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()  # removing extra spaces

    # Tokenize, remove stopwords, and lemmatize
    tokens = [
        lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words
    ]
    tokens = [word for word in tokens if len(word) > 2]

    return " ".join(tokens)


def cleanText(text):
    text = re.sub(r"^([0-9][0-9]|[A-Z]\)|@|©|\|\.|[0-9])\s*", "", text)
    text = re.sub(r"[+*]", "", text)
    return text


def num_of_syllables(text):
    text = re.sub(r"!(?!=)", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s\+\-\*\/\=\>\<\!]", "", text)
    #   print(text)
    words = text.split(" ")
    valid_operators = ["+", "-", "*", "/", "=", ">", "<", "<=", ">=", "==", "!="]
    operator_to_word = {
        "+": "plus",
        "-": "minus",
        "*": "times",
        "/": "divided by",
        "=": "equals",
        ">": "greater than",
        "<": "less than",
        "<=": "less than or equal to",
        ">=": "greater than or equal to",
        "==": "equal to",
        "!=": "not equal to",
    }
    words = text.split(" ")
    syl_count = 0
    inflector = inflect.engine()
    for w in words:
        if (
            len(w) > 2 and w[:2].isupper()
        ):  # Check if the first two letters are uppercase
            letters = list(w)  # Split the word into letters
            #   print(letters)
            syl_count += len(letters)
        elif w.isupper():
            if len(w) == 1:
                syl_count = syl_count + 1
                # print(w)
            else:
                letters = list(w)
                # print(letters)
                letter_count = len(letters)
                syl_count = syl_count + letter_count
        elif w.isdigit():
            num_to_text = inflector.number_to_words(w)
            #   print(num_to_text)
            syl_count = syl_count + textstat.syllable_count(num_to_text)
        elif w in valid_operators:
            operator_word = operator_to_word[w]
            #   print(operator_word)
            syl_count = syl_count + textstat.syllable_count(operator_word)
        else:
            #   print(w)
            syl_count = syl_count + textstat.syllable_count(w)

    return syl_count


def num_of_words(text):
    text_count = text.split(" ")
    #   print(f"word: {len(text_count)}")
    return len(text_count)


def num_of_sentences(text):
    sentences = re.split(r"(?<=[.?!])", text)
    sentences = [s for s in sentences if s.strip()]
    #   print(f"sentences: {len(sentences)}")
    return len(sentences)


def flesch_reading_ease_score(total_words, total_syllables, total_sentences):
    flesch_score = (
        206.835
        - 1.015 * (total_words / total_sentences)
        - 84.6 * (total_syllables / total_words)
    )
    if flesch_score < 0:
        flesch_score = 0
    elif flesch_score > 100:
        flesch_score = 100
    return flesch_score


def normalize(value, raw_min, raw_max, target_min, target_max):
    return target_min + (value + raw_min) * (target_max - target_min) / (
        raw_max - raw_min
    )


difficulty_ranges = {
    "Very Easy": (-5.0, -3.0),
    "Easy": (-2.9, -1.0),
    "Average": (-0.9, 1.0),
    "Hard": (1.1, 3.0),
    "Very Hard": (3.1, 5.0),
}


def predict_difficulty_value(text, difficulty_Level):
    total_words = num_of_words(text)
    total_syllables = num_of_syllables(text)
    total_sentences = num_of_sentences(text)
    flesch_score = flesch_reading_ease_score(
        total_words, total_syllables, total_sentences
    )
    target_min, target_max = difficulty_ranges[difficulty_Level]
    flesch_min, flesch_max = 0, 100
    return round(
        normalize(flesch_score, flesch_min, flesch_max, target_min, target_max), 1
    )


def flesch_reading_ease_score(total_words, total_syllables, total_sentences):
    flesch_score = (
        206.835
        - 1.015 * (total_words / total_sentences)
        - 84.6 * (total_syllables / total_words)
    )
    return max(0, min(100, flesch_score))


def preprocess_text(text):
    """
    Preprocess text with embedded code snippets.
    - Extract code enclosed in backticks (`).
    - Remove stopwords from the natural text except for important ones (e.g., 'who', 'what').
    - Concatenate preprocessed text and code snippets into a single sentence.
    - Returns tokens and the combined preprocessed text.
    """
    code_snippets = re.findall(r"`.*?`", text)
    text_without_code = re.sub(r"`.*?`", "", text) 
    text_without_code = re.sub(r"[^a-zA-Z0-9\s]", "", text_without_code.lower())
    tokens = word_tokenize(text_without_code)

    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in tokens if word not in stop_words]
    preprocessed_tokens = [lemmatizer.lemmatize(word) for word in filtered_words if word.isalpha()]

    preprocessed_text = " ".join(preprocessed_tokens)
    code_text = " ".join(code_snippets) 
    combined_text = f"{preprocessed_text} {code_text}".strip()
    
    return preprocessed_tokens, combined_text


def extract_features(question):
    """
    Extract features from a row using preprocessed and tokenized columns.
    """
    # Retrieve preprocessed and tokenized text
    tokenized_question, preprocessed_question = preprocess_text(question)

    # --- Initialize feature dictionary ---
    features = {}

    # --- Textual Features ---
    tfidf_vector = TFIDF_VECTORIZER.transform([preprocessed_question]).toarray()[0]  # Convert to 1D array
    features.update({f"tfidf_{i}": val for i, val in enumerate(tfidf_vector)})

    # --- Text Length Features ---
    features["word_count"] = len(tokenized_question)
    features["sentence_count"] = len(sent_tokenize(question))  # Use original text for sentence count
    features["avg_word_length"] = (
        np.mean([len(word) for word in tokenized_question]) if tokenized_question else 0
    )
    features["readability"] = flesch_kincaid_grade(question) if question.strip() else 0

    # --- Lexical Complexity ---
    unique_words = len(set(tokenized_question))
    features["unique_word_count"] = unique_words
    features["vocabulary_diversity"] = (
        unique_words / len(tokenized_question) if len(tokenized_question) > 0 else 0
    )
    features["complex_word_count"] = sum(
        syllable_count(word) > 3 for word in tokenized_question
    )

    return features


def process_and_predict(question):

    features = extract_features(question)
    # --- Convert to DataFrame ---
    feature_df = pd.DataFrame([features])

    # Normalize features
    pred_normalized = SCALER.transform(feature_df)

    # Make Prediction
    prediction_probs = RANDOM_FOREST_MODEL.predict_proba(pred_normalized)[0]
    logging.info(f"Raw model probabilities: {prediction_probs}")

    top_prediction_index = np.argmax(prediction_probs)

    top_prediction = match_difficulty(top_prediction_index)  # Assuming match_difficulty is defined elsewhere
    logging.info(f"Top predicted difficulty: {top_prediction}")

    return top_prediction


def match_difficulty(prediction):
    difficulty_mapping = {
        0: "Very Easy",
        1: "Easy",
        2: "Average",
        3: "Hard",
        4: "Very Hard",
    }
    return difficulty_mapping.get(prediction, "Unknown Difficulty") 


def get_discrimination(type):
    if type == "Very Easy":
        return 0.2
    elif type == "Easy":
        return 0.4
    elif type == "Average":
        return 0.6
    elif type == "Hard":
        return 0.8
    else:
        return 1.0


def checkExactMatch(query_text, similarity_threshold=0.90):
    """
    Check for similar questions in ChromaDB using cosine similarity.
    """
    try:
        all_documents = QUESTION_DOCUMENT.get()
        collection_size = len(all_documents.get("ids", []))

        if collection_size == 0:
            logging.warning(
                "The QUESTION_DOCUMENT collection is empty. Cannot perform similarity search."
            )
            return None
        k = min(5, collection_size)

        results = QUESTION_DOCUMENT.similarity_search_with_score(query=query_text, k=k)
        for doc, score in results:
            cosine_similarity = 1 - score
            if cosine_similarity >= similarity_threshold:
                logging.info(f"Found similar question: {doc.page_content}")
                return doc.page_content, doc.metadata

    except Exception as e:
        logging.error(f"Unexpected error during similarity search: {e}")
        return None
