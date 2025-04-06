from Setup import *

# regex
import re, uuid, requests
from pydantic import BaseModel
from datetime import datetime
import numpy as np
import random

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
from nltk import pos_tag
import string
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

def preprocess_text_with_code(text):
    """
    Preprocess text with embedded code snippets and preserve domain-specific terms:
    - Preserves ALLCAPS (e.g., API)
    - Preserves compound terms like CI/CD, client-server
    - Preserves MixedCase terms like DevOps, MicroService
    - Returns tokenized list and combined cleaned string
    """
    # Step 1: Extract code snippets
    code_snippets = re.findall(r'`.*?`', text)

    # Step 2: Remove code from main text for processing
    text_without_code = re.sub(r'`.*?`', '', text)

    # Step 3: Identify preserved terms
    preserved_terms = re.findall(
        r'\b(?:[A-Z]{2,}[0-9]*|[A-Z]{2,}/[A-Z0-9]{2,}|[a-zA-Z]{3,}[-/][a-zA-Z]{3,}|'  # API, CI/CD, client-server
        r'[A-Z][a-z]+(?:[A-Z][a-z]+)+)\b',                                             # DevOps, MicroService
        text_without_code
    )
    # Step 4: Clean unwanted punctuation (keep - and /)
    text_cleaned = re.sub(r'[^\w\s/-]', '', text_without_code)

    # Step 5: Tokenize
    tokens = word_tokenize(text_cleaned)

    # Step 6: Process tokens
    processed_tokens = []
    for token in tokens:
        if token in preserved_terms:
            processed_tokens.append(token)  # Keep preserved terms as-is
        else:
            token_lower = token.lower()
            lemmatized = lemmatizer.lemmatize(token_lower)
            if token_lower not in stop_words and lemmatized.isalpha() and len(lemmatized) > 1:
                processed_tokens.append(lemmatized)

    # Step 7: Combine tokens + code
    preprocessed_text = " ".join(processed_tokens)
    code_text = " ".join(code_snippets)
    combined_text = f"{preprocessed_text} {code_text}".strip()

    return processed_tokens, combined_text

def stopword_ratio(tokenized_question):
    return sum(1 for word in tokenized_question if word.lower() in stop_words) / len(tokenized_question) if tokenized_question else 0

def noun_verb_ratio(preprocessed_question):
    tokens = word_tokenize(preprocessed_question)
    pos_tags = pos_tag(tokens)
    nouns = sum(1 for _, tag in pos_tags if tag.startswith("NN"))
    verbs = sum(1 for _, tag in pos_tags if tag.startswith("VB"))
    return nouns / verbs if verbs > 0 else 0

def extract_keyword_features_for_all_categories(text):
    features = {}
    tokens = text.lower().split()
    weight = 3.0  # adjust as needed

    for category, keywords in KEYWORDS.items():
        keyword_count = sum(1 for token in tokens if token in keywords)
        keyword_frequency = keyword_count / len(tokens) if tokens else 0
        features[f'keyword_count_{category}'] = keyword_count * weight
        features[f'keyword_frequency_{category}'] = keyword_frequency * weight

    features['total_blooms_keyword_count'] = sum(
        features[f'keyword_count_{cat}'] for cat in KEYWORDS
    )

    return features

def extract_features(question):
    """
    Extract features from a row using preprocessed and tokenized columns.
    """
    # Retrieve preprocessed and tokenized text
    tokenized_question, preprocessed_question = preprocess_text_with_code(question)

    # --- Initialize feature dictionary ---
    features = {}

    # --- Textual Features ---
    tfidf_vector = TFIDF_VECTORIZER.transform([preprocessed_question]).toarray()[0]  # Convert to 1D array
    features.update({f"tfidf_{i}": val for i, val in enumerate(tfidf_vector)})
    features.update(extract_keyword_features_for_all_categories(question))
    
    # --- Text Length Features ---
    features["avg_word_length"] = (np.mean([len(word) for word in tokenized_question]) if tokenized_question else 0)
    features["readability"] = flesch_kincaid_grade(question) if question.strip() else 0

    # --- Lexical Complexity ---
    unique_words = len(set(tokenized_question))
    features["unique_word_count"] = unique_words
    features["vocabulary_diversity"] = (unique_words / len(tokenized_question) if len(tokenized_question) > 0 else 0)
    features["complex_word_count"] = sum(syllable_count(word) > 3 for word in tokenized_question)

    # --- New Advanced Linguistic Features ---
    features['stopword_ratio'] = stopword_ratio(question)
    features['noun_verb_ratio'] = noun_verb_ratio(question)


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

import warnings
from sklearn.exceptions import DataConversionWarning

def process_and_predict(question):
    # Extract features
    features = extract_features(question)
    feature_df = pd.DataFrame([features])

    # Normalize using fitted scaler
    scaled_array = SCALER.transform(feature_df)
    pred_normalized = pd.DataFrame(scaled_array, columns=feature_df.columns)

    # Predict class probabilities
    prediction_probs = RANDOM_FOREST_MODEL.predict_proba(pred_normalized.values)[0]
    top_prediction_index = np.argmax(prediction_probs)

    # Map to label
    top_prediction = match_difficulty(top_prediction_index)
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
        return round(random.uniform(0.1, 0.4), 2)
    elif type == "Easy":
        return round(random.uniform(0.4, 0.8), 2)
    elif type == "Average":
        return round(random.uniform(0.8, 1.2), 2)
    elif type == "Hard":
        return round(random.uniform(1.2, 1.6), 2)
    else:
        return round(random.uniform(1.6, 2.0), 2)



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
