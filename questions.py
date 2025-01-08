
from setup import *

#regex
import re,uuid,requests
from pydantic import BaseModel
from datetime import datetime
import numpy as np


#for question difficulty estimation
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
from nltk.tokenize import sent_tokenize,word_tokenize
from textstat import syllable_count

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

words_to_keep = {'how', 'what', 'where', 'why', 'when'}
stop_words = stop_words - words_to_keep


def clean_text(text):

    text = unicodedata.normalize('NFKC', text)
    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r'\s+', ' ', text).strip() #removing extra spaces

    # Tokenize, remove stopwords, and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]

    tokens = [word for word in tokens if len(word) > 2]

    return ' '.join(tokens)

def cleanText(text):
    text = re.sub(r'^([0-9][0-9]|[A-Z]\)|@|©|\|\.|[0-9])\s*', '', text)
    text = re.sub(r'[+*]', '', text)
    return text

def num_of_syllables(text):
  text = re.sub(r'!(?!=)', '', text)
  text = re.sub(r'[^a-zA-Z0-9\s\+\-\*\/\=\>\<\!]', '', text)
#   print(text)
  words = text.split(' ')
  valid_operators = ['+', '-', '*', '/', '=', '>', '<', '<=', '>=', '==', '!=']
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
      "!=": "not equal to"
  }
  words = text.split(' ')
  syl_count = 0
  inflector = inflect.engine()
  for w in words:
    if len(w) > 2 and w[:2].isupper():  # Check if the first two letters are uppercase
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
  sentences = re.split(r'(?<=[.?!])', text)
  sentences = [s for s in sentences if s.strip()]
#   print(f"sentences: {len(sentences)}")
  return len(sentences)

def flesch_reading_ease_score(total_words, total_syllables, total_sentences):
  #  print(f"flesch: {206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)}")
   flesch_score = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
   if flesch_score < 0:
     flesch_score = 0
   elif flesch_score > 100:
     flesch_score = 100
#    print(f"flesch: {flesch_score}")
   return flesch_score

def normalize(value, raw_min, raw_max, target_min, target_max):
  return target_min + (value + raw_min) * (target_max - target_min) / (raw_max - raw_min)

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
  flesch_score = flesch_reading_ease_score(total_words, total_syllables, total_sentences)
  target_min, target_max = difficulty_ranges[difficulty_Level]
  flesch_min, flesch_max = 0, 100
  return round(normalize(flesch_score, flesch_min, flesch_max ,target_min, target_max), 1)


def checkExactMatch(query_text):
    
  document_questions = vector_store_questions.get()['documents'] 
  
  if query_text in document_questions:
      exact_match = query_text
  else:
      exact_match = None

  return exact_match

def extract_keyword_features_for_all_categories(text):
    features = {}
    tokens = text.split()
    for category, keywords in loaded_keywords.items():
        keyword_count = sum(1 for token in tokens if token in keywords)
        keyword_frequency = keyword_count / len(tokens) if len(tokens) > 0 else 0
        features[f'keyword_count_{category}'] = keyword_count
        features[f'keyword_frequency_{category}'] = keyword_frequency
    return features


def preprocess_text(text):
    """
    Preprocess text to generate both tokenized and preprocessed versions.
    1. Lowercase the text for consistency.
    2. Remove all symbols using regex.
    3. Tokenized: Original text split into tokens.
    4. Preprocessed: Lowercased text without stopwords or special characters.
    """
    # Convert text to lowercase
    text = text.lower()

    # Remove all symbols using regex (keep only alphanumeric and spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    important_words = {"who", "what", "where", "how", "why", "when"}
    stop_words = stop_words - important_words
    preprocessed_tokens = [word for word in tokens if word not in stop_words]

    # Join tokens for clean preprocessed text
    preprocessed_text = " ".join(preprocessed_tokens)

    return tokens, preprocessed_text


def preprocess_and_predict(question, tfidf_vectorizer, example_embeddings):
    """
    Preprocess a new question and make a prediction using the trained pipeline.
    This function extracts features directly for prediction.
    Args:
        question (str): Input question as a string.
        tfidf_vectorizer: Pre-fitted TF-IDF vectorizer.
        example_embeddings (dict): Precomputed example embeddings for semantic similarity.
    Returns:
        prediction: Predicted class for the input question.
    """
    # --- Preprocessing ---
    # Lowercase and clean the input question
    question = question.lower()
    question_clean = re.sub(r'[^a-zA-Z0-9\s]', '', question)

    # Tokenize and remove stopwords
    tokens = word_tokenize(question_clean)
    stop_words = set(stopwords.words('english'))
    preprocessed_tokens = [word for word in tokens if word not in stop_words]

    # Reconstruct preprocessed question
    preprocessed_question = " ".join(preprocessed_tokens)

    # --- Initialize feature dictionary ---
    features = {}

    # --- Keyword Features ---
    # keyword_features = extract_keyword_features_for_all_categories(preprocessed_question)
    # features.update(keyword_features)

    # --- Textual Features ---
    tfidf_vector = tfidf_vectorizer.transform([preprocessed_question]).toarray()[0]
    features.update({f'tfidf_{i}': val for i, val in enumerate(tfidf_vector)})

    # --- Text Length Features ---
    features['word_count'] = len(tokens)
    features['sentence_count'] = len(sent_tokenize(question))  # Original question for sentence count
    features['avg_word_length'] = np.mean([len(word) for word in tokens]) if tokens else 0
    features['readability'] = flesch_kincaid_grade(question) if question.strip() else 0

    # --- Lexical Complexity ---
    unique_words = len(set(tokens))
    features['unique_word_count'] = unique_words
    features['vocabulary_diversity'] = unique_words / len(tokens) if len(tokens) > 0 else 0
    features['complex_word_count'] = sum(syllable_count(word) > 3 for word in tokens)

    # --- Semantic Similarity Features ---
    question_embedding = embedding_model.encode(preprocessed_question)
    similarity_scores = compute_semantic_similarity_from_embedding(question_embedding, example_embeddings)
    features.update(similarity_scores)
  
    # --- Convert to DataFrame ---
    feature_df = pd.DataFrame([features])

    # Fit and transform the training features
    pred_normalized = scaler.transform(feature_df)
    
    # --- Make Prediction ---
    prediction = classifier_model.predict(pred_normalized)
    return match_difficulty(prediction[0])

def compute_semantic_similarity_from_embedding(question_embedding, example_embeddings):
    """
    Compute the average semantic similarity of the precomputed question embedding 
    with each category's example embeddings.
    """
    # Ensure the question embedding is in 2D format
    question_embedding = question_embedding.reshape(1, -1)

    # Calculate similarity scores
    similarity_scores = {}
    for category, embeddings in example_embeddings.items():
        similarities = [
            cosine_similarity(question_embedding, example_emb.reshape(1, -1))[0][0]
            for example_emb in embeddings
        ]
        similarity_scores[f'similarity_category_{category}'] = np.mean(similarities)  # Average similarity

    return similarity_scores

 
def flesch_kincaid_grade(text):
    """
    Compute the Flesch-Kincaid Grade Level for a given text with debugging.
    """
    # Split text into sentences
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    total_sentences = len(sentences)

    # Split text into words
    words = re.findall(r'\b\w+\b', text)
    total_words = len(words)

    # Count total syllables
    total_syllables = sum(syllable_count(word) for word in words)

    # # Debugging: Print intermediate values
    # print(f"Text: {text}")
    # print(f"Total Sentences: {total_sentences}")
    # print(f"Total Words: {total_words}")
    # print(f"Total Syllables: {total_syllables}")

    # Calculate Flesch-Kincaid Grade Level
    if total_sentences == 0 or total_words == 0:  # Avoid division by zero
        return 0.0
    fkgl = (0.39 * (total_words / total_sentences)) + (11.8 * (total_syllables / total_words)) - 15.59

    # print(f"Flesch-Kincaid Grade Level: {fkgl}")
    return round(fkgl, 2)
  
def match_difficulty(prediction):
  difficulty_mapping = {
      1: "Very Easy",
      2: "Easy",
      3: "Average",
      4: "Hard",
      5: "Very Hard"
  }
  return difficulty_mapping.get(prediction, "Unknown Difficulty")  # Default for invalid input
