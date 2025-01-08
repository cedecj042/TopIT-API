import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from textstat import flesch_kincaid_grade, syllable_count
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string


# Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 

stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)


import joblib
# question difficulty estimation model 
reference_embeddings = joblib.load('models/embeddings.pkl')
classifier_model = joblib.load('models/trained_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
scaler = joblib.load('models/scaler.pkl')
# loaded_keywords = joblib.load('models/category_keywords.pkl')


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
    # stop_words = set(stopwords.words('english'))
    # important_words = {"who", "what", "where", "how", "why", "when"}
    # stop_words = stop_words - important_words
    # preprocessed_tokens = [word for word in tokens if word not in stop_words]

    # Join tokens for clean preprocessed text
    preprocessed_text = " ".join(tokens)

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
    tokens, preprocessed_question = preprocess_text(question)
    
    print(preprocessed_question)

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
    question_embedding = embedding_model.encode(question)
    similarity_scores = compute_semantic_similarity_from_embedding(question_embedding, example_embeddings)
    features.update(similarity_scores)
  
    # --- Convert to DataFrame ---
    feature_df = pd.DataFrame([features])

    # Fit and transform the training features
    pred_normalized = scaler.transform(feature_df)
    
    print(f"Shape of feature_df: {pred_normalized.shape}")
    
    
    # --- Make Prediction ---
    prediction = classifier_model.predict(pred_normalized)

    return prediction

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

    # Debugging: Print intermediate values
    print(f"Text: {text}")
    print(f"Total Sentences: {total_sentences}")
    print(f"Total Words: {total_words}")
    print(f"Total Syllables: {total_syllables}")

    # Calculate Flesch-Kincaid Grade Level
    if total_sentences == 0 or total_words == 0:  # Avoid division by zero
        return 0.0
    fkgl = (0.39 * (total_words / total_sentences)) + (11.8 * (total_syllables / total_words)) - 15.59

    print(f"Flesch-Kincaid Grade Level: {fkgl}")
    return round(fkgl, 2)


question = """
A company needs to calculate the total profit by subtracting total expenses from total revenue. Which of the following Python functions correctly performs this task?
"""
prediction = preprocess_and_predict(question, tfidf, reference_embeddings)
print(prediction)