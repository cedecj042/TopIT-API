
from Setup import *

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
from textstat import flesch_kincaid_grade, syllable_count
import torch




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
   flesch_score = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
   if flesch_score < 0:
     flesch_score = 0
   elif flesch_score > 100:
     flesch_score = 100 
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

def flesch_reading_ease_score(total_words, total_syllables, total_sentences):
  #  print(f"flesch: {206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)}")
   flesch_score = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
   if flesch_score < 0:
     flesch_score = 0
   elif flesch_score > 100:
     flesch_score = 100
#    print(f"flesch: {flesch_score}")
   return flesch_score

def preprocess_text(text):
    """
    Preprocess text with embedded code snippets:
    - Keep code snippets intact within <code> tags
    - Process natural language text with lemmatization
    - Preserve important question words
    - Remove unnecessary punctuation from natural text
    """
    # Define stopwords
    stop_words = set(stopwords.words('english')) - {"who", "what", "where", "how", "why", "when", "which"}

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # First, extract code blocks and replace with placeholders
    code_blocks = []

    def save_code(match):
        code = match.group(1)  # Get content between backticks
        code_blocks.append(code)
        return f"CODEMARKER{len(code_blocks)-1}ENDMARKER"

    # Replace code blocks with placeholders using unique markers
    text_with_placeholders = re.sub(r'`(.*?)`', save_code, text)

    # Process non-code text
    words = []
    # Split on whitespace while preserving our code markers
    tokens = re.split(r'(\bCODEMARKER\d+ENDMARKER\b|\s+)', text_with_placeholders)

    for token in tokens:
        if not token or token.isspace():  # Skip empty tokens and whitespace
            continue

        if token.startswith('CODEMARKER') and token.endswith('ENDMARKER'):
            # Extract the index from our marker
            index = int(re.search(r'CODEMARKER(\d+)ENDMARKER', token).group(1))
            words.append(f"<code>{code_blocks[index]}</code>")
        else:
            # Process natural language
            token = token.lower()
            # Remove punctuation except in numbers like 1.5
            if not any(c.isdigit() for c in token):
                token = re.sub(r'[^\w\s-]', '', token)

            if token and token not in stop_words:
                words.append(lemmatizer.lemmatize(token))

    # Filter out empty strings and join tokens
    words = [w for w in words if w.strip()]
    processed_text = ' '.join(words)

    return processed_text


def predict_question(text, device=DEVICE):
    """
    Predict the difficulty level of a question using the loaded DistilBERT model.
    """
    DISTILBERT_MODEL.eval()
    preprocessed_text = preprocess_text(text)
    
    # Tokenize input text
    inputs = DISTILBERT_TOKENIZER.encode_plus(
        preprocessed_text,
        None,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_token_type_ids=True,
        return_tensors="pt"
    )
    
    tokenized_question = preprocessed_text.split()
    # Extract handcrafted features
    raw_features = {
        'word_count': len(tokenized_question),
        'sentence_count': len(sent_tokenize(text)),
        'avg_word_length': np.mean([len(word) for word in tokenized_question]) if tokenized_question else 0,
        'readability': flesch_kincaid_grade(text) if text.strip() else 0,
        'unique_word_count': len(set(tokenized_question)),
        'vocabulary_diversity': len(set(tokenized_question)) / len(tokenized_question) if tokenized_question else 0,
        'complex_word_count': sum(num_of_syllables(word) > 3 for word in tokenized_question)
    }

    # Normalize handcrafted features
    normalized_features = torch.tensor([
        (raw_features[key] - FEATURE_STATS[key]['mean']) / FEATURE_STATS[key]['std']
        for key in raw_features
    ], dtype=torch.float).unsqueeze(0)  

    # Move inputs to the correct device
    ids = inputs['input_ids'].to(device)
    mask = inputs['attention_mask'].to(device)
    features = normalized_features.to(device)

    # Forward pass through the model
    with torch.no_grad():
        outputs = DISTILBERT_MODEL(ids, mask, features)
        
    # Apply softmax to get probabilities
    softmax = torch.nn.Softmax(dim=1)
    probabilities = softmax(outputs)

    # Get the predicted class index
    _, predicted_class = torch.max(outputs, dim=1)
    
    # Convert the class index to a human-readable label
    predicted_label = LABEL_ENCODER.inverse_transform([predicted_class.item()])[0]
    
        # Print probabilities for each class
    class_probabilities = probabilities.squeeze().cpu().numpy()  # Convert to numpy for easy display
    class_labels = LABEL_ENCODER.classes_

    # print(f"Predicted Label: {predicted_label}")
    # print("Class Probabilities:")
    # for class_name, prob in zip(class_labels, class_probabilities):
    #     print(f"{class_name}: {prob:.4f}")
    
    return match_difficulty(predicted_label)

def match_difficulty(prediction):
  difficulty_mapping = {
      1: "Very Easy",
      2: "Easy",
      3: "Average",
      4: "Hard",
      5: "Very Hard"
  }
  return difficulty_mapping.get(prediction, "Unknown Difficulty")  # Default for invalid input

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
    
    Args:
        query_text (str): The query text to check for similarity.
        similarity_threshold (float): The minimum cosine similarity to consider a match.
        
    Returns:
        tuple or None: (document, metadata) if a match is found, otherwise None.
    """
    results = QUESTION_DOCUMENT.similarity_search_with_score(
        query=query_text,
        k=5
    )
    for doc, score in results:
        cosine_similarity = 1 - score
        
        if cosine_similarity >= similarity_threshold:
            logging.info(f"Found similar question: {doc.page_content}")
            logging.info(f"Similarity: {cosine_similarity}")
            logging.info(f"Metadata: {doc.metadata}")
            return doc.page_content, doc.metadata
    
    return None