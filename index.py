import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def preprocess_text_with_code(text):
    """
    Preprocess text with embedded code snippets.
    - Extract code enclosed in backticks ().
    - Remove stopwords from the natural text except for important ones (e.g., 'who', 'what').
    - Concatenate preprocessed text and code snippets into a single sentence.
    """
    # Step 1: Extract code snippets enclosed in backticks
    code_snippets = re.findall(r'`.*?`', text)  # Matches text within backticks
    
    print(code_snippets)
    # Step 2: Remove code from the text for preprocessing
    text_without_code = re.sub(r'`.*?`', '', text)  # Remove code enclosed in backticks
    print(code_snippets)

    # Step 3: Preprocess the natural language text
    # Lowercase and remove non-alphanumeric characters (except spaces)
    text_without_code = re.sub(r'[^a-zA-Z0-9\s]', '', text_without_code.lower())
    tokens = word_tokenize(text_without_code)
    
    # Define stopwords and retain important words
    stop_words = set(stopwords.words('english'))
    important_words = {"who", "what", "where", "how", "why", "when"}
    stop_words = stop_words - important_words
    
    # Remove stopwords from tokens
    preprocessed_tokens = [word for word in tokens if word not in stop_words]
    
    # Combine preprocessed text and code snippets
    preprocessed_text = " ".join(preprocessed_tokens)
    code_text = " ".join(code_snippets)  # Combine all code snippets into a single string

    text = f"{preprocessed_text} {code_text}"
    # Step 4: Return the concatenated result
    return tokens, text


text = "What is your recommendation for improving the performance of this SQL query: `SELECT * FROM employees WHERE department = 'IT';`"
print(preprocess_text_with_code(text))