# for llamaparse
# from llama_parse import LlamaParse
# from llama_index.core import SimpleDirectoryReader
# langchain

# from langchain_chroma import Chroma
# from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
# from langchain.chains import RetrievalQA
# from langchain_community.llms import Ollama
# from langchain_core.documents import Document
# from sentence_transformers import SentenceTransformer
import chromadb.utils.embedding_functions as embedding_functions

from chromadb.config import Settings

#OpenAI
from openai import OpenAI

from uuid import uuid4

#Base64 images conversion
import base64, logging
from PIL import Image
from io import BytesIO
from fastapi import HTTPException

#handling json files
import json
from PIL import UnidentifiedImageError

#regex
import re,uuid,requests
from pydantic import BaseModel
from datetime import datetime

from setup import *

#for question difficulty estimation
import textstat
import inflect
import re
import joblib
from sklearn.preprocessing import LabelEncoder

# Initialize the encoder
# label_encoder = LabelEncoder()
import pandas as pd
# Configure logging
logging.basicConfig(
    filename="app.log",  # Path to the log file
    level=logging.INFO,  # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S",  # Date format for log messages
)

class QuestionFormat(BaseModel):
    course_id: int
    course_title: str
    questionType: str
    numOfVeryEasy: int
    numOfEasy: int
    numOfAverage: int
    numOfHard: int
    numOfVeryHard: int


class Difficulty(BaseModel):
    numOfVeryEasy: int
    numOfEasy: int
    numOfAverage: int
    numOfHard: int
    numOfVeryHard: int

# Define Questions model
class Question(BaseModel):
    type: str
    difficulty: Difficulty
    
# Define the main request model
class CreateQuestionsRequest(BaseModel):
    course_id: int
    course_title: str
    questions: list[Question]
    

import nltk
import unicodedata
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

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
    "very easy": (-5.0, -3.0),
    "easy": (-2.9, -1.0),
    "average": (-0.9, 1.0),
    "hard": (1.1, 3.0),
    "very hard": (3.1, 5.0),
}

def predict_difficulty_value(text, difficulty_Level):
  total_words = num_of_words(text)
  total_syllables = num_of_syllables(text)
  total_sentences = num_of_sentences(text)
  flesch_score = flesch_reading_ease_score(total_words, total_syllables, total_sentences)
  target_min, target_max = difficulty_ranges[difficulty_Level]
  flesch_min, flesch_max = 0, 100
  return round(normalize(flesch_score, flesch_min, flesch_max ,target_min, target_max), 1)

# Load the trained model and TF-IDF vectorizer  
rf_classifier = joblib.load('question_difficulty_model/random_forest_model.pkl')
tfidf = joblib.load('question_difficulty_model/tfidf_vectorizer.pkl')

def queryHierarchicalIndex(query_text, level=None):
    # Define filter criteria if the level is specified
    filter_criteria = {"level": level} if level else {}

    # Perform a similarity search using ChromaDB
    results = vector_store.similarity_search(
        query=query_text,  # The query string
        k=5,                    # Number of similar documents to retrieve
        filter=filter_criteria,   # Filter criteria if needed
        include=["documents"]
    )

    return results

def checkExactMatch(query_text):

  document_questions = vector_store_questions.get()['documents'] 

  if query_text in document_questions:
      exact_match = query_text
  else:
      exact_match = None

  return exact_match
def ModelQuerywithRAG(input, course_id):

    retriever = vector_store.as_retriever(search_kwargs={
        "filter": {
            "$and": [
                {"course_id": {"$eq": course_id}},
                {"type": {"$in": ["Module", "Lesson", "Section", "Subsection", "Table", "Figure", "Code"]}}
            ]
        }
    })
    retriever = vector_store.as_retriever()
    
    # Define prompt template
    template = """
    TOPCIT (Test of Practical Competency in IT) is designed to assess competencies in practical IT skills such as programming, algorithm problem-solving, and 
    IT business understanding. TOPCIT questions are typically scenario-based, requiring critical thinking and practical application of knowledge in areas like 
    software development, database management, algorithms, and IT ethics.
    You are a Information Technology College Teacher that is handling Reviewer for Information Technology certification reviewers.
    You are tasked to create questions for simulated exams for topcit.

    <context>
    {context}
    </context>

    Query: {input}
    """

    # Create a prompt template
    prompt_template = ChatPromptTemplate.from_template(template)
    # Create a chain 
    doc_chain = create_stuff_documents_chain(llm, prompt_template)
    chain = create_retrieval_chain(retriever, doc_chain)

    # User query 
    response = chain.invoke({"input": input})
    return response['answer']


multiple_choice_single = """{
    "question": "What is the primary function of a compiler?",
    "questionType": "Multiple Choice - Single",
    "answer": "Compilation",
    "difficulty_level": "very easy",
    "difficulty_value: -4.0,
    "discrimination": -4.5,
    "choices": ["Execution","Compilation","Interpretation","Debugging"]
},"""

multiple_choice_many = """{
    "question": "Which of the following programming languages is known for its simplicity and ease of use?",
    "questionType": "Multiple Choice - Many",
    "answer": ["Python","Ruby"],
    "difficulty_level": "easy",
    "difficulty_value: -2.9,
    "discrimination": 2.2,
    "choices": ["Java","C++","Python","Ruby"]
},"""

identification = """ {
    "question": "What is the term for a program that translates one programming language into another?",
    "questionType": "Identification",
    "answer": "Interpreter",
    "difficulty_level": "very easy",
    "difficulty_value: -4.0,
    "discrimination": 4.5
}"""
 
questionType = ['Identification','Multiple Choice - Single','Multiple Choice - Many']

correct_keys_for_multiple_choice = {
        "question",
        "questionType",
        "answer",
        "difficulty_level",
        "difficulty_value",
        "discrimination",
        "choices"
    }

correct_keys_for_identification = {
        "question",
        "questionType",
        "answer",
        "difficulty_level",
        "difficulty_value",
        "discrimination",
    }

def validate_question_format(result, question_type):
    # Define correct format based on question type
    if question_type == "Multiple Choice - Single":
        correct_format = {
            "question": str,
            "questionType": str,
            "answer": str,
            "difficulty": str,
            "discrimination": float,
            "choices": list
        }
    elif question_type == "Multiple Choice - Many":
        correct_format = {
            "question": str,
            "questionType": str,
            "answer": list,
            "difficulty": str,
            "discrimination": float,
            "choices": list
        }
    elif question_type == "identification":
        correct_format = {
            "question": str,
            "questionType": str,
            "answer": str,
            "difficulty": str,
            "discrimination": float
        }

    for key, expected_type in correct_format.items():
        if key not in result or not isinstance(result[key], expected_type):
            return False 

    return True 


def check_answers(result, questionType):
    #checking for answers 
    #pulihi lang ang "identification" sa ipass sa parameter nga questionType
    if questionType == "identification":
        answer = result['answer'].split()
        if not (1 <= len(answer) < 3):
            raise ValueError(f"Identification answer has {len(answer)} words; it should be atleast 1 or at most 2..")
        for key in result:
            if key not in correct_keys_for_identification:
                raise ValueError(f'{key} not in the correct keys for identifcation')


    #pulihi lang ang "multiple_choice_multiple_answer" sa ipass sa parameter nga questionType
    if questionType == "multiple_choice_multiple_answer":
        correct_answers_len = len(result['answer'])
        if correct_answers_len < 2: 
            raise ValueError("Multiple choice multiple answers is not atleast 2")
        
        for answer in result['answer']:
            if answer not in result['choices']:
                raise ValueError(f"Correct answer '{answer}' is not found in the choices.")
        for key in result:
            if key not in correct_keys_for_multiple_choice:
                raise ValueError(f'{key} not in the correct keys for multiple choice question type')

    #pulihi lang ang "multiple_choice_multiple_answer" sa ipass sa parameter nga questionType
    if questionType == "multiple_choice_single_answer":
        correct_answer = result["answer"]
        if correct_answer not in result['choices']:
            raise ValueError(f"Correct answer '{correct_answer}' is not found in the choices.")
        for key in result:
            if key not in correct_keys_for_multiple_choice:
                raise ValueError(f'{key} not in the correct keys for multiple choice question type')
        


def createQuestions(data: QuestionFormat):
    # Define the question type
    if data.questionType == "Multiple Choice - Single":
        example = multiple_choice_single
        questionTypewithDescription = "Multiple Choice - Single"
    elif data.questionType == "Multiple Choice - Many":
        example = multiple_choice_many
        questionTypewithDescription = "Multiple Choice - Many must have atleast 2 answers and don't put only 1 answer"
    elif data.questionType == "Identification":
        example = identification
        questionTypewithDescription = "Identification must have a maximum of 2 words in 1 correct answer"

    try:     
        result_questions = {
                "course_id": data.course_id,
                "course_title": data.course_title ,
                "questions": [   
                ]
        }

        counters = {
            'countOfVeryEasy': 0,
            'countOfEasy': 0,
            'countOfAverage': 0,
            'countOfHard': 0,
            'countOfVeryHard': 0
        }

        while True:
            instructions = f"""
            Generate a question and answer pairs in the form of {questionTypewithDescription} based on the content of the {data.course_title} course. 

            - **Create a realistic and practical scenario** related to {data.course_title} that tests knowledge, critical thinking, or application of concepts.  
            - **All questions must comply with Bloom's Taxonomy** as defined below. Each level is linked to its corresponding difficulty:  

            - **Very Easy (Remember):** Must include one or more of the following verbs: **state, list, recite, outline, define, name, match, quote, recall, identify, label, recognize**.  
            - **Easy (Understand):** Must include one or more of the following verbs: **complete, describe, explain, paraphrase, restate, summarize, contrast, interpret, discuss**. Avoid starting these questions with 'what' or 'which'.  
            - **Average (Apply):** Must include one or more of the following verbs: **calculate, predict, apply, solve, use, demonstrate, model, perform, present**.  
            - **Hard (Analyze):** Must include one or more of the following verbs: **distinguish, classify, break down, categorize, analyze, diagram, illustrate, criticize, simplify, associate**.  
            - **Very Hard (Evaluate):** Must include one or more of the following verbs: **justify, argue, choose, relate, determine, defend, judge, grade, compare, support, convince, select, evaluate**.  

            The difficulty and discrimination values **must strictly adhere** to the following ranges:  
            - Very Easy: -5.0 to -3.0  
            - Easy: -2.9 to -1.0  
            - Average: -0.9 to 1.0  
            - Hard: 1.1 to 3.0  
            - Very Hard: 3.1 to 5.0  

            The discrimination value **must be rounded to the nearest tenth**.  

            All questions must follow these rules:  
            - Avoid starting questions with "what, how, when, where, and which" to promote conceptual thinking.  
            - The question must be **simple, clear, concise**, and **directly related to the course content** ({data.course_title}).  
            - The question should be suitable for the **TOPCIT exam format**, balancing clarity and challenge.  
            - The question must be **aligned with the specified difficulty** and Bloom's category.  
            - **Avoid uncommon synonyms** and maintain a consistent, formal writing style.  
            

            For questions with Understand(easy) difficulty, follow the text format from this list of examples below:

                - Compare Calliope with Howie. Use the word bank.
                - Describe 4 types of coupling in software design.
                - Describe how Phillip and Timothy survived on the Cay.
                - Describe in prose what is shown in graph form.
                - Describe the Pareto Principle in statistical software quality assurance.
                - Describe what goes in each of the four areas on the first page of notes
                - Describe what took place as the Hato was sinking.
                - Describe how the linear polarization method can yield corrosion rates
                - Describe the major clinical differences between visceral and somatic pain
                - Define the transition state and activation energy in relation to catalytic power of an enzyme.
                - Compare and contrast two treatments of either drapery or animals by Greek sculptors, establishing their artistic contexts
                - Describe the circumstances when tertiary treatment of wastewater is necessary. 
                - Identify THREE (3) types of mortar used in wall construction. 
                - Identify which country has an absolute advantage and comparative advantage in the production of pizza and cloth. 
                - Identify the advantages and disadvantages of in situ and ex situ means of remediating polluted soils. 
                - Explain Machine to Machine Architecture
                - Describe what happens during the negative feedback loop of thermoregulation. 
                - What is reverberation time and how is it calculated?
                - Explain the phenomenon of spontaneous polarization in ferroelectric materials.
                - Define CDF and state any four properties of CDF.

            For questions with Understand(easy) difficulty, avoid generating questions about "version control systems".

            The AI should **generate questions** with **Question Type** of {questionTypewithDescription}. 
            
            The AI must generate exactly **50 questions** and each difficulty must align with the specified Bloom's Taxonomy level:  
            - **10 Very Easy questions**  
            - **10 Easy questions**  
            - **10 Average questions**  
            - **10 Hard questions**  
            - **10 Very Hard questions**  

            It should be stored in a JSON format like this and don't put any text beside this:
            The **output must be in this exact JSON format**:  
            
            ```json
            {{  
            {example}
            }}
            """

            response = ModelQuerywithRAG(instructions, data.course_id)

            logging.info(f"Response: {response}")
            if not response:
                logging.error("Error: Empty response received from model")
            else:
                cleaned_response = response.replace('```json', '').replace('```', '').strip()
                
            fixed_text = re.sub(r'(\}\s*|\]\s*|\w\s*")(\s*"|\s*\{)', r'\1,\2', cleaned_response)
            result = json.loads(fixed_text)
            print(f"result : \n {result}")
        
            if (
                counters['countOfVeryEasy'] == data.numOfVeryEasy and 
                counters['countOfEasy'] == data.numOfEasy and 
                counters['countOfAverage'] == data.numOfAverage and 
                counters['countOfHard'] == data.numOfHard and 
                counters['countOfVeryHard'] == data.numOfVeryHard
            ):
                return result_questions

            #check if result is stored in questions
            if 'questions' in result:
                result = result['questions']

            for res in result:
                try:
                    check_answers(result, questionType)  
                except ValueError as e:
                    print(f"Validation Error: {e}")
                    continue  
                
                #check if the generated question already exists
                if res['question'] in result_questions:
                    print("\n\nquestion already exist")
                    continue

                text = res['question']
                # if result['questions'][i]['question'] 
                predicted_class = rf_classifier.predict(tfidf.transform([text]))[0]
                difficulty_value  = predict_difficulty_value(text, predicted_class)
                print(f"\nllm prediction: {res['difficulty_level']} predicted class : {predicted_class}")

                #if res['question'] in list of questions
                # continue

                #checks if the question is already in the vector db
                exactMatch = checkExactMatch(text)

                if exactMatch:
                    continue
                
                if predicted_class == 'very easy':
                    if data.numOfVeryEasy == counters['countOfVeryEasy']: 
                        continue
                    else:
                        #check if the question already exist in the database
                        vector_store_questions.add_texts(
                            texts = [
                            text
                            ]
                        )
                        res['difficulty_level'] = predicted_class
                        res['difficulty_value'] = difficulty_value
                        # res['difficulty'] = predicted_class
                        result_questions["questions"].append(res)
                        counters['countOfVeryEasy']+=1
                        print(f"\nvery easy: {counters['countOfVeryEasy']}")

                elif predicted_class == 'easy':
                    if data.numOfEasy == counters['countOfEasy']:
                        continue
                    else:
                        vector_store_questions.add_texts(
                            texts = [
                            text
                            ]
                        )
                        res['difficulty_level'] = predicted_class
                        res['difficulty_value'] = difficulty_value
                        result_questions["questions"].append(res)
                        counters['countOfEasy']+=1
                        print(f"\neasy: {counters['countOfEasy']}")

                elif predicted_class == 'average':
                    if data.numOfAverage == counters['countOfAverage']:
                        continue
                    else:
                        vector_store_questions.add_texts(
                            texts = [
                            text
                            ]
                        )
                        res['difficulty_level'] = predicted_class
                        res['difficulty_value'] = difficulty_value
                        result_questions["questions"].append(res)
                        counters['countOfAverage']+=1
                        print(f"\naverage: {counters['countOfAverage']}")
                        
                elif predicted_class == 'hard':
                    if data.numOfHard == counters['countOfHard']:
                        continue
                    else:
                        vector_store_questions.add_texts(
                            texts = [
                            text
                            ]
                        )
                        res['difficulty_level'] = predicted_class
                        res['difficulty_value'] = difficulty_value
                        result_questions["questions"].append(res)
                        counters['countOfHard']+=1
                        print(f"\nhard: {counters['countOfHard']}")

                elif predicted_class == 'very hard':
                    if data.numOfVeryHard == counters['countOfVeryHard']:
                        continue
                    else:
                        vector_store_questions.add_texts(
                            texts = [
                            text
                            ]
                        )
                        res['difficulty_level'] = predicted_class
                        res['difficulty_value'] = difficulty_value
                        result_questions["questions"].append(res)
                        counters['countOfVeryHard']+=1
                        print(f"\nvery hard: {counters['countOfVeryHard']}")
                if (
                    counters['countOfVeryEasy'] == data.numOfVeryEasy and 
                    counters['countOfEasy'] == data.numOfEasy and 
                    counters['countOfAverage'] == data.numOfAverage and 
                    counters['countOfHard'] == data.numOfHard and 
                    counters['countOfVeryHard'] == data.numOfVeryHard
                ):
                    counters['countOfVeryEasy'] = 0
                    counters['countOfEasy'] = 0
                    counters['countOfAverage'] = 0
                    counters['countOfHard'] = 0
                    counters['countOfVeryHard'] = 0
                    return result_questions
        

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error processing the response: {e}")
        # continue

    

def send_questions_to_laravel(requests_list: list[CreateQuestionsRequest]):
    all_questions = []
    # Convert JSON input to Python objects
    parsed_data = [CreateQuestionsRequest.parse_obj(course) for course in requests_list]
    for course in parsed_data:
        logging.info(f"Accessing course: {course.course_title}")
        for question in course.questions:
            data = QuestionFormat(
                course_id=course.course_id,
                course_title=course.course_title,
                questionType=question.type,
                numOfVeryEasy=question.difficulty.numOfVeryEasy,
                numOfEasy=question.difficulty.numOfEasy,
                numOfAverage=question.difficulty.numOfAverage,
                numOfHard=question.difficulty.numOfHard,
                numOfVeryHard=question.difficulty.numOfVeryHard
            )
            questions = createQuestions(data)
            all_questions.append(questions)
    
    # Ensure the folder for storing JSON files exists
    folder_path = './json_files/'
    os.makedirs(folder_path, exist_ok=True)

    # Generate a unique name for the JSON file using UUID and timestamp
    file_name = f'questions_{uuid.uuid4().hex}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    file_path = os.path.join(folder_path, file_name)
    
    with open(file_path, "w") as json_file:
        json.dump(all_questions, json_file, indent=4)

    # Send the data to Laravel
    url = f"http://{ip}:8000/admin/store-questions/"
    
    
    try:
        response = requests.post(url, json=all_questions)
        logging.info(f"Response: {response.status_code} - {response.text}")
        if response.status_code == 200:
            return "Successfully sent the data to Laravel."
        else:
            return f"Failed to send data: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        logging.error(f"Error while sending data to Laravel: {e}")
        return "Failed to send data to Laravel due to a connection error."


# if __name__ == "__main__":
#     data = QuestionFormat(
#         course_id = 1,
#         course_title = "Software Development",
#         questionType = "Multiple Choice - Single",
#         numOfVeryEasy = 4,
#         numOfEasy = 4,
#         numOfAverage = 4,
#         numOfHard = 4,
#         numOfVeryHard = 4,
#     )

#     result = createQuestions(data)
#     print("Generated Questions:")
#     print(result)

# print(f"\n\n len: {len(vector_store_questions.get()['documents'])}")