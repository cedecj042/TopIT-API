# for getting environment variables
from dotenv import load_dotenv
load_dotenv() 

# for llamaparse
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

# langchain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
# from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
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
import re, os,uuid,requests,chromadb
from pydantic import BaseModel
from datetime import datetime


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
    
    
api_key = os.getenv("OPENAI_API_KEY")
ip = os.getenv('IP_ADDRESS')

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model_kwargs = {'device': 'cpu'} #use "cuda" if you have nvidia gpu otherwise use "cpu"
encode_kwargs = {'normalize_embeddings': True}

Sbert = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vector_store = Chroma(
    collection_name="TopIT",
    embedding_function=Sbert,
    persist_directory="./chroma_db1",  # Where to save data locally, remove if not neccesary
)

# Load the llm 
llm = ChatOpenAI(model_name="gpt-4o-mini",api_key=api_key)

def cleanText(text):
    text = re.sub(r'^([0-9][0-9]|[A-Z]\)|@|©|\|\.|[0-9])\s*', '', text)
    text = re.sub(r'[+*]', '', text)
    return text


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
    You are a Information Technology College Teacher that is handling Reviewer for Information Technology certification reviewers. 
    You should create an questions that is a college level for Test of Practical Competency in IT which is application, situational, and textual based questions.

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


multiple_choice_single_answer = """{
    "question": "What is the primary function of a compiler?",
    "questionType": "Multiple Choice - Single Answer",
    "correctAnswer": "Compilation",
    "difficulty": "very easy",
    "discrimination": -4.5,
    "choices": [
        {"choice_1": "Execution"},
        {"choice_2": "Compilation"},
        {"choice_3": "Interpretation"},
        {"choice_4": "Debugging"}
    ]
},"""

multiple_choice_multiple_answer = """{
    "question": "Which of the following programming languages is known for its simplicity and ease of use?",
    "questionType": "Multiple Choice - Multiple Answer",
    "correctAnswer": [
        {"Correct_answer_1": "Python"},
        {"Correct_answer_2": "Ruby"}
    ],
    "difficulty": "hard",
    "discrimination": 2.2,
    "choices": [
        {"choice_1": "Java"},
        {"choice_2": "C++"},
        {"choice_3": "Python"},
        {"choice_4": "Ruby"}
    ]
},"""

identification = """ {
    "question": "What is the term for a program that translates one programming language into another?",
    "questionType": "Identification",
    "correctAnswer": "Interpreter",
    "difficulty": "Very Hard",
    "discrimination": 4.5
}"""
 
    
questionType = ['Identification','Multiple Choice - Single Answer','Multiple Choice - Many Answer']
                    
def validate_question_format(result, question_type):
    # Define correct format based on question type
    if question_type == "Multiple Choice - Single Answer":
        correct_format = {
            "question": str,
            "questionType": str,
            "correctAnswer": str,
            "difficulty": str,
            "discrimination": float,
            "choices": list
        }
    elif question_type == "Multiple Choice - Multiple Answer":
        correct_format = {
            "question": str,
            "questionType": str,
            "correctAnswer": list,
            "difficulty": str,
            "discrimination": float,
            "choices": list
        }
    elif question_type == "identification":
        correct_format = {
            "question": str,
            "questionType": str,
            "correctAnswer": str,
            "difficulty": str,
            "discrimination": float
        }

    for key, expected_type in correct_format.items():
        if key not in result or not isinstance(result[key], expected_type):
            return False 
    
    return True 

def createQuestions(data: QuestionFormat):
    # Define the question type
    if data.questionType == "multiple-choice-single":
        example = multiple_choice_single_answer
        questionTypewithDescription = "Multiple Choice - Single Answer"
    elif data.questionType == "multiple-choice-many":
        example = multiple_choice_multiple_answer
        questionTypewithDescription = "Multiple Choice - Multiple Answer (correct answers should be between 2 - 3)"
    elif data.questionType == "identification":
        example = identification
        questionTypewithDescription = "Identification (maximum of 2 words for 1 correct answer)"
  
    try:
        instructions = f"""
            Generate a question and answer pairs in the form of {questionTypewithDescription} based on the content of the {data.course_title} course. The generated questions should include the following:
            - {data.numOfVeryEasy} Very Easy questions
            - {data.numOfEasy} Easy questions
            - {data.numOfAverage} Average questions
            - {data.numOfHard} Hard questions
            - {data.numOfVeryHard} Very Hard questions

            Each question must be assigned one of the following difficulty levels, with the corresponding discrimination level strictly within the specified range:
            - Very Easy: -5.0 to -3.0
            - Easy: -2.9 to -1.0
            - Average: -0.9 to 1.0
            - Hard: 1.1 to 3.0
            - Very Hard: 3.1 to 5.0
            Ensure that the discrimination values strictly stay within these ranges and round to the nearest tenth if necessary.

            It should be stored in a JSON format like this and don't put any text beside this:
            
            {{
                "course_id": "{data.course_id}",
                "course_title":"{data.course_title}",
                "questions": [
                    {example}
                ]
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
        
        # Validate the results
        # validate_keys(result)
        # validate_question_difficulty(result, question_level)
        # check_question_counts(result, question_level)
        # validate_question_content(result, questionType)

        # Save the results
        return result

    except (json.JSONDecodeError, ValueError) as e:
        return f"Error processing the response: {e}"

    

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
        logging.info(response)
        if response.status_code == 200:
            return "Successfully sent the data to Laravel."
        else:
            return f"Failed to send data: {response.status_code} - {response.text}"
    
    except requests.exceptions.RequestException as e:
        return f"Error occurred while sending data to Laravel: {e}"

# createQuestions(1, "Software Development", 10, "identification", 2, 2, 2, 2, 2)

# client.get()
