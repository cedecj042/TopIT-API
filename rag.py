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
    "discrimination": 0.5,
    "difficulty_type": "Easy",
    "difficulty_value":-2.5,
    "choices": ["Execution","Compilation","Interpretation","Debugging"]
},"""

multiple_choice_many = """{
    "question": "Which of the following programming languages is known for its simplicity and ease of use?",
    "questionType": "Multiple Choice - Many",
    "answer": ["Python","Ruby"],
    "difficulty_type": "Hard",
    "difficulty_value":1.3,
    "discrimination": 0.8,
    "choices": ["Java","C++","Python","Ruby"]
},"""

identification = """ {
    "question": "What is responsible for managing hardware resources in system architecture?,
    "questionType": "Identification",
    "answer": ["Operating System", "OS"],
    "difficulty_type": "Very Hard",
    "difficulty_value":3.4,
    "discrimination": 1.3
}"""
 
questionType = ['Identification','Multiple Choice - Single','Multiple Choice - Many']
                    
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
            "answer": list,
            "difficulty": str,
            "discrimination": float
        }

    for key, expected_type in correct_format.items():
        if key not in result or not isinstance(result[key], expected_type):
            return False 
    
    return True 

def createQuestions(data: QuestionFormat):
    # Define the question type
    if data.questionType == "Multiple Choice - Single":
        example = multiple_choice_single
        questionTypewithDescription = "Multiple Choice - Single"
    elif data.questionType == "Multiple Choice - Many":
        example = multiple_choice_many
        questionTypewithDescription = "Multiple Choice - Many must have atleast 2 answers and dont put only 1 answer"
    elif data.questionType == "Identification":
        example = identification
        questionTypewithDescription = """
        Identification answers must have a maximum of 3 words.
        Avoid questions that has a lot of subjective answers. 
        Structure of Description for Identification Questions:
        Directly ask the question, focusing on the specific term or concept.
        Use clear and concise language.
        Contextual Clue (Optional):
        If necessary, provide a brief context or category to guide the student.
        Keep it concise and relevant.
        Example: "In software development, what technique is used to simplify complex problems?"
        for its answers, list also the possible answers, for example the abbreviation(if there is one) or shortened word for the answer 
        """
  
    try:
        instructions = f"""
            Generate a question and answer pairs in the form of {questionTypewithDescription} based on the content of the {data.course_title} course. 
            Provide a realistic and practical scenario related to {data.course_title}. Formulate a question that tests critical thinking and application of knowledge. 
            Include coding examples, practical problems, and analytical questions in a certain situations.
            The generated questions should include the following:
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
            Ensure that difficulty_type and difficulty_value follows the assigned range and values.
            
            For Discrimination index is ranging from 0.5 (lowest index) to 2.0(high index);
            Ensure that the discrimination and difficulty values strictly stay within these ranges and round to the nearest tenth if necessary.
            Follow the bloombergs taxonomy for difficulty estimation. Very Easy for remember, Easy for Understand, Apply for Average, Analyze for Hard, Evaluate for Very Hard.
            Ensure the question is clear, concise, and within the context of the TOPCIT exam format.

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
        logging.info(f"Response: {response.status_code} - {response.text}")
        if response.status_code == 200:
            return "Successfully sent the data to Laravel."
        else:
            return f"Failed to send data: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        logging.error(f"Error while sending data to Laravel: {e}")
        return "Failed to send data to Laravel due to a connection error."


