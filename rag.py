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


# OpenAI
from openai import OpenAI

from uuid import uuid4

# Base64 images conversion
import base64, logging
from PIL import Image,UnidentifiedImageError
from io import BytesIO
from fastapi import HTTPException

# handling json files
import json

# regex
import re, uuid, requests
from datetime import datetime

from Setup import CONTENT_DOCUMENT,QUESTION_DOCUMENT,LLM,LARAVEL_IP
from Question import *
from Constants import *
from Models import QuestionFormat, CreateQuestionsRequest


def cleanText(text):
    text = re.sub(r"^([0-9][0-9]|[A-Z]\)|@|©|\|\.|[0-9])\s*", "", text)
    text = re.sub(r"[+*]", "", text)
    return text


def queryHierarchicalIndex(query_text, level=None):
    # Define filter criteria if the level is specified
    filter_criteria = {"level": level} if level else {}

    # Perform a similarity search using ChromaDB
    results = CONTENT_DOCUMENT.similarity_search(
        query=query_text,  # The query string
        k=5,  # Number of similar documents to retrieve
        filter=filter_criteria,  # Filter criteria if needed
        include=["documents"],
    )

    return results


def ModelQuerywithRAG(input, course_id):

    retriever = CONTENT_DOCUMENT.as_retriever(
        search_kwargs={
            "filter": {
                "$and": [
                    {"course_id": {"$eq": course_id}},
                    {
                        "type": {
                            "$in": [
                                "Module",
                                "Lesson",
                                "Section",
                                "Subsection",
                                "Table",
                                "Figure",
                                "Code",
                            ]
                        }
                    },
                ]
            }
        }
    )
    retriever = CONTENT_DOCUMENT.as_retriever()

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
    doc_chain = create_stuff_documents_chain(LLM, prompt_template)
    chain = create_retrieval_chain(retriever, doc_chain)

    # User query
    response = chain.invoke({"input": input})
    return response["answer"]


def check_answers(result, questionType):
    # checking for answers
    # pulihi lang ang "identification" sa ipass sa parameter nga questionType
    if questionType == "Identification":
        answer = result["answer"].split()
        if not (1 <= len(answer) < 3):
            raise ValueError(
                f"Identification answer has {len(answer)} words; it should be atleast 1 or at most 2.."
            )
        for key in result:
            if key not in IDENTIFICATION_KEYS:
                raise ValueError(f"{key} not in the correct keys for identifcation")

    # pulihi lang ang "multiple_choice_multiple_answer" sa ipass sa parameter nga questionType
    if questionType == "Multiple Choice - Many":
        correct_answers_len = len(result["answer"])
        if correct_answers_len < 2:
            raise ValueError("Multiple choice multiple answers is not atleast 2")

        for answer in result["answer"]:
            if answer not in result["c  hoices"]:
                raise ValueError(
                    f"Correct answer '{answer}' is not found in the choices."
                )
        for key in result:
            if key not in MULTIPLE_CHOICE_KEYS:
                raise ValueError(
                    f"{key} not in the correct keys for multiple choice question type"
                )

    # pulihi lang ang "multiple_choice_multiple_answer" sa ipass sa parameter nga questionType
    if questionType == "multiple_choice_single_answer":
        correct_answer = result["answer"]
        if correct_answer not in result["choices"]:
            raise ValueError(
                f"Correct answer '{correct_answer}' is not found in the choices."
            )
        for key in result:
            if key not in MULTIPLE_CHOICE_KEYS:
                raise ValueError(
                    f"{key} not in the correct keys for multiple choice question type"
                )

def createQuestions(data):
    try:
        max_iterations = 20
        iteration = 0
        DEFAULT_QUESTIONS_PER_DIFFICULTY = 10
        result_questions = {
            "course_id": data.course_id,
            "course_title": data.course_title,
            "questions": [],
        }
        
        needed = {
            "Very Easy": data.numOfVeryEasy,
            "Easy": data.numOfEasy,
            "Average": data.numOfAverage,
            "Hard": data.numOfHard,
            "Very Hard": data.numOfVeryHard,
        }
        
        existing_questions = set()
        warnings_feedback = []

        while iteration < max_iterations:
            iteration += 1
            
            remaining_questions = sum(needed.values())
            if remaining_questions <= 0:
                logging.info("All required questions have been generated.")
                break
            
            needed_difficulties = {diff: count for diff, count in needed.items() if count > 0}
            if not needed_difficulties:
                break
            logging.info(needed_difficulties)
            
            difficulty_list = list(needed_difficulties.keys())
            batch_needed_str = "\n".join(f"- **{diff}**" for diff in difficulty_list)
            
            logging.info(f"Iteration {iteration}: Remaining questions needed: {remaining_questions}")
            logging.info(f"Current needs by difficulty: {needed_difficulties}")
            
            filtered_blooms_descriptions = build_blooms_prompt(needed, BLOOMS_MAPPING)
            
            # Filter out irrelevant warnings
            relevant_feedback = [
                warning for warning in warnings_feedback
                if "No suitable Bloom's level" in warning or "Error processing question" in warning
            ]

            # Generate feedback string
            feedback_str = "\n".join(f"- {feedback}" for feedback in relevant_feedback)

            instructions = f"""
                Iteration {iteration}: 
                Generate questions and answers based on the content of the {data.course_title} course.
                Each question must align with one of the following Bloom's Taxonomy levels and associated cognitive processes.
                Provide realistic and practical scenarios that test critical thinking and application of knowledge.

                Generate {DEFAULT_QUESTIONS_PER_DIFFICULTY} questions per difficulty. The types of questions needed are:
                {batch_needed_str}
                
                Allowed Question Format:
                {QUESTION_TYPES}
                
                Rules for each question format:
                {QUESTION_RULES}
                                
                Descriptions and Examples for Needed Levels:
                {filtered_blooms_descriptions}
                
                Feedback from previous iterations:
                {feedback_str}
                         
                Ensure the question is clear, concise, and similar to those in the TOPCIT exam.

                It should be stored in a JSON format like this and don't put any text beside this:
                
                {{
                    "course_id": "{data.course_id}",
                    "course_title":"{data.course_title}",
                    "questions": [
                        {EXAMPLE}
                    ]
                }}
            """
            logging.info(instructions)
            
            
            response = ModelQuerywithRAG(instructions, data.course_id)
            logging.info(f"Response: {response}")

            if not response:
                logging.error("Error: Empty response received from model")
                if iteration >= max_iterations:
                    break
                continue

            # Process LLM response
            cleaned_response = response.replace("```json", "").replace("```", "").strip()

            try:
                result = json.loads(cleaned_response)
            except (json.JSONDecodeError, ValueError) as e:
                logging.error(f"JSON decoding error: {e}")
                if iteration >= max_iterations:
                    break
                continue

            # Ensure result has the expected structure
            if not isinstance(result, dict) or "questions" not in result:
                logging.error("Invalid result structure, expected a dictionary with 'questions' key.")
                continue

            questions_added = 0
            warnings_feedback.clear()
            for res in result["questions"]:
                question_text = res.get("question", "").strip()
                question_type = res.get("questionType","")

                logging.info(question_text)

                # Skip questions already in vector store or duplicate question
                if question_text in existing_questions or checkExactMatch(question_text):
                    warnings_feedback.append(f"Duplicate question skipped: {question_text}")
                    continue
                
                try:
                    # Predict the difficulty type as a string
                    predicted_class = predict_question(question_text)
                    logging.info(f"Predicted difficulty class: {predicted_class}")

                    if not predicted_class or predicted_class not in needed or needed[predicted_class] <= 0:
                        warning = f"No suitable Bloom's level found for question: '{question_text}'. Predicted: {predicted_class}"
                        logging.warning(warning)
                        warnings_feedback.append(warning)
                        continue

                    # Predict difficulty value based on the selected class
                    difficulty_value = predict_difficulty_value(question_text, predicted_class)

                    # Check for similar questions first
                    similar_question = checkExactMatch(question_text)
                    if similar_question:
                        warning = f"Similar question already exists: {similar_question[0]}"
                        logging.warning(warning)
                        warnings_feedback.append(warning)
                        continue

                    # Add to vector store - now using add_texts instead of add
                    QUESTION_DOCUMENT.add_texts(
                        texts=[question_text],
                        metadatas=[{
                            "difficulty": predicted_class,
                            "type": question_type,
                            "difficulty_value": difficulty_value,
                            "discrimination_index": get_discrimination(predicted_class)
                        }]
                    )
                    existing_questions.add(question_text)

                    # Update result with difficulty information
                    res.update({
                        "difficulty_type": predicted_class,
                        "difficulty_value": difficulty_value,
                        "discrimination_index": get_discrimination(predicted_class),
                    })

                    # Append question to results and update needed counts
                    result_questions["questions"].append(res)
                    needed[predicted_class] -= 1
                    questions_added += 1

                    logging.info(f"Added question of difficulty {predicted_class}. Remaining: {needed[predicted_class]}")

                except Exception as e:
                    warning = f"Error processing question '{question_text}': {e}"
                    logging.error(warning)
                    warnings_feedback.append(warning)
                    continue

            # If no questions were added in this iteration, include a general warning
            if questions_added == 0:
                warning = "No new questions added in this iteration. Consider refining the instructions for better alignment with Bloom's levels."
                logging.warning(warning)
                warnings_feedback.append(warning)


        logging.info(f"Final question counts by difficulty: {needed}")
        return result_questions

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return {}


def build_blooms_prompt(needed, blooms_mapping):
    """
    Build a prompt using Bloom's Taxonomy levels based on the needed difficulties.
    
    Args:
        needed (dict): Dictionary containing counts of needed questions for each difficulty.
        blooms_mapping (dict): Bloom's Taxonomy mapping with descriptions and examples.
    
    Returns:
        str: Formatted string to include in the prompt.
    """
    # Translate `needed` keys (e.g., "Very Easy") to Bloom's levels using the `difficulty` field
    included_levels = [
        level for level, details in blooms_mapping.items()
        if details["difficulty"] in needed and needed[details["difficulty"]] > 0
    ]

    # Build the string for the prompt
    filtered_blooms_descriptions = "\n".join(
        f"- {level} ({blooms_mapping[level]['difficulty']}): {blooms_mapping[level]['description']}\n"
        f"Examples:\n" +
        "\n".join(f"  - {example}" for example in blooms_mapping[level]["examples"])
        for level in included_levels
    )
    
    return filtered_blooms_descriptions

def send_questions_to_laravel(requests_list: list[CreateQuestionsRequest]):
    all_questions = []

    # Convert JSON input to Python objects
    parsed_data = [CreateQuestionsRequest.parse_obj(course) for course in requests_list]
    for course in parsed_data:
        data = QuestionFormat(
            course_id=course.course_id,
            course_title=course.course_title,
            numOfVeryEasy=course.difficulty.numOfVeryEasy,
            numOfEasy=course.difficulty.numOfEasy,
            numOfAverage=course.difficulty.numOfAverage,
            numOfHard=course.difficulty.numOfHard,
            numOfVeryHard=course.difficulty.numOfVeryHard,
        )
        questions = createQuestions(data)
        all_questions.append(questions)

    # Ensure the folder for storing JSON files exists
    folder_path = "./json_files/"
    os.makedirs(folder_path, exist_ok=True)

    # Generate a unique name for the JSON file using UUID and timestamp
    file_name = (
        f'questions_{uuid.uuid4().hex}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, "w") as json_file:
        json.dump(all_questions, json_file, indent=4)

    # Send the data to Laravel
    url = f"http://{LARAVEL_IP}:{LARAVEL_PORT}/{STORE_QUESTION_ROUTE}"

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
