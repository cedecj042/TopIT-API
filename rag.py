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
from PIL import Image
from io import BytesIO
from fastapi import HTTPException

# handling json files
import json
from PIL import UnidentifiedImageError

# regex
import re, uuid, requests
from datetime import datetime

from setup import *
from questions import *
from constants import *
from models import QuestionFormat, CreateQuestionsRequest


def cleanText(text):
    text = re.sub(r"^([0-9][0-9]|[A-Z]\)|@|©|\|\.|[0-9])\s*", "", text)
    text = re.sub(r"[+*]", "", text)
    return text


def queryHierarchicalIndex(query_text, level=None):
    # Define filter criteria if the level is specified
    filter_criteria = {"level": level} if level else {}

    # Perform a similarity search using ChromaDB
    results = vector_store.similarity_search(
        query=query_text,  # The query string
        k=5,  # Number of similar documents to retrieve
        filter=filter_criteria,  # Filter criteria if needed
        include=["documents"],
    )

    return results


def ModelQuerywithRAG(input, course_id):

    retriever = vector_store.as_retriever(
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
    return response["answer"]


# def validate_question_format(result, question_type):
#     # Define correct format based on question type
#     if question_type == "Multiple Choice - Single":
#         correct_format = {
#             "question": str,
#             "questionType": str,
#             "answer": str,
#             "difficulty": str,
#             "discrimination": float,
#             "choices": list
#         }
#     elif question_type == "Multiple Choice - Many":
#         correct_format = {
#             "question": str,
#             "questionType": str,
#             "answer": list,
#             "difficulty": str,
#             "discrimination": float,
#             "choices": list
#         }
#     elif question_type == "identification":
#         correct_format = {
#             "question": str,
#             "questionType": str,
#             "answer": list,
#             "difficulty": str,
#             "discrimination": float
#         }

#     for key, expected_type in correct_format.items():
#         if key not in result or not isinstance(result[key], expected_type):
#             return False

#     return True


def check_answers(result, questionType):
    # checking for answers
    # pulihi lang ang "identification" sa ipass sa parameter nga questionType
    if questionType == "identification":
        answer = result["answer"].split()
        if not (1 <= len(answer) < 3):
            raise ValueError(
                f"Identification answer has {len(answer)} words; it should be atleast 1 or at most 2.."
            )
        for key in result:
            if key not in IDENTIFICATION_KEYS:
                raise ValueError(f"{key} not in the correct keys for identifcation")

    # pulihi lang ang "multiple_choice_multiple_answer" sa ipass sa parameter nga questionType
    if questionType == "multiple_choice_multiple_answer":
        correct_answers_len = len(result["answer"])
        if correct_answers_len < 2:
            raise ValueError("Multiple choice multiple answers is not atleast 2")

        for answer in result["answer"]:
            if answer not in result["choices"]:
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

# def createQuestions(data, example, description):
#     try:
#         result_questions = {
#             "course_id": data.course_id,
#             "course_title": data.course_title,
#             "questions": [],
#         }

#         needed = {
#             "Very Easy": data.numOfVeryEasy,
#             "Easy": data.numOfEasy,
#             "Average": data.numOfAverage,
#             "Hard": data.numOfHard,
#             "Very Hard": data.numOfVeryHard,
#         }

#         while any(count > 0 for count in needed.values()):
#             instructions = f"""
#                 Generate questions and answers in the form of {description} based on the content of the {data.course_title} course. 
#                 Provide a realistic and practical scenario related to {data.course_title}. Formulate a question that tests critical thinking and application of knowledge. 
#                 Include coding examples, practical problems, and analytical questions in certain situations.

#                 The types of questions needed are:
#                 {', '.join([f'- {count} {key} questions' for key, count in needed.items() if count > 0])}

#                 Ensure the question is clear, concise, and similar to questions in the TOPCIT exam.

#                 It should be stored in a JSON format like this and don't put any text beside this:
                
#                 {{
#                     "course_id": "{data.course_id}",
#                     "course_title":"{data.course_title}",
#                     "questions": [
#                         {example}
#                     ]
#                 }}
#             """

#             response = ModelQuerywithRAG(instructions, data.course_id)
#             logging.info(f"Response: {response}")

#             if not response:
#                 logging.error("Error: Empty response received from model")
#                 break

#             # Process LLM response
#             cleaned_response = response.replace("```json", "").replace("```", "").strip()

#             try:
#                 parsed_response = json.loads(cleaned_response)
#                 logging.info(f"parsing response:  {parsed_response}")
#             except (json.JSONDecodeError, ValueError) as e:
#                 logging.error(f"JSON decoding error: {e}")
#                 break

#             if isinstance(parsed_response, dict) and "questions" in parsed_response:
#                 questions = parsed_response["questions"]
#                 logging.info(f"parsing response:  {parsed_response}")
                
#             else:
#                 logging.error("Invalid format received from model.")
#                 break
            
#             logging.info(questions)
#             for question in questions:
#                 question_text = question.get("question", "")
#                 logging.info(f"current question: {question_text}")
                
#                 difficulty_type = question.get("difficulty_type", "")

#                 if not question_text or not difficulty_type:
#                     logging.warning("Skipping question due to missing fields.")
#                     continue

#                 if any(q["question"] == question_text for q in result_questions["questions"]):
#                     logging.info("Duplicate question found, skipping.")
#                     continue

#                 if checkExactMatch(question_text):
#                     logging.info("Exact match in vector store, skipping.")
#                     continue

#                 difficulty_value = predict_difficulty_value(question_text,difficulty_type)
#                 discrimination_value = get_discrimination(difficulty_type)

#                 if difficulty_type in needed and needed[difficulty_type] > 0:
#                     vector_store_questions.add_texts(texts=[question_text])

#                     question["difficulty_value"] = difficulty_value
#                     question["discrimination_index"] = discrimination_value

#                     result_questions["questions"].append(question)
#                     needed[difficulty_type] -= 1
                    

#         logging.info(f"Final questions: {result_questions}")
#         return result_questions

#     except Exception as e:
#         logging.error(f"Unexpected error: {e}")
#         return [{}]


# def createQuestions(data, example, description):
#     try:
#         max_iterations = 100
#         iteration = 0
#         result_questions = {
#             "course_id": data.course_id,
#             "course_title": data.course_title,
#             "questions": [],
#         }
#         needed = {
#             "very easy": data.numOfVeryEasy,
#             "easy": data.numOfEasy,
#             "average": data.numOfAverage,
#             "hard": data.numOfHard,
#             "very hard": data.numOfVeryHard,
#         }

#         while True:
#             iteration += 1
#              # Identify remaining difficulties
#             needed_difficulties = {diff: count for diff, count in needed.items() if count > 0}
#             if not needed_difficulties:
#                 break  # Exit when no questions are needed
            
#             # Prepare batch instructions
#             batch_needed = {}
#             for diff, count in needed_difficulties.items():
#                 batch_needed[diff] = min(30, count)  # Limit to a batch of 20 per difficulty
            
#             batch_needed_str = "\n".join(
#                 f"- **{diff.capitalize()} ({diff.replace(' ', '_').title()}):** {count} questions"
#                 for diff, count in batch_needed.items()
#             )
# Examples:
#                 Very Easy: [
#                     "What is the default port for HTTP connections in web servers?",
#                     "Which of the following is an example of a relational database?",
#                     "What is the command to list files and directories in Linux?",
#                     "Which layer of the OSI model is responsible for routing data between networks?",
#                     "Name the data type in Python used to store a collection of unique items.",
#                     "Explain the designation of coated electrode.",
#                     "Write down various applications of Nanomaterials in Chemical industries.",
#                     "What is functional independence?"
#                 ],
#                 Easy: [
#                     "Explain the difference between client-side and server-side scripting.",
#                     "Which of the following best describes a foreign key in a relational database?",
#                     "What is the purpose of normalization in database design?",
#                     "Which statement about REST APIs is correct?",
#                     "In object-oriented programming, what is encapsulation?",
#                     "Explain why mycorrhizal fungi are important for phosphorus availability in soil.",
#                     "Which of the following demonstrates the core concept of systems design?"
#                 ],
#                 Average: [
#                     "Write a SQL query to retrieve all employees earning more than $50,000 from a table named employees.",
#                     "Which of the following algorithms would you use to sort a large dataset efficiently?",
#                     "What is the output of the following Python code: `a = [1, 2, 3, 4]; print(a[::-1])`?",
#                     "Which command in Git is used to combine changes from multiple branches?",
#                     "Describe how to implement a `for` loop to print numbers from 1 to 10 in Python.",
#                     "Which of the following selects the core concept of data analysis?",
#                     "Apply the principles of fluid dynamics to design an efficient water distribution system."
#                 ],
#                 Hard: [
#                     "Given this function, identify the time complexity: `def sum_array(arr): for i in range(len(arr)): for j in range(len(arr)): print(i, j)`",
#                     "Which of the following is NOT a type of database normalization?",
#                     "What are the components of a basic software architecture diagram?",
#                     "A critical bug in production has been identified. What is the FIRST step in handling the issue?",
#                     "What is the purpose of using pseudocode in system design?",
#                     "Which of the following themes the core concept of software development?",
#                     "Compare the security features and vulnerabilities of different authentication methods (e.g., passwords, biometrics, multi-factor authentication) in network access control."
#                 ],
#                 Very Hard: [
#                     "Evaluate the pros and cons of using NoSQL databases over relational databases.",
#                     "Which of the following factors is MOST important when choosing a cloud service provider?",
#                     "What is your recommendation for improving the performance of this SQL query: SELECT * FROM employees WHERE department = 'IT';",
#                     "Which of the following is an ethical consideration in software development?",
#                     "Why is conducting a code review critical for software quality assurance?",
#                     "Assess the scalability of a serverless computing platform for deploying and running event-driven applications.",
#                     "Choose the key principles of software development."
#                 ]

            
#             instructions = f"""
#                 Iteration {iteration}: Generate questions and answers in the form of {description} based on the content of the {data.course_title} course. 
#                 Provide a realistic and practical scenario related to {data.course_title}. Formulate a question that tests critical thinking and application of knowledge. 
#                 Include coding examples, practical problems, and analytical questions in a certain situations.
#                 The types of questions needed are:
#                 {batch_needed_str}

#                 Ensure the question is clear, concise, and within the like in the TOPCIT exam.

#                 It should be stored in a JSON format like this and don't put any text beside this:
                
#                 {{
#                     "course_id": "{data.course_id}",
#                     "course_title":"{data.course_title}",
#                     "questions": [
#                         {example}
#                     ]
#                 }}
#             """
            
#             response = ModelQuerywithRAG(instructions, data.course_id)
#             logging.info(f"Response: {response}")

#             if not response:
#                 logging.error("Error: Empty response received from model")
#                 if iteration >= max_iterations:
#                     break
#                 continue

#             # Process LLM response
#             cleaned_response = response.replace("```json", "").replace("```", "").strip()

#             try:
#                 result = json.loads(cleaned_response)
#             except (json.JSONDecodeError, ValueError) as e:
#                 logging.error(f"JSON decoding error: {e}")
#                 if iteration >= max_iterations:
#                     break
#                 continue

#             if isinstance(result, dict) and "questions" in result:
#                 result = result["questions"]
                
                
#             result_questions_list = result["questions"]
#             existing_questions = {q["question"] for q in result_questions["questions"]}
#             for res in result:
#                 question_text = res.get("question", "").strip()

#                 # Skip duplicate questions in result_questions
#                 if question_text in existing_questions:
#                     logging.info("Duplicate question found, skipping.")
#                     continue

#                 # Skip questions already in vector store
#                 if checkExactMatch(question_text):
#                     logging.info("Exact match in vector store, skipping.")
#                     continue

#                 # Predict difficulty type and value
#                 try:
#                     predicted_class = preprocess_and_predict(question_text)
#                     difficulty_value = predict_difficulty_value(question_text, predicted_class)
#                 except Exception as e:
#                     logging.error(f"Error during prediction for question '{question_text}': {e}")
#                     continue

#                 # Check if the predicted class is still needed
#                 if predicted_class in needed and needed[predicted_class] > 0:
#                     # Add to vector store
#                     vector_store_questions.add_texts(texts=[question_text])

#                     # Update result with difficulty information
#                     res.update({
#                         "difficulty_type": predicted_class,
#                         "difficulty_value": difficulty_value,
#                         "discrimination_index": get_discrimination(predicted_class),
#                     })

#                     # Append question to results and update needed counts
#                     result_questions["questions"].append(res)
#                     existing_questions.add(question_text)  # Update the tracker
#                     needed[predicted_class] -= 1

#                 # Check if all difficulties are satisfied
#                 if all(count == 0 for count in needed.values()):
#                     logging.info("All needed difficulties satisfied.")
#                     break

#                 # Break if max iterations reached
#                 if iteration >= max_iterations:
#                     logging.warning("Reached maximum iterations without fulfilling all difficulties.")
#                     break

#         return result_questions

#     except Exception as e:
#         logging.error(f"Unexpected error: {e}")
#         return {}

def createQuestions(data, example, description):
    try:
        max_iterations = 100
        iteration = 0
        DEFAULT_QUESTIONS_PER_DIFFICULTY = 50
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

            instructions = f"""
                Iteration {iteration}: Generate questions and answers in the form of {description} based on the content of the {data.course_title} course. 
                Provide a realistic and practical scenario related to {data.course_title}. Formulate a question that tests critical thinking and application of knowledge. 
                Include coding examples, practical problems, and analytical questions in a certain situations.
                Generate 20 questions per type. The types of questions needed are:
                {batch_needed_str}
                
                Use this as a guideline for generating questions
                Blooms Level | Difficulty Level | Description 
                Remember | Very Easy | Retrieving relevant knowledge from long-term memory. 
                Understand | Easy | Constructing meaning from oral, written, and graphic messages. 
                Apply | Average | Carrying out or using a procedure in a given situation.
                Analyze | Hard | Breaking material into constituent parts and detecting how the parts relate to one another and to an overall structure or purpose. 
                Evaluate | Very Hard | Making judgments based on criteria and standards.
                
                
                Generate the question base on the Blooms Level not the difficulty.
                                       
                Ensure the question is clear, concise, and similar to those in the TOPCIT exam.

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

            for res in result["questions"]:
                question_text = res.get("question", "").strip()
                logging.info(question_text)
                
                # Skip duplicate questions in result_questions
                if question_text in existing_questions:
                    logging.info("Duplicate question found, skipping.")
                    continue

                # Skip questions already in vector store
                if checkExactMatch(question_text):
                    logging.info("Exact match in vector store, skipping.")
                    continue

                # Predict difficulty type and value
                try:
                    predicted_class = preprocess_and_predict(question_text,tfidf,reference_embeddings)
                    difficulty_value = predict_difficulty_value(question_text, predicted_class)
                    # Check if the predicted class is still needed
                    if predicted_class in needed and needed[predicted_class] > 0:
                        # Add to vector store
                        vector_store_questions.add_texts(texts=[question_text])
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
                        logging.error(f"Error processing question '{question_text}': {e}")
                        continue
                
                logging.info(f"Added {questions_added} questions in this iteration")
                
                if questions_added == 0:
                    logging.warning("No new questions added in this iteration")
                    if iteration >= 3:  # Give it a few tries before giving up
                        break
        logging.info(f"Final question counts by difficulty: {needed}")
        return result_questions

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return {}



def send_questions_to_laravel(requests_list: list[CreateQuestionsRequest]):
    all_questions = []

    # Convert JSON input to Python objects
    parsed_data = [CreateQuestionsRequest.parse_obj(course) for course in requests_list]
    for course in parsed_data:
        logging.info(f"Accessing course: {course.course_title}")
        for question in course.questions:

            # getting the example and description base on question type
            example, description = get_question_type_data(question.type)

            data = QuestionFormat(
                course_id=course.course_id,
                course_title=course.course_title,
                questionType=question.type,
                numOfVeryEasy=question.difficulty.numOfVeryEasy,
                numOfEasy=question.difficulty.numOfEasy,
                numOfAverage=question.difficulty.numOfAverage,
                numOfHard=question.difficulty.numOfHard,
                numOfVeryHard=question.difficulty.numOfVeryHard,
            )
            questions = createQuestions(data, example, description)
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


def get_question_type_data(question_type):
    # Dictionary to map question types to their templates and descriptions
    question_types = {
        "Multiple Choice - Single": (
            MULTIPLE_CHOICE_SINGLE,
            "Multiple Choice - Single",
        ),
        "Multiple Choice - Many": (
            MULTIPLE_CHOICE_MANY,
            "Multiple Choice - Many must have at least 2 answers and don't put only 1 answer",
        ),
        "Identification": (
            IDENTIFICATION,
            """
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
            """,
        ),
    }
    return question_types.get(question_type, (None, None))


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