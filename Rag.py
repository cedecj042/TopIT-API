# for llamaparse
# from llama_parse import LlamaParse
# from llama_index.core import SimpleDirectoryReader

# langchain

# from langchain_chroma import Chroma
# from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.retrievers import BaseRetriever
# from langchain.chains import RetrievalQA
# from langchain_community.llms import Ollama
# from langchain_core.documents import Document
# from sentence_transformers import SentenceTransformer
import chromadb.utils.embedding_functions as embedding_functions
from chromadb.config import Settings

# OpenAI
from openai import OpenAI

from uuid import uuid4
from random import shuffle

# Base64 images conversion
import base64, logging
from PIL import Image,UnidentifiedImageError
from io import BytesIO
from fastapi import HTTPException

# handling json files
import json
import uuid

# regex
import re, uuid, requests, random
from datetime import datetime
from random import shuffle
from Setup import CONTENT_DOCUMENT,QUESTION_DOCUMENT,LLM,LARAVEL_IP
# from RandomForest import *
from RandomForest import *
from Constants import *
from Utils import create_module_text
from Models import QuestionFormat, CreateQuestionsRequest


def generate_custom_short_uuid(course_id: int) -> str:
    course_id_str = str(course_id)
    original_uuid = uuid.uuid4().bytes
    short_uuid = base64.b64encode(original_uuid).decode('utf-8').rstrip('==')
    custom_short_uuid = f"C{course_id_str}-{short_uuid}"
    return custom_short_uuid

def clean_answer(answer):
    if answer and answer.endswith('.'):
        return answer[:-1]
    else:
        return answer

def get_sorted_modules(course_id):
    try:
        module_response = CONTENT_DOCUMENT._collection.get(
            where={"course_id": course_id},  
            include=["metadatas"] 
        )

        module_metadatas = module_response.get("metadatas", [])

        module_uids = {
            metadata.get("module_uid") 
            for metadata in module_metadatas 
            if metadata.get("type") == "Module" 
        }

        if not module_uids:
            logging.warning(f"No modules found for course ID {course_id}.")
            return [] 

        # Convert set to list and shuffle
        module_uids = list(module_uids)
        shuffle(module_uids)

        return module_uids

    except Exception as e:
        logging.error(f"Error fetching modules for course ID {course_id}: {str(e)}")
        return []


async def ModelQuerywithRAG(input, module_uid):
    try:
        module_context_response = await create_module_text(module_uid)

        context = module_context_response.get("module_text", "")
        if not context:
            raise ValueError(f"No context found for module ID {module_uid}")

        template = """
        TOPCIT (Test of Practical Competency in IT) is designed to assess competencies in practical IT skills such as programming, algorithm problem-solving, and 
        IT business understanding. TOPCIT questions are typically scenario-based, requiring critical thinking and practical application of knowledge in areas like 
        software development, database management, algorithms, and IT ethics.  Use the context to generate scenarios before creating a question.
        You are an Information Technology College Teacher tasked to create simulated exam questions for TOPCIT.

        The context provides key information for creating exam questions. Ensure every question explicitly references or relies on details from the context.
        Always use the information provided in <context> to generate your answers. Do not include any details that are not directly supported by the context.
        Don't reference the context without explicitly putting it in the question.
        <context>
        {context}
        </context>

        Query: {input}
        """

        prompt_template = PromptTemplate(template=template)
        
        prompt_text = prompt_template.invoke({"context": context, "input": input})
        
        response = LLM.invoke(prompt_text)
        return response.content

    except Exception as e:
        raise ValueError(f"An error occurred while querying the RAG model: {str(e)}")

def check_answers(result, questionType):
    try:
        if questionType == "Identification":
            if not isinstance(result["answer"], list):
                raise ValueError(f"Identification answer must be a list, got {type(result['answer'])}.")
            for answer in result["answer"]:
                if len(answer.split()) >= 4:
                    raise ValueError(f"Answer '{answer}' has 4 or more words. Answers must be less than 3 words.")
            for key in result:
                if key not in IDENTIFICATION_KEYS:
                    raise ValueError(f"Unexpected key '{key}' in Identification question.")

        elif questionType == "Multiple Choice - Many":
            if not isinstance(result["answer"], list):
                raise ValueError(f"Answer for 'Multiple Choice - Many' must be a list, got {type(result['answer'])}.")
            if not isinstance(result["choices"], list):
                raise ValueError(f"Choices for 'Multiple Choice - Many' must be a list, got {type(result['choices'])}.")
            correct_answers_len = len(result["answer"])
            if correct_answers_len < 2:
                raise ValueError("Multiple Choice - Many requires at least 2 correct answers.")

            original_answers = result["answer"][:] #create a copy to avoid in place modification problems.
            for i, answer in enumerate(original_answers):
                if answer not in result["choices"]:
                    cleaned_answer = clean_answer(answer)
                    if cleaned_answer in result["choices"]:
                        result["answer"][i] = cleaned_answer #update the answer in the original dict.
                    else:
                      raise ValueError(f"Correct answer '{answer}' is not found in the choices: {result['choices']}.")
            for key in result:
                if key not in MULTIPLE_CHOICE_KEYS:
                    raise ValueError(f"Unexpected key '{key}' in Multiple Choice - Many question.")

        elif questionType == "Multiple Choice - Single":
            if not isinstance(result["answer"], str):
                raise ValueError(f"Answer for 'Multiple Choice - Single' must be a string, got {type(result['answer'])}.")
            if not isinstance(result["choices"], list):
                raise ValueError(f"Choices for 'Multiple Choice - Single' must be a list, got {type(result['choices'])}.")

            correct_answer = result["answer"]
            if correct_answer not in result["choices"]:
                cleaned_answer = clean_answer(correct_answer)
                if cleaned_answer in result["choices"]:
                    result["answer"] = cleaned_answer # update the answer in the original dict.
                else:
                    raise ValueError(f"Correct answer '{correct_answer}' is not found in the choices: {result['choices']}.")
            for key in result:
                if key not in MULTIPLE_CHOICE_KEYS:
                    raise ValueError(f"Unexpected key '{key}' in Multiple Choice - Single question.")

        else:
            raise ValueError(f"Unsupported question type: {questionType}.")

    except KeyError as e:
        raise ValueError(f"Missing required key in result: {e}.")
    except Exception as e:
        raise ValueError(f"Error validating answers: {e}.")

async def createQuestionPerDifficulty(data):
    try:
        result_questions = {
            "course_id": data.course_id,
            "course_title": data.course_title,
            "questions": [],
        }

        # Updated difficulties as a dictionary
        difficulties = {
            "Very Easy": {"name": "Very Easy", "remaining": data.numOfVeryEasy, "level": "Remember"},
            "Easy": {"name": "Easy", "remaining": data.numOfEasy, "level": "Understand"},
            "Average": {"name": "Average", "remaining": data.numOfAverage, "level": "Apply"},
            "Hard": {"name": "Hard", "remaining": data.numOfHard, "level": "Analyze"},
            "Very Hard": {"name": "Very Hard", "remaining": data.numOfVeryHard, "level": "Evaluate"},
        }
        
        module_uids  = get_sorted_modules(data.course_id)
        module_index = 0
        logging.info(f"Module UIDs: {module_uids}\n")
        max_attempts_per_difficulty = 30
        existing_questions = set()

        # Iterate through difficulties directly as a dictionary
        for difficulty_name, difficulty in difficulties.items():
            
            if difficulty["remaining"] <= 0:
                logging.info(f"Skipping {difficulty_name} as there are no remaining questions.")
                continue
            iteration = 0
            while difficulty["remaining"] > 0 and iteration < max_attempts_per_difficulty:
                DEFAULT_QUESTION_SIZE = max(10,difficulty["remaining"])
                
                iteration += 1
                logging.info(f"Attempt {iteration}: Generating up to {DEFAULT_QUESTION_SIZE} questions for difficulty: {difficulty_name}")
                module_uid = module_uids[module_index]
                module_index = (module_index + 1) % len(module_uids)
                
                # Build instructions for generating questions
                filtered_blooms_descriptions = build_blooms_prompt(difficulty["level"], BLOOMS_MAPPING)
                  
                instructions = f"""   
                    Generate {DEFAULT_QUESTION_SIZE} questions for the Cognitive Level {difficulty['level']}.
                    Each question must follow the cognitive level’s difficulty guidelines and be stored in a JSON structure.
                    
                    Instructions:
                    {filtered_blooms_descriptions}

                    Rule: Ensure questions are stored in JSON format like this and don't add any text after:
                    {{
                        "course_id": "{data.course_id}",
                        "course_title": "{data.course_title}",
                        "questions": [
                            {EXAMPLE}
                        ]
                    }}
                """
                
                response = await ModelQuerywithRAG(instructions, module_uid)
                logging.info(f"Response for difficulty {difficulty_name}, iteration {iteration}: {response}")

                try:
                    if not response or response.strip() == "":
                        logging.error(f"Empty response received for difficulty: {difficulty_name}, iteration {iteration}")
                        continue  # Skip processing if response is empty

                    # Remove markdown code block markers (```json and ```)
                    cleaned_response = response.strip("`")  # Removes backticks if present
                    if cleaned_response.startswith("json"):
                        cleaned_response = cleaned_response[len("json"):].strip()

                    result = json.loads(cleaned_response)

                except json.JSONDecodeError as e:
                    logging.error(f"JSON decoding error for difficulty: {difficulty_name}, iteration {iteration}: {e}")
                    logging.error(f"Response content: {response}")  # Log raw response for debugging
                    continue
                
                if not isinstance(result, dict) or "questions" not in result:
                    logging.error(f"Invalid response structure for difficulty: {difficulty_name}, iteration {iteration}")
                    continue

                questions_added = 0

                for question in result["questions"]:
                    if difficulty["remaining"] <= 0:
                        logging.info(f"No more questions needed for {difficulty_name}, moving to next difficulty.")
                        break  # Stop processing this difficulty and move to the next one

                    question_text = question.get("question", "").strip()
                    question_type = question.get("questionType", "")
                    logging.info(f"Question: {question_text}")
                    # Skip duplicate or invalid questions
                    if question_text in existing_questions or checkExactMatch(question_text):
                        logging.info(f"Duplicate question skipped: {question_text}")
                        continue
                    
                    try:
                        check_answers(question, question_type)
                        logging.info("Validation successful!")
                    except ValueError as e:
                        logging.error(f"Validation error: {e}")
                        continue

                    try:
                        predicted_class = process_and_predict(question_text)
                        if predicted_class in difficulties and difficulties[predicted_class]["remaining"] > 0:
                            difficulties[predicted_class]["remaining"] -= 1
                        else:
                            logging.warning(f"No reliable class found for question '{question_text}'. Skipping.")
                            continue   
                        
                        try:
                            question_uid = generate_custom_short_uuid(data.course_id)
                            difficulty_value = predict_difficulty_value(question_text, predicted_class)

                            QUESTION_DOCUMENT.add_texts(
                                texts=[question_text],
                                metadatas=[{
                                    "question_uid": question_uid,
                                    "course_id" : data.course_id,
                                    "difficulty": predicted_class,
                                    "type": question_type,
                                    "difficulty_value": difficulty_value,
                                    "discrimination_index": get_discrimination(predicted_class)
                                }]
                            )
                        except Exception as e:
                            
                            logging.error(f"Error adding question to vector store: {e}")
                            continue
                        
                        question.update({
                            "question_uid" : question_uid,
                            "difficulty_type": predicted_class,
                            "difficulty_value": difficulty_value,
                            "discrimination_index": get_discrimination(predicted_class),
                        })
                        result_questions["questions"].append(question)
                        questions_added += 1
                        existing_questions.add(question_text)

                        logging.info(f"Added question of difficulty {predicted_class}. Remaining: {difficulties[predicted_class]['remaining']}")
                    except Exception as e:
                        logging.error(f"Error processing question '{question_text}': {e}")
                        continue

                if questions_added == 0:
                    warning = "No new questions added in this iteration. Consider refining the instructions for better alignment with Bloom's levels."
                    logging.warning(warning)

                logging.info(f"Questions remaining for {difficulty_name}: {difficulty['remaining']}")
                
        logging.info(f"Done generating questions for all difficulties.")
        
        print(type(result_questions))
        updated_result_questions = updateIdentificationAnswers(result_questions, data.course_id)
        return updated_result_questions

    except Exception as e:
        raise Exception(f"Exception error: {e}")

def build_blooms_prompt(needed, blooms_mapping):
    if needed not in blooms_mapping:
        return f"No Bloom's level found for: {needed}"

    level_details = blooms_mapping[needed]
    difficulty = level_details["difficulty"]

    prompt = (
        f"- {needed} ({difficulty}):\n"
        f"  Description: {level_details['description']}\n"
        f"  **Instructions**: {level_details.get('instructions', 'N/A')}\n"
        f"  **Question Types**:\n{level_details['question_types']}"
    )

    return prompt

def clean_and_parse_json(response, difficulty_name, iteration):
    try:
        # Remove the word "json" (case-insensitive) and strip whitespace
        cleaned_response = response.replace("json", "").strip()
        result = json.loads(cleaned_response)
        return result
    except json.JSONDecodeError as e:
        # Log detailed error information
        logging.error(
            f"JSON decoding error for difficulty: {difficulty_name}, iteration {iteration}. "
            f"Error: {e}. Response: {cleaned_response}"
        )
    except ValueError as e:
        # Handle other value-related errors
        logging.error(
            f"Value error for difficulty: {difficulty_name}, iteration {iteration}. "
            f"Error: {e}. Response: {cleaned_response}"
        )
    except Exception as e:
        # Catch any other unexpected errors
        logging.error(
            f"Unexpected error for difficulty: {difficulty_name}, iteration {iteration}. "
            f"Error: {e}. Response: {cleaned_response}"
        )
    return None  # Return None if parsing fails

def updateIdentificationAnswers(result_questions, course_id):
    
    remaining_questions_set = set()
    remaining_questions = ""
        
    #get all identification questions
    for question in result_questions["questions"]:
        if question['questionType'] == "Identification":
            remaining_questions_set.add(question['question'])
            remaining_questions = remaining_questions + "\n"+ question['question']

    # initializing remaining questions
    remaining_questions_count = len(remaining_questions_set)
    logging.info("updating identification answers")
    while remaining_questions_count > 0:
        try: 
          #instructions for RAG
          instructions = f"""
                      Answer each question with all possible correct answers: 

                      {remaining_questions}


                      It should be stored in a JSON format like this and don 't put any text beside this:
                      The **output must be in this exact JSON format stored in an array []**:

                      ```json
                          {{
                          {IDENTIFICATION_SAMPLE_QA}
                          }}
                      """
          
          response = RAGForIdentificationQuestions(instructions, course_id, remaining_questions_count)

          #cleaning response
          cleaned_response = response.strip("`")  # Removes backticks if present
          if cleaned_response.startswith("json"):
              cleaned_response = cleaned_response[len("json"):].strip()

          IDENTIFICATION_ANSWERS = json.loads(cleaned_response)
          
          remaining_questions_set.clear() #reset remaining questions set

          #updating identification answers and checking for remaining questions
          for answer in IDENTIFICATION_ANSWERS:
              exist = False
              for result in result_questions['questions']:
                  if answer['question'] == result['question']:
                      exist = True
                      result['answer'] = list(set(result['answer'] + answer['answer']))
                      break
              if not exist: 
                  remaining_questions_set.add(result['question'])

          
          remaining_questions = ""   #reset remaining questions
          remaining_questions = "\n".join(remaining_questions_set) 
          remaining_questions_count = len(remaining_questions_set) #update remaining_questions_count
    
        except json.JSONDecodeError as e:
          logging.error(f"Error: {e}. Response: {cleaned_response}")
          print(f"Error processing the response: {e}")
          continue
      
    return result_questions
    
def RAGForIdentificationQuestions(input, course_id, number_of_docs):

    retriever = CONTENT_DOCUMENT.as_retriever(
        search_kwargs={
            "filter": {"ids": {"$eq": course_id}},
            "k": number_of_docs + 10, #add 10 more documents to be retrieved 
        }
    )

    # Define prompt template
    template = """
    You are Information Technology College Teacher that is handling Reviewer for Information Technology certification reviewers.
    You are tasked to answer each of the question with all possible correct answers for each question.
    Each answer should be concise, limited to a maximum of 3 words.

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

async def send_questions_to_laravel(requests_list: list[CreateQuestionsRequest]):
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
        questions = await createQuestionPerDifficulty(data)
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
        # logging.info(f"Response: {response.status_code} - {response.text}")
        if response.status_code == 200:
            return "Successfully sent the data to Laravel."
        else:
            return f"Failed to send data: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        logging.error(f"Error while sending data to Laravel: {e}")
        return "Failed to send data to Laravel due to a connection error."
    

import asyncio

if __name__ == "__main__":
    data = QuestionFormat(
        course_id=1,
        course_title="Software Development",
        numOfVeryEasy=10,
        numOfEasy=10,
        numOfAverage=10,
        numOfHard=10,
        numOfVeryHard=10,
    )

    async def main():
        result = await createQuestionPerDifficulty(data)
        print("Generated Questions:")
        print(result)

    asyncio.run(main())


