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

from setup import CONTENT_DOCUMENT,QUESTION_DOCUMENT,LLM,LARAVEL_IP
from Question import *
from constants import *
from models import QuestionFormat, CreateQuestionsRequest



def clean_text(text):
    #lower text
    text = text.lower()

    # allow only letters from a-z
    text = re.sub(r'[^a-z]', ' ', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # tokenize words
    tokens = word_tokenize(text)

    #lemmatizing tokens
    lemmatized_tokens = []
    for token in tokens:
       lemmatized_tokens.append(lemmatizer.lemmatize(token, pos="v"))

    # removing stop words
    text_stop_rem = []
    for word in lemmatized_tokens:
      if word not in stop_words:
         text_stop_rem.append(word)

    text = ' '.join(text_stop_rem)

    return text

def validate_answers_and_keys(result, questionType):

    if questionType == "Identification":
        #remove trailing period for answer
        result['answer'] = re.sub(r"\.$", "", result['answer'])

        answer = result['answer'].split()

        #check if identification answer word lenght is atleast 1 but less than 3
        if not (1 <= len(answer) < 4):
            raise ValueError(f"Identification answer has {len(answer)} words; it should be atleast 1 or at most 3..")
        for key in result:
            if key not in correct_keys_for_identification:
                raise ValueError(f'{key} not in the correct keys for identifcation')


    if questionType == "Multiple Choice - Many":
        #remove trailing period for answer
        result['answer'] = result['answer'] = [re.sub(r"\.$", "", res) for res in result['choices']
                                               ]
        #removing trailing period for choices
        result['choices'] = [re.sub(r"\.$", "", res) for res in result['choices'] ]

        #check if answers is not aleast 2
        correct_answers_len = len(result['answer'])
        if correct_answers_len < 2:
            raise ValueError("Multiple choice multiple answers is not atleast 2")

        #check if answers is in the choices
        for answer in result['answer']:
            if answer not in result['choices']:
                raise ValueError(f"Correct answer '{answer}' is not found in the choices.")

        #check if all keys are correct
        for key in result:
            if key not in correct_keys_for_multiple_choice:
                raise ValueError(f'{key} not in the correct keys for multiple choice question type')

    if questionType == "Multiple Choice - Single":
        #remove trailing period for answer
        result['answer'] = re.sub(r"\.$", "", result['answer'])

        #removing trailing period for choices
        result['choices'] = [re.sub(r"\.$", "", res) for res in result['choices'] ]

        correct_answer = result["answer"]

        #check if answers is in the choices
        if correct_answer not in result['choices']:
            raise ValueError(f"Correct answer '{correct_answer}' is not found in the choices.")

        #check if all keys are correct
        for key in result:
            if key not in correct_keys_for_multiple_choice:
                raise ValueError(f'{key} not in the correct keys for multiple choice question type')

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

# Global storage for modules and last used index per course
Stored_Modules = {}  # {course_title: [modules]}
last_module_index_per_course = {}  # {course_title: last_used_index}

def getModules(course_title):
    global Stored_Modules

    # Get stored dodcuments
    Stored_Content = getStoredDocuments()

    # Store module ID if for a course title if it did not exist
    if course_title not in Stored_Modules:
      Stored_Modules[course_title] = []
      for content in Stored_Content['metadatas']:
          if content.get('course_id') == course_title:
              if content['module_id'] not in Stored_Modules[course_title]:
                  Stored_Modules[course_title].append(content['module_id'])

    return Stored_Modules.get(course_title, [])

def getNextModule(course_title):
    global last_module_index_per_course
    
    #get modules from the course
    modules = getModules(course_title) 

    # Get last used module index, set 0 if first time
    last_module_index_per_course.setdefault(course_title, 0)

    # Select the next module
    module_index = last_module_index_per_course[course_title]
    module = modules[module_index]
    
    # Update index and loop back to 0 if needed
    last_module_index_per_course[course_title] = (module_index + 1) % len(modules)

    return module

def getStoredDocuments():
    return CONTENT_DOCUMENT.get()


def ModelQuerywithRAG(input, course_title, module_id):
    stored_docs = getStoredDocuments()

    #count number of documents inside a course_id
    filtered_docs = [metadata for metadata in stored_docs["metadatas"]
                    if metadata.get("course_id") == course_title]

    Number_of_docs = len(filtered_docs)
    retriever = CONTENT_DOCUMENT.as_retriever(
        search_kwargs={
            "filter": {
                "$and": [
                    {"course_id": {"$eq": course_title}},
                    {"module_id": {"$eq": module_id}}
                ]
            },
            "k": Number_of_docs,
        },
    )   

    # Define prompt template
    template = """
    TOPCIT (Test of Practical Competency in IT) is designed to assess competencies in practical IT skills such as programming, algorithm problem-solving, and 
    IT business understanding. TOPCIT questions are typically scenario-based, requiring critical thinking and practical application of knowledge in areas like 
    software development, database management, algorithms, and IT ethics.
    You are Information Technology College Teacher that is handling Reviewer for Information Technology certification reviewers.
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


correct_keys_for_multiple_choice = {
        "question",
        "answer",
        "choices",
        "questionType"
    }

correct_keys_for_identification = {
        "question",
        "answer",
        "questionType"
    }

def validate_answers_and_keys(result, questionType):

    if questionType == "Identification":
        #remove trailing period for answer
        result['answer'] = re.sub(r"\.$", "", result['answer'])

        answer = result['answer'].split()

        #check if identification answer word lenght is atleast 1 but less than 3
        if not (1 <= len(answer) < 4):
            raise ValueError(f"Identification answer has {len(answer)} words; it should be atleast 1 or at most 3..")
        for key in result:
            if key not in correct_keys_for_identification:
                raise ValueError(f'{key} not in the correct keys for identifcation')


    if questionType == "Multiple Choice - Many":

        if not isinstance(result['answer'], list):
            raise ValueError("Answer should be a list")

        #check if all keys are correct
        for key in result:
            if key not in correct_keys_for_multiple_choice:
                raise ValueError(f'{key} not in the correct keys for multiple choice question type')

        #check if all keys exist
        for key in correct_keys_for_multiple_choice:
            if key not in result:
                raise ValueError(f'{key} not in the result')

        #remove trailing period for answer
        result['answer'] = result['answer'] = [re.sub(r"\.$", "", res) for res in result['choices']
                                               ]
        #removing trailing period for choices
        result['choices'] = [re.sub(r"\.$", "", res) for res in result['choices'] ]

        #check if answers is not aleast 2
        correct_answers_len = len(result['answer'])
        if correct_answers_len < 2:
            raise ValueError("Multiple choice multiple answers is not atleast 2")

        #check if answers is in the choices
        for answer in result['answer']:
            if answer not in result['choices']:
                raise ValueError(f"Correct answer '{answer}' is not found in the choices.")

        #check if all keys are correct
        for key in result:
            if key not in correct_keys_for_multiple_choice:
                raise ValueError(f'{key} not in the correct keys for multiple choice question type')

    if questionType == "Multiple Choice - Single":

        #check if all keys exist
        for key in correct_keys_for_multiple_choice:
            if key not in result:
                raise ValueError(f'{key} not in the result')

        #check if all keys are correct
        for key in result:
            if key not in correct_keys_for_multiple_choice:
                raise ValueError(f'{key} not in the correct keys for multiple choice question type')

        if isinstance(result['answer'], list):
            raise ValueError("Answer should not be a list")

        #remove trailing period for answer
        result['answer'] = re.sub(r"\.$", "", result['answer'])

        #removing trailing period for choices
        result['choices'] = [re.sub(r"\.$", "", res) for res in result['choices'] ]

        correct_answer = result["answer"]

        #check if answers is in the choices
        if correct_answer not in result['choices']:
            raise ValueError(f"Correct answer '{correct_answer}' is not found in the choices.")



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
            # logging.info(f"Found similar question: {doc.page_content}")
            # logging.info(f"Similarity: {cosine_similarity}")
            # logging.info(f"Metadata: {doc.metadata}")
            return doc.page_content, doc.metadata

    return None



class QuestionFormat(BaseModel):
    course_id: int
    course_title: str
    numOfVeryEasy: int
    numOfEasy: int
    numOfAverage: int
    numOfHard: int
    numOfVeryHard: int

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

def generate_questions_with_retry(instructions, data, module, max_retries=3):
    for attempt in range(max_retries):
        try:
            # Generate response
            response = ModelQuerywithRAG(instructions, data.course_title, module)
            print(response)

            # Clean the response
            cleaned_response = response.replace('```json', '').replace('```', '').strip()
            print(cleaned_response)
            # Try parsing methods
            try:
                # First attempt: direct parsing
                result = json.loads(cleaned_response)
            except json.JSONDecodeError:
                try:
                    # Second attempt: fix potential JSON formatting issues
                    if isinstance(cleaned_response, str):
                      fixed_text = re.sub(r'(\}\s*|\]\s*|\w\s*")(\s*"|\s*\{)', r'\1,\2', cleaned_response)
                    else:
                      fixed_text = cleaned_response

                    result = json.loads(fixed_text)
                except json.JSONDecodeError:
                    print(f"Attempt {attempt + 1}: Failed to parse JSON")
                    continue

            # check if questions exist in result, assign in to result
            if 'questions' in result:
                result = result['questions']

            #check if result is list
            if not isinstance(result, list):
                result = [result]

            # Validate non-empty result
            if result and len(result) > 0:
                return result

            print(f"Attempt {attempt + 1}: Empty result")

        except Exception as e:
            print(f"Attempt {attempt + 1}: Error - {e}")

    # If all attempts fail
    print("Failed to generate questions after multiple attempts")
    return []


def createQuestions(data: QuestionFormat):

   #for iteration through the difficulty levels
    difficulty_levels = [
          {'level': 'Very Easy(Remember)', 'max_count': data.numOfVeryEasy, 'counter': 0, 'predicted_class': 'very easy'},
          {'level': 'Easy(Understand)', 'max_count': data.numOfEasy, 'counter': 0, 'predicted_class': 'easy'},
          {'level': 'Average(Apply)', 'max_count': data.numOfAverage, 'counter': 0, 'predicted_class': 'average'},
          {'level': 'Hard(Analyze)', 'max_count': data.numOfHard, 'counter': 0, 'predicted_class': 'hard'},
          {'level': 'Very Hard(Evaluate)', 'max_count': data.numOfVeryHard, 'counter': 0, 'predicted_class': 'very hard'},
      ]

    try:
        result_questions = {
                "course_id": data.course_id,
                "course_title": data.course_title ,
                "questions": [
                ]
        }

        #Get modules for the course 
        Stored_Modules = getModules(data.course_title)
        print(f"\n\nlength of Stored_modules: {len(Stored_Modules)}")

        #iterating through the difficulty levels
        for level in difficulty_levels:

          #subtract max count to counter to check how many counts are left for questions to be generated
          count = level['max_count'] - level['counter']

          while count > 0:
            
            #get next module 
            module = getNextModule(data.course_title)
            print(f'\n\nModule_id: {module}')

            #check if level is easy or hard, and add 10 questions to be generated
            if level['level'] == 'Easy(Understand)' or level['level'] == 'Hard(Analyze)' and count < 6:
              numberOfQuestions = count + 10
            else:
              numberOfQuestions = count + 3

            #number of questions per multiple choice
            half = round(numberOfQuestions/2)

            #number of questions per identication sub category(fill in the blanks, interogative, complete teh sentence)
            num_of_identification = round(numberOfQuestions/3)

            if level['level'] == 'Very Easy(Remember)':
              instructions = f"""
              Generate {numberOfQuestions} **objective test question and answer pair** in the form of Identification (must have a maximum of 3 words in 1 correct answer) based only on the provided content from the module {module} in {data.course_title} course.
              Objective test questions are questions that have specific answer/s (meaning they are not subjective).

              Avoid generating questions that uses "how" as it requires a subjective answer. Meaning, it does not result to identification questions.

              **Strictly** ensure to divide all questions into a form of: ** Fill in the blanks**(Don't include "fill in the blanks" in the question sentence), **Questions that uses interrogative pronouns(Don't use 'how')**, ** Complete the sentence questions(Don't include "complete the sentence" in the question sentence)** where the answer logically completes the idea."

              Follow this format:

              {num_of_identification} fill in the blanks questions
              {num_of_identification} Questions that uses interrogative pronouns
              {num_of_identification} Complete the sentence questions
              """
              
              instructions += identification_very_easy

            else:
              instructions = f"""
              Generate {numberOfQuestions} **objective test question and answer pair** in the form of Multiple Choice - Single(must have only 1 correct answer) and Multiple Choice - Many(must have atleast 2 answers and a maxium of 4 answers) based only on the provided content from the module {module} in {data.course_title} course.
              Objective test questions are questions that have specific answer/s (meaning they are not subjective).

              **Strictly** Ensure the answers for each question are in the choices.

              Follow this structure when generating questions: 
              
              """
              if level['level'] == 'Easy(Understand)':
                  instructions += multiple_choice_easy

              elif level['level'] == 'Average(Apply)':
                  instructions += multiple_choice_average.format(numberOfQuestions=numberOfQuestions, module=module)
                 
              elif level['level'] == 'Hard(Analyze)':
                  instructions += multiple_choice_hard.format(numberOfQuestions=numberOfQuestions, module=module)

              elif level['level'] == 'Very Hard(Evaluate)':
                  instructions += multiple_choice_very_hard.format(numberOfQuestions=numberOfQuestions, module=module)
              
            
            instructions += f"""
            Ensure the answers are not stated in the question.
            All questions should be suitable for the **TOPCIT exam format**, balancing clarity and challenge.

            It should be stored in a JSON format like this and don't put any text beside this:
            The **output must be in this exact JSON format stored in an array []**:

            """
            if level['level'] == 'Very Easy(Remember)':
              instructions += f"""
            ```json
            {{
            {identification}
            }}"""

            else: 
              instructions += f"""
            ```json
            {{
              {multiple_choice_single} 
              {multiple_choice_many}
            }}"""

            generated_questions = generate_questions_with_retry(instructions, data, module)

            if not generated_questions:
              continue

            #loop through the generated questions
            for res in generated_questions:

                #assigning multiple-single if answer is in not a list otherwise multiple-many
                if isinstance(res['answer'], list):
                  res['questionType'] = 'Multiple Choice - Many'
                else:
                  res['questionType'] = 'Multiple Choice - Single'
                
                #assign identification questiontype if generated questions are very easy(remember)
                if level['level'] == 'Very Easy(Remember)':
                    res['questionType'] = 'Identification'

                #check for correct keys
                try:
                    validate_answers_and_keys(res, res['questionType'])
                except ValueError as e:
                    print(f"Validation Error: {e}")
                    continue

                # Skip if question already exists
                if res['question'] in result_questions:
                    print("\n\nquestion already exist")
                    continue

                #predict difficulty
                question_text = clean_text(res['question'])
                predicted_class = RF_CLASSIFIER.predict(TFIDF.transform([question_text]))[0]
                difficulty_value  = predict_difficulty_value(question_text, predicted_class)


                #checks if the question is already in the questions vectordb
                exactMatch = checkExactMatch(question_text)
                if exactMatch:
                    continue

                res['difficulty_level'] = predicted_class
                res['difficulty_value'] = difficulty_value
                res['discrimination'] = get_discrimination(predicted_class)

                #check if max count of the difficulty level has been reached
                if predicted_class == level['predicted_class'] and level['counter'] < level['max_count'] :
                    print(res['question']) 
                    result_questions["questions"].append(res)
                    level['counter']+=1
                    print(f"{level['level']} : {level['counter']}")
                    
                    #store the question in vectordb for exactmatch finding
                    QUESTION_DOCUMENT.add_texts(
                        [res['question']]
                    )

            count = level['max_count'] - level['counter']
            print(f"remaining count: {count}")

        return result_questions

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error processing the response: {e}")




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


if __name__ == "__main__":
    data = QuestionFormat(
        course_id=1,
        course_title="Software Development",
        numOfVeryEasy=30,
        numOfEasy=30,
        numOfAverage=30,
        numOfHard=30,
        numOfVeryHard=30,
    )

    result = createQuestions(data)
    print("Generated Questions:")
    print(result)

    
