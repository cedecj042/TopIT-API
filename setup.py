import os
import layoutparser as lp
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
# for llamaparse
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

import joblib
import logging 
import torch
import easyocr
import pickle

from sklearn.preprocessing import StandardScaler

from dotenv import load_dotenv
load_dotenv() 

import nltk
nltk_data_dir = "nltk_data/" 
nltk.data.path.append(nltk_data_dir)

api_key = os.getenv("OPENAI_API_KEY")
ip = os.getenv('LARAVEL_IP_ADDRESS')
API_KEY_LlamaParse = os.getenv("LLAMA_CLOUD_API_KEY")
# Set up the parser
parser = LlamaParse(
    api_key=API_KEY_LlamaParse,
    result_type="text"
)

model = lp.Detectron2LayoutModel(
    config_path='faster_rcnn/config.yaml',
    model_path='faster_rcnn/model_final.pth',
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7],
    label_map={0: "Caption", 1: "Code", 2: "Figures", 3: "Header", 4: "Lesson", 5: "Module", 6: "Section", 7: "Subsection", 8: "Tables", 9: "Text"}
)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 

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

vector_store_questions = Chroma(
    collection_name="Questions",
    embedding_function=Sbert,
    persist_directory="./chroma_db1",
)

# question difficulty estimation model 
reference_embeddings = joblib.load('models/embeddings.pkl')
classifier_model = joblib.load('models/trained_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
scaler = joblib.load('models/scaler.pkl')
# loaded_keywords = joblib.load('models/category_keywords.pkl')

# with open('models/category_keywords.pkl', 'rb') as file:
#     loaded_keywords = pickle.load(file)

# Load the llm 
llm = ChatOpenAI(model_name="gpt-4o-mini",api_key=api_key)

#EasyOCR
gpu_available = torch.cuda.is_available()
reader = easyocr.Reader(['en'],gpu=gpu_available,model_storage_directory=None,download_enabled=True)

# Configure logging
logging.basicConfig(
    filename="app.log",  # Path to the log file
    level=logging.INFO,  # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S",  # Date format for log messages
)