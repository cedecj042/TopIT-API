import os
from huggingface_hub import hf_hub_download
import layoutparser as lp
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb import PersistentClient, HttpClient
import chromadb
# from langchain_community.chat_models import ChatOpenAI
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

from huggingface_hub import hf_hub_download

import torch
from PIL import Image
import requests


from dotenv import load_dotenv
load_dotenv() 

import nltk
nltk.download('stopwords',quiet=True)
nltk.download('punkt',quiet=True)
nltk.download('punkt_tab',quiet=True)
nltk.download('wordnet',quiet=True)


API_KEY = os.getenv("OPENAI_API_KEY")
LARAVEL_IP = os.getenv('LARAVEL_IP_ADDRESS')
LARAVEL_PORT = os.getenv("LARAVEL_PORT")
API_KEY_LLAMAPARSE = os.getenv("LLAMA_CLOUD_API_KEY")

# Hugging Face model repository details
repo_id = "lepuer/layout_parser"
config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml")
model_path = hf_hub_download(repo_id=repo_id, filename="model_final.pth")
# chroma_client = HttpClient(host="chromadb", port=8000)

# Initialize Detectron2LayoutModel using downloaded files
DETECTRON_MODEL = lp.Detectron2LayoutModel(
    config_path=config_path,
    model_path=model_path,
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7],
    label_map={0: "Caption", 1: "Code", 2: "Figures", 3: "Header", 4: "Lesson", 5: "Module", 6: "Section", 7: "Subsection", 8: "Tables", 9: "Text"}
)

STORE_PDF_ROUTE = "admin/store-processed-pdf/"
UPDATE_MODULE_STATUS_ROUTE = "admin/update-module-status"
STORE_QUESTION_ROUTE = "admin/store-questions"

FASTER_RCNN_REPO_ID = "lepuer/layout_parser"
FASTER_RCNN_FILENAME = "model_final.pth" 
FASTER_RCNN_MODEL_PATH = hf_hub_download(repo_id=FASTER_RCNN_REPO_ID, filename=FASTER_RCNN_FILENAME)
FASTER_RCNN_CONFIG_PATH = hf_hub_download(repo_id=FASTER_RCNN_REPO_ID, filename="config.yaml")


CHROMA_PERSIST_DIR = os.getenv('CHROMA_PERSIST_DIR')
NLTK_DATA_DIR = os.getenv('NLTK_DATA_DIR')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# Set up the parser
PARSER = LlamaParse(
    api_key=API_KEY_LLAMAPARSE,
    result_type="text"
)
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2') 

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_KWARGS = {'device': 'cpu'} #use "cuda" if you have nvidia gpu otherwise use "cpu"
ENCODE_KWARGS = {'normalize_embeddings': True}

SBERT = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs=MODEL_KWARGS,
    encode_kwargs=ENCODE_KWARGS
)

# question difficulty estimation model 
SCALER = joblib.load("RandomForest/updated_scaler.pkl")
TFIDF_VECTORIZER = joblib.load("RandomForest/updated_tfidf_vectorizer.pkl")
RANDOM_FOREST_MODEL = joblib.load("RandomForest/updated_trained_model.pkl")
client = chromadb.PersistentClient(path="chroma_backup/chroma_db1")
print(client.list_collections())

CONTENT_DOCUMENT = Chroma(
    # client=chroma_client,
    collection_name="TopIT1",
    embedding_function=SBERT,
    persist_directory=CHROMA_PERSIST_DIR,
)

QUESTION_DOCUMENT = Chroma(
    # client=chroma_client,
    collection_name="Questions1",
    embedding_function=SBERT,
    persist_directory=CHROMA_PERSIST_DIR
)

# Load the llm 
LLM = ChatOpenAI(model_name="gpt-4o-mini",api_key=API_KEY, temperature=0.5, top_p=0.9)

#EasyOCR
EASY_READER = easyocr.Reader(['en'],gpu=DEVICE,model_storage_directory=None,download_enabled=True)

# Configure logging
logging.basicConfig(
    filename="app.log",  # Path to the log file
    level=logging.INFO,  # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S",  # Date format for log messages
)