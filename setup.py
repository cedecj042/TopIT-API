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
from models import CombinedDistilBERT
from paths import *

from sklearn.preprocessing import StandardScaler

from dotenv import load_dotenv
load_dotenv() 

import nltk
nltk_data_dir = "nltk_data/" 
nltk.data.path.append(nltk_data_dir)

API_KEY = os.getenv("OPENAI_API_KEY")
LARAVEL_IP = os.getenv('LARAVEL_IP_ADDRESS')
API_KEY_LLAMAPARSE = os.getenv("LLAMA_CLOUD_API_KEY")
LABEL_ENCODER_PATH = os.getenv("LABEL_ENCODER_PATH")
DISTILBERT_MODEL_PATH = os.getenv("DISTILBERT_MODEL_PATH")

# Set up the parser
PARSER = LlamaParse(
    api_key=API_KEY_LLAMAPARSE,
    result_type="text"
)

DETECTRON_MODEL = lp.Detectron2LayoutModel(
    config_path='faster_rcnn/config.yaml',
    model_path='faster_rcnn/model_final.pth',
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7],
    label_map={0: "Caption", 1: "Code", 2: "Figures", 3: "Header", 4: "Lesson", 5: "Module", 6: "Section", 7: "Subsection", 8: "Tables", 9: "Text"}
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

CONTENT_DOCUMENT = Chroma(
    collection_name="TopIT",
    embedding_function=SBERT,
    persist_directory="./chroma_db1",  # Where to save data locally, remove if not neccesary
)

QUESTION_DOCUMENT = Chroma(
    collection_name="Questions",
    embedding_function=SBERT,
    persist_directory="./chroma_db1",
)

# question difficulty estimation model 
LABEL_ENCODER = joblib.load(LABEL_ENCODER_PATH)
DISTILBERT_MODEL = CombinedDistilBERT(num_classes=len(LABEL_ENCODER.classes_) , feature_dim=7)

# Load the saved weights
DISTILBERT_MODEL.load_state_dict(torch.load(DISTILBERT_MODEL_PATH))
DISTILBERT_MODEL.eval()  # Set to evaluation mode
print("Model loaded successfully!")


# Load the llm 
LLM = ChatOpenAI(model_name="gpt-4o-mini",api_key=API_KEY)

#EasyOCR
gpu_available = torch.cuda.is_available()
EASY_READER = easyocr.Reader(['en'],gpu=gpu_available,model_storage_directory=None,download_enabled=True)

# Configure logging
logging.basicConfig(
    filename="app.log",  # Path to the log file
    level=logging.INFO,  # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S",  # Date format for log messages
)