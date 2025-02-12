import os
import layoutparser as lp
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
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

from  models import CombinedDistilBERT, QuestionEmbeddings
from transformers import DistilBertTokenizer

from dotenv import load_dotenv
load_dotenv() 

import nltk
nltk.download('stopwords',quiet=True)
nltk.download('punkt',quiet=True)
nltk.download('punkt_tab',quiet=True)
nltk.download('wordnet',quiet=True)
# nltk_data_dir = "nltk_data/" 
# nltk.data.path.append(nltk_data_dir)

API_KEY = os.getenv("OPENAI_API_KEY")
LARAVEL_IP = os.getenv('LARAVEL_IP_ADDRESS')
LARAVEL_PORT = os.getenv("LARAVEL_PORT")
API_KEY_LLAMAPARSE = os.getenv("LLAMA_CLOUD_API_KEY")

# Routes
STORE_PDF_ROUTE = "admin/store-processed-pdf/"
UPDATE_MODULE_STATUS_ROUTE = "admin/update-module-status"
STORE_QUESTION_ROUTE = "admin/store-questions"

FASTER_RCNN_CONFIG_PATH = os.getenv('FASTER_RCNN_CONFIG_PATH')
FASTER_RCNN_MODEL_PATH = os.getenv('FASTER_RCNN_MODEL_PATH')
CHROMA_PERSIST_DIR = os.getenv('CHROMA_PERSIST_DIR')
NLTK_DATA_DIR = os.getenv('NLTK_DATA_DIR')

RANDOM_FOREST_CLASSIFIER_PATH = os.getenv('RF_CLASSIFIER_PATH')
TFIDF_PATH = os.getenv('TFIDF_PATH')

LABEL_ENCODER_PATH = os.getenv('LABEL_ENCODER_PATH')
FEATURE_STATS_PATH = os.getenv('FEATURE_STATS_PATH')
DISTILBERT_MODEL_PATH = os.getenv('DISTILBERT_MODEL_PATH')
DISTILBERT_TOKENIZER_PATH = os.getenv('DISTILBERT_TOKENIZER_PATH')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# Set up the parser
PARSER = LlamaParse(
    api_key=API_KEY_LLAMAPARSE,
    result_type="text"
)

DETECTRON_MODEL = lp.Detectron2LayoutModel(
    config_path=FASTER_RCNN_CONFIG_PATH,
    model_path=FASTER_RCNN_MODEL_PATH,
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

# question difficulty estimation model 
MAX_LEN = 128
LABEL_ENCODER = joblib.load(LABEL_ENCODER_PATH)
LABEL_CLASSES = len(LABEL_ENCODER.classes_) 
FEATURE_STATS = joblib.load(FEATURE_STATS_PATH)

DISTILBERT_MODEL = CombinedDistilBERT(num_classes=len(LABEL_ENCODER.classes_) , feature_dim=7)
DISTILBERT_TOKENIZER = DistilBertTokenizer.from_pretrained(DISTILBERT_TOKENIZER_PATH, clean_up_tokenization_spaces=True)

# Load the saved weights
DISTILBERT_MODEL.load_state_dict(torch.load(DISTILBERT_MODEL_PATH,map_location=DEVICE,weights_only=True))
DISTILBERT_MODEL.eval()  # Set to evaluation mode

DISTILBERT_EMBEDDING = QuestionEmbeddings(
    tokenizer=DISTILBERT_TOKENIZER,
    model=DISTILBERT_MODEL,
    max_len=MAX_LEN
)

RF_CLASSIFIER = joblib.load(RANDOM_FOREST_CLASSIFIER_PATH)
TFIDF = joblib.load(TFIDF_PATH)

CONTENT_DOCUMENT = Chroma(
    collection_name="TopIT",
    embedding_function=SBERT,
    persist_directory=CHROMA_PERSIST_DIR,  # Loaded from .env
)

QUESTION_DOCUMENT = Chroma(
    collection_name="Questions",
    embedding_function=SBERT,
    persist_directory=CHROMA_PERSIST_DIR,
)

# Load the llm 
LLM = ChatOpenAI(model_name="gpt-4o-mini",api_key=API_KEY)

#EasyOCR
# gpu_available = torch.cuda.is_available()
EASY_READER = easyocr.Reader(['en'],gpu=DEVICE,model_storage_directory=None,download_enabled=True)

# Configure logging
logging.basicConfig(
    filename="app.log",  # Path to the log file
    level=logging.INFO,  # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S",  # Date format for log messages
)