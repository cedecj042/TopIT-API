from fastapi import FastAPI, File, UploadFile,Form, HTTPException, BackgroundTasks,Request, status, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import os
import cv2
import layoutparser as lp
from PIL import Image
import pytesseract
import easyocr
import json
import requests
from typing import List
from uuid import uuid4
import shutil
import fitz  # PyMuPDF
import base64
from io import BytesIO
import logging 
import re
import torch
from dotenv import load_dotenv

load_dotenv() 
ip = os.getenv('IP_ADDRESS')

model = lp.Detectron2LayoutModel(
    config_path='faster_rcnn/config.yaml',
    model_path='faster_rcnn/model_final.pth',
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7],
    label_map={0: "Caption", 1: "Code", 2: "Figures", 3: "Header", 4: "Lesson", 5: "Module", 6: "Section", 7: "Subsection", 8: "Tables", 9: "Text"}
)

# Configure logging
logging.basicConfig(
    filename="app.log",  # Path to the log file
    level=logging.INFO,  # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S",  # Date format for log messages
)

    # Check if GPU is available
gpu_available = torch.cuda.is_available()
reader = easyocr.Reader(['en'],gpu=gpu_available,model_storage_directory=None,download_enabled=False)

def natural_sort_key(s):
    """Helper function to extract numbers from a string for natural sorting."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def sort_textblocks_by_y1(textblocks):
    """Sort TextBlocks by their y_1 coordinate in ascending order."""
    return sorted(textblocks, key=lambda tb: tb.block.y_1, reverse=False)


def clean_and_sanitize_text(text):
    # Remove unwanted characters and patterns
    unwanted_chars = ['>>>', '>>', '>', '+', '*', '«', '@', '©']
    for char in unwanted_chars:
        text = text.replace(char, ' ')
    
    # Remove specific newline patterns
    text = re.sub(r"\n+", ' ', text)  # Replacing newline with a space instead of removing

    # Strip harmful HTML tags, but allow basic formatting
    allowed_tags = ['<b>', '<i>', '<p>', '<strong>', '<em>', '<ul>', '<ol>', '<li>', '<br>', '<br />']
    tag_re = re.compile(r'(<!--.*?-->|<[^>]*>)')
    
    # Replace disallowed tags with a space
    text = tag_re.sub(lambda m: m.group(0) if m.group(0) in allowed_tags else ' ', text)

    # Trim any extra whitespace
    return text.strip()

def process_image_with_layoutparser(image, model):
    layout = model.detect(image)
    sorted_layout = sort_textblocks_by_y1(layout)
    return sorted_layout

def extract_text_from_layout(sorted_layout, image):
    items = {}
    relevant_types = ["Text", "Tables", "Module", "Lesson", "Section", "Subsection", "Caption", "Code", "Figures", "Header"]
    
    for idx, lay in enumerate(sorted_layout):
        if lay.type in relevant_types:
            rect = lay.block
            x1, y1, x2, y2 = int(rect.x_1), int(rect.y_1), int(rect.x_2), int(rect.y_2)
            cropped_image = image[y1:y2, x1:x2]

            if lay.type in ["Tables", "Figures", "Code"]:
                # Convert image to Base64 string
                pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                buffered = BytesIO()
                pil_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                items[f"Block_{idx+1}"] = {
                    "type": lay.type,
                    "image_base64": img_str,
                    "coordinates": [x1, y1, x2, y2]
                }
            else:
                # Use EasyOCR to extract text
                result = reader.readtext(cropped_image, detail=0)  # detail=0 for text only
                text = ' '.join(result)
                cleaned_text = clean_and_sanitize_text(text)
                items[f"Block_{idx+1}"] = {
                    "type": lay.type,
                    "text": cleaned_text,
                    "coordinates": [x1, y1, x2, y2]
                }
    return items

# Initialize order counter
order_counter = 1
def add_to_last_object(course_structure, key, value):
    """Recursively add content to the correct level in the nested structure following the course hierarchy."""
    
    def find_last_object(structure, object_type):
        if isinstance(structure, dict):
            logging.info(f"Searching for last {object_type} in {list(structure.keys())}")
            if object_type in structure and structure[object_type]:
                logging.info(f"Found {object_type} in structure, returning the last one")
                return structure[object_type][-1]  # Safely return the last object if it exists
            for key in ["Modules", "Lessons", "Sections", "Subsections"]:
                if key in structure and structure[key]:
                    logging.info(f"Recursively searching in {key}")
                    return find_last_object(structure[key][-1], object_type)  # Recursively find the last object
        logging.info(f"Could not find {object_type} in the structure")
        return None 
    
    def get_total(items):
        if items is None:
            return 1  # Return a default value if items are None
        return len(items) + 1

    logging.info(f"Adding {key} to the course structure")
    
    global order_counter
    
    # Reset the order when we encounter a new level (Module, Lesson, Section, Subsection)
    if key in ["Module", "Lesson", "Section", "Subsection"]:
        order_counter = 1
        
        if key == "Module": 
            module = {"Title": value["text"], "Type": value["type"], "Lessons": [], "Content": []}
            course_structure["Modules"].append(module)
        elif key == "Lesson":
            last_module = find_last_object(course_structure, "Modules")
            if last_module and "Lessons" in last_module:
                lesson = {"Title": value["text"], "Type": value["type"], "Sections": [], "Content": []}
                last_module["Lessons"].append(lesson)
            else:
                new_module = {"Title": f"Module {get_total(course_structure['Modules'])}", "Type": 'Module', "Lessons": [], "Content": []}
                lesson = {"Title": value["text"], "Type": value["type"], "Sections": [], "Content": []}
                new_module["Lessons"].append(lesson)
                course_structure["Modules"].append(new_module)
        elif key == "Section":
            last_lesson = find_last_object(course_structure, "Lessons")
            if last_lesson:
                section = {"Title": value["text"], "Type": value["type"], "Subsections": [], "Content": []}
                last_lesson["Sections"].append(section)
            else:
                new_lesson = {"Title": f"Lesson {get_total(course_structure.get('Lessons', []))}", "Type": "Lesson", "Sections": [], "Content": []}
                section = {"Title": value["text"], "Type": value["type"], "Subsections": [], "Content": []}
                new_lesson["Sections"].append(section)
                last_module = find_last_object(course_structure, "Modules")
                if last_module:
                    last_module["Lessons"].append(new_lesson)
                else:
                    new_module = {"Title": f"Module {get_total(course_structure['Modules'])}", "Type": "Module", "Lessons": [], "Content": []}
                    new_module["Lessons"].append(new_lesson)
                    course_structure["Modules"].append(new_module)
        elif key == "Subsection":
            last_section = find_last_object(course_structure, "Sections")
            if last_section:
                subsection = {"Title": value["text"], "Content": []}
                last_section["Subsections"].append(subsection)
            else:
                new_section = {"Title": f"Section {get_total(course_structure.get('Sections', []))}", "Type": "Section", "Subsections": [], "Content": []}
                subsection = {"Title": value["text"], "Type": value["type"], "Content": []}
                new_section["Subsections"].append(subsection)
                last_lesson = find_last_object(course_structure, "Lessons")
                if last_lesson:
                    last_lesson["Sections"].append(new_section)
                else:
                    logging.debug(f"Creating a new lesson and adding a section with subsection")
                    new_lesson = {"Title": f"Lesson {get_total(course_structure.get('Lessons', []))}", "Type": "Lesson", "Sections": [], "Content": []}
                    new_lesson["Sections"].append(new_section)
                    last_module = find_last_object(course_structure, "Modules")
                    if last_module:
                        last_module["Lessons"].append(new_lesson)
                    else:
                        new_module = {"Title": f"Module {get_total(course_structure['Modules'])}", "Type": "Module", "Lessons": [], "Content": []}
                        new_module["Lessons"].append(new_lesson)
                        course_structure["Modules"].append(new_module)

    # Track order for content items like Tables, Figures, Text, Header
    else:
        last_obj = find_last_object(course_structure, "Subsections") or \
                   find_last_object(course_structure, "Sections") or \
                   find_last_object(course_structure, "Lessons") or \
                   find_last_object(course_structure, "Modules") or \
                   course_structure
        value["order"] = order_counter
        if key == "Tables":
            if "Tables" not in last_obj:
                last_obj["Tables"] = []
            last_obj["Tables"].append(value)
        elif key == "Figures":
            if "Figures" not in last_obj:
                last_obj["Figures"] = []
            last_obj["Figures"].append(value)
        elif key == "Code":
            if "Codes" not in last_obj:
                last_obj["Codes"] = []
            last_obj["Codes"].append(value)
        else:
            if "Content" not in last_obj:
                last_obj["Content"] = []
            last_obj["Content"].append(value)

        # Increment order for each content element added
        order_counter += 1

def build_json_structure(extracted_data, course_structure=None, course_name=""):
    if not isinstance(extracted_data, dict):
        logging.error(f"Expected extracted_data to be a dict, got {type(extracted_data)}")
        return course_structure  # Or raise an exception
    
    # Initialize the course structure with the course name
    if course_structure is None:
        course_structure = {"Course": course_name, "Modules": []}

    # To keep track of unassigned captions (before and after)
    before_caption = None
    last_block_info = None

    for block_id, data in extracted_data.items():
        text_type = data.get("type")
        
        # Handle caption before the table/figure/code
        if text_type == "Caption":
            if last_block_info and last_block_info["type"] in ["Tables", "Figures", "Code"]:
                # Associate this caption as an after caption
                last_block_info["after_caption"] = data.get("text")
                last_block_info = None  # Reset after use
            else:
                before_caption = data.get("text")
            continue

        # Handle tables, figures, and code blocks
        if text_type in ["Tables", "Figures", "Code"]:
            block_info = {
                "type": text_type,
                "image_base64": data.get("image_base64"),
                "coordinates": data.get("coordinates"),
                "before_caption": before_caption,  # Caption that appeared before this block
                "after_caption": None  # Placeholder for caption that may appear after
            }
            before_caption = None  # Reset the before caption after using it
            last_block_info = block_info  # Temporarily store this block info for possible after caption
            
            # Add the block_info to the structure immediately
            add_to_last_object(course_structure, text_type, block_info)
            
        else:
            # Handle other text types like headers or regular text
            block_info = {
                "type": text_type,
                "text": data.get("text")
            }
            last_block_info = None  # Reset, as this is not a table/figure/code

            # Add the block_info to the structure immediately
            add_to_last_object(course_structure, text_type, block_info)

    return course_structure


def process_pdf_images(pdf_folder_path, model, course_name):
    course_structure = None

    # Sort files by numeric part of the filename
    for image_file in sorted(os.listdir(pdf_folder_path), key=natural_sort_key):
        image_path = os.path.join(pdf_folder_path, image_file)
        image = cv2.imread(image_path)

        sorted_layout = process_image_with_layoutparser(image, model)
        extracted_text = extract_text_from_layout(sorted_layout, image)
        
        # Build or update the JSON structure with the new data
        course_structure = build_json_structure(extracted_text, course_structure, course_name)

    return course_structure


def convert_pdf_to_images(pdf_path):
    # Get the base name of the PDF file without the directory components and extension
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    images_dir = f"images/{pdf_name}"
    # Create a folder for the converted images
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    pdf_document = fitz.open(pdf_path)
    
    for page_index in range(len(pdf_document)):
        page = pdf_document.load_page(page_index)
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.save(os.path.join(images_dir, f"page_{page_index + 1}.png"), "PNG")

    pdf_document.close()

    #print(f"PDF conversion complete for {pdf_path}. Images are saved in the '{images_dir}' folder.")
    return images_dir


def actual_pdf_processing_function(file_content, course_name: str, filename: str, course_id: int):
    try:
        upload_dir = f"uploads/{uuid4()}"
        os.makedirs(upload_dir, exist_ok=True)
        pdf_path = os.path.join(upload_dir, filename)
        with open(pdf_path, "wb") as buffer:
            buffer.write(file_content)

        images_folder = convert_pdf_to_images(pdf_path)

        course_json = process_pdf_images(images_folder, model, course_name=course_name)

        output_path = os.path.join(upload_dir, 'output.json')
        with open(output_path, 'w') as json_file:
            json.dump(course_json, json_file)
            
        logging.info("Sending data to Laravel.")
        
        laravel_endpoint = f"http://{ip}:8000/admin/store-processed-pdf/"
        response = requests.post(laravel_endpoint, json={
            "course_id": course_id,
            "file_name":filename,
            "processed_data": course_json
        })
        
        shutil.rmtree(upload_dir)
        
        if response.status_code == 201:
            logging.info("Data stored successfully in Laravel.")
        else:
            logging.error(f"Failed to store data: {response.text}")

        # Return the response details for logging or further use
        return {"status": "success", "details": response.json() if response.status_code == 201 else response.text}

    except Exception as e:
        logging.error(f"An error occurred during PDF processing: {str(e)}")
        return {"status": "error", "details": str(e)}

def process_content(content_list):
    """Processes the content at any level (module, lesson, section, subsection)."""
    processed_content = []
    
    for item in content_list:
        # Check if it's a header or text and process it accordingly
        if item["type"] == "Header":
            processed_content.append(f"Header: {item['text']}")
        elif item["type"] == "Text":
            processed_content.append(f"Text: {item['text']}")
    
    return processed_content