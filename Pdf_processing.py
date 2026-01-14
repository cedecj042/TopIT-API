import os
import cv2
from PIL import Image
from uuid import uuid4
import shutil
import fitz
import base64
import json
import requests
import re
import asyncio
import uuid
from io import BytesIO

from Setup import logging, LARAVEL_IP, LARAVEL_PORT, STORE_PDF_ROUTE, DETECTRON_MODEL, EASY_READER

# Check if GPU is available

def natural_sort_key(s):
    """Helper function to extract numbers from a string for natural sorting."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def sort_textblocks_by_y1(textblocks):
    """Sort TextBlocks by their y_1 coordinate in ascending order."""
    return sorted(textblocks, key=lambda tb: tb.block.y_1, reverse=False)

def decode_base64_image(base64_str):
    """Convert a base64 string into a PIL image."""
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data))

def image_to_base64(image):
    """Convert a PIL image to a base64 encoded string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def merge_images_centered(images, background_color=(255, 255, 255)):
    """
    Merge images vertically, centered horizontally.
    Takes a list of PIL Images or base64 strings.
    """
    # Handle both PIL Images and base64 strings
    imgs = []
    for img in images:
        if isinstance(img, Image.Image):
            imgs.append(img.convert("RGBA"))
        else:
            # If it's a base64 string
            imgs.append(decode_base64_image(img).convert("RGBA"))
    
    # Calculate max dimensions
    widths, heights = zip(*(img.size for img in imgs))
    max_width = max(widths)
    total_height = sum(heights)

    # Create a new blank image
    new_image = Image.new('RGBA', (max_width, total_height), background_color + (255,))

    # Center each image horizontally
    y_offset = 0
    for img in imgs:
        x_offset = (max_width - img.width) // 2
        new_image.paste(img, (x_offset, y_offset), img)
        y_offset += img.height

    # Convert to RGB
    new_image = new_image.convert("RGB")

    return new_image


def clean_and_sanitize_text(text):
    # Remove unwanted characters and patterns
    unwanted_chars = ['>>>', '>>', '>', '+', '*', '«', '@', '©']
    for char in unwanted_chars:
        text = text.replace(char, ' ')
    
    # Remove specific newline patterns
    text = re.sub(r"\n+", ' ', text)
    # Replace semicolons with commas
    text = text.replace(';', ',')
    # Strip harmful HTML tags, but allow basic formatting
    allowed_tags = ['<b>', '<i>', '<p>', '<strong>', '<em>', '<ul>', '<ol>', '<li>', '<br>', '<br />']
    tag_re = re.compile(r'(<!--.*?-->|<[^>]*>)')
    
    # Replace disallowed tags with a space
    text = tag_re.sub(lambda m: m.group(0) if m.group(0) in allowed_tags else ' ', text)

    # Trim any extra whitespace
    return text.strip()

async def process_image_with_layoutparser(image, model):
    loop = asyncio.get_event_loop()
    layout = await loop.run_in_executor(None, model.detect, image)
    sorted_layout = sort_textblocks_by_y1(layout)
    return sorted_layout

async def extract_text_from_layout(sorted_layout, image):
    items = {}
    relevant_types = ["Text", "Tables", "Module", "Lesson", "Section", "Subsection", "Caption", "Code", "Figures", "Header"]

    height, width, _ = image.shape
    y_threshold = int(0.9 * height)

    for idx, lay in enumerate(sorted_layout):
        if lay.type in relevant_types:
            rect = lay.block
            x1, y1, x2, y2 = int(rect.x_1), int(rect.y_1), int(rect.x_2), int(rect.y_2)
            
            if y1 > y_threshold:
                continue

            cropped_image = image[y1:y2, x1:x2]

            if lay.type in ["Tables", "Figures", "Code"]:
                pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                buffered = BytesIO()
                pil_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                items[f"Block_{idx+1}"] = {
                    "type": lay.type,
                    "image_base64": img_str,
                }
            else:
                result = await asyncio.to_thread(EASY_READER.readtext, cropped_image, detail=0)
                text = ' '.join(result)
                cleaned_text = clean_and_sanitize_text(text)
                items[f"Block_{idx+1}"] = {
                    "type": lay.type,
                    "text": cleaned_text,
                }
    return items

async def add_to_last_object(course_structure, key, value, order_counter):
    """Recursively add content to the correct level in the nested structure following the course hierarchy."""
    
    def generate_unique_id(prefix):
        """Generate a unique ID for each entity (Module, Lesson, Section, Subsection, Content)."""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

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
            return 1 
        return len(items) + 1

    logging.info(f"Adding {key} to the course structure")
    
    # Reset the order when we encounter a new level (Module, Lesson, Section, Subsection)
    if key in ["Module", "Lesson", "Section", "Subsection"]:
        order_counter[0] = 1
        
        if key == "Module": 
            module_uid = generate_unique_id("MD")
            module = {
                "module_uid": module_uid,
                "Title": value["text"],
                "Type": value["type"],
                "Lessons": [],
                "Content": []
            }
            course_structure["Modules"].append(module)
        elif key == "Lesson":
            last_module = find_last_object(course_structure, "Modules")
            if last_module and "Lessons" in last_module:
                lesson_uid = generate_unique_id("LS")
                lesson = {
                    "lesson_uid": lesson_uid,
                    "module_uid": last_module["module_uid"],
                    "Title": value["text"],
                    "Type": value["type"],
                    "Sections": [],
                    "Content": []
                }
                last_module["Lessons"].append(lesson)
            else:
                new_module = {
                    "module_uid": generate_unique_id("MD"),
                    "Title": f"Module {get_total(course_structure['Modules'])}",
                    "Type": 'Module',
                    "Lessons": [],
                    "Content": []
                }
                lesson_uid = generate_unique_id("LS")
                lesson = {
                    "lesson_uid": lesson_uid,
                    "module_uid": new_module["module_uid"],
                    "Title": value["text"],
                    "Type": value["type"],
                    "Sections": [],
                    "Content": []
                }
                new_module["Lessons"].append(lesson)
                course_structure["Modules"].append(new_module)
        elif key == "Section":
            last_lesson = find_last_object(course_structure, "Lessons")
            if last_lesson:
                section_uid = generate_unique_id("SC")
                section = {
                    "section_uid": section_uid,
                    "lesson_uid": last_lesson["lesson_uid"],
                    "Title": value["text"],
                    "Type": value["type"],
                    "Subsections": [],
                    "Content": []
                }
                last_lesson["Sections"].append(section)
            else:
                new_lesson = {
                    "lesson_uid": generate_unique_id("LS"),
                    "Title": "",
                    "Type": "Lesson",
                    "Sections": [],
                    "Content": []
                }
                section_uid = generate_unique_id("SC")
                section = {
                    "section_uid": section_uid,
                    "lesson_uid": new_lesson["lesson_uid"],
                    "Title": value["text"],
                    "Type": value["type"],
                    "Subsections": [],
                    "Content": []
                }
                new_lesson["Sections"].append(section)
                last_module = find_last_object(course_structure, "Modules")
                if last_module:
                    last_module["Lessons"].append(new_lesson)
                else:
                    new_module = {
                        "module_uid": generate_unique_id("MD"),
                        "Title": f"Module {get_total(course_structure['Modules'])}",
                        "Type": "Module",
                        "Lessons": [],
                        "Content": []
                    }
                    new_module["Lessons"].append(new_lesson)
                    course_structure["Modules"].append(new_module)
        elif key == "Subsection":
            last_section = find_last_object(course_structure, "Sections")
            if last_section:
                subsection_uid = generate_unique_id("SS")
                subsection = {
                    "subsection_uid": subsection_uid,
                    "section_uid": last_section["section_uid"],
                    "Title": value["text"],
                    "Content": []
                }
                last_section["Subsections"].append(subsection)
            else:
                new_section = {
                    "section_uid": generate_unique_id("SC"),
                    "Title": "",
                    "Type": "Section",
                    "Subsections": [],
                    "Content": []
                }
                subsection_uid = generate_unique_id("SS")
                subsection = {
                    "subsection_uid": subsection_uid,
                    "section_uid": new_section["section_uid"],
                    "Title": value["text"],
                    "Type": value["type"],
                    "Content": []
                }
                new_section["Subsections"].append(subsection)
                last_lesson = find_last_object(course_structure, "Lessons")
                if last_lesson:
                    last_lesson["Sections"].append(new_section)
                else:
                    logging.debug(f"Creating a new lesson and adding a section with subsection")
                    new_lesson = {
                        "lesson_uid": generate_unique_id("LS"),
                        "Title": "",
                        "Type": "Lesson",
                        "Sections": [],
                        "Content": []
                    }
                    new_lesson["Sections"].append(new_section)
                    last_module = find_last_object(course_structure, "Modules")
                    if last_module:
                        last_module["Lessons"].append(new_lesson)
                    else:
                        new_module = {
                            "module_uid": generate_unique_id("MD"),
                            "Title": f"Module {get_total(course_structure['Modules'])}",
                            "Type": "Module",
                            "Lessons": [],
                            "Content": []
                        }
                        new_module["Lessons"].append(new_lesson)
                        course_structure["Modules"].append(new_module)

    # Track order for content items like Tables, Figures, Text, Header
    else:
        last_obj = find_last_object(course_structure, "Subsections") or \
                   find_last_object(course_structure, "Sections") or \
                   find_last_object(course_structure, "Lessons") or \
                   find_last_object(course_structure, "Modules") or \
                   course_structure

        # Creating content with unique content_uid
        if key == "Text":
            # Check if there's a previous Text item in Content with an adjacent order
            if last_obj["Content"] and last_obj["Content"][-1]["type"] == "Text" and last_obj["Content"][-1]["order"] == order_counter[0] - 1:
                last_obj["Content"][-1]["text"] += " " + value["text"]
            else:
                value["order"] = order_counter[0]
                content_uid = f"CT_{uuid.uuid4().hex[:8]}"  # Generate a content UID
                value["content_uid"] = content_uid
                if "Content" not in last_obj:
                    last_obj["Content"] = []
                last_obj["Content"].append(value)
                order_counter[0] += 1

        elif key == "Tables":
            content_uid = f"CT_{uuid.uuid4().hex[:8]}"  # Generate a content UID
            value["content_uid"] = content_uid
            if "Tables" not in last_obj:
                last_obj["Tables"] = []
            last_obj["Tables"].append(value)
            order_counter[0] += 1
        elif key == "Figures":
            content_uid = f"CT_{uuid.uuid4().hex[:8]}"  # Generate a content UID
            value["content_uid"] = content_uid
            if "Figures" not in last_obj:
                last_obj["Figures"] = []
            last_obj["Figures"].append(value)
            order_counter[0] += 1
        elif key == "Code":
            content_uid = f"CT_{uuid.uuid4().hex[:8]}"  # Generate a content UID
            value["content_uid"] = content_uid
            if "Codes" not in last_obj:
                last_obj["Codes"] = []
            last_obj["Codes"].append(value)
            order_counter[0] += 1
        else:
            content_uid = f"CT_{uuid.uuid4().hex[:8]}"  # Generate a content UID
            value["content_uid"] = content_uid
            if "Content" not in last_obj:
                last_obj["Content"] = []
            last_obj["Content"].append(value)
            order_counter[0] += 1

async def build_json_structure(extracted_data, course_structure=None, course_name=""):
    if not isinstance(extracted_data, dict):
        logging.error(f"Expected extracted_data to be a dict, got {type(extracted_data)}")
        return course_structure  # Or raise an exception
    
    # Initialize the course structure with the course name
    if course_structure is None:
        course_structure = {"Course": course_name, "Modules": []}

    # To keep track of unassigned captions (before and after)
    before_caption = None
    last_block_info = None
    order_counter = [1] 
    
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
                "before_caption": before_caption,  # Caption that appeared before this block
                "after_caption": None  # Placeholder for caption that may appear after
            }
            before_caption = None  # Reset the before caption after using it
            last_block_info = block_info  # Temporarily store this block info for possible after caption
            
            # Add the block_info to the structure immediately
            await add_to_last_object(course_structure, text_type, block_info, order_counter)
            
        else:
            # Handle other text types like headers or regular text
            block_info = {
                "type": text_type,
                "text": data.get("text")
            }
            last_block_info = None  # Reset, as this is not a table/figure/code

            # Add the block_info to the structure immediately
            await add_to_last_object(course_structure, text_type, block_info, order_counter)
    
    return course_structure

async def process_pdf_images(pdf_folder_path, model, course_name):
    course_structure = None

    for image_file in sorted(os.listdir(pdf_folder_path), key=natural_sort_key):
        image_path = os.path.join(pdf_folder_path, image_file)
        image = await asyncio.to_thread(cv2.imread, image_path)

        sorted_layout = await process_image_with_layoutparser(image, model)
        extracted_text = await extract_text_from_layout(sorted_layout, image)
        
        course_structure = await build_json_structure(extracted_text, course_structure, course_name)

    return course_structure

async def convert_pdf_to_images(pdf_path):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    images_dir = f"images/{pdf_name}"
    os.makedirs(images_dir, exist_ok=True)

    pdf_document = fitz.open(pdf_path)
    
    for page_index in range(len(pdf_document)):
        page = pdf_document.load_page(page_index)
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.save(os.path.join(images_dir, f"page_{page_index + 1}.png"), "PNG")

    pdf_document.close()
    logging.info(f"PDF conversion complete for {pdf_path}. Images are saved in the '{images_dir}' folder.")
    return images_dir

async def actual_pdf_processing_function(file_content, course_name: str, filename: str, course_id: int, pdf_id: int):
    try:
        upload_dir = f"uploads/{uuid4()}"
        os.makedirs(upload_dir, exist_ok=True)
        pdf_path = os.path.join(upload_dir, filename)
        
        with open(pdf_path, "wb") as buffer:
            buffer.write(file_content)

        images_folder = await convert_pdf_to_images(pdf_path)
        course_json = await process_pdf_images(images_folder, DETECTRON_MODEL, course_name=course_name)

        output_path = os.path.join(upload_dir, 'output.json')
        with open(output_path, 'w') as json_file:
            json.dump(course_json, json_file)
            
        logging.info("Sending data to Laravel.")
        logging.info(f"Pdf id: {pdf_id}")
        laravel_endpoint = f"http://{LARAVEL_IP}:{LARAVEL_PORT}/{STORE_PDF_ROUTE}"
        
        response = requests.post(laravel_endpoint, json={
            "course_id": course_id,
            "file_name": filename,
            "processed_data": course_json,
            "pdf_id": pdf_id
        })

        if response.status_code == 201:
            logging.info("Data stored successfully in Laravel.")
        else:
            logging.error(f"Failed to store data: {response.text}")

        shutil.rmtree(upload_dir)
        images_dir = f"images/{filename.replace('.pdf', '')}"
        shutil.rmtree(images_dir)
        
        return {"status": "success", "details": response.json() if response.status_code == 201 else response.text}

    except Exception as e:
        logging.error(f"An error occurred during PDF processing: {str(e)}")
        return {"status": "error", "details": str(e)}

def process_content(content_list):
    """Processes the content at any level (module, lesson, section, subsection)."""
    processed_content = []
    
    for item in content_list:
        # Access fields with dot notation instead of bracket notation
        if item.type == "Header":
            processed_content.append(f"Header: {item.description}")
        elif item.type == "Text":
            processed_content.append(f"Text: {item.description}")
        
        # Handle Table, Figure, and Code content types
        elif item.type in ["Table", "Figure", "Code"]:
            content_str = f"{item.type}: {item.description}"
            if item.caption:
                content_str += f". Caption: {item.caption}"
            processed_content.append(content_str)
    
    return processed_content
