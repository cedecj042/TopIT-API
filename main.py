from fastapi import FastAPI, File, UploadFile,Form, HTTPException, BackgroundTasks,Request, status, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from Pdf_processing import actual_pdf_processing_function,process_content
from Rag import *
from Setup import *
from Models import CreateQuestionsRequest, QueryRequest, ModuleModel,CourseModel

import nest_asyncio
import asyncio
import shutil
import time
from typing import List

nest_asyncio.apply()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specific origins like ["http://localhost"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
    
@app.get("/view-data/")
def view_data():
    # Access the stored documents from the Chroma collection
    all_data = CONTENT_DOCUMENT._collection.get(include=["documents"])  # Retrieves all documents
    return {"data": all_data["documents"]}

@app.get("/view-questions/")
def view_data():
    # Access the stored documents from the Chroma collection
    all_data = QUESTION_DOCUMENT._collection.get(include=["documents"])  # Retrieves all documents
    return {"data": all_data["documents"]}

@app.post("/process-pdf/")
async def process_pdf(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    course_name: str = Form(...), 
    course_id: int = Form(...)
):
    # Read the file content
    file_content = await file.read()
    
    # Schedule the background task for processing the PDF
    background_tasks.add_task(actual_pdf_processing_function, file_content, course_name, file.filename, course_id)
    
    return {"message": "Processing started, you'll be notified when it's done."}
     
@app.delete("/delete/{pdf_name}")
async def delete_pdf_images(pdf_name: str):
    images_dir = f"images/{pdf_name}"
    if os.path.exists(images_dir):
        try:
            shutil.rmtree(images_dir)
            return JSONResponse(content={
                "message": f"Images for {pdf_name} deleted successfully"
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred while deleting: {str(e)}")
    else:
        raise HTTPException(status_code=404, detail=f"No images found for {pdf_name}")


@app.post("/create-questions/")
async def create_questions_endpoint(
    background_tasks: BackgroundTasks, 
    data: list[CreateQuestionsRequest]
):
    # Add the createQuestions function to the background tasks
    background_tasks.add_task(
        send_questions_to_laravel, 
        data
    )
    return {"message": "Question generation task started in the background."}

@app.post("/query/")
def query_hierarchical_index_endpoint(request: QueryRequest, level: str = None):
    try:
        results = queryHierarchicalIndex(request.query, level)
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 

async def process_image_with_retries(image: UploadFile, retries: int = 3, delay: int = 2):
    file_location = f"/tmp/{image.filename}"

    # Save the uploaded file to a temporary location
    try:
        with open(file_location, "wb") as buffer:
            buffer.write(await image.read())
    except Exception as e:
        logging.error(f"Failed to save the uploaded image: {e}")
        return {"error": "Failed to save the uploaded image."}

    for attempt in range(retries):
        try:
            logging.info(f"Processing image attempt {attempt + 1}...")

            # Define the file extractor for supported image types
            file_extractor = {".jpg": PARSER, ".jpeg": PARSER, ".png": PARSER}  # Support more formats

            # Load the document using the SimpleDirectoryReader
            documents = SimpleDirectoryReader(input_files=[file_location], file_extractor=file_extractor).load_data()

            # Extract the text content
            text_content = "\n".join([doc.text for doc in documents])

            logging.info(f"Text extracted: {text_content}")

            if text_content.strip():
                if os.path.exists(file_location):
                    os.remove(file_location)
                return {"text": text_content}
            else:
                logging.warning(f"Attempt {attempt + 1}: No text extracted, retrying...")
                # Wait for a short delay before retrying
                await asyncio.sleep(delay)

        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay)  # Wait before retrying
            else:
                logging.error(f"Retries exhausted. Final error: {e}")
                if os.path.exists(file_location):
                    os.remove(file_location)
                return {"error": "Failed to process the image after multiple attempts."}

    # If retries are exhausted and no valid text is found, return an error message
    if os.path.exists(file_location):
        os.remove(file_location)
    return {"error": "Text extraction failed after multiple attempts."}


# FastAPI route to handle image uploads
@app.post("/process-image/")
async def imagetoText(image: UploadFile = File(...)):
    try:
        # Call the retry function with a maximum of 3 retries
        return await process_image_with_retries(image, retries=3, delay=2)
    except Exception as e:
        logging.error(f"Error processing image to text: {e}")
        return {"error": str(e)}

@app.post("/add-modules/")
async def add_modules_bulk(modules: List[ModuleModel]):
    try:
        logging.info("Batch processing started for modules.")
        all_documents = []
        all_ids = []

        for module in modules:
            logging.info(f"Processing module with ID: {module.module_id}")
            # Check if the course exists for each module
            course = CONTENT_DOCUMENT._collection.get(
                where={"course_id": module.course_id}
            )
            if not course['documents']:
                raise HTTPException(status_code=404, detail=f"Course with ID {module.course_id} not found. Please create the course first.")

            # Process module content
            module_content = process_content(module.contents)
            module_document = Document(
                page_content=f"Module: {module.title}. " + " ".join(module_content),
                metadata={"module_id": module.module_id, "course_id": module.course_id, "type": "Module"}
            )
            all_documents.append(module_document)
            all_ids.append(str(module.module_id))

            # Add lessons, sections, subsections, and associated content
            for lesson in module.lessons:
                lesson_content = process_content(lesson.contents)
                lesson_document = Document(
                    page_content=f"Lesson: {lesson.title}. " + " ".join(lesson_content),
                    metadata={"module_id": module.module_id, "lesson_id": lesson.lesson_id, "course_id": module.course_id, "type": "Lesson"}
                )
                all_documents.append(lesson_document)
                all_ids.append(f"{module.module_id}_{lesson.lesson_id}")

                for section in lesson.sections:
                    section_content = process_content(section.contents)
                    section_document = Document(
                        page_content=f"Section: {section.title}. " + " ".join(section_content),
                        metadata={"module_id": module.module_id, "lesson_id": lesson.lesson_id, "section_id": section.section_id, "course_id": module.course_id, "type": "Section"}
                    )
                    all_documents.append(section_document)
                    all_ids.append(f"{module.module_id}_{lesson.lesson_id}_{section.section_id}")

                    for subsection in section.subsections:
                        subsection_content = process_content(subsection.contents)
                        subsection_document = Document(
                            page_content=f"Subsection: {subsection.title}. " + " ".join(subsection_content),
                            metadata={"module_id": module.module_id, "lesson_id": lesson.lesson_id, "section_id": section.section_id, "subsection_id": subsection.subsection_id, "course_id": module.course_id, "type": "Subsection"}
                        )
                        all_documents.append(subsection_document)
                        all_ids.append(f"{module.module_id}_{lesson.lesson_id}_{section.section_id}_{subsection.subsection_id}")

        # Add all documents, ids, and metadata to the Chroma collection
        CONTENT_DOCUMENT.add_documents(documents=all_documents, ids=all_ids)
        logging.info("All documents successfully added to Chroma collection.")
        module_ids = [module.module_id for module in modules]
        # Notify Laravel
        await update_module_status(module_ids)
        
        return {"message": "Modules added successfully"}

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
async def update_module_status(module_ids: List[int]):
    """
    Notify Laravel that modules have been vectorized.
    """

    laravel_endpoint = f"http://{LARAVEL_IP}:{LARAVEL_PORT}/{UPDATE_MODULE_STATUS_ROUTE}"
    try:
        # Notify Laravel about the completed vectorization
        response = requests.post(
            laravel_endpoint,
            json={"module_ids": module_ids},
        )
        response.raise_for_status()
        logging.info(f"Laravel notified successfully for module IDs: {module_ids}")
        return {"message": "Laravel notified successfully"}
    except Exception as e:
        logging.error(f"Failed to notify Laravel: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to notify Laravel: {str(e)}"
        )
    

@app.post("/create_course/")
async def create_course(course: CourseModel):
    try:
        logging.info('Creating new course')
        # Query the collection to see if a course with the same course_id already exists
        existing_course = CONTENT_DOCUMENT._collection.get(
            where={"course_id": course.course_id}
        )

        # Check if the course already exists
        if existing_course['documents']:
            raise HTTPException(status_code=400, detail="Course with this ID already exists.")

        # Create a Document object for the course
        course_document = Document(
            page_content=f"Course Title: {course.title}. Course Description: {course.description}",
            metadata={"course_id": course.course_id}
        )
        
        # Add the document to the vector store
        CONTENT_DOCUMENT.add_documents(
            documents=[course_document],  # Pass Document object
            ids=[str(course.course_id)],
        )
        logging.info('Course Created successfully')
        return {"message": "Course created successfully", "course_id": course.course_id}

    except Exception as e:
        logging.error(f"Error in creating course {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

from fastapi import HTTPException

@app.delete("/delete_module/{module_id}")
async def delete_module(module_id: int):
    try:
        # Retrieve all documents associated with the module_id
        module_related_documents = CONTENT_DOCUMENT._collection.get(
            where=lambda doc: doc['metadata'].get('module_id') == module_id
        )
        
        # Extract document IDs to delete
        document_ids = [doc['id'] for doc in module_related_documents if 'id' in doc]

        # If no documents are found, raise a 404 error
        if not document_ids:
            raise HTTPException(status_code=404, detail=f"No module or associated content found for Module ID {module_id}.")

        # Delete all documents related to the module_id
        CONTENT_DOCUMENT._collection.delete(ids=document_ids)
        
        return {"message": f"Module {module_id} and all associated documents deleted successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.delete("/delete_course/{course_id}")
async def delete_course(course_id: int):  # Ensure course_id is an integer
    try:
        # Retrieve all documents associated with the course_id
        course_documents = CONTENT_DOCUMENT._collection.get(
            where={"course_id": course_id}  # Query with integer course_id
        )
        # Check if any documents are found for the course_id
        document_ids = course_documents.get('ids', [])

        # If no documents are found, raise a 404 error
        if not document_ids:
            raise HTTPException(status_code=404, detail=f"No course or associated content found for Course ID {course_id}.")

        # Delete all documents related to the course_id
        CONTENT_DOCUMENT._collection.delete(ids=document_ids)

        return {"message": f"Course {course_id} and all associated documents deleted successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/reset_collection/")
async def reset_collection():
    try:
        global CONTENT_DOCUMENT
        # Get all documents in the collection
        all_documents = CONTENT_DOCUMENT._collection.get()

        # Extract document IDs from the 'ids' key
        document_ids = all_documents.get('ids', [])

        # Delete all documents by their IDs
        if document_ids:
            CONTENT_DOCUMENT._collection.delete(ids=document_ids)

        return {"message": "All documents cleared from the collection successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
@app.post("/reset_question_collection/")
async def reset_collection():
    try:
        global QUESTION_DOCUMENT
        # Get all documents in the collection
        all_documents = QUESTION_DOCUMENT._collection.get()

        # Extract document IDs from the 'ids' key
        document_ids = all_documents.get('ids', [])

        # Delete all documents by their IDs
        if document_ids:
            QUESTION_DOCUMENT._collection.delete(ids=document_ids)

        return {"message": "All documents cleared from the collection successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)