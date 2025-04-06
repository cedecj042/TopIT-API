from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Form,
    HTTPException,
    BackgroundTasks,
)
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from Pdf_processing import actual_pdf_processing_function, process_content
from Rag import *
from Setup import *
from Models import CreateQuestionsRequest, QueryRequest, ModuleModel, CourseModel, QuestionDataRequest
from Utils import *

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
    all_data = CONTENT_DOCUMENT._collection.get(
        include=["documents"]
    )  # Retrieves all documents
    return {"data": all_data["documents"]}


@app.get("/view-questions/")
def view_data():
    # Access the stored documents from the Chroma collection
    all_data = QUESTION_DOCUMENT._collection.get(
        include=["documents"]
    )  # Retrieves all documents
    return {"data": all_data["documents"]}


@app.post("/process-pdf/")
async def process_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    course_name: str = Form(...),
    course_id: int = Form(...),
    pdf_id: int = Form(...),
):
    # Read the file content
    file_content = await file.read()

    # Schedule the background task for processing the PDF
    background_tasks.add_task(
        actual_pdf_processing_function,
        file_content,
        course_name,
        file.filename,
        course_id,
        pdf_id,
    )

    return {"message": "Processing started, you'll be notified when it's done."}


@app.delete("/delete/{pdf_name}")
async def delete_pdf_images(pdf_name: str):
    images_dir = f"images/{pdf_name}"
    if os.path.exists(images_dir):
        try:
            shutil.rmtree(images_dir)
            return JSONResponse(
                content={"message": f"Images for {pdf_name} deleted successfully"}
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"An error occurred while deleting: {str(e)}"
            )
    else:
        raise HTTPException(status_code=404, detail=f"No images found for {pdf_name}")


@app.post("/create-questions/")
async def create_questions_endpoint(
    background_tasks: BackgroundTasks, data: list[CreateQuestionsRequest]
):
    # Add the createQuestions function to the background tasks
    background_tasks.add_task(send_questions_to_laravel, data)
    return {"message": "Question generation task started in the background."}


@app.post("/query/")
def query_hierarchical_index_endpoint(request: QueryRequest, level: str = None):
    try:
        results = queryHierarchicalIndex(request.query, level)
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def process_image_with_retries(
    image: UploadFile, retries: int = 3, delay: int = 2
):
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
            file_extractor = {
                ".jpg": PARSER,
                ".jpeg": PARSER,
                ".png": PARSER,
            }

            # Load the document using the SimpleDirectoryReader
            documents = SimpleDirectoryReader(
                input_files=[file_location], file_extractor=file_extractor
            ).load_data()

            # Extract the text content
            text_content = "\n".join([doc.text for doc in documents])

            logging.info(f"Text extracted: {text_content}")

            if text_content.strip():
                if os.path.exists(file_location):
                    os.remove(file_location)
                return {"text": text_content}
            else:
                logging.warning(
                    f"Attempt {attempt + 1}: No text extracted, retrying..."
                )
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

        # Process the modules using the utility function
        module_uids = await process_module_hierarchy(modules, CONTENT_DOCUMENT)

        # Notify Laravel
        await update_module_status(module_uids)

        return {"message": "Modules added successfully"}

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    

@app.post("/create_course/")
async def create_course(course: CourseModel):
    try:
        logging.info("Creating new course")
        # Query the collection to see if a course with the same course_id already exists
        existing_course = CONTENT_DOCUMENT._collection.get(
            where={"course_id": course.course_id}
        )

        # Check if the course already exists
        if existing_course["documents"]:
            raise HTTPException(
                status_code=400, detail="Course with this ID already exists."
            )

        # Create a Document object for the course
        course_document = Document(
            page_content=f"Course Title: {course.title}. Course Description: {course.description}",
            metadata={"course_id": course.course_id},
        )

        # Add the document to the vector store
        CONTENT_DOCUMENT.add_documents(
            documents=[course_document],  # Pass Document object
            ids=[str(course.course_id)],
        )
        logging.info("Course Created successfully")
        return {"message": "Course created successfully", "course_id": course.course_id}

    except Exception as e:
        logging.error(f"Error in creating course {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.delete("/delete_module/{module_uid}")
async def delete_module(module_uid: str):
    try:
        logging.info(f"Starting deletion process for Module ID: {module_uid}")

        # Fetch all related IDs for the module hierarchy
        hierarchy_ids = await fetch_module_hierarchy_ids(module_uid, CONTENT_DOCUMENT)
        logging.info(f"Ids: {hierarchy_ids}")
        # Delete the module hierarchy
        await delete_module_hierarchy(hierarchy_ids, CONTENT_DOCUMENT)

        return {
            "message": f"Module {module_uid} and all related content deleted successfully"
        }

    except Exception as e:
        logging.error(f"Error occurred while deleting Module ID {module_uid}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/fetch_module/{module_uid}")
async def fetch_module(module_uid: str):
    try:
        # Fetch module hierarchy
        return await fetch_module_hierarchy(module_uid, CONTENT_DOCUMENT)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/get_module_text/{module_uid}")
async def get_module_text(module_uid: str):
    try:
        # Generate module text
        module_text = await create_module_text(module_uid)
        return {"module_text": module_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    

@app.delete("/delete_course/{course_id}")
async def delete_course(course_id: int):  # Ensure course_id is an integer
    try:
        logging.info(f"Course received: {course_id}")
        # Retrieve all documents associated with the course_id
        course_documents = CONTENT_DOCUMENT._collection.get(
            where={"course_id": course_id}  # Query with integer course_id
        )
        # Check if any documents are found for the course_id
        document_ids = course_documents.get("ids", [])

        # If no documents are found, raise a 404 error
        if not document_ids:
            raise HTTPException(
                status_code=404,
                detail=f"No course or associated content found for Course ID {course_id}.",
            )

        # Delete all documents related to the course_id
        CONTENT_DOCUMENT._collection.delete(ids=document_ids)

        return {
            "message": f"Course {course_id} and all associated documents deleted successfully."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.delete("/delete_question/{question_uid}")
async def delete_question(question_uid: str):
    try:
        
        question = QUESTION_DOCUMENT._collection.get(
            where={"question_uid": question_uid}  # Query with integer course_id
        )
        
        if not question["ids"]:  # Check if any documents were found
            raise HTTPException(
                status_code=404,
                detail=f"Question with UID {question_uid} not found.",
            )

        QUESTION_DOCUMENT.delete(ids=question["ids"])
        return {"message": f"Question with UID {question_uid} deleted successfully."}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
@app.post('/bulk_delete_questions/')
async def bulk_delete_questions(question_uids: List[str]):
    try:
        # Retrieve all documents associated with the question_uids
        question_documents = QUESTION_DOCUMENT._collection.get(
            where={"question_uid": {"$in": question_uids}}  # Query with integer course_id
        )
        
        # Check if any documents are found for the question_uids
        document_ids = question_documents.get("ids", [])

        # If no documents are found, raise a 404 error
        if not document_ids:
            raise HTTPException(
                status_code=404,
                detail=f"No questions found for the provided UIDs.",
            )

        # Delete all documents related to the question_uids
        QUESTION_DOCUMENT._collection.delete(ids=document_ids)

        return {
            "message": f"Questions with UIDs {question_uids} deleted successfully."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


    
@app.put('/update_question/{question_uid}')
async def update_quesiton(question_uid:str, question:QuestionDataRequest):
    try:
        result = QUESTION_DOCUMENT._collection.get(where={"question_uid": question_uid})
        ids = result.get("ids", [])
        
        if not ids:
            raise HTTPException(
                status_code=404,
                detail=f"Question with UID {question_uid} not found."
            )

        # Update the document in ChromaDB
        QUESTION_DOCUMENT._collection.update(
            ids=ids,
            texts=[question.question],
            metadatas=[{
                "question_uid": question.question_uid,
                "course_id": question.course_id,
                "difficulty": question.difficulty_type,
                "type": question.question_type,
                "difficulty_value": question.difficulty_value,
                "discrimination_index": question.discrimination_index
            }]
        )
        return {"message": f"Question with UID {question_uid} updated successfully."}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
  

@app.post("/reset_collection/")
async def reset_collection():
    try:
        global CONTENT_DOCUMENT
        # Get all documents in the collection
        all_documents = CONTENT_DOCUMENT._collection.get()

        # Extract document IDs from the 'ids' key
        document_ids = all_documents.get("ids", [])

        # Delete all documents by their IDs
        if document_ids:
            CONTENT_DOCUMENT._collection.delete(ids=document_ids)

        return {"message": "All documents cleared from the collection successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/reset_question_collection/")
async def reset_collection():
    """
    Deletes all documents from the Chroma QUESTION_DOCUMENT collection.
    """
    try:
        global QUESTION_DOCUMENT
        
        all_documents = QUESTION_DOCUMENT._collection.get()

        # Extract document IDs from the 'ids' key
        document_ids = all_documents.get("ids", [])

        # Delete all documents by their IDs
        if document_ids:
            QUESTION_DOCUMENT.delete(ids=document_ids)
            message = f"All {len(document_ids)} documents cleared from the collection successfully."
        else:
            message = "The collection is already empty."

        return {"message": message}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
