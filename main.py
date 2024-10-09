from fastapi import FastAPI, File, UploadFile,Form, HTTPException, BackgroundTasks,Request, status, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

# for llamaparse
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

from typing import Dict, Any
from functions import *
from rag import *
import nest_asyncio
import time
import asyncio

nest_asyncio.apply()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specific origins like ["http://localhost"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    filename="app.log",  # Path to the log file
    level=logging.INFO,  # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S",  # Date format for log messages
)

# Load environment variables
load_dotenv()
API_KEY_LlamaParse = os.getenv("LLAMA_CLOUD_API_KEY")
# Set up the parser
parser = LlamaParse(
    api_key=API_KEY_LlamaParse,
    result_type="text"
)

class QueryRequest(BaseModel):
    query: str

class CourseDataRequest(BaseModel):
    course_data: dict
    
# Define the expected structure of the input JSON using Pydantic
# class CreateQuestionsRequest(BaseModel):
#     course_id: int
#     course_title: str
#     questionType: str
#     numOfVeryEasy: int
#     numOfEasy: int
#     numOfAverage: int
#     numOfHard: int
#     numOfVeryHard: int
    
class TableModel(BaseModel):
    table_id: int
    description: str
    caption: str

class FigureModel(BaseModel):
    figure_id: int
    description: str
    caption: str

class CodeModel(BaseModel):
    code_id: int
    description: str
    caption: str

class SubsectionModel(BaseModel):
    subsection_id: int
    title: str
    content: list[dict]
    tables: list[TableModel]
    figures: list[FigureModel]
    codes: list[CodeModel]

class SectionModel(BaseModel):
    section_id: int
    title: str
    content: list[dict]
    tables: list[TableModel]
    figures: list[FigureModel]
    codes: list[CodeModel]
    subsections: list[SubsectionModel]

class LessonModel(BaseModel):
    lesson_id: int
    title: str
    content: list[dict]
    sections: list[SectionModel]

class ModuleModel(BaseModel):
    module_id: int
    course_id: int  # Associate the module with a course
    title: str
    content: list[dict]
    lessons: list[LessonModel]
    
class CourseModel(BaseModel):
    course_id: int
    title: str
    description: str
    
@app.get("/view-data/")
def view_data():
    # Access the stored documents from the Chroma collection
    all_data = vector_store._collection.get(include=["documents"])  # Retrieves all documents
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
            file_extractor = {".jpg": parser, ".jpeg": parser, ".png": parser}  # Support more formats

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
        all_documents = []
        all_metadatas = []
        all_ids = []

        for module in modules:
            # Check if the course exists for each module
            course = vector_store._collection.get(
                where={"course_id": module.course_id}
            )
            if not course['documents']:
                raise HTTPException(status_code=404, detail=f"Course with ID {module.course_id} not found. Please create the course first.")

            # Process module content
            module_content = process_content(module.content)
            module_document = Document(
                page_content=f"Module: {module.title}. " + " ".join(module_content),
                metadata={"module_id": module.module_id, "course_id": module.course_id, "type": "Module"}
            )
            all_documents.append(module_document)
            all_ids.append(str(module.module_id))

            # Add lessons, sections, subsections, and associated content
            for lesson in module.lessons:
                lesson_content = process_content(lesson.content)
                lesson_document = Document(
                    page_content=f"Lesson: {lesson.title}. " + " ".join(lesson_content),
                    metadata={"module_id": module.module_id, "lesson_id": lesson.lesson_id, "course_id": module.course_id, "type": "Lesson"}
                )
                all_documents.append(lesson_document)
                all_ids.append(f"{module.module_id}_{lesson.lesson_id}")

                for section in lesson.sections:
                    section_content = process_content(section.content)
                    section_document = Document(
                        page_content=f"Section: {section.title}. " + " ".join(section_content),
                        metadata={"module_id": module.module_id, "lesson_id": lesson.lesson_id, "section_id": section.section_id, "course_id": module.course_id, "type": "Section"}
                    )
                    all_documents.append(section_document)
                    all_ids.append(f"{module.module_id}_{lesson.lesson_id}_{section.section_id}")

                    for subsection in section.subsections:
                        subsection_content = process_content(subsection.content)
                        subsection_document = Document(
                            page_content=f"Subsection: {subsection.title}. " + " ".join(subsection_content),
                            metadata={"module_id": module.module_id, "lesson_id": lesson.lesson_id, "section_id": section.section_id, "subsection_id": subsection.subsection_id, "course_id": module.course_id, "type": "Subsection"}
                        )
                        all_documents.append(subsection_document)
                        all_ids.append(f"{module.module_id}_{lesson.lesson_id}_{section.section_id}_{subsection.subsection_id}")

                        # Add tables, figures, codes as separate entries with metadata
                        for table in subsection.tables:
                            table_document = Document(
                                page_content=f"Table: {table.description}. Caption: {table.caption}",
                                metadata={"module_id": module.module_id, "lesson_id": lesson.lesson_id, "section_id": section.section_id, "subsection_id": subsection.subsection_id, "table_id": table.table_id, "course_id": module.course_id, "type": "Table"}
                            )
                            all_documents.append(table_document)
                            all_ids.append(f"{module.module_id}_{lesson.lesson_id}_{section.section_id}_{subsection.subsection_id}_table_{table.table_id}")

                        for figure in subsection.figures:
                            figure_document = Document(
                                page_content=f"Figure: {figure.description}. Caption: {figure.caption}",
                                metadata={"module_id": module.module_id, "lesson_id": lesson.lesson_id, "section_id": section.section_id, "subsection_id": subsection.subsection_id, "figure_id": figure.figure_id, "course_id": module.course_id, "type": "Figure"}
                            )
                            all_documents.append(figure_document)
                            all_ids.append(f"{module.module_id}_{lesson.lesson_id}_{section.section_id}_{subsection.subsection_id}_figure_{figure.figure_id}")

                        for code in subsection.codes:
                            code_document = Document(
                                page_content=f"Code: {code.description}. Caption: {code.caption}",
                                metadata={"module_id": module.module_id, "lesson_id": lesson.lesson_id, "section_id": section.section_id, "subsection_id": subsection.subsection_id, "code_id": code.code_id, "course_id": module.course_id, "type": "Code"}
                            )
                            all_documents.append(code_document)
                            all_ids.append(f"{module.module_id}_{lesson.lesson_id}_{section.section_id}_{subsection.subsection_id}_code_{code.code_id}")

        # Add all documents, ids, and metadata to the Chroma collection
        vector_store.add_documents(documents=all_documents, ids=all_ids)

        return {"message": "Modules added successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/create_course/")
async def create_course(course: CourseModel):
    try:
        # Query the collection to see if a course with the same course_id already exists
        existing_course = vector_store._collection.get(
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
        vector_store.add_documents(
            documents=[course_document],  # Pass Document object
            ids=[str(course.course_id)],
        )
        return {"message": "Course created successfully", "course_id": course.course_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.delete("/delete_course/{course_id}")
async def delete_course(course_id: int):  # Ensure course_id is an integer
    try:
        # Retrieve all documents associated with the course_id
        course_documents = vector_store._collection.get(
            where={"course_id": course_id}  # Query with integer course_id
        )
        # Check if any documents are found for the course_id
        document_ids = course_documents.get('ids', [])

        # If no documents are found, raise a 404 error
        if not document_ids:
            raise HTTPException(status_code=404, detail=f"No course or associated content found for Course ID {course_id}.")

        # Delete all documents related to the course_id
        vector_store._collection.delete(ids=document_ids)

        return {"message": f"Course {course_id} and all associated documents deleted successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/reset_collection/")
async def reset_collection():
    try:
        global vector_store
        # Get all documents in the collection
        all_documents = vector_store._collection.get()

        # Extract document IDs from the 'ids' key
        document_ids = all_documents.get('ids', [])

        # Delete all documents by their IDs
        if document_ids:
            vector_store._collection.delete(ids=document_ids)

        return {"message": "All documents cleared from the collection successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    

    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)