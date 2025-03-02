from fastapi import HTTPException
from Setup import logging
from typing import List
from Setup import LARAVEL_IP, LARAVEL_PORT, UPDATE_MODULE_STATUS_ROUTE
from Pdf_processing import process_content
from langchain.schema import Document
import requests
from Setup import CONTENT_DOCUMENT

async def update_module_status(module_uids: List[str]):
    """
    Notify Laravel that modules have been vectorized.
    """

    laravel_endpoint = (
        f"http://{LARAVEL_IP}:{LARAVEL_PORT}/{UPDATE_MODULE_STATUS_ROUTE}"
    )
    try:
        # Notify Laravel about the completed vectorization
        response = requests.post(
            laravel_endpoint,
            json={"module_uids": module_uids},
        )
        response.raise_for_status()
        logging.info(f"Laravel notified successfully for module IDs: {module_uids}")
        return {"message": "Laravel notified successfully"}
    
    except Exception as e:
        logging.error(f"Failed to notify Laravel: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to notify Laravel: {str(e)}"
        )


async def fetch_module_hierarchy_ids(module_uid: str, content_document):
    """
    Fetch all IDs for lessons, sections, and subsections related to the given module.

    Args:
        module_id (int): The module ID to fetch.
        content_document: The content document collection.

    Returns:
        dict: A dictionary containing IDs for the module, lessons, sections, and subsections.
    """
    try:
        # Fetch the module document by its ID
        module_response = content_document._collection.get(
            ids=[module_uid], include=["documents", "metadatas"]
        )
        if not module_response.get("documents"):
            raise HTTPException(status_code=404, detail=f"Module {module_uid} not found")

        # logging.info(f"Module document fetched: {module_response}")

        # Fetch lesson IDs associated with the module using 'ids'
        lesson_response = content_document._collection.get(
            where={"$and": [{"module_uid": module_uid}, {"type": "Lesson"}]},
            include=["documents","metadatas"],
        )
        # logging.info(f"Lesson response: {lesson_response}")

        # Extract lesson IDs directly from 'ids'
        lesson_uids = lesson_response.get("ids", [])
        # logging.info(f"Lesson IDs: {lesson_uids}")

        section_uids = []
        subsection_uids = []

        # Fetch sections and subsections for each lesson
        for lesson_uid in lesson_uids:
            # logging.info(f"Processing Lesson ID: {lesson_uid}")

            # Fetch sections for this lesson
            section_response = content_document._collection.get(
                where={"$and": [{"lesson_uid": lesson_uid}, {"type": "Section"}]},
                include=["documents","metadatas"],
            )
            section_uids.extend(section_response.get("ids", []))

            # Fetch subsections for each section
            for section_uid in section_response.get("ids", []):
                subsection_response = content_document._collection.get(
                    where={"$and": [{"section_uid": section_uid}, {"type": "Subsection"}]},
                    include=["documents","metadatas"],
                )
                subsection_uids.extend(subsection_response.get("ids", []))
                

        return {
            "module_uid": module_uid,
            "lesson_uids": lesson_uids,
            "section_uids": section_uids,
            "subsection_uids": subsection_uids,
        }

    except Exception as e:
        raise Exception(f"Error fetching module hierarchy IDs: {str(e)}")


async def delete_module_hierarchy(hierarchy_ids: dict, content_document):
    """
    Delete a module and all its related lessons, sections, and subsections.

    Args:
        hierarchy_ids (dict): A dictionary containing IDs for the module, lessons, sections, and subsections.
        content_document: The content document collection.
    """
    try:
        # Delete all subsections
        subsection_uids = hierarchy_ids.get("subsection_uids", [])
        if subsection_uids:
            content_document._collection.delete(ids=subsection_uids)
            logging.info(f"Deleted {len(subsection_uids)} subsections")
        logging.info(f"subsectionuids: {subsection_uids}")
        # Delete all sections
        section_uids = hierarchy_ids.get("section_uids", [])
        if section_uids:
            content_document._collection.delete(ids=section_uids)
            logging.info(f"Deleted {len(section_uids)} sections")
        logging.info(f"sectionuids: {section_uids}")
        # Delete all lessonsd
        lesson_uids = hierarchy_ids.get("lesson_uids", [])
        if lesson_uids:
            content_document._collection.delete(ids=lesson_uids)
            logging.info(f"Deleted {len(lesson_uids)} lessons")
        logging.info(f"lessonuids: {lesson_uids}")
        # Delete the module itself
        module_uid = hierarchy_ids.get("module_uid")
        if module_uid:
            content_document._collection.delete(ids=[module_uid])
            logging.info(f"Deleted module with ID {module_uid}")
        logging.info(f"module_uids: {module_uid}")

    except Exception as e:
        raise Exception(f"Error deleting module hierarchy: {str(e)}")
    

async def fetch_module_hierarchy(module_uid: str, content_document):
    """
    Fetch the module hierarchy (module, lessons, sections, subsections) based on the module_id.

    Args:
        module_id (int): The module ID to fetch.
        content_document: The content document collection.

    Returns:
        dict: The module hierarchy with lessons, sections, and subsections.
    """
    try:
        # Step 1: Fetch the module by its ID
        module_response = content_document._collection.get(
            ids=[module_uid], include=["documents"]
        )
        if not module_response["documents"]:
            raise HTTPException(status_code=404, detail=f"Module {module_uid} not found")

        # Extract the module document
        module_document = module_response["documents"][0]

        # Initialize module hierarchy
        module_hierarchy = {"module": module_document, "lessons": []}
        # logging.info(f"module document {module_document}")
        
        lesson_response = content_document._collection.get(
            where={"$and": [{"module_uid": module_uid}, {"type": "Lesson"}]},
            include=["documents", "metadatas"],
        )
        lesson_documents = lesson_response.get("documents", [])
        lesson_metadatas = lesson_response.get("metadatas", [])
        
        # logging.info(lesson_metadatas)

        # Process lessons
        for i, lesson_document in enumerate(lesson_documents):
            # Extract lesson ID from metadata
            lesson_uid = lesson_metadatas[i].get("lesson_uid")
            lesson_content = lesson_document

            # Initialize lesson entry
            lesson_entry = {
                "lesson_uid": lesson_uid,
                "content": lesson_content,
                "sections": [],
            }

            # Step 3: Fetch all sections related to this lesson
            section_response = content_document._collection.get(
                where={"$and": [{"lesson_uid": lesson_uid}, {"type": "Section"}]},
                include=["documents", "metadatas"],
            )
            section_documents = section_response.get("documents", [])
            section_metadatas = section_response.get("metadatas", [])
            # logging.info(f"Section metadatas: {section_metadatas}")

            # Process sections
            for j, section_document in enumerate(section_documents):
                # Extract section ID from metadata
                section_uid = section_metadatas[j].get("section_uid")
                section_content = section_document

                # Initialize section entry
                section_entry = {
                    "section_uid": section_uid,
                    "content": section_content,
                    "subsections": [],
                }

                # Step 4: Fetch all subsections related to this section
                subsection_response = content_document._collection.get(
                    where={
                        "$and": [{"section_uid": section_uid}, {"type": "Subsection"}]
                    },
                    include=["documents", "metadatas"],
                )
                subsection_documents = subsection_response.get("documents", [])
                subsection_metadatas = subsection_response.get("metadatas", [])

                # Process subsections
                for k, subsection_document in enumerate(subsection_documents):
                    subsection_id = subsection_metadatas[k].get("subsection_uid")
                    subsection_content = subsection_document

                    # Append subsection to section entry
                    section_entry["subsections"].append(
                        {"subsection_uid": subsection_id, "content": subsection_content}
                    )

                # Append section to lesson entry
                lesson_entry["sections"].append(section_entry)

            # Append lesson to module hierarchy
            module_hierarchy["lessons"].append(lesson_entry)

        return module_hierarchy

    except Exception as e:
        raise Exception(f"An error occurred: {str(e)}")


async def create_module_text(module_uid: int, fetch_module_func):
    """
    Generate text representation of a module and its hierarchy.

    Args:
        module_id (int): The module ID to fetch.
        fetch_module_func (function): A function to fetch the module hierarchy.

    Returns:
        str: The text representation of the module.
    """
    try:
        # Step 1: Fetch the module hierarchy
        module_hierarchy = await fetch_module_func(module_uid)

        # Initialize the text output
        text_output = f"Module: {module_hierarchy['module']}\n"

        # Step 2: Append lessons, sections, and subsections
        for lesson in module_hierarchy["lessons"]:
            text_output += f"  Lesson: {lesson['content']}\n"
            for section in lesson["sections"]:
                text_output += f"    Section: {section['content']}\n"
                for subsection in section["subsections"]:
                    text_output += f"      Subsection: {subsection['content']}\n"

        return text_output

    except Exception as e:
        raise Exception(f"An error occurred: {str(e)}")


async def process_module_hierarchy(modules: List, content_document):
    """
    Process and add modules, lessons, sections, and subsections to the Chroma collection.

    Args:
        modules (List[ModuleModel]): List of modules to process.
        content_document: The Chroma content document collection.

    Returns:
        List[int]: List of module IDs that were processed.
    """
    try:
        all_documents = []
        all_ids = []

        for module in modules:
            logging.info(f"Processing module with ID: {module.module_uid}")

            # Check if the course exists
            course = content_document._collection.get(
                ids=[str(module.course_id)]  # Use course_id directly
            )
            if not course["documents"]:
                raise HTTPException(
                    status_code=404,
                    detail=f"Course with ID {module.course_id} not found. Please create the course first.",
                )

            # Process module content
            module_content = process_content(module.contents)
            module_document = Document(
                page_content=f"Module: {module.title}. " + " ".join(module_content),
                metadata={
                    "module_id": module.module_id,
                    "module_uid": module.module_uid,
                    "course_id": module.course_id,
                    "type": "Module",
                },
            )
            all_documents.append(module_document)
            all_ids.append(str(module.module_uid))

            # Process lessons, sections, and subsections
            for lesson in module.lessons:
                lesson_content = process_content(lesson.contents)
                lesson_document = Document(
                    page_content=f"Lesson: {lesson.title}. " + " ".join(lesson_content),
                    metadata={
                        "module_uid": module.module_uid,
                        "lesson_uid": lesson.lesson_uid,
                        "lesson_id": lesson.lesson_id,
                        "course_id": module.course_id,
                        "type": "Lesson",
                    },
                )
                all_documents.append(lesson_document)
                all_ids.append(str(lesson.lesson_uid))

                for section in lesson.sections:
                    section_content = process_content(section.contents)
                    section_document = Document(
                        page_content=f"Section: {section.title}. "
                        + " ".join(section_content),
                        metadata={
                            "module_uid": module.module_uid,
                            "lesson_uid": lesson.lesson_uid,
                            "section_id": section.section_id,
                            "section_uid": section.section_uid,
                            "course_id": module.course_id,
                            "type": "Section",
                        },
                    )
                    all_documents.append(section_document)
                    all_ids.append(str(section.section_uid))

                    for subsection in section.subsections:
                        subsection_content = process_content(subsection.contents)
                        subsection_document = Document(
                            page_content=f"Subsection: {subsection.title}. "
                            + " ".join(subsection_content),
                            metadata={
                                "module_uid": module.module_uid,
                                "lesson_uid": lesson.lesson_uid,
                                "section_uid": section.section_uid,
                                "subsection_id": subsection.subsection_id,
                                "subsection_uid": subsection.subsection_uid,
                                "course_id": module.course_id,
                                "type": "Subsection",
                            },
                        )
                        all_documents.append(subsection_document)
                        all_ids.append(str(subsection.subsection_uid))

        # Add all documents and IDs to the Chroma collection
        content_document.add_documents(documents=all_documents, ids=all_ids)
        logging.info("All documents successfully added to Chroma collection.")
        # logging.info(f"Processed the modules {all_ids}")

        # Return the module IDs
        return [module.module_uid for module in modules]

    except Exception as e:
        raise Exception(f"An error occurred while processing modules: {str(e)}")
    
async def create_module_text(module_uid: str):
    try:
        # Step 1: Fetch the module hierarchy
        module_hierarchy = await fetch_module_hierarchy(module_uid,CONTENT_DOCUMENT)

        # Initialize the text output
        text_output = f"Module: {module_hierarchy['module']}\n"

        # Step 2: Append lessons, sections, and subsections
        for lesson in module_hierarchy["lessons"]:
            text_output += f"  Lesson: {lesson['content']}\n"
            for section in lesson["sections"]:
                text_output += f"    Section: {section['content']}\n"
                for subsection in section["subsections"]:
                    text_output += f"      Subsection: {subsection['content']}\n"

        # Return the text output
        return {"module_text": text_output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")