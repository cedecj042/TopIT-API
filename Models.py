
from pydantic import BaseModel
from typing import List,Optional
from transformers import DistilBertModel
import torch.nn as nn
import torch
from langchain.embeddings.base import Embeddings

class QuestionFormat(BaseModel):
    course_id: int
    course_title: str
    question_job_id: int
    numOfVeryEasy: int
    numOfEasy: int
    numOfAverage: int
    numOfHard: int
    numOfVeryHard: int


class Difficulty(BaseModel):
    numOfVeryEasy: int
    numOfEasy: int
    numOfAverage: int
    numOfHard: int
    numOfVeryHard: int

# Define Questions model
class Question(BaseModel):
    type: str
    difficulty: Difficulty
    
# Define the main request model
class CreateQuestionsRequest(BaseModel):
    course_id: int
    course_title: str
    question_job_id: int
    difficulty: Difficulty

class ContentModel(BaseModel):
    content_id: int
    type: str  # Enum could also be used here
    description: str
    caption: Optional[str] = None
    
class SubsectionModel(BaseModel):
    subsection_id: int
    subsection_uid: str
    title: Optional[str] = None
    contents: List[ContentModel]  # List of content items

class SectionModel(BaseModel):
    section_id: int
    section_uid: str
    title: Optional[str] = None
    contents: List[ContentModel]
    subsections: List[SubsectionModel]

class LessonModel(BaseModel):
    lesson_id: int
    lesson_uid: str
    title: Optional[str] = None
    contents: List[ContentModel]
    sections: List[SectionModel]

class ModuleModel(BaseModel):
    module_id: int
    module_uid: str
    course_id: int  # Associate the module with a course
    title: Optional[str] = None
    contents: List[ContentModel]
    lessons: List[LessonModel]
    
class CourseModel(BaseModel):
    course_id: int
    title: str
    description: str
    

class CourseDataRequest(BaseModel):
    course_data: dict
      
class QuestionDataRequest(BaseModel):
    question: str
    question_id: int
    question_uid: str
    question_type: str
    difficulty_type: str
    difficulty_value: float
    discrimination_index: float
    answer: List[str]
    choices: Optional[List[str]] = None
    course_id:int
    test_type: str
    