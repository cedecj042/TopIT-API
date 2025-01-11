
from pydantic import BaseModel
from typing import List, Dict, Any,Optional
from transformers import DistilBertModel
import torch.nn as nn

class QuestionFormat(BaseModel):
    course_id: int
    course_title: str
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
    difficulty: Difficulty

class ContentModel(BaseModel):
    content_id: int
    type: str  # Enum could also be used here
    description: str
    order: int
    caption: Optional[str] = None
    
class SubsectionModel(BaseModel):
    subsection_id: int
    title: Optional[str] = None
    contents: List[ContentModel]  # List of content items

class SectionModel(BaseModel):
    section_id: int
    title: Optional[str] = None
    contents: List[ContentModel]
    subsections: List[SubsectionModel]

class LessonModel(BaseModel):
    lesson_id: int
    title: Optional[str] = None
    contents: List[ContentModel]
    sections: List[SectionModel]

class ModuleModel(BaseModel):
    module_id: int
    course_id: int  # Associate the module with a course
    title: Optional[str] = None
    contents: List[ContentModel]
    lessons: List[LessonModel]
    
class CourseModel(BaseModel):
    course_id: int
    title: str
    description: str
    
class QueryRequest(BaseModel):
    query: str

class CourseDataRequest(BaseModel):
    course_data: dict


class CombinedDistilBERT(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(CombinedDistilBERT, self).__init__()
        # Load DistilBERT model
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        
        # Feed-forward network for handcrafted features
        self.feature_layer = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Final classifier
        self.classifier = nn.Linear(768 + 32, num_classes)

    def forward(self, ids, mask, features):
        # Pass tokenized inputs through BERT
        bert_outputs = self.bert(input_ids=ids, attention_mask=mask)
        pooled_output = bert_outputs.last_hidden_state[:, 0, :]  # CLS token embedding
        
        # Pass handcrafted features through feed-forward layer
        feature_output = self.feature_layer(features)
        
        # Concatenate BERT and feature outputs
        combined_output = torch.cat((pooled_output, feature_output), dim=1)
        
        # Final classification
        logits = self.classifier(combined_output)
        return logits
