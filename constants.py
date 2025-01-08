MULTIPLE_CHOICE_SINGLE = """{
    "question": "What is the primary function of a compiler?",
    "questionType": "Multiple Choice - Single",
    "answer": "Compilation",
    "choices": ["Execution","Compilation","Interpretation","Debugging"]
},"""

MULTIPLE_CHOICE_MANY = """{
    "question": "Which of the following programming languages is known for its simplicity and ease of use?",
    "questionType": "Multiple Choice - Many",
    "answer": ["Python","Ruby"],
    "choices": ["Java","C++","Python","Ruby"]
},"""

IDENTIFICATION = """ {
    "question": "What is the output of the following Python code: `a = [1, 2, 3, 4]; print(a[::-1])`?",
    "questionType": "Identification",
    "answer": ["[4, 3, 2, 1]", "4,3,2,1"],
}"""

QUESTION_TYPES = [
    "Identification",
    "Multiple Choice - Single",
    "Multiple Choice - Many",
]

MULTIPLE_CHOICE_KEYS = {
    "question",
    "questionType",
    "answer",
    "choices",
}

IDENTIFICATION_KEYS = {
    "question",
    "questionType",
    "answer",
}
