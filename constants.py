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


QUESTION_RULES = {
    """
    Multiple Choice - Single:
    - Must have only 1 answer.
    Keys: 
    {MULTIPLE_CHOICE_KEYS}
    Example: 
    {MULTIPLE_CHOICE_SINGLE}
    Multiple Choice - Many:
    - Must have atleast 2 answers and dont put only 1 answer
    Keys: 
    {MULTIPLE_CHOICE_KEYS}
    Example:
    {MULTIPLE_CHOICE_MANY}
    Identification - Identification answers must have a maximum of 3 words.
                    Avoid questions that has subjective answers and only concrete and objective answers are needed.
                    Keep it concise and relevant.
                    Keys:
                    {IDENTIFICATION_KEYS}
                    Example:
                    {IDENTIFICATION}
    """
}


EXAMPLE = {
    """
    {
        "question": "What is the primary function of a compiler?",
        "questionType": "Multiple Choice - Single",
        "answer": "Compilation",
        "choices": ["Execution","Compilation","Interpretation","Debugging"]
    },
    {
        "question": "Which of the following programming languages is known for its simplicity and ease of use?",
        "questionType": "Multiple Choice - Many",
        "answer": ["Python","Ruby"],
        "choices": ["Java","C++","Python","Ruby"]
    },
    {
        "question": "What is the output of the following Python code: `a = [1, 2, 3, 4]; print(a[::-1])`?",
        "questionType": "Identification",
        "answer": ["[4, 3, 2, 1]", "4,3,2,1"],
    }
    """
    
}