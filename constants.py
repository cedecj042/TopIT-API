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

BLOOMS_MAPPING = {
    "Remember": {
        "difficulty": "Very Easy",
        "description": "Retrieving relevant knowledge from long-term memory.",
        "examples": [
            "Name the data type in Python used to store a collection of unique items.",
            "What is the default port for HTTP connections in web servers?",
            "Which of the following is an example of a relational database?",
            "What is the command to list files and directories in Linux?",
            "Which layer of the OSI model is responsible for routing data between networks?",
        ],
    },
    "Understand": {
        "difficulty": "Easy",
        "description": "Constructing meaning from oral, written, and graphic messages.",
        "examples": [
            "Explain the difference between client-side and server-side scripting.",
            "Which of the following best describes a foreign key in a relational database?",
            "What is the purpose of normalization in database design?",
            "Which statement about REST APIs is correct?",
            "In object-oriented programming, what do you understand by encapsulation?",
        ],
    },
    "Apply": {
        "difficulty": "Average",
        "description": "Applying knowledge to solve problems or execute tasks.",
        "examples": [
            "Write a SQL query to retrieve all employees earning more than $50,000 from a table named employees.",
            "Which of the following algorithms would you use to sort a large dataset efficiently?",
            "What is the output of the following Python code: a = [1, 2, 3, 4]; print(a[::-1])?",
            "Solve a budget overrun in party planning by showing how to reduce costs on decorations or catering.",
            "Calculate average waiting time and turnaround time using the Round Robin Algorithm with Quantum time = 2.",
        ],
    },
    "Analyze": {
        "difficulty": "Hard",
        "description": "Breaking down information into components and analyzing relationships.",
        "examples": [
            "Analyze the space complexity of def sum_list(lst): return sum(lst).",
            "Identify the time complexity of this function: def sum_array(arr): for i in range(len(arr)): for j in range(len(arr)): print(i, j).",
            "Troubleshoot a web application crash during peak hours and suggest the most likely root cause.",
            "Compare air cooling versus liquid cooling systems for electric vehicle battery packs.",
            "Analyze challenges in microservices architecture, such as inconsistent data synchronization.",
        ],
    },
    "Evaluate": {
        "difficulty": "Very Hard",
        "description": "Making judgments and recommendations based on criteria and standards.",
        "examples": [
            "Evaluate the pros and cons of using NoSQL databases over relational databases.",
            "Which factor is most important when choosing a cloud service provider?",
            "Recommend improvements for this SQL query: SELECT * FROM employees WHERE department = 'IT'.",
            "Suggest solutions for data synchronization issues in an analytics dashboard.",
            "Argue which programming paradigm is best for projects requiring high concurrency and why.",
        ],
    },
}

DIFFICULTY_TO_BLOOMS = {
    "Very Easy": "Remember",
    "Easy": "Understand",
    "Average": "Apply",
    "Hard": "Analyze",
    "Very Hard": "Evaluate",
}