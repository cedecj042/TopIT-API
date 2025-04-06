MULTIPLE_CHOICE_SINGLE = """{
    "question": "In a software development lifecycle, during the compilation process, source code is transformed into a machine-readable format. What is the primary function of a compiler in this context?",
    "questionType": "Multiple Choice - Single",
    "answer": "Compilation",
    "choices": ["Execution", "Compilation", "Interpretation", "Debugging"]
},"""

MULTIPLE_CHOICE_MANY = """{
    "question": "A company is selecting programming languages for developing a new web application. Which of the following programming languages is known for its simplicity, versatility, and ease of use, making it ideal for rapid development?",
    "questionType": "Multiple Choice - Many",
    "answer": ["Python", "Ruby"],
    "choices": ["Java", "C++", "Python", "Ruby"]
},"""

IDENTIFICATION = """{
    "question": "Identify the primary characteristics of NoSQL databases.",
    "questionType": "Identification",
    "answer": ["Non-relational", "Distributed","Scalable"]
    "requires_all_answer": false
}"""

QUESTION_TYPES = [
    "Identification",
    "Multiple Choice - Single",
    "Multiple Choice - Many",
]

MULTIPLE_CHOICE_KEYS = {"question", "questionType", "answer", "choices"}

IDENTIFICATION_KEYS = {
    "question",
    "questionType",
    "requires_all_answer",
    "answer",
}

VALID_QUESTION_TYPES = {
    "Identification",
    "Multiple Choice - Many",
    "Multiple Choice - Single",
}

EXAMPLE = {
    """
    {
        "question": "",
        "questionType": "Multiple Choice - Single",
        "answer": "",
        "choices": [""]
    },
    {
        "question": "",
        "questionType": "Multiple Choice - Many",
        "answer": [""],
        "choices": [""]
    },
    {
        "question": "",
        "questionType": "Identification",
        "answer": [""]
        "requires_all_answer":false
    }
    """
}


DIFFICULTY_TEMPLATES = {
    "Very Easy": "Multiple Choice and Identification questions using verbs: state, list, recite, define, name, quote, recall, identify, label, recognize. Focus on basic comprehension or recall. Use one of the formats: fill in the blanks, interrogative pronouns, or complete the sentence.",
    "Easy": "Multiple Choice and Identification questions using verbs with suffixes (e.g., classified, compared, contrasted, etc.). Emphasize basic interpretative understanding. Use one of the formats: fill in the blanks, interrogative pronouns (avoid 'how'), or complete the sentence.",
    "Average": "Multiple Choice questions using verbs like calculate, predict, apply, solve, etc., with realistic scenarios or basic coding problems. Identification questions should also use verbs like calculated, predicted, applied, etc. Use one of the formats: fill in the blanks, interrogative pronouns (avoid 'how'), or complete the sentence.",
    "Hard": "Multiple Choice questions with strict word limits (max 45 words/3 sentences) using verbs such as distinguish, classify, break down, etc. Alternate with questions using verbs like compared, differentiated, etc. Scenarios should be situational, testing critical thinking or complex coding problems.",
    "Very Hard": "Multiple Choice questions using verbs like critique, decide, justify, etc., with sophisticated, realistic scenarios and advanced coding/analytical problems. Identification questions must incorporate evaluation verbs (e.g., appraised, assessed, compared, etc.) with clear measurement criteria.",
}

BLOOMS_MAPPING = {
    "Remember": {
        "difficulty": "Very Easy",
        "description": "Retrieving relevant knowledge from long-term memory.",
        "keywords": [
            "List",
            "Recall",
            "Identify",
            "Name",
        ],
        "instructions": """

        **General Instructions:**
        * Focus on direct, precise questions that require selecting the correct answer from a list, identifying a specific term, concept, or fact, or completing a sentence.
        * Avoid subjective or opinion-based questions. All questions must be factual and objective.
        * PROVIDE ALL POSSIBLE ANSWERS FOR EACH IDENTIFICATION QUESTION.
        * Generate identification questions that require a specific term as the answer. Use prompts like 
            - "What is the term for...", 
            - "What do you call...", 
            - "What is the name of..." 
            - "Enumerate one..."
            - "List one..."
            - "Identify one..."
            - "Name one..."
            - "Recall one..
        to ensure the question leads to an specific term.
        * If using enumerate, identify, list, name, recall. Indicate all answers.
        * Limit each question to 5-10 words.
        * The answer should not be found in the question.
        * DO NOT INCLUDE subjective or conceptual questions where:
            - The answer is paraphrasable or interpretative.
            - The question asks about a purpose, benefit, goal,principle etc...
            - The answer is more than **3 words** or not a term. 
        * If a question has **more than one valid phrasing of the correct answer** (e.g., "dynamic" and "dynamic routing"):
            - Include **all accepted variations** in the "answer" list.
            - Limit to **1–3 acceptable synonyms** that reflect real-world terminology, not paraphrases.
        """,
        "question_types": """
            **1. Identification Questions:**
            * **Keywords:** Use keywords such as "List," "Name," "Recall," "Identify,", "Enumerate","The term for".
            * **Answer Format:** The "answer" field should be an array containing all possible answers. Include both the full expanded answer and its short form of answer.
            * **Answer Length:** Each answer should be concise, limited to a maximum of 3 words.
            * **Required Keys:** "question", "questionType", "answer"
                        
            Keys:
            {"question","questionType","answer",}
            Example:
            {
                "question": "Identify the primary characteristics of NoSQL databases.",
                "questionType": "Identification",
                "answer": ["Non-relational", "Distributed","Scalable"]
            }
            
        """,
    },
    "Understand": {
        "difficulty": "Easy",
        "description": "Constructing meaning from oral, written, and graphic messages.",
        "keywords": [
            "Summarize",
            "Explain",
            "Describe",
            "Compare",
            "Classify",
            "Paraphrase",
            "Discuss",
            "Illustrate",
            "Interpret",
            "Clarify",
        ],
        "examples": [
            "A developer is comparing client-side and server-side technologies. What is the difference between client-side and server-side scripting.",
            "A database designer is defining table relationships. Which of the following best describes a foreign key in a relational database?",
            "A team is optimizing a database for scalability. What is the purpose of normalization in database design?",
            "A developer is working on a REST API for an application. Which statement about REST APIs is correct?",
            "A new programmer is learning object-oriented programming. In object-oriented programming, what do you understand by encapsulation?",
        ],
        "instructions": """
        * Easy (Understand) questions with exactly 2 sentences.
        * Questions should test the ability to explain, compare, or interpret information beyond simple recall.
        * Use practical scenarios to make questions more engaging and relevant.
        * Ensure distractors (incorrect choices) are plausible and align with the "Understand" cognitive level.
        * Keywords: "Summarize", "Explain","Describe","Classify", "Paraphrase", "Discuss", "Illustrate", "Interpret", "Clarify",
        * Ensure that the answer can be found in the choices.
        * **Distractor Quality (VERY IMPORTANT):**
            - Each incorrect choice (distractor) must be **clearly incorrect** and **not similar** in phrasing, logic, or context to the correct answer.
            - Do NOT include distractors that:
                * Use the same keywords or structure as the correct answer.
                * Sound "technically plausible" but are unrelated or vague.
                * Could be partially correct depending on interpretation.

            *Each distractor must be clearly eliminated by someone who understands the concept.*

            Good Distractors:
            {
                "question": "Which function allows data insertion into a SQL table?",
                "answer": "INSERT INTO",
                "choices": ["INSERT INTO", "SELECT FROM", "GROUP BY", "WHERE"]
            }

            Bad Distractors:
            {
                "question": "Which function allows data insertion into a SQL table?",
                "answer": "INSERT INTO",
                "choices": ["INSERT INTO", "INSERT ALL", "INSERT", "SELECT INTO"]
                # All choices seem close, but only one is actually valid — confusing
            }
        """,
        "question_types": """
            **1. Multiple Choice - Single Questions:**
            * Each question must have only one correct answer.
            * Frame questions to require evaluating relationships or categorizing elements within the provided scenario.
            * Generate multiple choice single questions that using prompts like 
                - "Explain why [concept/process] is important in [context].", 
                - "Describe what happens when [event/process] occurs.", 
                - "Clarify the purpose of [X] in [Y]." 
                - "Interpret the meaning of the output/result below."
                - "Identify the correct statement about [concept]."
                - "Select the best explanation for the following result."
                - "Determine which phase comes next in the [process/workflow].
                - "Illustrate how [concept] applies to [real-world scenario]."
            * **Required Keys:** "question", "questionType", "answer", "choices"
            * **QuestionType:** The questionType key must always be "Multiple Choice - Single"
            Example: 
            {"question": "Explain why version control is important in collaborative software development.",
            "questionType": "Multiple Choice - Single",
            "answer": "It helps manage code changes and prevents conflicts when multiple developers work together.",
            "choices": [
                "It helps manage code changes and prevents conflicts when multiple developers work together.",
                "It automates the testing process.",
                "It encrypts files for security.",
                "It converts code into machine language."
            ]}
            2. Multiple Choice - Many:
            * Each question must have at least two - three CORRECT ANSWERS.
            * Generate multiple choice single questions that using prompts like 
                - "Summarize the key principles of [framework/concept].", 
                - "Classify the following items as [Type A] or [Type B].", 
                - "Select the following that [keywords] the [concept]."
                - "Clarify the purpose of [X] in [Y]." 
                - "Paraphrase the following statement: "[X]" (choose all valid rephrasings)."
                - "Identify the correct statement about [concept]."
                - "Which of the following describe [feature/behavior]?"
                - "Which of the following illustrate [concept] in action?"
                - "Which of the following sentences summarize... [concept]"
                - "Which of the following statements best illustrates [concept]"
                - "Discuss the valid implications of [scenario]."
            * Required Keys: "question", "questionType", "answer", "choices"
            * QuestionType: The questionType key must always be "Multiple Choice - Many"
            Example:
            {
                "question": "A company is selecting programming languages for developing a new web application. Which of the following programming languages is known for its simplicity, versatility, and ease of use, making it ideal for rapid development?",
                "questionType": "Multiple Choice - Many",
                "answer": ["Python", "Ruby"],
                "choices": ["Java", "C++", "Python", "Ruby"]
            }
        """,
    },
    "Apply": {
        "difficulty": "Average",
        "description": "Applying knowledge to solve problems or execute tasks.",
        "keywords": [
            "Solve",
            "Use",
            "Implement",
            "Execute",
            "Demonstrate",
            "Perform",
            "Apply",
            "Calculate",
            "Operate",
            "Utilize",
        ],
        "examples": [
            "An HR department wants a list of employees with high salaries. Which SQL query should be used to retrieve all employees earning more than $50,000 from a table named employees?",
            "A software engineer is sorting a dataset for an analytics project. Which of the following algorithms would you use to sort a large dataset efficiently?",
            "A Python developer is reversing the contents of a list. What is the output of the following Python code: a = [1, 2, 3, 4]; print(a[::-1])?",
            "A party planner is facing a budget overrun and needs to reduce costs. The current budget breakdown shows 40'%' spent on decorations and 50'%' on catering. Which steps should the planner take to reduce expenses while maintaining the quality of the event?",
            "A computer science student is calculating Round Robin scheduling times. Calculate average waiting time and turnaround time using the Round Robin Algorithm with Quantum time = 2.",
        ],
        "question_types": """
            * Generate Identification questions adhering to the following guidelines:
                ** Accepted Formats Based on Course Type:**
                    - **FOR SOFTWARE DEVELOPMENT COURSE ONLY:**
                        - Predict the output of code written in C, Python, or pseudocode
                        - Complete missing parts of code snippets
                        - Identify algorithms from implementation
                        - Determine time complexity of code
                        - Debug or correct code errors
                        - Be specific about what format the output should take (e.g., "Enter the numeric value", "Enter the string that would be printed")
                    
                    - **FOR UNDERSTANDING AND USING DATA COURSE ONLY:**
                        - Complete SQL queries with the correct keywords or clauses
                        - Predict the output of SQL queries given sample data
                        - Identify the appropriate database operation for a scenario
                        - Select the correct table or column name for a specific operation
                        - Always provide context — such as sample table structure, current values, or a scenario
                        - Clearly indicate what part of the query is missing (e.g., "Fill in the correct [JOIN type]", "Complete the WHERE clause with the correct [condition]")
                    
                    - **FOR ALL OTHER COURSES:**
                        - DO NOT use identification questions
                        - Use Multiple Choice - Single questions instead
                        
                * **VERY IMPORTANT: Every Identification question must include detailed context in the question string itself.**
                    - If the question is based on **code, pseudocode, SQL query, SQL table**, include the actual code block using `\\`\\`python\\n#code\\n\\`\\`` format.
                        - Note: When embedding code in JSON strings, use double backslashes before backticks to ensure proper escaping.
                    - If the question is about **SQL**, always provide:
                        * Table name
                        * Relevant column names
                        * Query or partial query
                        * Sample data (if needed for clarity)
                    - If the question is about a **scenario**, describe the real-world use case before asking the question.

                    Good Example:
                    {
                        "question": "A developer wants to calculate the average salary from the 'employees' table using SQL.\\n\\n\\`\\`\\`sql\\nSELECT AVG(salary) FROM employees;\\n\\`\\`\\`\\nWhat will the query return?",
                        "questionType": "Identification",
                        "answer": ["Average salary value"]
                    }

                    Bad Example:
                    {
                        "question": "What is the average salary query?",
                        "questionType": "Identification",
                        "answer": ["AVG(salary)"]
                    }
                * **Response Format Guidance:**
                    - Include a hint about the expected format without revealing the answer: "Enter a single number", "Enter the function name", etc.
                    - For numerical answers, indicate if precision matters (e.g., "Round to 2 decimal places")
                    - For string answers, specify case sensitivity (e.g., "Case-sensitive, enter exactly as it would appear")
                    - DO NOT include or hint at the actual answer in the question or instructions
                    
                * **Example Phrasings:**
                    - "What would be the output? Enter the exact value that would be displayed."
                    - "Which SQL keyword should replace the [blank]? Enter just the keyword."
                    - "What is the time complexity of this algorithm? Use Big O notation (e.g., O(n))."
                    - "What single value would this function return? Enter only the number."
                * **Preserve Formatting for Code Snippets - CRITICAL FOR JSON PARSING:**
                    - Always use **double backslashes for special characters** inside the code block. Example: `\\n` for newlines and `\\t` for indentation.
                    - When embedding code in question strings, use four backslashes before backticks: `\\\\`\\\\`python\\n# code here\\n\\\\`\\\\``
                    - Example:
                    {
                        "question": "Predict the output of the following code:\\n\\n\\\\`\\\\`\\\\`python\\n\\\\def add(a, b):\\n\\\\\\\\treturn a + b\\n\\\\\\\\nprint(add(2, 3))\\n\\\\`\\\\`\\\\`",
                        "questionType": "Identification",
                        "answer": ["5"]
                    }
                    - **CRITICALLY IMPORTANT:** When using backticks inside JSON strings, use `\\\\`\\\\`` not ``` ``` to avoid breaking JSON format.
    
                - Do **not** generate questions with multiple correct answers. The output must be a single value (string or number) unless you're explicitly using a list in the output.
                -Example: 
                {
                    "question": "Given the following code that multiplies two numbers, what is the output?\n\n```python\ndef multiply(a, b):\n\treturn a * b\n\nprint(multiply(4, 5))\n```",
                    "questionType": "Identification",
                    "answer": ["20"]
                },
                    
                * **Answer Format:** Provide an array containing the exact output or solution.
                * **Answer Length:** Answers should be concise and directly address the question.
                * **Required Keys:** "question", "questionType", "answer"
                * **QuestionType:** The questionType key must always be "Identification"
                * If a question has **more than one valid phrasing of the correct answer** (e.g., "dynamic" and "dynamic routing"):
                    - Include **all accepted variations** in the "answer" list.
                    - Limit to **1–3 acceptable synonyms** that reflect real-world terminology, not paraphrases.
                Example:
                {
                    "question": "Calculate the output of the following Python code:\n```\nprint(2 + 3 * 4)\n```",
                    "questionType": "Identification",
                    "answer": ["14"]
                }
            Generate Multiple Choice - Single questions adhering to the following guidelines:
            * **Keywords:** "Solve", "Use", "Implement", "Execute", "Demonstrate", "Perform", "Apply", "Calculate", "Operate", "Utilize",
            * **Practical Scenarios:** Incorporate realistic, practical scenarios to enhance engagement and relevance.
            * **Single Correct Answer:** Each question must have only one definitively correct answer.
            * **Scenario-Based Evaluation:** Frame questions to require evaluating relationships, categorizing elements, or applying concepts within the provided scenario.
            * **Plausible Distractors:** Ensure that the incorrect answer choices (distractors) are plausible and logically related to the question, reflecting the appropriate cognitive level of analysis.
            * **Output Format:** Provide the question in JSON format with the following keys: "question", "questionType", "answer", and "choices". The "answer" key should contain the single correct answer and the "choices" key should contain an array of all possible answers including the correct one.
            * **QuestionType:** The questionType key must always be "Multiple Choice - Single"
            * **Clarity and Conciseness:** Questions should be clear, concise, and easy to understand.
            * **Avoid ambiguity:** Ensure that the correct answer is clearly the best answer, and that distractor answers are clearly wrong.
            * **Distractor Quality (VERY IMPORTANT):**
                - Each incorrect choice (distractor) must be **clearly incorrect** and **not similar** in phrasing, logic, or context to the correct answer.
                - Do NOT include distractors that:
                    * Use the same keywords or structure as the correct answer.
                    * Sound "technically plausible" but are unrelated or vague.
                    * Could be partially correct depending on interpretation.

                *Each distractor must be clearly eliminated by someone who understands the concept.*

                Good Distractors:
                {
                    "question": "Which function allows data insertion into a SQL table?",
                    "answer": "INSERT INTO",
                    "choices": ["INSERT INTO", "SELECT FROM", "GROUP BY", "WHERE"]
                }

                Bad Distractors:
                {
                    "question": "Which function allows data insertion into a SQL table?",
                    "answer": "INSERT INTO",
                    "choices": ["INSERT INTO", "INSERT ALL", "INSERT", "SELECT INTO"]
                    # All choices seem close, but only one is actually valid — confusing
                }
            Example:
            {
                "question": "Define the software development cycle.",
                "questionType": "Multiple Choice - Single",
                "answer": "A structured process for planning, creating, testing, and deploying software.",
                "choices": ["A structured process for planning, creating, testing, and deploying software.","The process of writing code in a programming language.","The act of only debugging and testing software.","The process of only designing the user interface."]
            }
        """,
    },
    "Analyze": {
        "difficulty": "Hard",
        "description": "Breaking down information into components and analyzing relationships.",
        "keywords": [
            "Differentiate",
            "Organize",
            "Compare",
            "Contrast",
            "Categorize",
            "Examine",
            "Distinguish",
            "Investigate",
            "Decompose",
        ],
        "examples": [
            "A programmer is evaluating the efficiency of a Python function. Analyze the space complexity of def sum_list(lst): return sum(lst).",
            "A software engineer is assessing the time complexity of a nested loop. Identify the time complexity of this function: def sum_array(arr): for i in range(len(arr)): for j in range(len(arr)): print(i, j).",
            "A web application crashes during peak usage. Troubleshoot a web application crash during peak hours and suggest the most likely root cause.",
            "An automotive engineer is evaluating cooling systems for EV batteries. Compare air cooling versus liquid cooling systems for electric vehicle battery packs.",
            "A software architect is facing data synchronization issues in a distributed system. Analyze challenges in microservices architecture, such as inconsistent data synchronization.",
        ],
        "instructions": """
        **Instructions:**
        * **Deep Analysis:** Questions must require students to perform deep analysis, break down information into its components, and identify underlying patterns or relationships.
        * **Complexities and Relationships:** Questions should challenge students to recognize complexities and understand how different elements interact.
        * **Problem Decomposition:** Students should be required to break down problems into smaller parts and examine each component thoroughly.
        * **Situational Questions:** Prioritize situational questions that test critical thinking and problem-solving abilities in real-world or complex scenarios.
        * **Coding/Computational Questions:** If applicable to the subject matter, include complex coding/programming or computational/analytical questions.
        * **Minimum 3 Sentences:** Ensure each question is a minimum of 3 sentences to provide sufficient context and complexity.
        * **Plausible Distractors:** Ensure that the incorrect answer choices (distractors) are plausible and logically related to the question, reflecting the appropriate cognitive level of analysis.
        * **Keywords:** "Differentiate", "Organize", "Compare", "Contrast", "Categorize", "Examine", "Distinguish", "Investigate", "Decompose",
        * Ensure that the answer can be found in the choices.
        * **Distractor Quality (VERY IMPORTANT):**
            - Each incorrect choice (distractor) must be **clearly incorrect** and **not similar** in phrasing, logic, or context to the correct answer.
            - Do NOT include distractors that:
                * Use the same keywords or structure as the correct answer.
                * Sound "technically plausible" but are unrelated or vague.
                * Could be partially correct depending on interpretation.

            *Each distractor must be clearly eliminated by someone who understands the concept.*

            Good Distractors:
            {
                "question": "Which function allows data insertion into a SQL table?",
                "answer": "INSERT INTO",
                "choices": ["INSERT INTO", "SELECT FROM", "GROUP BY", "WHERE"]
            }

            Bad Distractors:
            {
                "question": "Which function allows data insertion into a SQL table?",
                "answer": "INSERT INTO",
                "choices": ["INSERT INTO", "INSERT ALL", "INSERT", "SELECT INTO"]
                # All choices seem close, but only one is actually valid — confusing
            }
        """,
        "question_types": """
            **1. Multiple Choice - Single Questions:**
            * Each question must have only one correct answer.
            * Frame questions to require evaluating relationships or categorizing elements within the provided scenario.
            * **Required Keys:** "question", "questionType", "answer", "choices"
            * **QuestionType:** The questionType key must always be "Multiple Choice - Single"
            Example: 
            {
                "question": "In a software development lifecycle, during the compilation process, source code is transformed into a machine-readable format. What is the primary function of a compiler in this context?",
                "questionType": "Multiple Choice - Single",
                "answer": "Compilation",
                "choices": ["Execution", "Compilation", "Interpretation", "Debugging"]
            }
            2. Multiple Choice - Many:
            * Each question must have at least two correct answers.
            * Ensure that the question is designed to have multiple correct answers and not just one.
            * Required Keys: "question", "questionType", "answer", "choices"
            * QuestionType: The questionType key must always be "Multiple Choice - Many"
            Example:
            {
                "question": "A company is selecting programming languages for developing a new web application. Which of the following programming languages is known for its simplicity, versatility, and ease of use, making it ideal for rapid development?",
                "questionType": "Multiple Choice - Many",
                "answer": ["Python", "Ruby"],
                "choices": ["Java", "C++", "Python", "Ruby"]
            }
        """,
    },
    "Evaluate": {
        "difficulty": "Very Hard",
        "description": "Making judgments and recommendations based on criteria and standards.",
        "keywords": [
            "Judge",
            "Critique",
            "Recommend",
            "Justify",
            "Prioritize",
            "Argue",
            "Assess",
            "Defend",
            "Evaluate",
            "Validate",
        ],
        "examples": [
            "A company is deciding between NoSQL and relational databases for a new project. Evaluate the pros and cons of using NoSQL databases over relational databases.",
            "An IT manager is choosing a cloud service provider for enterprise solutions. Which factor is most important when choosing a cloud service provider?",
            "A database administrator wants to optimize query performance. Recommend improvements for this SQL query: SELECT * FROM employees WHERE department = 'IT'.",
            "A software engineer is fixing synchronization issues in an analytics dashboard. Suggest solutions for data synchronization issues in an analytics dashboard.",
            "A technical lead is evaluating programming paradigms for a high-concurrency project. Argue which programming paradigm is best for projects requiring high concurrency and why.",
        ],
        "instructions": """
        **Instructions:**
        * **Critical Thinking and Justification:** Questions must require students to engage in critical thinking, justify their choices, and make informed decisions based on provided criteria.
        * **Complex Scenarios:** Present complex, realistic scenarios that demand a thorough evaluation of options.
        * **Recommendation and Rationale:** Students should be required to recommend a specific solution or course of action and provide a clear rationale for their choice.
        * **Objective Evaluation Criteria:** Each scenario must include specific, measurable criteria, metrics, or standards to facilitate objective evaluation.
        * **Minimum 4 Sentences:** Ensure each question is a minimum of 4 sentences to provide sufficient context and complexity.
        * **Plausible Distractors:** Ensure that the incorrect answer choices (distractors) are plausible and logically related to the question, reflecting the appropriate cognitive level of analysis.
        * **Keywords:** "Judge", "Critique", "Recommend", "Justify", "Prioritize", "Argue", "Assess", "Defend", "Evaluate", "Validate",
        * Ensure that the answer can be found in the choices.
        * **Distractor Quality (VERY IMPORTANT):**
            - Each incorrect choice (distractor) must be **clearly incorrect** and **not similar** in phrasing, logic, or context to the correct answer.
            - Do NOT include distractors that:
                * Use the same keywords or structure as the correct answer.
                * Sound "technically plausible" but are unrelated or vague.
                * Could be partially correct depending on interpretation.

            *Each distractor must be clearly eliminated by someone who understands the concept.*

            Good Distractors:
            {
                "question": "Which function allows data insertion into a SQL table?",
                "answer": "INSERT INTO",
                "choices": ["INSERT INTO", "SELECT FROM", "GROUP BY", "WHERE"]
            }

            Bad Distractors:
            {
                "question": "Which function allows data insertion into a SQL table?",
                "answer": "INSERT INTO",
                "choices": ["INSERT INTO", "INSERT ALL", "INSERT", "SELECT INTO"]
                # All choices seem close, but only one is actually valid — confusing
            }
        """,
        "question_types": """
            **1. Multiple Choice - Single Questions:**
            * Each question must have only one correct answer.
            * Frame questions to require evaluating relationships or categorizing elements within the provided scenario.
            * **Required Keys:** "question", "questionType", "answer", "choices"
            * **QuestionType:** The questionType key must always be "Multiple Choice - Single"
            Example: 
            {
                "question": "In a software development lifecycle, during the compilation process, source code is transformed into a machine-readable format. What is the primary function of a compiler in this context?",
                "questionType": "Multiple Choice - Single",
                "answer": "Compilation",
                "choices": ["Execution", "Compilation", "Interpretation", "Debugging"]
            }
            2. Multiple Choice - Many:
            * Each question must have at least two correct answers.
            * Ensure that the question is designed to have multiple correct answers and not just one.
            * Required Keys: "question", "questionType", "answer", "choices"
            * QuestionType: The questionType key must always be "Multiple Choice - Many"
            Example:
            {
                "question": "A company is selecting programming languages for developing a new web application. Which of the following programming languages is known for its simplicity, versatility, and ease of use, making it ideal for rapid development?",
                "questionType": "Multiple Choice - Many",
                "answer": ["Python", "Ruby"],
                "choices": ["Java", "C++", "Python", "Ruby"]
            }
        """,
    },
}


DIFFICULTY_TO_BLOOMS = {
    "Very Easy": "Remember",
    "Easy": "Understand",
    "Average": "Apply",
    "Hard": "Analyze",
    "Very Hard": "Evaluate",
}


"""
Very Easy
Create Identification questions that test basic recall of factual information. 
Focus on direct, precise questions that require selecting the correct answer from a list or identifying a specific term, concept, or fact. 
Use one of the formats: fill in the blanks, interrogative pronouns, or complete the sentence. 
**List all possible answers in the given questions and add required_all_answer to be true if all answers are required.** 
Don't make it subjective and focus on factual and objective questions. Ensure to generate exactly 1 sentence with 5-10 words.

**2. Multiple Choice - Single**
* Each question must have only one correct answer.
* Frame questions to require evaluating relationships or categorizing elements within the provided scenario.
* Keywords: "Recognize", "Retrieve", "Memorize", "Outline", "State",
* **Required Keys:** "question", "questionType", "answer", "choices"
* **QuestionType:** The questionType key must always be "Multiple Choice - Single"
Example: 
{
    "question": "Define the software development cycle.",
    "questionType": "Multiple Choice - Single",
    "answer": A structured process for planning, creating, testing, and deploying software.",
    "choices": ["A structured process for planning, creating, testing, and deploying software.","The process of writing code in a programming language.","The act of only debugging and testing software.","The process of only designing the user interface."]
}

Easy

        Design only multiple-choice questions that require comprehension, interpretation, or explanation of concepts. 
        Questions should test the ability to explain, compare, or interpret information beyond simple recall. 
        
Hard        
   Develop only multiple-choice questions that require deep analysis, identifying complexities, and understanding underlying patterns or relationships. 
        Questions should challenge students to break down problems and examine their components. 
        Question should have a minimum of 3 sentences. 
        Provide **situational questions** that will test an individual's critical thinking and problem-solving abilities.
        Provide complex coding/programming questions if applicable in {module} and Computational/analytical questions.      
Very hard

        Create only multiple-choice questions that require critical thinking, justification, and making informed decisions. 
        Questions should present complex scenarios where students must evaluate options, recommend solutions, and provide rationale. 
        Question should have a minimum of 4 sentences.  All scenarios must include specific measurement criteria, metrics, or standards for objective evaluation.
"""
"""
- 1. **Code Output Prediction** (C or pseudocode)
                * Include a code snippet.
                * Ask for the **output** using prompts like:
                    - "What is the output?"
                    - "Predict the result..."
                * Output must be **one value** or **2–3 words max**.
            - 2. **SQL Query Completion (Fill-in-the-Blanks)**
                * Provide **context** based on the SQL table.:
                    - Table name and column(s)
                    - Sample data (when needed)
                    - Goal of the query
                * Blanks may involve: `*`, table name, column name, or clauses like `WHERE`, `ORDER BY`, `GROUP BY`, etc.
            - 3. **SQL Calculation Questions**
                * Include schema and sample table data.
                * Ask for a computed result using aggregate functions (`AVG()`, `SUM()`, etc.).
            - 4. **Algorithm Guessing**
                * Provide pseudocode or C-style logic.
                * Ask: "Which algorithm does this represent?"
            - 5. **Data Structure Selection**
                * Provide a real-world scenario.
                * Ask: "Which data structure is best suited for this case?"
            - 6. **Time Complexity Estimation**
                * Show a loop or recursive pattern (C/pseudocode).
                * Ask: "What is the time complexity in Big O notation?"
            - 7. **Algorithm Identification**
                * Provide logic or code (C/pseudocode).
                * Ask: "Which algorithm is implemented here?"
            - 8. **Code Correction**
                * Provide a snippet with a clear error.
                * Ask: "Which word should be replaced?"
                * Focus on **one specific token** (not a whole line).
            - 9. **Code Completion**
                * Provide a snippet with a missing word.
                * Ask: "What word completes this code?"
                * **Answer must be a single token**, e.g., `while`, `print`, `int`.
            - 10. **Code Debugging**
                * Show a broken snippet and a brief error context.
                * Ask: "What should be the correct output after fixing?"
"""

import json
IDENTIFICATION_SAMPLE_QA = json.dumps([
    {
        "question": "Identify an example of a non-functional requirement.",
        "answer": ["Performance", "Scalability"]
    },
    {
        "question": "What is a key aspect of architecture verification?",
        "answer": ["Approval process"]
    },
    {
        "question": "What is the main purpose of the context diagram?",
        "answer": ["System boundary"]
    },
    {
        "question": "What is the output of architecture design?",
        "answer": ["Detailed architecture"]
    },
    {
        "question": "Identify a method of expressing architecture design.",
        "answer": ["Context model", "Component diagram"]
    },
    {
        "question": "What is a key feature of the client-server model?",
        "answer": ["Service requests"]
    }
], indent=4)