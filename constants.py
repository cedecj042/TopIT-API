multiple_choice_easy = """
                   Easy (Understand) questions with exactly 1 sentence with 10 - 18 words.
                  - You must *strictly* use at least 1 of the following verbs within the question: Classify, Compare, Contrast, Demonstrate, Explain, Illustrate, Infer, Interpret, Outline, Relate, Rephrase, Show, Summarize, discuss
                  - Ensure the answer is in the choices
                  - Ensure the questions are gramatically correct.
                  - Ensure question is objective, not subjective.
                  - CRITICAL GUIDELINES FOR QUESTION CONSTRUCTION:
                      - The answer MUST NOT be directly stated in the question
                      - Ensure the correct answer is NOT obvious from the question's wording

                  Example: 
                  - Explain the difference between client-side and server-side scripting.
                  - Interpret the following code snippet: for (int i = 0; i < 10; i++) { System.out.println(i); }
                  - Illustrate the concept of inheritance in object-oriented programming.
                  - Discuss the importance of code reviews in the software development process.
                  """
                
multiple_choice_average = """
                {numberOfQuestions} Average (Apply) with exactly 30 words(maximum of 2 sentences).
                  - You must *strictly* use atleast 1 of the following verbs: implement, calculate, predict, apply, solve, use, demonstrate, model, perform, present.
                  - Provide **realistic and practical scenarios** related to {module}.
                  - ensure the answer is in the choices
                  - When creating questions, ensure to alternate between these 2 types:
                    1. **Situational questions** that will test an individual's critical thinking and problem-solving abilities. Start the questions with scenario(don't include the word 'scenario') followed with situational questions about the scenarios.
                    2. Coding/programming/computational questions if applicable in {module}.  Example: 
                            - Implement a function in Java to calculate the factorial of a number. Which of the following correctly implements this
                            - Demonstrate the use of a for loop in C++ to print numbers from 1 to 5. Which code snippet accomplishes this?
                            - Solve the problem of finding the largest number in an array of integers using JavaScript. Which function does this correctly?
                            - Use the concept of recursion to write a Python function that calculates the nth Fibonacci number. Which implementation is correct?
                   """

multiple_choice_hard = """
               {numberOfQuestions} Hard (Analyze) with exactly 45 words(3 sentences).
                    - You must *strictly* use atleast 1 of the following verbs: distinguish, classify, break down, categorize, analyze, diagram, illustrate, criticize, simplify, associate.
                    - When creating questions, ensure to alternate between these 2 types:
                        1. Provide **situational questions** that will test an individual's critical thinking and problem-solving abilities. Sentence 1: Scenario (don't include the word 'scenario'), Sentence 2: Continuation of scenario Sentence 3: Objective Question about the scenario. 
                        2. Provide complex coding/programming questions if applicable in {module} and Computational questions. Example:
                                    - Distinguish the following Java methods based on their access modifiers.
                                    - Break down the components of a SQL SELECT statement. Which component is optional?
                              """


multiple_choice_very_hard = """
                {numberOfQuestions} Very Hard (Evaluate) with exactly 3 sentences(maximum of 60 words).
                    - You must *strictly* use atleast 1 of the following verbs:  critique, decide, justify, argue, choose, relate, determine, defend, judge, grade, compare, support, convince, select, evaluate.
                    - When creating questions, ensure to alternate between these 2 types:
                      1. Provide **situational questions** that will test an individual's critical thinking and problem-solving abilities. Sentence 1: Scenario (don't include the word 'scenario'), Sentence 2: Continuation of scenario Sentence 3: Objective Question about the scenario. 
                      2. Provide sophisticated coding/programming questions if applicable in {module}. Example:
                        - Critique the following code snippet for potential issues:
                          def divide(a, b):
                              return a / b
                        - Judge the efficiency of the following code snippet for finding the maximum value in a list. Which statement is true?
                          def find_max(numbers):
                              max_value = numbers[0]
                              for num in numbers:
                                  if num > max_value:
                                      max_value = num
                              return max_value
                    """
               
identification_very_easy = """
                Very Easy (Remember) questions with exactly 1 sentence with 13 words.
                  - You must *strictly* use at least 1 of the following verbs: define, name, quote, recall, identify, label, recognize. 
                  - Use 'list' or state' before the word 'one' only. For Example: 'list one' and 'state one'. 
                  - Simple, clear, concise, and focused on basic Comprehension or recall.
                  - Avoid unnecessary complexity or ambiguous phrasing.
                  - Generate objective identification questions only. Don't use verbs that requires subjective answer.
                  """

multiple_choice_single = """{
    "question": "{input_the_question_in_here}",
    "answer": "{answer_in_here}",
    "choices": ["{choice_in_here}","{choice_in_here}","{choice_in_here}","{choice_in_here}"],
},"""

multiple_choice_many = """{
    "question": "{input_the_question_in_here}",
    "answer": ["{answer_in_here}","{answer_in_here}"],
    "choices": ["{choice_in_here}","{choice_in_here}","{choice_in_here}","{choice_in_here}"],
},"""

identification = """ {
    "question": "{input_the_question_in_here}",
    "answer": "{answer_in_here}",
}"""

