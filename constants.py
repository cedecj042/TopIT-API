#constants

multiple_choice_very_easy = """
                  {numberOfQuestions} Very Easy (Remember) questions
                  - Ensure to generate exactly 1 sentence with 5-10 words.
                  - You must *strictly* use at least 1 of the following verbs: state, list, recite, define, name, quote, recall, identify, label, recognize.
                  - Simple, clear, concise, and focused on basic Comprehension or recall.
                  - Avoid unnecessary complexity or ambiguous phrasing.
                  - Ensure the questions are gramatically correct.
                """


multiple_choice_easy = """{numberOfQuestions} Easy (Understand) questions with exactly 2 sentences with 18 words.
                  - **Strictly** Use the following verbs within the question with their suffixes. Example use case of the verbs: "when classifying...", "In comparing...", "In contrasting...", "Client explained...", "",
                  - Refer to these verbs with their suffix:
                    classifying, classified, comparing, compared, contrasting, contrasted, demonstrating, demonstrated,
                    explaining,explained, illustrating,illustrated, inferring,inferred, interpreting, interpreted, outlining,outlined,
                    relating,related, rephrasing,rephrased, showing,shown, summarizing,summarized, translating,translated
                    discussing,discussed, describing,describedl, using,used, writing,write, giving,gave, defining,defined, identify, determinining,determined

                  - ensure to use the verbs appropriately for each question. It should be grammatically correct.
                  - Emphasize basic interpretative understanding
                  - Avoid unnecessary complexity or ambiguous phrasing.
                  - ensure the difficulty level is 'Easy'
                  - **Strictly** ensure to divide all questions into a form of: ** Fill in the blanks**(Don't include "fill in the blanks" in the question sentence), **Questions that uses interrogative pronouns(Don't use 'how')**, ** Complete the sentence questions(Don't include "complete the sentence" in the question sentence)** where the answer logically completes the idea."
                  - CRITICAL GUIDELINES FOR QUESTION CONSTRUCTION:
                       - The answer MUST NOT be directly stated within the question
                       - Create questions that require careful reading and interpretation
                       - Ensure the correct answer is NOT obvious from the question's wording
                       - Construct choices that require thoughtful analysis
                 """
multiple_choice_average = """
                {numberOfQuestions} Average (Apply) with exactly 30 words(maximum of 2 sentences).
                  - You must *strictly* use atleast 1 of the following verbs: calculate, predict, apply, solve, use, demonstrate, model, perform, present.
                  - Provide **realistic and practical scenarios** related to {module}.
                  - ensure the answer is in the choices
                  - When creating questions, ensure to alternate between these 2 types:
                    1. **Situational questions** that will test an individual's critical thinking and problem-solving abilities. Start the questions with scenario(don't include the word 'scenario') followed with situational questions about the scenarios.
                    2. Basic coding/programming/computational questions if applicable in {module}.
                   """
multiple_choice_hard = """
                {numberOfQuestions} Hard (Analyze) with exactly 45 words(maximum of 3 sentences).
                    - You must *strictly* use atleast 1 of the following verbs: distinguish, classify, break down, categorize, analyze, diagram, illustrate, criticize, simplify, associate.
                    - Provide **realistic and practical scenarios** related to {module}.
                    - Provide **situational questions** that will test an individual's critical thinking and problem-solving abilities. Start the questions with a scenario (don't include the word 'scenario') followed with situational questions about the scenarios.
                    - Provide complex coding/programming questions if applicable in {module} and Computational/analytical questions. """


multiple_choice_very_hard = """
                {numberOfQuestions} Very Hard (Evaluate) with exactly 3 sentences(maximum of 60 words).
                    - You must *strictly* use atleast 1 of the following verbs:  critique, decide, justify, argue, choose, relate, determine, defend, judge, grade, compare, support, convince, select, evaluate.
                    - Provide **realistic and practical scenarios** related to {module}.
                    - Provide **situational questions** that will test an individual's critical thinking and problem-solving abilities. Start the questions with scenario(don't include the word 'scenario') followed with situational questions about the scenarios.
                    - Provide sophisticated coding/programming questions if applicable in {module} and Computational/analytical questions.
                    - ensure the difficulty level is 'very hard' """
identification_very_easy = """
                {numberOfQuestions} Very Easy (Remember) questions with exactly 1 sentence with 13 words.
                  - You must *strictly* use at least 1 of the following verbs: state, list, recite, outline, define, name, quote, recall, identify, label, recognize.
                  - Simple, clear, concise, and focused on basic Comprehension or recall.
                  - Avoid unnecessary complexity or ambiguous phrasing.
                  - Generate objective identification questions only. Don't use verbs that requires subjective answer.
                  - **Strictly** ensure to divide all questions into a form of: ** Fill in the blanks**(Don't include "fill in the blanks" in the question sentence), **Questions that uses interrogative pronouns**, ** Complete the sentence questions(Don't include "complete the sentence" in the question sentence)** where the answer logically completes the idea."
                """

identification_easy = """
                {numberOfQuestions} Easy (Understand) questions with exactly 2 sentences with 18 words.
                  - **Strictly** Use the following verbs within the question with their suffixes to ensure questions being generated are identification questions with 1 -3 words as the answer. Example use case of the verbs: "when classifying...", "In comparing...", "In contrasting...", "Client explained...", "",
                  - Refer to these verbs with their suffix:
                    classifying, classified, comparing, compared, contrasting, contrasted, demonstrating, demonstrated,
                    explaining,explained, illustrating,illustrated, inferring,inferred, interpreting, interpreted, outlining,outlined,
                    relating,related, rephrasing,rephrased, showing,shown, summarizing,summarized, translating,translated
                    discussing,discussed, describing,describedl, using,used, writing,write, giving,gave, defining,defined, identify, determinining,determined

                  - ensure to use the verbs appropriately for each question. It should be grammatically correct.
                  - Emphasize basic interpretative understanding
                  - Avoid unnecessary complexity or ambiguous phrasing.
                  - ensure the difficulty level is 'Easy'
                  - **Strictly** ensure to divide all questions into a form of: ** Fill in the blanks**(Don't include "fill in the blanks" in the question sentence), **Questions that uses interrogative pronouns(Don't use 'how')**, ** Complete the sentence questions(Don't include "complete the sentence" in the question sentence)** where the answer logically completes the idea."
                  - CRITICAL GUIDELINES FOR QUESTION CONSTRUCTION:
                       - The answer MUST NOT be directly stated within the question
                       - Create questions that require careful reading and interpretation
                       - Ensure the correct answer is NOT obvious from the question's wording
                       - Construct choices that require thoughtful analysis
                 """

identification_average = """
                {numberOfQuestions} Average (Apply)
                  - Total words should be 20-30 words
                  - You must *strictly* use atleast 1 of the following verbs (with suffix): calculating, calculated, predicting,predicted, applying, applied, solving,solved, using, used, demonstrating, demonstrated, modeling, modeled, performing, performed, presenting, presented.
                  - Ensure to use the verbs appropriately to the questions. It must be gramatically correct.
                  - Provide **situational questions** that will test an individual's critical thinking and problem-solving abilities. Sentence 1: Start the questions with scenario(don't include the word 'scenario') Sentence 2: followed with situational questions about the scenarios.
                  - Provide basic coding/programming questions if applicable in {module}.

                  - **Strictly** ensure to divide all questions into a form of: ** Fill in the blanks**(Don't include "fill in the blanks" in the question sentence), **Questions that uses interrogative pronouns(Don't use 'how')**, ** Complete the sentence questions(Don't include "complete the sentence" in the question sentence)** where the answer logically completes the idea."
                   """
identification_hard = """
                {numberOfQuestions} Hard (Analyze)
                     - total words should be 35 - 45 words.
                     - **Strictly** Use the following verbs within the question with their suffixes to ensure questions being generated are identification questions with 1 -3 words as the answer. Example use case of the verbs: "when distinguishing...", "When examining...", "Client emphasized...", "Client outlined...", "",
                     - Refer to these verbs with their suffix:
                      comparing, compared, distinguishing, distinguished, break down, categorizing, categorized, analyzing, analyzed, diagram, illustrating, illustrated, criticizing, criticized,
                      discussing, discussed, differentiating, differentiate, developing, developed, deriving, derived, outlining, outlined, determining, determined, examining, examined, investigation, investigate
                    - Ensure to use the verbs appropriately to the questions. It must be gramatically correct.
                    - Provide **situational questions** that will test an individual's critical thinking and problem-solving abilities. Sentence 1: Start the questions with scenario(don't include the word 'scenario') Sentence 2: Continuation of the scenario Sentence 3: followed with situational questions about the scenarios. It can be in a form of ** Fill in the blanks**(Don't include "fill in the blanks" in the question sentence), **Questions that uses interrogative pronouns(Don't use 'how')**, ** Complete the sentence questions(Don't include "complete the sentence" in the question sentence)** where the answer logically completes the idea."
                    - Provide complex coding/programming questions if applicable in {module} and Computational/analytical questions.

                    """

identification_very_hard = """
                {numberOfQuestions} Very Hard (Evaluate)
                    - Questions must have exactly 3 sentences, 16-20 words each.
                    - Each sentence must incorporate evaluation verbs with appropriate suffixes:
                    (e.g., appraising/appraised, arguing/argued, assessing/assessed, comparing/compared, concluding/concluded, considering/considered,
                    contrasting/contrasted, convincing/convinced, criticizing/criticized, critiquing/critiqued, deciding/decided, determining/determined,
                    discriminating/discriminated, evaluating/evaluated, grading/graded, judging/judged, justifying/justified, measuring/measured,
                    ranking/ranked, rating/rated, recommending/recommended, reviewing/reviewed, scoring/scored, selecting/selected, standardizing/standardized,
                    supporting/supported, testing/tested, validating/validated, choosing/chosen, relating/related, defending/defended)
                    - All scenarios must include specific measurement criteria, metrics, or standards for objective evaluation.

                    Question Structure:

                    For Situational Questions:
                    Sentence 1: Context with defined evaluation criteria/metrics
                    Sentence 2: Specific situation using evaluation verb with suffix
                    Sentence 3: Objective identification question (Fill-in-blanks/Which/What/Complete-the-sentence)

                    For Programming Questions:
                    Sentence 1: Context with performance metrics/requirements
                    Sentence 2: Code snippet with evaluation parameters
                    Sentence 3: Objective identification question about code outcomes

                    Requirements:
                    - Use 1-2 verbs from: appraise, assess, compare, evaluate, grade, judge, measure, rank, rate, review, score, validate
                     """

multiple_choice_single = """{
    "question": "{input_the_question_in_here}",
    "answer": "{answer_in_here}",
    "choices": ["{choice_in_here}","{choice_in_here}","{choice_in_here}","{choice_in_here}"]
},"""

multiple_choice_many = """{
    "question": "{input_the_question_in_here}",
    "answer": ["{answer_in_here}","{answer_in_here}"],
    "choices": ["{choice_in_here}","{choice_in_here}","{choice_in_here}","{choice_in_here}"]
},"""

identification = """ {
    "question": "{input_the_question_in_here}",
    "answer": "{answer_in_here}",
}"""
