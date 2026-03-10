You are preparing the final exam questions for the lesson {lesson_num} in module {module_num}. The subject is {subject} and the language of the quiz is {language}.

[Context: Materials]
The quizzes have to be designed to test the theoretical and practical knowledge of the following handouts:

{handout}

[Context: Quizzes Guidelines]
The quizzes have to follow this guide:

{guide}

[Context: Formatting]
In addtion, the quizzes have to comply with the following formatting structure:

{format}

[Context: Self-evaluation Quizzes]
The students already have some self-evaluation quizzes, that follow the same aformentioned structure and constraints. if the quizzes are available, they will be presented here for your reference:
---

{se_quizzes}

---
It is of PARAMOUNT importance for the final quizzes to BE DIFFERENT from the self-evaluation ones. More specifically, 
- The topics of the questions can be the same, but the exact question has to be different.
- No good or correct answwer should identically appear on the self-evaluation quizzes and on the final quizzes. 
Givent that, you are freee to use the self.evaluation quizzes to gauge the difficulty level of the final exam questions.

[Task: Quiz Creation Plan]
1. Conceptual Analysis: Identify of 5 key concepts from the lecture notes.
2. Definition of Difficulty Level (1-5): The quizzes should be designed keeping in mind a difficulty scale from 1 to 5. 
3. Formulation of Questions and Distractors: Draft {num_questions} questions of DIFFICULTY LEVEL {difficulty} in affirmative form that are clear and understandable
even out of context, with 4 plausible but DISTINCT answer options, avoiding "all/none of the above."
5. CSV Generation: Compilation of the final CSV string in the exact required format, considering that the "video lesson title" is the title of the handout, and the reference paragraph titles are the handouts sections (1. 2. 3. etch) titles.

## Task
please, generate the quizzes in {language} language following the plan.  