from pathlib import Path
from config.definitions import ROOT_DIR
from config.config_loader import get_config
from src.pipeline_manager import SelfEvalQuizPipelineManager
from src.quizzes_gen import (generate_self_eval_quizzes, show_quiz_pipeline_status, reset_quiz_pipeline,
                             generate_exam_quizzes)
import os
import re
from typing import Optional


def get_lessons_list(module_num: int, module_path: Path) -> Optional[list]:
    handouts_dir = module_path / "handouts"
    if not handouts_dir.exists():
        return None

    handout_pattern = re.compile(rf"handout_m{module_num:03}_l(\d+)_(\d+)\.md")
    lessons_list = list()
    for file in handouts_dir.glob("*.md"):
        match = handout_pattern.match(file.name)
        if match:
            lesson = int(match.group(1))
            lessons_list.append(lesson)
    return sorted(lessons_list)


def main():
    """
    This main can be used to generate questions for final exams.
    The questions can be generated for any list or interval of lessons.
    For each lesson, the agents will load the existing material (handout) and the previously generated
    self-evaluation questions, if any. The generation process will then generate new questions, with a
    given difficulty level, that will cover the same, or new topics w.r.t. the self-evaluation questions.

    The generation process has different steps to refine the difficulty level and maintain consistency in
    format and style.

    Since it's agentic, it will work, hopefully :)
    :return:
    """

    # Load configuration
    config = get_config()

    # Extract values from config
    subject = config.course.subject
    language = config.course.language
    subfolder_name = config.course.subfolder_name

    # Exam-specific settings (could be added to config.yaml if needed)
    modules = [3]  # list(range(1, 6))
    lessons = "all"  # TODO add fine grained control
    num_questions = config.quiz.num_questions
    difficulty = 5
    generate_se_quizzes = False
    reset_quiz_stage = None  # this is effective in combination with generate _self_eval_quizzes

    # Agent configuration from config
    quiz_agent_config = config.get_agent_config('quiz_generator')
    quiz_agent_type = quiz_agent_config.type
    quiz_model_name = quiz_agent_config.get_model()

    resume_quiz = config.resume.quiz

    print("=" * 60)
    print(f"Class Notes Distiller - Exam quizzes generation for Modules {modules}")
    print("=" * 60)

    for module_num in modules:

        module_folder = Path(ROOT_DIR) / f"data/output/{subfolder_name}/module {module_num:03}"
        lessons_list = None
        if isinstance(lessons, str):
            if lessons == "all":
               lessons_list = get_lessons_list(module_num, module_folder)
        elif isinstance(lessons, list):
            lessons_list = lessons
        elif isinstance(lessons, int):
            lessons_list = [lessons]

        if lessons_list is None:
            print("!" * 60)
            print(f"No lessons in module {module_num}|")
            print("!" * 60)
            continue  # go to the next module

        for lesson_num in lessons_list:
            print(f"Starting pipeline for Module {module_num}, Lesson {lesson_num}")

            # ============================================================
            # RESET PIPELINES
            # ============================================================
            if generate_se_quizzes and reset_quiz_stage is not None:
                quiz_stages = {str(i): s for i, s in enumerate(SelfEvalQuizPipelineManager.STAGES)}
                if isinstance(reset_quiz_stage, int):
                    stage_name = quiz_stages[str(reset_quiz_stage)]
                else:
                    stage_name = reset_quiz_stage  # if the stage is given as a string
                reset_quiz_pipeline(lesson_num, module_num, from_stage=stage_name, output_dir=module_folder)

            # ============================================================
            # RUN PIPELINE
            # ============================================================
            if generate_se_quizzes:
                generate_self_eval_quizzes(
                    config=config,
                    lesson_num=lesson_num,
                    module_num=module_num,
                    resume=resume_quiz,
                    output_folder=module_folder,
                    num_questions=num_questions,
                    agent_type=quiz_agent_type,
                    model_name=quiz_model_name
                )
                print(f"\n✅ Self-eval Quiz pipeline for lesson {lesson_num} completed!")

            generate_exam_quizzes(
                config=config,
                lesson_num=lesson_num,
                module_num=module_num,
                resume=resume_quiz,
                output_folder=module_folder,
                num_questions=num_questions,
                difficulty_level=difficulty,
                agent_type=quiz_agent_type,
                model_name=quiz_model_name
            )
            print(f"\n✅ Exam Quiz pipeline for lesson {lesson_num} completed!")



if __name__ == "__main__":
    main()
