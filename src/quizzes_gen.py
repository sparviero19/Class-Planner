

import os
import re
from pathlib import Path
from time import time
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown

from config.definitions import ROOT_DIR, load_api_keys
from src.agents import GeminiAgent, OpenAIAgent, OllamaAgent
from src.pipeline_manager import SelfEvalQuizPipelineManager, ExamQuizPipelineManager


def load_prompt(prompt_path, **kwargs):
    """Load prompt from file and format with kwargs"""
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    return prompt.format(**kwargs)


def find_latest_X(X: str, lesson_num: int, module_num: int, output_folder: Path) -> Optional[Path]:
    """
    Find the latest generated X for a given lesson
    where X could be
        handout
        quizes
    """
    if X == "handouts":
        prefix = "handout"
        suffix = "md"
    else:
        prefix = X
        suffix = "csv"
    X_dir = output_folder / f"{X}"

    if not X_dir.exists():
        return None

    pattern = re.compile(rf"{prefix}_m{module_num:03}_l{lesson_num:03}_(\d+)\.{suffix}")
    latest_file = None
    latest_ts = -1

    for file in X_dir.glob(f"*.{suffix}"):
        match = pattern.match(file.name)
        if match:
            ts = int(match.group(1))
            if ts > latest_ts:
                latest_ts = ts
                latest_file = file

    return latest_file


def generate_self_eval_quizzes(subject, language, lesson_num, module_num, resume=True, output_folder=None,
                               quiz_guide_path=None, quiz_format_path=None, num_questions=5, agent_type="gemini", model_name=None):
    """
    Generate self-evaluation quizzes from a handout with checkpoint/resume capability.
    """
    if not output_folder:
        output_folder = Path(ROOT_DIR) / f"data/output/module {module_num:03}"
    if not quiz_guide_path:
        quiz_guide_path = Path(ROOT_DIR) / "src/prompts/quiz_generator/guide.txt"
    if not quiz_format_path:
        quiz_format_path = Path(ROOT_DIR) / "src/prompts/quiz_generator/quiz_format.txt"

    if agent_type == "ollama" and model_name is None: # use only for debug! it produces wrong or gibberish quizzes
        model_name = "ministral-3:latest"
    elif agent_type == "gemini" and model_name is None:
        model_name = "gemini-2.5-flash-lite"

    api_keys = load_api_keys()
    console = Console()

    console.print(Markdown("# Self-Evaluation Quiz Generation Pipeline"))
    console.print(f"Using agent: {agent_type} ({model_name})")

    # Initialize pipeline manager
    pipeline = SelfEvalQuizPipelineManager(lesson_num, module_num, output_folder)

    if not resume:
        console.print(Markdown("**Starting fresh pipeline (not resuming)**"))
        pipeline.clear_all()

    # Find the handout
    handout_path = find_latest_X("handouts", lesson_num, module_num, output_folder)
    if not handout_path:
        raise FileNotFoundError(
            f"No handout found for lesson {lesson_num}, module {module_num}. "
            f"Expected in: {output_folder}/handouts/"
        )
    console.print(f"Using handout: {handout_path.name}")
    with open(handout_path, 'r') as f:
        handout_content = f.read()

    with open(quiz_guide_path, 'r') as f:
        quiz_guide = f.read()

    with open(quiz_format_path, 'r') as f:
        quiz_format = f.read()

    # Define agents
    def get_teacher():
        system_prompt_T = load_prompt(Path(ROOT_DIR) / "src/prompts/system.teacher.md",
                                      subject=subject, language=language)
        if agent_type == "ollama":
            return OllamaAgent("T", model_name, system_prompt_T)
        else:
            return GeminiAgent("T", model_name, system_prompt_T, manage_history=False, tools=None,
                               api_key=api_keys['google'])

    def get_evaluator():
        system_prompt_E = load_prompt(Path(ROOT_DIR) / "src/prompts/quiz_generator/system.quiz_evaluator.md",
                                      course_subject=subject, guide=quiz_guide, format=quiz_format)
        return OpenAIAgent("E", "gpt-4o-mini", system_prompt_E)

    # Pipeline logic
    next_stage = pipeline.get_next_stage()
    if next_stage:
        console.print(Markdown(f"**Resuming from stage: {next_stage}**"))
    else:
        console.print(Markdown("**All stages completed!**"))
        return

    # Stage 1: Generate quiz draft
    if not pipeline.is_stage_completed("generating_quiz_draft"):
        console.print(Markdown(f"## Step 1: Generating {num_questions} quesitons"))

        teacher = get_teacher()
        draft_quiz_prompt_template = Path(ROOT_DIR) / "src/prompts/quiz_generator/self_eval_quiz.teacher.sl.md"
        draft_quiz_prompt = load_prompt(draft_quiz_prompt_template,
                                     num_questions=num_questions,
                                     course_subject=subject,
                                     handout=handout_content,
                                     guide=quiz_guide,
                                     format=quiz_format,
                                     language=language)


        console.print(f"Generating initial quiz draft...")
        quiz_draft = teacher.chat(draft_quiz_prompt)

        pipeline.save_stage_output("generating_quiz_draft", quiz_draft)
        console.print("✓ Quiz draft generated.")
    else:
        # pipeline step already completed, loading corresponding output files
        quiz_draft = pipeline.get_stage_output("generating_quiz_draft")

    # Stage 2: reformat CSV according to avoid separators errors
    if not pipeline.is_stage_completed("reformat_csv"):
        console.print(Markdown("## Step 2: Re-format quiz CSV"))

        reformat_quizzes_prompt_template = Path(ROOT_DIR) / "src/prompts/quiz_generator/reformat_quizzes.teacher.sl.md"
        reformat_quizzes_prompt = load_prompt(reformat_quizzes_prompt_template,
                                              quiz_draft=quiz_draft,
                                              format=quiz_format)
        teacher = get_teacher()
        formatted_quiz = teacher.chat(reformat_quizzes_prompt)
        pipeline.save_stage_output("reformat_csv", formatted_quiz)
        console.print("✓ Quiz format updated.")

    else:
        formatted_quiz = pipeline.get_stage_output("reformat_csv")

    # Stage 3: Evaluate quizzes
    if not pipeline.is_stage_completed("evaluating_quiz_draft"):
        console.print(Markdown("## Step 3: Evaluating quiz"))

        evaluator = get_evaluator()
        eval_quiz_prompt_template = Path(ROOT_DIR) / "src/prompts/quiz_generator/eval_quiz.evaluator.sl.md"
        evaluation_prompt = load_prompt(eval_quiz_prompt_template,
                                        handout=handout_content,
                                        quiz_draft=formatted_quiz,)
        evaluation_result = evaluator.chat(evaluation_prompt)

        pipeline.save_stage_output("evaluating_quiz_draft", evaluation_result)
        console.print("✓ Quizzes evaluated.")
    else:
        evaluation_result = pipeline.get_stage_output("evaluating_quiz_draft")

    # Stage 4: Final Quiz Extraction (Simplified for now, picking the best and saving as CSV)
    if not pipeline.is_stage_completed("final_quiz"):
        console.print(Markdown("## Step 4: Finalizing CSV output"))

        final_quiz_template_prompt = Path(ROOT_DIR) / "src/prompts/quiz_generator/final_quiz.teacher.sl.md"
        final_quiz_csv_prompt = load_prompt(final_quiz_template_prompt,
                                     lesson_num=lesson_num,
                                     formatted_quiz=formatted_quiz,
                                     evaluation=evaluation_result,
                                     format=quiz_format,
                                     language=language)

        teacher = get_teacher()
        final_csv = teacher.chat(final_quiz_csv_prompt)

        # Cleanup CSV if there are markdown blocks
        if "```csv" in final_csv:
            final_csv = final_csv.split("```csv")[1].split("```")[0].strip()
        elif "```" in final_csv:
            final_csv = final_csv.split("```")[1].split("```")[0].strip()
        final_csv = final_csv.strip()

        timestamp = int(time())
        final_path = pipeline.get_final_output_path(timestamp)
        pipeline.save_stage_output("final_quiz", final_csv, final_path)

        console.print(f"✓ Final quiz saved to {final_path}")

def show_quiz_pipeline_status(lesson_num, module_num, output_folder):
    """Display the current status of the quiz pipeline"""
    pipeline = SelfEvalQuizPipelineManager(lesson_num, module_num, output_folder)
    console = Console()
    
    console.print(f"\n[bold]Quiz Pipeline Status (Lesson {lesson_num}, Module {module_num}):[/bold]")
    for stage in SelfEvalQuizPipelineManager.STAGES:
        status = "✅" if pipeline.is_stage_completed(stage) else "⏳"
        file_path = pipeline.get_stage_file(stage)
        file_info = f" ({file_path.name})" if file_path.exists() else ""
        console.print(f"{status} {stage}{file_info}")

#TODO: super redundant: create a single function for resetting pipelines
def reset_quiz_pipeline(lesson_num, module_num, from_stage=None, output_dir=None):
    """Reset the quiz pipeline from a specific stage"""
    pipeline = SelfEvalQuizPipelineManager(lesson_num, module_num, output_dir)
    if from_stage:
        pipeline.reset_from_stage(from_stage)
    else:
        pipeline.clear_all()

def generate_exam_quizzes(subject, language, lesson_num, module_num, resume=True, output_folder=None,
                          quiz_guide_path=None, quiz_format_path=None, num_questions=5, difficulty_level=5,
                          agent_type="gemini", model_name=None):
    """
    Generate final exam quizzes from a handout and self-eval quizzes (optional) with checkpoint/resume capability.
    """

    if not output_folder:
        output_folder = Path(ROOT_DIR) / f"data/output/module {module_num:03}"
    if not quiz_guide_path:
        quiz_guide_path = Path(ROOT_DIR) / "src/prompts/quiz_generator/guide.txt"
    if not quiz_format_path:
        quiz_format_path = Path(ROOT_DIR) / "src/prompts/quiz_generator/quiz_format.txt"

    if agent_type == "ollama" and model_name is None: # use only for debug! it produces wrong or gibberish quizzes
        model_name = "ministral-3:latest"
    elif agent_type == "gemini" and model_name is None:
        model_name = "gemini-2.5-flash-lite"

    api_keys = load_api_keys()
    console = Console()

    console.print(Markdown("# Final Exam Quiz Generation Pipeline"))
    console.print(f"Using agent: {agent_type} ({model_name})")

    # Initialize pipeline manager
    pipeline = ExamQuizPipelineManager(lesson_num, module_num, output_folder)

    if not resume:
        console.print(Markdown("**Starting fresh pipeline (not resuming)**"))
        pipeline.clear_all()

    # Find the handout
    handout_path = find_latest_X("handouts", lesson_num, module_num, output_folder)
    if not handout_path:
        raise FileNotFoundError(
            f"No handout found for lesson {lesson_num}, module {module_num}. "
            f"Expected in: {output_folder}/handouts/"
        )

    # Find any self-eval quizzes
    se_quizzes_path = find_latest_X("quizzes", lesson_num, module_num, output_folder)
    se_quizzes = ""
    if not se_quizzes_path:
        print(
            f"No self-evaluation quizzes found for lesson {lesson_num}, module {module_num}. "
            f"Continuing generation of exam quizzes from scratch."
        )
    else:
        with open(se_quizzes_path, 'r') as f:
            se_quizzes = f.read()

    console.print(f"Using handout: {handout_path.name}")
    with open(handout_path, 'r') as f:
        handout_content = f.read()

    with open(quiz_guide_path, 'r') as f:
        quiz_guide = f.read()

    with open(quiz_format_path, 'r') as f:
        quiz_format = f.read()

    # Define agents
    def get_teacher():
        system_prompt_T = load_prompt(Path(ROOT_DIR) / "src/prompts/system.teacher.md",
                                      subject=subject, language=language)
        if agent_type == "ollama":
            return OllamaAgent("T", model_name, system_prompt_T)
        else:
            return GeminiAgent("T", model_name, system_prompt_T, manage_history=False)

    def get_evaluator():
        system_prompt_E = load_prompt(Path(ROOT_DIR) / "src/prompts/quiz_generator/system.quiz_evaluator.md",
                                      course_subject=subject, guide=quiz_guide, format=quiz_format)
        return OpenAIAgent("E", "gpt-4o-mini", system_prompt_E)

    # Pipeline logic
    next_stage = pipeline.get_next_stage()
    if next_stage:
        console.print(Markdown(f"**Resuming from stage: {next_stage}**"))
    else:
        console.print(Markdown("**All stages completed!**"))
        return

    if not pipeline.is_stage_completed("generating_exam_quiz"):
        teacher = get_teacher()
        prompt_template_path = Path(ROOT_DIR) / "src/prompts/quiz_generator/exam_draft.teacher.sl.md"
        prompt = load_prompt(prompt_template_path,
                             lesson_num=lesson_num,
                             module_num=module_num,
                             subject=subject,
                             language=language,
                             handout=handout_content,
                             guide=quiz_guide,
                             format=quiz_format,
                             se_quizzes=se_quizzes,
                             num_questions=num_questions,
                             difficulty=difficulty_level
                             )
        exam_first_draft = teacher.chat(prompt)
        pipeline.save_stage_output("generating_exam_quiz", exam_first_draft)
        console.print("✓ First draft of the exam quizzes completed!")
    else:
        exam_first_draft = pipeline.get_stage_output("generating_exam_quiz")

    if not pipeline.is_stage_completed("reformat_csv"):
        console.print(Markdown("## Step 2: Re-format quiz CSV"))

        reformat_quizzes_prompt_template = Path(ROOT_DIR) / "src/prompts/quiz_generator/reformat_quizzes.teacher.sl.md"
        reformat_quizzes_prompt = load_prompt(reformat_quizzes_prompt_template,
                                              quiz_draft=exam_first_draft,
                                              format=quiz_format)
        teacher = get_teacher()
        formatted_exam = teacher.chat(reformat_quizzes_prompt)
        pipeline.save_stage_output("reformat_csv", formatted_exam)
        console.print("✓ Quiz format updated.")
    else:
        formatted_exam = pipeline.get_stage_output("reformat_csv")

    if not pipeline.is_stage_completed("evaluating_exam_quiz"):
        evaluator = get_evaluator()
        prompt_template_path = Path(ROOT_DIR) / "src/prompts/quiz_generator/eval_quiz.evaluator.sl.md"
        prompt = load_prompt(prompt_template_path,
                             handout=handout_content,
                             quiz_draft=formatted_exam,
                             num_questions=num_questions,
                             difficulty=difficulty_level
                             )
        exam_evaluation = evaluator.chat(prompt) #the format and guid are in the system prompt!
        pipeline.save_stage_output("evaluating_exam_quiz", exam_evaluation)
        console.print("✓ Exam evaluated")
    else:
        exam_evaluation = pipeline.get_stage_output("evaluating_exam_quiz")

    if not pipeline.is_stage_completed("final_exam_quiz"):
        teacher = get_teacher()
        prompt_template_path = Path(ROOT_DIR) / "src/prompts/quiz_generator/final_quiz.teacher.sl.md"
        prompt = load_prompt(prompt_template_path,
                             lesson_num=lesson_num,
                             formatted_quiz=formatted_exam,
                             evaluation=exam_evaluation,
                             format=quiz_format,
                             language=language
                             )
        final_exam = teacher.chat(prompt)

        # Cleanup CSV if there are markdown blocks
        if "```csv" in final_exam:
            final_csv = final_exam.split("```csv")[1].split("```")[0].strip()
        elif "```" in final_exam:
            final_csv = final_exam.split("```")[1].split("```")[0].strip()
        else:
            final_csv = final_exam.strip()

        timestamp = int(time())
        final_path = pipeline.get_final_output_path(timestamp)
        pipeline.save_stage_output("final_exam_quiz", final_csv, final_path)

        console.print(f"✓ Final quiz saved to {final_path}")
