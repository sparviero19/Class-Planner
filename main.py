from pathlib import Path
from config.definitions import ROOT_DIR, load_api_keys
from src.pipeline_manager import HandoutPipelineManager, SlidesPipelineManager, SelfEvalQuizPipelineManager
from src.handout_distiller import generate_handout, reset_pipeline, show_pipeline_status
from src.slides_distiller import generate_slides, reset_slides_pipeline, show_slides_pipeline_status
from src.quizzes_gen import generate_self_eval_quizzes, show_quiz_pipeline_status, reset_quiz_pipeline


def main():
    """
    Main entry point for running handout and slides generation pipelines.

    Configure the settings below to:
    - Choose which pipeline(s) to run
    - Set lesson and module numbers
    - Enable/disable resume functionality
    - Reset pipelines from specific stages
    """

    # ============================================================
    # CONFIGURATION
    # ============================================================

    subject = "Computer Vision"
    language = "Italian"
    module_num = 2
    lesson_num = 9

    # Pipeline selection
    run_handout_pipeline = False    # Set to True to run handout generation
    run_slides_pipeline = False     # Set to True to run slides generation
    run_quiz_pipeline = True       # Set to True to run quiz generation

    # Agent settings
    quiz_agent_type = "gemini"    # "gemini" or "ollama"
    if quiz_agent_type == "ollama":
        quiz_model_name = "ministral-3:latest" # e.g. "ministral-3:latest" for ollama or "gemini-2.0-flash"
    else:
        quiz_model_name = "gemini-2.5-flash-lite"

    # Resume settings
    resume_handout = True          # Resume from last checkpoint (handout)
    resume_slides = True           # Resume from last checkpoint (slides)
    resume_quiz = True             # Resume from last checkpoint (quiz)

    # Reset pipelines from specific stages:
        # Reset handout pipeline from a specific stage if needed
        # HANDOUT STAGES:
        #     "first_draft",          0
        #     "review",               1
        #     "summary",              2
        #     "handout_draft",        3
        #     "editing_instructions", 4
        #     "final_handout"         5
        # SLIDES STAGES:
        #     "pedagogical_analysis",  0
        #     "slide_budget",          1
        #     "visual_inventory",      2
        #     "content_draft",         3
        #     "review",                4
        #     "final_slides",          5
        #     "slides_distillation"    6
    reset_handout_stage = None  # Set to stage number (0-5) or stage name to reset from
    reset_slides_stage = None  # Set to stage number (0-6) or stage name to reset from

    # Slide budget (only for slides pipeline)
    slide_budget = 35              # Total number of content slides to generate

    # Number of questions to generate
    num_questions = 10

    # ============================================================
    # PATHS
    # ============================================================

    input_folder = Path(ROOT_DIR) / f"data/input/module {module_num:03}/Lez {lesson_num:03} materials"
    output_folder = Path(ROOT_DIR) / f"data/output/module {module_num:03}"

    # ============================================================
    # SHOW PIPELINE STATUS (Optional)
    # ============================================================

    # Uncomment to see current pipeline status before running
    # show_pipeline_status(lesson_num, module_num, output_folder)
    # show_slides_pipeline_status(lesson_num, module_num, output_folder)
    # show_quiz_pipeline_status(lesson_num, module_num, output_folder)

    # ============================================================
    # RESET PIPELINES (Optional)
    # ============================================================

    # Reset handout pipeline from a specific stage if needed
    handout_stages = {str(i): s for i, s in enumerate(HandoutPipelineManager.STAGES)}

    if reset_handout_stage is not None:
        if isinstance(reset_handout_stage, int):
            stage_name = handout_stages[str(reset_handout_stage)]
        else:
            stage_name = reset_handout_stage
        reset_pipeline(lesson_num, module_num, from_stage=stage_name, output_dir=output_folder)
        print(f"Handout pipeline reset from stage: {stage_name}")

    # Reset slides pipeline from a specific stage if needed
    slides_stages = {str(i): s for i, s in enumerate(SlidesPipelineManager.STAGES)}

    if reset_slides_stage is not None:
        if isinstance(reset_slides_stage, int):
            stage_name = slides_stages[str(reset_slides_stage)]
        else:
            stage_name = reset_slides_stage
        reset_slides_pipeline(lesson_num, module_num, from_stage=stage_name, output_dir=output_folder)
        print(f"Slides pipeline reset from stage: {stage_name}")

    # Reset quiz pipeline from a specific stage if needed
    quiz_stages = {str(i): s for i, s in enumerate(SelfEvalQuizPipelineManager.STAGES)}
    reset_quiz_stage = None # Set to stage number (0-2) or stage name to reset from

    if reset_quiz_stage is not None:
        if isinstance(reset_quiz_stage, int):
            stage_name = quiz_stages[str(reset_quiz_stage)]
        else:
            stage_name = reset_quiz_stage
        reset_quiz_pipeline(lesson_num, module_num, from_stage=stage_name, output_dir=output_folder)
        print(f"Quiz pipeline reset from stage: {stage_name}")

    # ============================================================
    # RUN PIPELINES
    # ============================================================

    print("=" * 60)
    print(f"Class Notes Distiller - Module {module_num}, Lesson {lesson_num}")
    print("=" * 60)

    # Run handout generation pipeline
    if run_handout_pipeline:
        print("\n🔄 Running HANDOUT generation pipeline...")
        print("-" * 60)
        generate_handout(
            subject=subject,
            language=language,
            lesson_num=lesson_num,
            module_num=module_num,
            resume=resume_handout,
            override_files=None,
            input_folder=input_folder,
            output_folder=output_folder
        )
        print("\n✅ Handout pipeline completed!")
    else:
        print("\n⏭️  Skipping handout generation (run_handout_pipeline=False)")

    # Run slides generation pipeline
    if run_slides_pipeline:
        print("\n🔄 Running SLIDES generation pipeline...")
        print("-" * 60)
        generate_slides(
            subject=subject,
            language=language,
            lesson_num=lesson_num,
            module_num=module_num,
            resume=resume_slides,
            override_files=None,
            input_folder=input_folder,
            output_folder=output_folder,
            slide_budget=slide_budget
        )
        print("\n✅ Slides pipeline completed!")
    else:
        print("\n⏭️  Skipping slides generation (run_slides_pipeline=False)")

    # Run quiz generation pipeline
    if run_quiz_pipeline:
        print("\n🔄 Running QUIZ generation pipeline...")
        print("-" * 60)
        generate_self_eval_quizzes(
            subject=subject,
            language=language,
            lesson_num=lesson_num,
            module_num=module_num,
            resume=resume_quiz,
            output_folder=output_folder,
            num_questions=num_questions,
            agent_type=quiz_agent_type,
            model_name=quiz_model_name
        )
        print("\n✅ Quiz pipeline completed!")
    else:
        print("\n⏭️  Skipping quiz generation (run_quiz_pipeline=False)")

    # ============================================================
    # COMPLETION
    # ============================================================

    print("\n" + "=" * 60)
    print("All selected pipelines completed!")
    print("=" * 60)

    # Show final status
    if run_handout_pipeline or run_slides_pipeline or run_quiz_pipeline:
        print("\nFinal pipeline status:")
        if run_handout_pipeline:
            print("\n📄 Handout Pipeline:")
            show_pipeline_status(lesson_num, module_num, output_folder)
        if run_slides_pipeline:
            print("\n📊 Slides Pipeline:")
            show_slides_pipeline_status(lesson_num, module_num, output_folder)
        if run_quiz_pipeline:
            print("\n❓ Quiz Pipeline:")
            show_quiz_pipeline_status(lesson_num, module_num, output_folder)


if __name__ == "__main__":
    main()
