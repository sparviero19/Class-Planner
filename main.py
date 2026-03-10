from pathlib import Path
from config.definitions import ROOT_DIR
from config.config_loader import get_config
from src.pipeline_manager import HandoutPipelineManager, SlidesPipelineManager, SelfEvalQuizPipelineManager
from src.handout_distiller import generate_handout, reset_pipeline, show_pipeline_status
from src.slides_distiller import generate_slides, reset_slides_pipeline, show_slides_pipeline_status
from src.quizzes_gen import generate_self_eval_quizzes, show_quiz_pipeline_status, reset_quiz_pipeline


def main():
    """
    Main entry point for running handout and slides generation pipelines.

    All configuration is now managed in config.yaml.
    Edit that file to:
    - Choose which pipeline(s) to run
    - Set lesson and module numbers
    - Configure agent models
    - Enable/disable resume functionality
    - Reset pipelines from specific stages
    """

    # ============================================================
    # LOAD CONFIGURATION
    # ============================================================

    config = get_config()

    # Extract configuration values
    subject = config.course.subject
    subfolder_name = config.course.subfolder_name
    language = config.course.language
    module_num = config.lesson.module_num
    lesson_num = config.lesson.lesson_num

    input_folder = Path(ROOT_DIR) / f"data/input/{subfolder_name}/module {module_num:03}/Lez {lesson_num:03} materials"
    output_folder = Path(ROOT_DIR) / f"data/output/{subfolder_name}/module {module_num:03}"

    # Pipeline selection
    run_handout_pipeline = config.pipelines.run_handout
    run_slides_pipeline = config.pipelines.run_slides
    run_quiz_pipeline = config.pipelines.run_quiz

    # Agent settings (quiz generator)
    quiz_agent_config = config.get_agent_config('quiz_generator')
    quiz_agent_type = quiz_agent_config.type
    quiz_model_name = quiz_agent_config.get_model()

    # Resume settings
    resume_handout = config.resume.handout
    resume_slides = config.resume.slides
    resume_quiz = config.resume.quiz

    # Reset settings
    reset_handout_stage = config.reset.handout_stage
    reset_slides_stage = config.reset.slides_stage

    # Pipeline-specific settings
    slide_budget = config.slides.slide_budget
    num_questions = config.quiz.num_questions

    # ============================================================
    # SHOW PIPELINE STATUS (Optional)
    # ============================================================

    # Show pipeline status based on config
    if config.advanced.show_status.handout:
        show_pipeline_status(lesson_num, module_num, output_folder)
    if config.advanced.show_status.slides:
        show_slides_pipeline_status(lesson_num, module_num, output_folder)
    if config.advanced.show_status.quiz:
        show_quiz_pipeline_status(lesson_num, module_num, output_folder)

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
            config=config,
            lesson_num=lesson_num,
            module_num=module_num,
            resume=resume_handout,
            override_files=None,
            input_folder=input_folder,
            output_folder=output_folder
        )
        print("\n✅ Handout pipeline completed!")
    else:
        print("\n⏭️  Skipping handout generation (config: pipelines.run_handout=false)")

    # Run slides generation pipeline
    if run_slides_pipeline:
        print("\n🔄 Running SLIDES generation pipeline...")
        print("-" * 60)
        generate_slides(
            config=config,
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
        print("\n⏭️  Skipping slides generation (config: pipelines.run_slides=false)")

    # Run quiz generation pipeline
    if run_quiz_pipeline:
        print("\n🔄 Running QUIZ generation pipeline...")
        print("-" * 60)
        generate_self_eval_quizzes(
            config=config,
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
        print("\n⏭️  Skipping quiz generation (config: pipelines.run_quiz=false)")

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
