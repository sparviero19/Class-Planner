from pathlib import Path
from config.definitions import ROOT_DIR, load_api_keys
from src.pipeline_manager import HandoutPipelineManager, SlidesPipelineManager
from src.handout_distiller import generate_handout, reset_pipeline, show_pipeline_status
from src.slides_distiller import generate_slides, reset_slides_pipeline, show_slides_pipeline_status


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
    module_num = 1
    lesson_num = 6

    # Pipeline selection
    run_handout_pipeline = True    # Set to True to run handout generation
    run_slides_pipeline = True     # Set to True to run slides generation

    # Resume settings
    resume_handout = True          # Resume from last checkpoint (handout)
    resume_slides = True           # Resume from last checkpoint (slides)

    # Slide budget (only for slides pipeline)
    slide_budget = 35              # Total number of content slides to generate

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

    # ============================================================
    # RESET PIPELINES (Optional)
    # ============================================================

    # Reset handout pipeline from a specific stage if needed
    """
    HANDOUT STAGES:
        "first_draft",          0
        "review",               1
        "summary",              2
        "handout_draft",        3
        "editing_instructions", 4
        "final_handout"         5
    """
    handout_stages = {str(i): s for i, s in enumerate(HandoutPipelineManager.STAGES)}
    reset_handout_stage = None  # Set to stage number (0-5) or stage name to reset from

    if reset_handout_stage is not None:
        if isinstance(reset_handout_stage, int):
            stage_name = handout_stages[str(reset_handout_stage)]
        else:
            stage_name = reset_handout_stage
        reset_pipeline(lesson_num, module_num, from_stage=stage_name, output_dir=output_folder)
        print(f"Handout pipeline reset from stage: {stage_name}")

    # Reset slides pipeline from a specific stage if needed
    """
    SLIDES STAGES:
        "pedagogical_analysis",  0
        "slide_budget",          1
        "visual_inventory",      2
        "content_draft",         3
        "review",                4
        "final_slides"           5
    """
    slides_stages = {str(i): s for i, s in enumerate(SlidesPipelineManager.STAGES)}
    reset_slides_stage = None  # Set to stage number (0-5) or stage name to reset from

    if reset_slides_stage is not None:
        if isinstance(reset_slides_stage, int):
            stage_name = slides_stages[str(reset_slides_stage)]
        else:
            stage_name = reset_slides_stage
        reset_slides_pipeline(lesson_num, module_num, from_stage=stage_name, output_dir=output_folder)
        print(f"Slides pipeline reset from stage: {stage_name}")

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

    # ============================================================
    # COMPLETION
    # ============================================================

    print("\n" + "=" * 60)
    print("All selected pipelines completed!")
    print("=" * 60)

    # Show final status
    if run_handout_pipeline or run_slides_pipeline:
        print("\nFinal pipeline status:")
        if run_handout_pipeline:
            print("\n📄 Handout Pipeline:")
            show_pipeline_status(lesson_num, module_num, output_folder)
        if run_slides_pipeline:
            print("\n📊 Slides Pipeline:")
            show_slides_pipeline_status(lesson_num, module_num, output_folder)


if __name__ == "__main__":
    main()
