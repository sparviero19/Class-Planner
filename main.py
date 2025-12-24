from pathlib import  Path
from config.definitions import ROOT_DIR, load_api_keys
from src.pipeline_manager import PipelineManager
from src.handout_distiller import generate_handout, reset_pipeline


def main():
    module_num = 7
    lesson_num = 11

    input_folder = Path(ROOT_DIR) / f"data/input/module {module_num:03}/Lez {lesson_num:03} materials"
    output_folder = Path(ROOT_DIR) / f"data/output/module {module_num:03}"

    # Show current status
    # show_pipeline_status(lesson_num, module_num, output_folder)
    
    # Reset from a specific stage if needed
    """
    STAGES = [
        "first_draft",          0
        "review",               1
        "summary",              2
        "handout_draft",        3
        "editing_instructions", 4
        "final_handout"         5
    ]
    """
    stages = {str(i):s for i,s in enumerate(PipelineManager.STAGES)}
    reset_stage = None
    if reset_stage is not None:
        reset_pipeline(lesson_num, module_num, from_stage=stages[str(reset_stage)], output_dir=output_folder)
    
    # Run the pipeline (will resume from last checkpoint)
    generate_handout(lesson_num, module_num, resume=True, override_files=None, input_folder=input_folder, output_folder=output_folder)

if __name__ == "__main__":
    main()
