from config.definitions import ROOT_DIR, load_api_keys
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown


load_api_keys()

def generate_slides(lesson_num, module_num, resume=True, override_files=None, input_folder=None, output_folder=None,
                     manage_history=True):
    """
        Generate slides from handouts and module structure with checkpoint/resume capability

        Args:
            lesson_num: Lesson number
            module_num: Module number
            resume: If True, resume from last checkpoint. If False, start fresh.
            override_files: Dict mapping stage names to file paths for pre-existing files
                           Example: {"summary": "path/to/my_edited_summary.md"}
            manage_history: If True, use google api automatic chat history. If False, history is handled manually.
    """
    # check inputs #todo: this is in common with generate_handout - refactor
    if not input_folder:
        input_folder = Path(ROOT_DIR) / f"data/input/module {module_num:03}/Lez {lesson_num:03} materials"
    if not output_folder:
        output_folder = Path(ROOT_DIR) / f"data/output/module {module_num:03}"
    stateless = ".sl"
    if manage_history and not resume:
        stateless = ""
    api_keys = load_api_keys()
    console = Console()

    # init agents
        # teacher assistant
        # reviewer
        # editor

    pass