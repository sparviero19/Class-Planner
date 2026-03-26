# src/pipeline_manager.py
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import abc

# ------------------------------------------------------------
# Abstract class for state management
# ------------------------------------------------------------
class AbstractPipelineManager(abc.ABC):

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.intermediate_dir = output_dir / "intermediate"
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        self.state = {}

    @abc.abstractmethod
    def _load_state(self) -> dict[str, Any]:
        pass

    @abc.abstractmethod
    def clear_all(self):
        pass

    def _save_state(self):
        """Save pipeline state to disk"""
        self.state["last_updated"] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def get_stage_file(self, stage: str) -> Path:
        """Get the standard filename for a stage"""
        return self.intermediate_dir / f"{stage}.md"

    def is_stage_completed(self, stage: str) -> bool:
        """Check if a stage has been completed"""
        return stage in self.state.get("completed_stages", [])

    def get_stage_output(self, stage: str) -> Optional[str]:
        """Load output from a completed stage"""
        if stage in self.state.get("stage_files", {}):
            file_path = Path(self.state["stage_files"][stage])
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return f.read()
        return None

    def save_stage_output(self, stage: str, content: str, file_path: Optional[Path] = None):
        """Save output for a stage and mark it as completed"""
        if file_path is None:
            file_path = self.get_stage_file(stage)

        # Save the content
        with open(file_path, 'w') as f:
            f.write(content)

        # Update state
        if stage not in self.state["completed_stages"]:
            self.state["completed_stages"].append(stage)
        self.state["stage_files"][stage] = str(file_path)
        self._save_state()

        return file_path

    def use_existing_file(self, stage: str, file_path: Path) -> bool:
        """Register an existing file as the output for a stage"""
        if not file_path.exists():
            return False

        # Mark stage as completed with this file
        if stage not in self.state["completed_stages"]:
            self.state["completed_stages"].append(stage)
        self.state["stage_files"][stage] = str(file_path)
        self._save_state()

        return True

    def get_next_stage(self) -> Optional[str]:
        """Get the next stage that needs to be completed"""
        completed = set(self.state.get("completed_stages", []))
        for stage in self.STAGES:
            if stage not in completed:
                return stage
        return None

    def reset_from_stage(self, stage: str):
        """Reset pipeline from a specific stage onwards"""
        try:
            stage_index = self.STAGES.index(stage)
            stages_to_remove = self.STAGES[stage_index:]

            for s in stages_to_remove:
                if s in self.state["completed_stages"]:
                    self.state["completed_stages"].remove(s)
                if s in self.state["stage_files"]:
                    del self.state["stage_files"][s]

            self._save_state()
        except ValueError:
            print(f"Unknown stage: {stage}")


# ------------------------------------------------------------
class HandoutPipelineManager (AbstractPipelineManager):
    """Manages pipeline state and intermediate file checkpoints for HANDOUTS"""

    STAGES = [
        "summary",
        "review",
        "revised_summary",
        "handout_draft",
        "editing_instructions",
        "final_handout"
    ]

    def __init__(self, lesson_num: int, module_num: int, output_dir: Path):
        # Create handouts subdirectory
        handouts_dir = output_dir / "handouts"
        super().__init__(handouts_dir)

        self.lesson_num = lesson_num
        self.module_num = module_num

        # State file for this specific lesson in handouts/intermediate/
        self.state_file = self.intermediate_dir / f"lesson_{module_num:03}_{lesson_num:03}_state.json"
        self.state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        """Load pipeline state from disk"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            "lesson_num": self.lesson_num,
            "module_num": self.module_num,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "completed_stages": [],
            "stage_files": {}
        }

    def get_stage_file(self, stage: str) -> Path:
        """Get the standard filename for a stage"""
        return self.intermediate_dir / f"{stage}_m{self.module_num:03}_l{self.lesson_num:03}.md"

    def get_final_output_path(self, timestamp: int) -> Path:
        """Get path for final handout output"""
        return self.output_dir / f"handout_m{self.module_num:03}_l{self.lesson_num:03}_{timestamp}.md"

    def clear_all(self):
        """Clear all pipeline state"""
        self.state = {
            "lesson_num": self.lesson_num,
            "module_num": self.module_num,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "completed_stages": [],
            "stage_files": {}
        }
        self._save_state()

# ------------------------------------------------------------
class SlidesPipelineManager(AbstractPipelineManager):
    """Manages pipeline state and intermediate file checkpoints for SLIDES"""

    STAGES = [
        "pedagogical_analysis",
        "slide_budget",
        "visual_inventory",
        "content_draft",
        "review",
        "final_slides",
        "slides_distillation"
    ]

    def __init__(self, lesson_num: int, module_num: int, output_dir: Path):
        # Create slides subdirectory
        slides_dir = output_dir / "slides"
        super().__init__(slides_dir)
        self.distilled_dir = slides_dir / "distilled"
        self.distilled_dir.mkdir(parents=True, exist_ok=True)

        self.lesson_num = lesson_num
        self.module_num = module_num

        # State file for this specific lesson in slides/intermediate/
        self.state_file = self.intermediate_dir / f"slides_lesson_{module_num:03}_{lesson_num:03}_state.json"
        self.state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        """Load pipeline state from disk"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            "lesson_num": self.lesson_num,
            "module_num": self.module_num,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "completed_stages": [],
            "stage_files": {}
        }

    def get_stage_file(self, stage: str) -> Path:
        """Get the standard filename for a stage"""
        return self.intermediate_dir / f"{stage}_m{self.module_num:03}_l{self.lesson_num:03}.md"

    def get_final_output_path(self, timestamp: int) -> Path:
        """Get path for final slides output"""
        return self.output_dir / f"slides_m{self.module_num:03}_l{self.lesson_num:03}_{timestamp}.md"

    def get_final_distilled_output_path(self, timestamp: int) -> Path:
        """Get path for distilled instruction for final slides output"""
        return self.distilled_dir / f"distilled_slides_m{self.module_num:03}_l{self.lesson_num:03}_{timestamp}.md"

    def clear_all(self):
        """Clear all pipeline state"""
        self.state = {
            "lesson_num": self.lesson_num,
            "module_num": self.module_num,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "completed_stages": [],
            "stage_files": {}
        }
        self._save_state()

# ------------------------------------------------------------
class SelfEvalQuizPipelineManager(AbstractPipelineManager):
    """Manages pipeline state and intermediate file checkpoints for QUIZZES"""

    STAGES = [
        "generating_quiz_draft",
        "reformat_csv",
        "evaluating_quiz_draft",
        "final_quiz"
    ]

    def __init__(self, lesson_num: int, module_num: int, output_dir: Path):
        # Create quizzes subdirectory
        quizzes_dir = output_dir / "quizzes"
        super().__init__(quizzes_dir)

        self.lesson_num = lesson_num
        self.module_num = module_num

        # State file for this specific lesson
        self.state_file = self.intermediate_dir / f"quizzes_lesson_{module_num:03}_{lesson_num:03}_state.json"
        self.state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        """Load pipeline state from disk"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            "lesson_num": self.lesson_num,
            "module_num": self.module_num,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "completed_stages": [],
            "stage_files": {}
        }

    def get_stage_file(self, stage: str) -> Path:
        """Get the standard filename for a stage"""
        return self.intermediate_dir / f"{stage}_m{self.module_num:03}_l{self.lesson_num:03}.md"

    def get_final_output_path(self, timestamp: int) -> Path:
        """Get path for final quiz output"""
        return self.output_dir / f"quiz_m{self.module_num:03}_l{self.lesson_num:03}_{timestamp}.csv"

    def clear_all(self):
        """Clear all pipeline state"""
        self.state = {
            "lesson_num": self.lesson_num,
            "module_num": self.module_num,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "completed_stages": [],
            "stage_files": {}
        }
        self._save_state()

class ExamQuizPipelineManager(AbstractPipelineManager):
    """Manages pipeline state and intermediate file checkpoints for EXAM QUIZZES"""

    STAGES = [
        "generating_exam_quiz",
        "reformat_csv",
        "evaluating_exam_quiz",
        "final_exam_quiz",
    ]

    def __init__(self, lesson_num: int, module_num: int, output_dir: Path):
        # Create exams subdirectory
        exams_dir = output_dir / "exams"
        super().__init__(exams_dir)

        self.lesson_num = lesson_num
        self.module_num = module_num

        # State file for this specific lesson in exams/intermediate/
        self.state_file = self.intermediate_dir / f"exam_lesson_{module_num:03}_{lesson_num:03}_state.json"
        self.state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        """Load pipeline state from disk"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            "lesson_num": self.lesson_num,
            "module_num": self.module_num,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "completed_stages": [],
            "stage_files": {}
        }

    def get_stage_file(self, stage: str) -> Path:
        """Get the standard filename for a stage"""
        return self.intermediate_dir / f"{stage}_m{self.module_num:03}_l{self.lesson_num:03}.md"

    def get_final_output_path(self, timestamp: int) -> Path:
        """Get path for final exam quiz output"""
        return self.output_dir / f"exam_m{self.module_num:03}_l{self.lesson_num:03}_{timestamp}.csv"

    def clear_all(self):
        """Clear all pipeline state"""
        self.state = {
            "lesson_num": self.lesson_num,
            "module_num": self.module_num,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "completed_stages": [],
            "stage_files": {}
        }
        self._save_state()

