import os
from pathlib import Path
from typing import Optional
from config.definitions import ROOT_DIR, load_api_keys
from src.agents import OpenAIAgent, GeminiAgent
from src.pipeline_manager import SlidesPipelineManager
from time import time
from rich.markdown import Markdown
from rich.console import Console
import re


def load_prompt(prompt_path, **kwargs):
    """Load and format a prompt file"""
    with open(prompt_path, "r") as f:
        prompt = f.read()
    prompt = prompt.format(**kwargs)
    return prompt


def extract_topics(topics_file_path):
    """Extract topics from topics file"""
    topics = []
    with open(topics_file_path, "r") as file:
        for line in file:
            topics.append(line.strip())
    return topics


def extract_module_structure(module_file_path):
    """Extract module structure from module topics file"""
    module_structure = {}
    with open(module_file_path, "r") as file:
        # get title
        module_structure["title"] = file.readline().strip()
        for line in file:
            if line.strip() == "":
                continue
            if re.search(r"\b" + re.escape("lezione") + r"\b", line.lower()):
                # Extract lesson number and title
                parts = line.split(":")
                lesson_num = parts[0].lower().replace("lezione", "").strip()
                module_structure[str(lesson_num)] = {"title": parts[1].strip(), "topics": []}
            else:
                module_structure[str(lesson_num)]["topics"].append(line.strip())
    return module_structure


def find_latest_handout(lesson_num: int, module_num: int, output_folder: Path) -> Optional[Path]:
    """
    Find the most recent handout for a lesson

    Returns:
        Path to handout file, or None if not found
    """
    handout_dir = output_folder / "handouts"
    if not handout_dir.exists():
        return None

    # Look for handout files matching pattern
    pattern = f"handout_m{module_num:03}_l{lesson_num:03}_*.md"
    handout_files = list(handout_dir.glob(pattern))

    if not handout_files:
        return None

    # Return most recent (highest timestamp)
    return max(handout_files, key=lambda p: p.stat().st_mtime)


def load_materials_paths(input_folder):
    """Load all material paths from input folder"""
    all_materials = [Path(os.path.join(input_folder, file)) for file in os.listdir(input_folder)]
    topics = None
    for material in all_materials[:]:
        if material.suffix == ".txt":
            topics = material
            all_materials.remove(material)
    return all_materials, topics


def generate_slides(subject, language, lesson_num, module_num, resume=True, override_files=None, input_folder=None, output_folder=None,
                     manage_history=True, slide_budget=30):
    """
    Generate slides from handouts and module structure with checkpoint/resume capability

    Args:
        subject: Subject of the lesson (e.g. "Computer Vision")
        language: Language of the lesson (e.g. "Italian")
        lesson_num: Lesson number
        module_num: Module number
        resume: If True, resume from last checkpoint. If False, start fresh.
        override_files: Dict mapping stage names to file paths for pre-existing files
                       Example: {"pedagogical_analysis": "path/to/my_analysis.md"}
        manage_history: If True, use google api automatic chat history. If False, history is handled manually.
        slide_budget: Total number of slides to generate (default: 30)
    """
    # Check inputs
    if not input_folder:
        input_folder = Path(ROOT_DIR) / f"data/input/module {module_num:03}/Lez {lesson_num:03} materials"
    if not output_folder:
        output_folder = Path(ROOT_DIR) / f"data/output/module {module_num:03}"

    stateless = ".sl"
    if manage_history and not resume:
        stateless = ""

    api_keys = load_api_keys()
    console = Console()

    console.print(Markdown("# Slides Generation Pipeline"))

    # Initialize SLIDES pipeline manager (creates slides/ and slides/intermediate/)
    pipeline = SlidesPipelineManager(lesson_num, module_num, output_folder)

    # Clear pipeline if not resuming
    if not resume:
        console.print(Markdown("**Starting fresh pipeline (not resuming)**"))
        pipeline.clear_all()

    # Register any override files
    if override_files:
        console.print(Markdown("## Registering pre-existing files"))
        for stage, file_path in override_files.items():
            file_path = Path(file_path)
            if pipeline.use_existing_file(stage, file_path):
                console.print(f"✓ Using existing file for '{stage}': {file_path.name}")
            else:
                console.print(f"✗ File not found for '{stage}': {file_path}")

    # Locate the handout from handout_distiller output
    handout_path = find_latest_handout(lesson_num, module_num, output_folder)
    if not handout_path:
        raise FileNotFoundError(
            f"No handout found for lesson {lesson_num}, module {module_num}. "
            f"Expected in: {output_folder}/handouts/"
        )

    console.print(f"Using handout: {handout_path.name}")

    # Load handout content
    with open(handout_path, 'r') as f:
        handout_content = f.read()

    # Module folder
    m_folder = Path(ROOT_DIR) / f"data/input/module {module_num:03}"

    # Load materials and topics
    material_paths, topics_file = load_materials_paths(input_folder)
    topics = extract_topics(topics_file)

    module_structure = {}
    if os.path.exists(m_folder / "module_topics.md"):
        module_structure = extract_module_structure(m_folder / "module_topics.md")

    # Define agents (lazy initialization)
    teacher = None
    reviewer = None

    def get_teacher(manage_history=True):
        nonlocal teacher
        if teacher is None:
            system_prompt_T = load_prompt(Path(ROOT_DIR) / "src/prompts/system.teacher.md",
                                          subject=subject, language=language)
            teacher = GeminiAgent("T", "gemini-2.5-flash", system_prompt_T, manage_history, None,
                                      api_key=api_keys['google'])
        return teacher

    def get_reviewer():
        nonlocal reviewer
        if reviewer is None:
            system_prompt_R = load_prompt(Path(ROOT_DIR) / "src/prompts/system.reviewer.md",
                                          subject=subject, language=language)
            reviewer = OpenAIAgent("R", "gpt-4o-mini", system_prompt_R, None)
        return reviewer

    # Check what stage we're at
    next_stage = pipeline.get_next_stage()
    if next_stage:
        console.print(Markdown(f"**Resuming from stage: {next_stage}**"))
    else:
        console.print(Markdown("**All stages completed!**"))
        return

    console.print(Markdown("## Step 0: Uploading PDF resources"))
    material_files = get_teacher().load_pdfs(material_paths, use_cache=True)
    materials = {m.name: p.name for m, p in zip(material_files, material_paths)}

    # Stage 1: Pedagogical Analysis
    if not pipeline.is_stage_completed("pedagogical_analysis"):
        console.print(Markdown("## Step 1: Analyzing pedagogical requirements"))

        analysis_prompt = load_prompt(
            Path(ROOT_DIR) / f"src/prompts/slides_distiller/pedagogical_analysis.teacher{stateless}.md",
            handout=handout_content,
            topics=topics,
            lesson_num=lesson_num,
            module_num=module_num,
            module_structure=module_structure
        )

        pedagogical_analysis = get_teacher().chat(analysis_prompt)
        saved_path = pipeline.save_stage_output("pedagogical_analysis", pedagogical_analysis)
        console.print(f"✓ Pedagogical analysis saved to: {saved_path}")
    else:
        console.print(Markdown("## Step 1: ✓ Pedagogical analysis already exists (skipping)"))
        pedagogical_analysis = pipeline.get_stage_output("pedagogical_analysis")

    # Stage 2: Slide Budget Allocation
    if not pipeline.is_stage_completed("slide_budget"):
        console.print(Markdown(f"## Step 2: Allocating slide budget ({slide_budget} slides)"))

        budget_prompt = load_prompt(
            Path(ROOT_DIR) / f"src/prompts/slides_distiller/budget_allocation.teacher{stateless}.md",
            pedagogical_analysis=pedagogical_analysis,
            slide_budget=slide_budget,
            topics=topics,
            lesson_num=lesson_num
        )

        slide_budget_allocation = get_teacher().chat(budget_prompt)
        saved_path = pipeline.save_stage_output("slide_budget", slide_budget_allocation)
        console.print(f"✓ Slide budget allocation saved to: {saved_path}")
    else:
        console.print(Markdown("## Step 2: ✓ Slide budget allocation already exists (skipping)"))
        slide_budget_allocation = pipeline.get_stage_output("slide_budget")

    # Stage 3: Visual Inventory
    if not pipeline.is_stage_completed("visual_inventory"):
        console.print(Markdown("## Step 3: Extracting visual inventory from materials"))

        visual_prompt = load_prompt(
            Path(ROOT_DIR) / f"src/prompts/slides_distiller/visual_extraction.teacher{stateless}.md",
            materials=materials,
            pedagogical_analysis=pedagogical_analysis,
            topics=topics
        )

        visual_inventory = get_teacher().chat(visual_prompt)
        saved_path = pipeline.save_stage_output("visual_inventory", visual_inventory)
        console.print(f"✓ Visual inventory saved to: {saved_path}")
    else:
        console.print(Markdown("## Step 3: ✓ Visual inventory already exists (skipping)"))
        visual_inventory = pipeline.get_stage_output("visual_inventory")

    # Stage 4: Content Draft with Visual Assignment
    if not pipeline.is_stage_completed("content_draft"):
        console.print(Markdown("## Step 4: Generating slide content with visual assignments"))

        content_prompt = load_prompt(
            Path(ROOT_DIR) / f"src/prompts/slides_distiller/content_with_visuals.teacher{stateless}.md",
            slide_budget_allocation=slide_budget_allocation,
            visual_inventory=visual_inventory,
            handout=handout_content,
            pedagogical_analysis=pedagogical_analysis,
            materials=materials,
            lesson_num=lesson_num,
            topics=topics
        )

        content_draft = get_teacher().chat(content_prompt)
        saved_path = pipeline.save_stage_output("content_draft", content_draft)
        console.print(f"✓ Content draft saved to: {saved_path}")
    else:
        console.print(Markdown("## Step 4: ✓ Content draft already exists (skipping)"))
        content_draft = pipeline.get_stage_output("content_draft")

    # Stage 5: Review and Balance Check
    if not pipeline.is_stage_completed("review"):
        console.print(Markdown("## Step 5: Reviewing slides for balance and pedagogy"))

        review_prompt = load_prompt(
            Path(ROOT_DIR) / f"src/prompts/slides_distiller/slides_review.reviewer{stateless}.md",
            content_draft=content_draft,
            slide_budget_allocation=slide_budget_allocation,
            visual_inventory=visual_inventory,
            slide_budget=slide_budget,
            topics=topics
        )

        review = get_reviewer().chat(review_prompt)
        console.print(Markdown(review))
        saved_path = pipeline.save_stage_output("review", review)
        console.print(f"✓ Review saved to: {saved_path}")
    else:
        console.print(Markdown("## Step 5: ✓ Review already exists (skipping)"))
        review = pipeline.get_stage_output("review")

    # Stage 6: Final Slides
    if not pipeline.is_stage_completed("final_slides"):
        console.print(Markdown("## Step 6: Finalizing slides presentation"))

        finalize_prompt = load_prompt(
            Path(ROOT_DIR) / f"src/prompts/slides_distiller/finalize_slides.teacher{stateless}.md",
            content_draft=content_draft,
            review=review,
            visual_inventory=visual_inventory,
            lesson_num=lesson_num,
            module_structure=module_structure,
            module_num=module_num,
            slide_budget_allocation=slide_budget_allocation,
        )

        final_slides = get_teacher().chat(finalize_prompt)
        # Save final slides with timestamp in slides/ folder
        final_path = pipeline.get_final_output_path(round(time()))
        pipeline.save_stage_output("final_slides", final_slides, final_path)
        console.print(f"✓ Final slides saved to: {final_path}")
    else:
        console.print(Markdown("## Step 6: ✓ Final slides already exist"))
        final_slides = pipeline.get_stage_output("final_slides")

    # stage 7: Slides instruction distillation
    if not pipeline.is_stage_completed("slides_distillation"):
        console.print(Markdown("## Step 7: Distillation of slides assembly instructions"))

        distillation_prompt = load_prompt(
            Path(ROOT_DIR) / f"src/prompts/slides_distiller/slides_distillation.teacher{stateless}.md",
            slides_deck_plan= final_slides,
            lesson_num=lesson_num,
            module_num=module_num,
        )

        distilled_slides = get_teacher().chat(distillation_prompt)
        # Save final slides with timestamp in slides/ folder
        distilled_path = pipeline.get_final_distilled_output_path(round(time()))
        pipeline.save_stage_output("distilled_slides", distilled_slides, distilled_path)
        console.print(f"✓ Distilled instructions for Final slides saved to: {distilled_path}")
    else:
        console.print(Markdown("## Step 6: ✓ Final slides already exist"))


    console.print(Markdown("## ✓ Slides Generation Completed!"))


def reset_slides_pipeline(lesson_num, module_num, from_stage=None, output_dir=None):
    """Reset slides pipeline from a specific stage or completely"""
    if not output_dir:
        output_dir = Path(ROOT_DIR) / f"data/output/module {module_num:03}"
    pipeline = SlidesPipelineManager(lesson_num, module_num, output_dir)

    if from_stage:
        pipeline.reset_from_stage(from_stage)
        print(f"Slides pipeline reset from stage '{from_stage}' onwards")
    else:
        pipeline.clear_all()
        print("Slides pipeline completely reset")


def show_slides_pipeline_status(lesson_num, module_num, output_folder):
    """Show current slides pipeline status"""
    pipeline = SlidesPipelineManager(lesson_num, module_num, output_folder)
    console = Console()

    console.print(Markdown(f"## Slides Pipeline Status: Module {module_num}, Lesson {lesson_num}"))
    console.print(f"State file: {pipeline.state_file}")
    console.print(f"\nCompleted stages:")

    for stage in SlidesPipelineManager.STAGES:
        status = "✓" if pipeline.is_stage_completed(stage) else "○"
        file_path = pipeline.state.get("stage_files", {}).get(stage, "N/A")
        console.print(f"  {status} {stage}: {file_path}")

    next_stage = pipeline.get_next_stage()
    if next_stage:
        console.print(f"\n**Next stage to run: {next_stage}**")
    else:
        console.print("\n**All stages completed!**")