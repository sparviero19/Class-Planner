import os
import sys
from pathlib import  Path
from config.definitions import ROOT_DIR, load_api_keys
from main_bkp import input_folder
from src.agents import OpenAIAgent, GeminiAgent
from src.pipeline_manager import PipelineManager
from time import time
from rich.markdown import Markdown
from rich.console import Console


def load_materials_paths(input_folder):
    all_materials = [Path(os.path.join(input_folder, file)) for file in os.listdir(input_folder)]
    for material in all_materials:
        if material.suffix == ".txt":
            topics = material
            all_materials.remove(material)
    return all_materials, topics

def extract_topics(topics_file_path):
    topics = []
    with open(topics_file_path, "r") as file:
        for line in file:
            topics.append(line.strip())
    return topics

def extract_module_structure(module_file_path):
    module_structure = {}
    with open(module_file_path, "r") as file:
        # get title
        module_structure["title"] = file.readline().strip()
        for line in file:
            if line.strip() == "":
                continue
            if "lezione" in line.lower():
                # Extract lesson number and title
                parts = line.split(":")
                lesson_num = parts[0].lower().replace("Lezione", "").strip()
                module_structure[str(lesson_num)] = {"title": parts[1].strip(), "topics": []}
            else:
                module_structure[str(lesson_num)]["topics"].append(line.strip())
    return module_structure

def load_prompt(prompt_path, **kwargs):
    with open(prompt_path, "r") as f:
        prompt = f.read()
    prompt = prompt.format(**kwargs)
    return prompt

def save_output(path, text):
    with open(path, "w") as f:
        f.write(text)

def generate_handout(lesson_num, module_num, resume=True, override_files=None):
    """
    Generate handout with checkpoint/resume capability
    
    Args:
        lesson_num: Lesson number
        module_num: Module number
        resume: If True, resume from last checkpoint. If False, start fresh.
        override_files: Dict mapping stage names to file paths for pre-existing files
                       Example: {"summary": "path/to/my_edited_summary.md"}
    """
    input_folder = Path(ROOT_DIR) / f"data/input/module {module_num:03}/Lez {lesson_num:03} materials"
    output_dir = Path(ROOT_DIR) / f"data/output/module {module_num:03}"
    api_keys = load_api_keys()
    console = Console()

    console.print(Markdown("# Hello from class-notes-distiller!"))

    # Initialize pipeline manager
    pipeline = PipelineManager(lesson_num, module_num, output_dir)
    
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

    # General variables
    subject = "Computer Vision"
    language = "Italian"

    # module folder
    m_folder = Path(ROOT_DIR) / f"data/input/module {module_num:03}"

    # Define Teacher (lazy initialization - only when needed)
    teacher = None
    reviewer = None
    editor = None
    
    def get_teacher():
        nonlocal teacher
        if teacher is None:
            system_prompt_T = load_prompt(Path(ROOT_DIR) / "src/prompts/system.teacher.md", subject=subject, language=language)
            teacher = GeminiAgent("T", "gemini-2.5-pro", system_prompt_T, True, None)
        return teacher
    
    def get_reviewer():
        nonlocal reviewer
        if reviewer is None:
            system_prompt_R = load_prompt(Path(ROOT_DIR) / "src/prompts/system.reviewer.md", subject=subject, language=language)
            reviewer = OpenAIAgent("R", "gpt-4o-mini", system_prompt_R, None)
        return reviewer
    
    def get_editor():
        nonlocal editor
        if editor is None:
            system_prompt_E = load_prompt(Path(ROOT_DIR) / "src/prompts/system.editor.md", subject=subject, language=language)
            editor = GeminiAgent("E", "gemini-2.5-flash", system_prompt_E, None)
        return editor

    # Check what stage we're at
    next_stage = pipeline.get_next_stage()
    if next_stage:
        console.print(Markdown(f"**Resuming from stage: {next_stage}**"))
    else:
        console.print(Markdown("**All stages completed!**"))
        return

    # First step: draft the summary from the input materials
    if not pipeline.is_stage_completed("first_draft"):
        console.print(Markdown("## Step 1: Uploading pdf resources and generating first draft"))
        material_paths, topics_file = load_materials_paths(input_folder)
        module_structure = {}
        if os.path.exists(m_folder / "module_topics.md"):
            module_structure = extract_module_structure(m_folder / "module_topics.md")
        topics = extract_topics(topics_file)
        material_files = get_teacher().load_pdfs(material_paths, use_cache=True)
        materials = {m.name: p.name for m, p in zip(material_files, material_paths)}
        instructions = load_prompt(Path(ROOT_DIR) / "src/prompts/summary.teacher.md", topics=topics, subject=subject, language=language, materials=materials, lesson_num=lesson_num)
        if module_structure:
            instructions += f"\n\n##Here I give you the topics for all the lessons in the module: \n{module_structure}"
        
        first_draft = get_teacher().chat(instructions)
        saved_path = pipeline.save_stage_output("first_draft", first_draft)
        console.print(f"✓ First draft saved to: {saved_path}")
    else:
        console.print(Markdown("## Step 1: ✓ First draft already exists (skipping)"))
        first_draft = pipeline.get_stage_output("first_draft")

    # Second step: review the summary with a different model
    if not pipeline.is_stage_completed("review"):
        console.print(Markdown("## Step 2: Reviewing first draft"))
        # Reload materials info for instructions
        material_paths, topics_file = load_materials_paths(input_folder)
        module_structure = {}
        if os.path.exists(m_folder / "module_topics.md"):
            module_structure = extract_module_structure(m_folder / "module_topics.md")
        topics = extract_topics(topics_file)
        
        # Recreate instructions (needed for review context)
        materials_info = {p.name: p.name for p in material_paths}
        instructions = load_prompt(Path(ROOT_DIR) / "src/prompts/summary.teacher.md", topics=topics, subject=subject, language=language, materials=materials_info, lesson_num=lesson_num)
        if module_structure:
            instructions += f"\n\n##Here I give you the topics for all the lessons in the module: \n{module_structure}"
        
        review_instructions = load_prompt(Path(ROOT_DIR) / "src/prompts/review.reviewer.md", summary_instructions=instructions, summary_draft=first_draft, lesson_num=lesson_num)
        review = get_reviewer().chat(review_instructions)
        console.print(Markdown(review))
        saved_path = pipeline.save_stage_output("review", review)
        console.print(f"✓ Review saved to: {saved_path}")
    else:
        console.print(Markdown("## Step 2: ✓ Review already exists (skipping)"))
        review = pipeline.get_stage_output("review")

    # Third step: create the revised summary
    if not pipeline.is_stage_completed("summary"):
        console.print(Markdown("## Step 3: Updating draft based on review"))
        update_instructions = review
        revised_summary = get_teacher().chat(update_instructions)
        saved_path = pipeline.save_stage_output("summary", revised_summary)
        console.print(f"✓ Summary saved to: {saved_path}")
    else:
        console.print(Markdown("## Step 3: ✓ Summary already exists (skipping)"))
        revised_summary = pipeline.get_stage_output("summary") # it is unused since the teacher agent has internal history management

    # Fourth step: Write notes
    if not pipeline.is_stage_completed("handout_draft"):
        console.print(Markdown("## Step 4: Writing Handout"))
        handout_instructions = load_prompt(Path(ROOT_DIR) / f"src/prompts/notes.teacher.md", language=language)
        handout = get_teacher().chat(handout_instructions)
        saved_path = pipeline.save_stage_output("handout_draft", handout)
        console.print(f"✓ Handout draft saved to: {saved_path}")
    else:
        console.print(Markdown("## Step 4: ✓ Handout draft already exists (skipping)"))
        handout = pipeline.get_stage_output("handout_draft")

    # Fifth step: revise the notes for editorial modifications
    if not pipeline.is_stage_completed("editing_instructions"):
        console.print(Markdown("## Step 5: Checking Editorial Constraints"))
        handout_instructions = load_prompt(Path(ROOT_DIR) / f"src/prompts/notes.teacher.md", language=language)
        editing_instructions = load_prompt(Path(ROOT_DIR) / "src/prompts/editing.editor.md", instructions=handout_instructions, handout=handout)
        editing_response = get_editor().chat(editing_instructions)
        saved_path = pipeline.save_stage_output("editing_instructions", editing_response)
        console.print(f"✓ Editing instructions saved to: {saved_path}")
    else:
        console.print(Markdown("## Step 5: ✓ Editing instructions already exist (skipping)"))
        editing_response = pipeline.get_stage_output("editing_instructions")

    # Sixth step: final revision
    if not pipeline.is_stage_completed("final_handout"):
        console.print(Markdown("## Step 6: Updating Final Handout"))
        editorial_corrections = load_prompt(Path(ROOT_DIR) / "src/prompts/final_notes.teacher.md", lesson_num=lesson_num, review=editing_response)
        final_handout = get_teacher().chat(editorial_corrections)
        # Save final handout with timestamp in main output folder
        final_path = output_dir / f"handout_m{module_num:03}_l{lesson_num:03}_{round(time())}.md"
        pipeline.save_stage_output("final_handout", final_handout, final_path)
        console.print(f"✓ Final handout saved to: {final_path}")
    else:
        console.print(Markdown("## Step 6: ✓ Final handout already exists"))

    console.print(Markdown("## ✓ Handout Generation Completed!"))

def clear_cache():
    """Utility function to clear the PDF cache"""
    api_keys = load_api_keys()
    temp_agent = GeminiAgent("temp", "gemini-2.5-flash", "", False, None)
    temp_agent.clear_cache()
    print("Cache cleared successfully!")

def reset_pipeline(lesson_num, module_num, from_stage=None):
    """Reset pipeline from a specific stage or completely"""
    output_dir = Path(ROOT_DIR) / "data/output"
    pipeline = PipelineManager(lesson_num, module_num, output_dir)
    
    if from_stage:
        pipeline.reset_from_stage(from_stage)
        print(f"Pipeline reset from stage '{from_stage}' onwards")
    else:
        pipeline.clear_all()
        print("Pipeline completely reset")

def show_pipeline_status(lesson_num, module_num):
    """Show current pipeline status"""
    output_dir = Path(ROOT_DIR) / "data/output"
    pipeline = PipelineManager(lesson_num, module_num, output_dir)
    console = Console()
    
    console.print(Markdown(f"## Pipeline Status: Module {module_num}, Lesson {lesson_num}"))
    console.print(f"State file: {pipeline.state_file}")
    console.print(f"\nCompleted stages:")
    
    for stage in PipelineManager.STAGES:
        status = "✓" if pipeline.is_stage_completed(stage) else "○"
        file_path = pipeline.state.get("stage_files", {}).get(stage, "N/A")
        console.print(f"  {status} {stage}: {file_path}")
    
    next_stage = pipeline.get_next_stage()
    if next_stage:
        console.print(f"\n**Next stage to run: {next_stage}**")
    else:
        console.print("\n**All stages completed!**")

def main():
    module_num = 3
    lesson_num = 5
    
    # Show current status
    # show_pipeline_status(lesson_num, module_num)
    
    # Reset from a specific stage if needed
    # reset_pipeline(lesson_num, module_num, from_stage="summary")
    
    # Run the pipeline (will resume from last checkpoint)
    generate_handout(lesson_num, module_num, resume=True)

if __name__ == "__main__":
    main()
