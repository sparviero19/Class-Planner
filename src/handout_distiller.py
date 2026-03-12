import os
from pathlib import Path
from config.definitions import ROOT_DIR, load_api_keys
from src.agents import OpenAIAgent, GeminiAgent, AnthropicAgent, OllamaAgent
from src.pipeline_manager import HandoutPipelineManager
from time import time
from rich.markdown import Markdown
from rich.console import Console
import re


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
            if re.search(r"\b" + re.escape("lezione") + r"\b", line.lower()):
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


def generate_handout(config, lesson_num, module_num, resume=True, override_files=None, input_folder=None, output_folder=None,):
    """
    Generate handout with checkpoint/resume capability

    Args:
        config: Configuration object from config_loader
        lesson_num: Lesson number
        module_num: Module number
        resume: If True, resume from last checkpoint. If False, start fresh.
        override_files: Dict mapping stage names to file paths for pre-existing files
                       Example: {"summary": "path/to/my_edited_summary.md"}
    """
    # Extract from config
    subject = config.course.subject
    language = config.course.language
    subfolder_name = config.course.subfolder_name

    if not input_folder:
        input_folder = Path(ROOT_DIR) / f"data/input/{subfolder_name}/module {module_num:03}/Lez {lesson_num:03} materials"
    if not output_folder:
        output_folder = Path(ROOT_DIR) / f"data/output/{subfolder_name}/module {module_num:03}"

    stateless = ".sl"
    api_keys = load_api_keys()
    console = Console()

    console.print(Markdown("# Hello from class-notes-distiller!"))

    # Initialize pipeline manager
    pipeline = HandoutPipelineManager(lesson_num, module_num, output_folder)

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

    # module folder
    m_folder = Path(ROOT_DIR) / f"data/input/module {module_num:03}"

    # Define Teacher (lazy initialization - only when needed)
    teacher = None
    reviewer = None
    editor = None

    def get_teacher(manage_history=True):
        nonlocal teacher
        if teacher is None:
            system_prompt_T = load_prompt(Path(ROOT_DIR) / "src/prompts/system.teacher.md", subject=subject,
                                          language=language)
            teacher_config = config.get_agent_config('teacher')
            agent_type = teacher_config.type
            agent_model = teacher_config.models[agent_type]
            if agent_type == "ollama":
                teacher = OllamaAgent("T", agent_model, system_prompt_T)
            elif agent_type == "gemini":
                teacher = GeminiAgent("T", agent_model, system_prompt_T, None,
                                  api_key=api_keys['google'])
            elif agent_type == "anthropic":
                teacher = AnthropicAgent("T", agent_model, system_prompt_T, None,
                                  api_key=api_keys['anthropic'])
            elif agent_type == "openai":
                teacher = OpenAIAgent("T", agent_model, system_prompt_T, None,
                                  api_key=api_keys['openai'])
            else:
                raise ValueError(f"The chosen model type ({agent_type}) for the teacher agent is not supported!")
        return teacher

    def get_reviewer():
        nonlocal reviewer
        if reviewer is None:
            system_prompt_R = load_prompt(Path(ROOT_DIR) / "src/prompts/system.reviewer.md", subject=subject,
                                          language=language)
            reviewer_config = config.get_agent_config('reviewer')

            agent_type = reviewer_config.type
            agent_model = reviewer_config.models[agent_type]
            if agent_type == "ollama":
                reviewer = OllamaAgent("R", agent_model, system_prompt_R)
            elif agent_type == "gemini":
                reviewer = GeminiAgent("R", agent_model, system_prompt_R, None, api_key=api_keys['google'])
            elif agent_type == "anthropic":
                reviewer = AnthropicAgent("R", agent_model, system_prompt_R, None, api_key=api_keys['anthropic'])
            elif agent_type == "openai":
                reviewer = OpenAIAgent("R", agent_model, system_prompt_R, None, api_key=api_keys['openai'])
            else:
                raise ValueError(f"The chosen model type ({agent_type}) for the reviewer agent is not supported!")

        return reviewer

    def get_editor():
        nonlocal editor
        if editor is None:
            system_prompt_E = load_prompt(Path(ROOT_DIR) / "src/prompts/system.editor.md", subject=subject,
                                          language=language)
            editor_config = config.get_agent_config('editor')
            editor = OpenAIAgent("E", editor_config.get_model(), system_prompt_E, None)

            agent_type = editor_config.type
            agent_model = editor_config.models[agent_type]
            if agent_type == "ollama":
                editor = OllamaAgent("E", agent_model, system_prompt_E)
            elif agent_type == "gemini":
                editor = GeminiAgent("E", agent_model, system_prompt_E, None, api_key=api_keys['google'])
            elif agent_type == "anthropic":
                editor = AnthropicAgent("E", agent_model, system_prompt_E, None, api_key=api_keys['anthropic'])
            elif agent_type == "openai":
                editor = OpenAIAgent("E", agent_model, system_prompt_E, None, api_key=api_keys['openai'])
            else:
                raise ValueError(f"The chosen model type ({agent_type}) for the editor agent is not supported!")

        return editor

    # Check what stage we're at
    next_stage = pipeline.get_next_stage()
    if next_stage:
        console.print(Markdown(f"**Resuming from stage: {next_stage}**"))
    else:
        console.print(Markdown("**All stages completed!**"))
        return

    console.print(Markdown("## Step 0: Uploading pdf resources"))
    material_paths, topics_file = load_materials_paths(input_folder)
    module_structure = {}
    if os.path.exists(m_folder / "module_topics.md"):
        module_structure = extract_module_structure(m_folder / "module_topics.md")
    topics = extract_topics(topics_file)
    material_files = get_teacher().load_pdfs(material_paths, use_cache=True)
    materials = {m.name: p.name for m, p in zip(material_files, material_paths)}
    materials_info = {p.name: p.name for p in material_paths}

    # First step: draft the summary from the input materials
    if not pipeline.is_stage_completed("first_draft"):
        console.print(Markdown("## Step 1: Generating first draft"))
        summary_instructions = load_prompt(Path(ROOT_DIR) / f"src/prompts/summary.teacher{stateless}.md", topics=topics,
                                           subject=subject, language=language, materials=materials,
                                           lesson_num=lesson_num)
        if module_structure:
            summary_instructions += f"\n\n##Here I give you the topics for all the lessons in the module: \n{module_structure}"

        first_draft = get_teacher().chat(summary_instructions)
        saved_path = pipeline.save_stage_output("first_draft", first_draft)
        console.print(f"✓ First draft saved to: {saved_path}")
    else:
        console.print(Markdown("## Step 1: ✓ First draft already exists (skipping)"))
        summary_instructions = load_prompt(Path(ROOT_DIR) / f"src/prompts/summary.teacher{stateless}.md", topics=topics,
                                           subject=subject, language=language, materials=materials,
                                           lesson_num=lesson_num)
        first_draft = pipeline.get_stage_output("first_draft")

    # Second step: review the summary with a different model
    if not pipeline.is_stage_completed("review"):
        console.print(Markdown("## Step 2: Reviewing first draft"))
        # Reload materials info for instructions
        # material_paths, topics_file = load_materials_paths(input_folder)
        # module_structure = {}
        # if os.path.exists(m_folder / "module_topics.md"):
        #     module_structure = extract_module_structure(m_folder / "module_topics.md")
        # topics = extract_topics(topics_file)

        # # Recreate instructions (needed for review context)

        # summarY_instructions = load_prompt(Path(ROOT_DIR) / "src/prompts/summary.teacher.md", topics=topics, subject=subject, language=language, materials=materials_info, lesson_num=lesson_num)
        # if module_structure:
        #     summarY_instructions += f"\n\n##Here I give you the topics for all the lessons in the module: \n{module_structure}"

        review_instructions = load_prompt(Path(ROOT_DIR) / "src/prompts/review.reviewer.md",
                                          summary_instructions=summary_instructions, summary_draft=first_draft,
                                          lesson_num=lesson_num)
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
        update_instructions = load_prompt(Path(ROOT_DIR) / f"src/prompts/review_summary.teacher{stateless}.md",
                                          instructions=summary_instructions, summary=first_draft, review=review)
        revised_summary = get_teacher().chat(update_instructions)
        saved_path = pipeline.save_stage_output("summary", revised_summary)
        console.print(f"✓ Summary saved to: {saved_path}")
    else:
        console.print(Markdown("## Step 3: ✓ Summary already exists (skipping)"))
        revised_summary = pipeline.get_stage_output(
            "summary")  # it is unused since the teacher agent has internal history management

    # Fourth step: Write notes
    if not pipeline.is_stage_completed("handout_draft"):
        console.print(Markdown("## Step 4: Writing Handout"))
        handout_instructions = load_prompt(Path(ROOT_DIR) / f"src/prompts/notes.teacher{stateless}.md",
                                           lesson_num=lesson_num, summary=revised_summary, materials=materials,
                                           language=language, summary_instructions=summary_instructions)
        console.print(Markdown(handout_instructions))
        handout = get_teacher().chat(handout_instructions)
        saved_path = pipeline.save_stage_output("handout_draft", handout)
        console.print(f"✓ Handout draft saved to: {saved_path}")
    else:
        handout_instructions = load_prompt(Path(ROOT_DIR) / f"src/prompts/notes.teacher{stateless}.md",
                                           lesson_num=lesson_num, summary=revised_summary, materials=materials,
                                           language=language, summary_instructions=summary_instructions)
        console.print(Markdown("## Step 4: ✓ Handout draft already exists (skipping)"))
        handout = pipeline.get_stage_output("handout_draft")

    # Fifth step: revise the notes for editorial modifications
    if not pipeline.is_stage_completed("editing_instructions"):
        console.print(Markdown("## Step 5: Checking Editorial Constraints"))
        handout_instructions = load_prompt(Path(ROOT_DIR) / f"src/prompts/notes.teacher{stateless}.md",
                                           lesson_num=lesson_num, summary=revised_summary, materials=materials,
                                           language=language, summary_instructions=summary_instructions)
        editing_instructions = load_prompt(Path(ROOT_DIR) / "src/prompts/editing.editor.md",
                                           instructions=handout_instructions, handout=handout)
        editing_response = get_editor().chat(editing_instructions)
        saved_path = pipeline.save_stage_output("editing_instructions", editing_response)
        console.print(f"✓ Editing instructions saved to: {saved_path}")
    else:
        console.print(Markdown("## Step 5: ✓ Editing instructions already exist (skipping)"))
        editing_response = pipeline.get_stage_output("editing_instructions")

    # Sixth step: final revision
    if not pipeline.is_stage_completed("final_handout"):
        console.print(Markdown("## Step 6: Updating Final Handout"))
        editorial_corrections = load_prompt(Path(ROOT_DIR) / f"src/prompts/final_notes.teacher{stateless}.md",
                                            lesson_num=lesson_num, handout_draft=handout, review=editing_response,
                                            summary_instructions=summary_instructions,
                                            handout_instructions=handout_instructions)
        console.print(Markdown(editorial_corrections))
        final_handout = get_teacher().chat(editorial_corrections)
        # Save final handout with timestamp in handouts/ folder
        final_path = pipeline.get_final_output_path(round(time()))
        pipeline.save_stage_output("final_handout", final_handout, final_path)
        console.print(f"✓ Final handout saved to: {final_path}")
    else:
        console.print(Markdown("## Step 6: ✓ Final handout already exists"))

    console.print(Markdown("## ✓ Handout Generation Completed!"))


def clear_cache():
    """Utility function to clear the PDF cache"""
    from config.config_loader import get_config
    api_keys = load_api_keys()
    config = get_config()
    teacher_config = config.get_agent_config('teacher')
    temp_agent = GeminiAgent("temp", teacher_config.get_model(), "", False, None, api_key=api_keys['google'])
    temp_agent.clear_cache()
    print("Cache cleared successfully!")


def reset_pipeline(lesson_num, module_num, from_stage=None, output_dir=None):
    """Reset pipeline from a specific stage or completely"""
    if not output_dir:
        output_dir = Path(ROOT_DIR) / "data/output"
    pipeline = HandoutPipelineManager(lesson_num, module_num, output_dir)

    if from_stage:
        pipeline.reset_from_stage(from_stage)
        print(f"Pipeline reset from stage '{from_stage}' onwards")
    else:
        pipeline.clear_all()
        print("Pipeline completely reset")


def show_pipeline_status(lesson_num, module_num, pipeline_status_folder):
    """Show current pipeline status"""
    pipeline = HandoutPipelineManager(lesson_num, module_num, pipeline_status_folder)
    console = Console()

    console.print(Markdown(f"## Pipeline Status: Module {module_num}, Lesson {lesson_num}"))
    console.print(f"State file: {pipeline.state_file}")
    console.print(f"\nCompleted stages:")

    for stage in HandoutPipelineManager.STAGES:
        status = "✓" if pipeline.is_stage_completed(stage) else "○"
        file_path = pipeline.state.get("stage_files", {}).get(stage, "N/A")
        console.print(f"  {status} {stage}: {file_path}")

    next_stage = pipeline.get_next_stage()
    if next_stage:
        console.print(f"\n**Next stage to run: {next_stage}**")
    else:
        console.print("\n**All stages completed!**")
