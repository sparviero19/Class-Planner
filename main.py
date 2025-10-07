import os
import sys
from pathlib import  Path
from config.definitions import ROOT_DIR, load_api_keys
from main_bkp import input_folder
from src.agents import OpenAIAgent, GeminiAgent
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

def generate_handout(lesson_num, module_num):
    if module_num is None:
        module_num = 0
        input_folder = Path(ROOT_DIR) / f"data/input/Lez {lesson_num:03} materials"
    else:
        input_folder = Path(ROOT_DIR) / f"data/input/module {module_num:03}/Lez {lesson_num:03} materials"
    api_keys = load_api_keys()
    console = Console()

    console.print(Markdown("# Hello from class-notes-distiller!"))

    # General variables
    subject = "Computer Vision"
    language = "Italian"

    # module folder
    m_folder = Path(ROOT_DIR) / f"data/input/module {module_num:03}"

    # Define Teacher
    system_prompt_T = load_prompt(Path(ROOT_DIR) / "src/prompts/system.teacher.md", subject=subject, language=language)
    teacher = GeminiAgent("T", "gemini-2.5-pro", system_prompt_T, True, None)

    # define reviewer
    system_prompt_R = load_prompt(Path(ROOT_DIR) / "src/prompts/system.reviewer.md", subject=subject, language=language)
    reviewer = OpenAIAgent("R", "gpt-4o-mini", system_prompt_R, None)

    # define the editor
    system_prompt_E = load_prompt(Path(ROOT_DIR) / "src/prompts/system.editor.md", subject=subject, language=language)
    editor = GeminiAgent("E", "gemini-2.5-flash", system_prompt_E, None)

    # First step: draft the summary from the input materials
    console.print(Markdown("## Uploading pdf resources"))
    material_paths, topics_file = load_materials_paths(input_folder)
    module_structure = {}
    if os.path.exists(m_folder / "module_topics.md"):
        module_structure = extract_module_structure(m_folder / "module_topics.md")
    else:
        module_structure["title"] = "No tile"
    topics = extract_topics(topics_file)
    material_files = teacher.load_pdfs(material_paths, use_cache=True)
    materials = {m.name: p.name for m, p in zip(material_files, material_paths)}
    instructions = load_prompt(Path(ROOT_DIR) / "src/prompts/summary.teacher.md", topics=topics, subject=subject,
                               language=language, materials=materials, lesson_num=lesson_num)
    # if module_structure:
    #     instructions += f"\n\n##Here I give you the topics for all the lessons in the module: \n{module_structure}"
    console.print(Markdown("## Instructions"))
    console.print(Markdown(instructions))
    first_draft = teacher.chat(instructions)
    save_output(Path(ROOT_DIR) / f"data/output/intermediate/first_draft_{round(time())}.md", first_draft)

    # Second step: review the summary with a different model
    console.print(Markdown("## Reviewing first draft"))
    review_instructions = load_prompt(Path(ROOT_DIR) / "src/prompts/review.reviewer.md", summary_instructions=instructions, summary_draft=first_draft, lesson_num=lesson_num)
    review = reviewer.chat(review_instructions)
    console.print(Markdown(review))
    save_output(Path(ROOT_DIR) / f"data/output/intermediate/review_{round(time())}.md", review)

    # Third step: create the draft of the notes
    console.print(Markdown("## Updating draft"))
    update_instructions = review
    revised_summary = teacher.chat(update_instructions)
    save_output(Path(ROOT_DIR) / f"data/output/intermediate/summary_{round(time())}.md", revised_summary)

    # Fourth step: Write notes
    console.print(Markdown("## Writing Handout"))
    handout_instructions = load_prompt(Path(ROOT_DIR) / f"src/prompts/notes.teacher.md", language=language)
    handout = teacher.chat(handout_instructions)
    save_output(Path(ROOT_DIR) / f"data/output/intermediate/handout_draft_{round(time())}.md", handout)

    # Fifth step: revise the notes for editorial modifications
    console.print(Markdown("## Checking Editorial Constraints"))
    editing_instructions = load_prompt(Path(ROOT_DIR) / "src/prompts/editing.editor.md", instructions=handout_instructions, handout=handout)
    editing_response = editor.chat(editing_instructions)
    save_output(Path(ROOT_DIR) / f"data/output/intermediate/editing_instructions_{round(time())}.md", editing_response)

    # Sixth step: final revision
    console.print(Markdown("## Updating Final Handout"))
    editorial_corrections = load_prompt(Path(ROOT_DIR) / "src/prompts/final_notes.teacher.md", review=review, lesson_num=lesson_num)
    final_handout = teacher.chat(editorial_corrections)
    if not os.path.exists(Path(ROOT_DIR) / f"data/output/module {module_num:03}/"):
        os.makedirs(Path(ROOT_DIR) / f"data/output/module {module_num:03}/")
    save_output(Path(ROOT_DIR) / f"data/output/module {module_num:03}/handout_{lesson_num:03}_{round(time())}.md", final_handout)

    console.print(Markdown("## Handout Generation Completed!"))

def clear_cache():
    """Utility function to clear the PDF cache"""
    api_keys = load_api_keys()
    temp_agent = GeminiAgent("temp", "gemini-2.5-flash", "", False, None)
    temp_agent.clear_cache()
    print("Cache cleared successfully!")

def main():
    # Uncomment the next line if you want to clear cache
    # clear_cache()
    
    module_num = 3 # a number or None
    lesson_num = 2
    generate_handout(lesson_num, module_num)

if __name__ == "__main__":
    main()
