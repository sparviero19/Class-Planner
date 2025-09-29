import os
from pathlib import  Path
from config.definitions import ROOT_DIR, load_api_keys
from src.agents import OpenAIAgent, GeminiAgent
from time import time
from rich.markdown import Markdown
from rich.console import Console

input_folder = Path(ROOT_DIR) / "data/input/Lez 09 materials"

def load_materials_paths(input_folder):
    return [Path(os.path.join(input_folder, file)) for file in os.listdir(input_folder)]

def load_prompt(prompt_path, **kwargs):
    with open(prompt_path, "r") as f:
        prompt = f.read()
    prompt = prompt.format(**kwargs)
    return prompt

def save_output(path, text):
    with open(path, "w") as f:
        f.write(text)


def main():
    api_keys = load_api_keys()
    console = Console()

    console.print(Markdown("# Hello from class-notes-distiller!"))

    # General variables
    lesson_num = 9
    subject = "Computer Vision"
    language = "Italian"

    # materials
    m_folder = Path(ROOT_DIR) / f"data/input/Lez {lesson_num:03} materials"

    # Define Teacher
    system_prompt_T = load_prompt(Path(ROOT_DIR) / "src/prompts/system.teacher.md", subject=subject, language=language)
    teacher = GeminiAgent("T", "gemini-2.5-flash", system_prompt_T, True, None)

    # define reviewer
    system_prompt_R = load_prompt(Path(ROOT_DIR) / "src/prompts/system.reviewer.md", subject=subject, language=language)
    reviewer = OpenAIAgent("R", "gpt-4o-mini", system_prompt_R, None)

    # define the editor
    system_prompt_E = load_prompt(Path(ROOT_DIR) / "src/prompts/system.editor.md", subject=subject, language=language)
    editor = GeminiAgent("E", "gemini-2.5-flash", system_prompt_E, None)

    # First step: draft the summary from the input materials
    console.print(Markdown("## Uploading pdf resources"))
    material_paths = load_materials_paths(input_folder)
    material_files = teacher.load_pdfs(material_paths)
    materials = {m.name: p.name for m, p in zip(material_files, material_paths)}
    instructions = load_prompt(Path(ROOT_DIR) / "src/prompts/summary.teacher.md", subject=subject, language=language, materials=materials, lesson_num=lesson_num)
    console.print(Markdown("## Instructions"))
    # console.print(Markdown(instructions))
    first_draft = teacher.chat(instructions)
    save_output(Path(ROOT_DIR) / f"data/output/intermediate/first_draft_{round(time())}.md", first_draft)

    # Second step: review the summary with a different model
    console.print(Markdown("## Reviewing first draft"))
    review_instructions = load_prompt(Path(ROOT_DIR) / "src/prompts/review.reviewer.md", summary_instructions=instructions, summary_draft=first_draft)
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
    editorial_corrections = editing_response
    final_handout = teacher.chat(editorial_corrections)
    save_output(Path(ROOT_DIR) / f"data/output/handout_{lesson_num:03}_{round(time())}.md", final_handout)

    console.print(Markdown("## Handout Generation Completed!"))



if __name__ == "__main__":
    main()
