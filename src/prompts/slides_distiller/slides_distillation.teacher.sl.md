Context:
The following is the detailed plan for the slides of a lesson:
{slides_deck_plan}

The plan has a lot of useful information, but is difficult to read. 

Task:
Distill the slides structure, removing the markdown modifiers (e.g. for italics of for bold text) for the content, except for:
1. lists and numbered lists
2. equations: in particular, use the "$$" delimiter instead of the single one "$". 
leaving only the information needed to assemble the slide deck. 

The format should be:
```markdown

# Lesson {lesson_num}, Module {module_num}:

## Slide 1: [Title]
**Section**: [Topic/Section Name]

**Content**:
[Complete, polished slide text]
- [Well-formatted bullet points]
- [Clear explanations]
- [Proper equation formatting if applicable]

**Visuals**:
- **Primary**: [Visual ID or NEW VISUAL] - [Description] - [Source]
- **Secondary** (if applicable): [Additional visual details]

---
---

## Slide 2: [Title]
**Section**: [Topic/Section Name]

[...complete slide specification...]

---

[...continue for ALL slides...]

---

## Slide N: [Final Slide Title]
[...final slide content...]

---

END OF SLIDE DECK
```
## Important Notes:
- Don't modify the original slide content and structure, just distill the schema following the structure above.
- Remember to remove formatting modifiers from the slide content, except for equations.

Begin creating the distilled slide deck: