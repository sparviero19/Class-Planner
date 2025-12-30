Context:
You need to allocate a total of {slide_budget} slides for lesson {lesson_num}.
The lesson topics are: {topics}

The titles of the presentation's sections are exactly the same as the lesson topics.

A detailed pedagogical Analysis has already been designed, and it's provided below:
{pedagogical_analysis}

Constraints:
1. Exactly {slide_budget} slides (excluding title/section divider slides)
2. At least 6 content slides per major topic/section
3. No bibliography slides, exercise slides, or "thank you" slides
4. Allocation should reflect pedagogical complexity and teaching needs

Task:
Create a detailed slide budget allocation that distributes the {slide_budget} slides across the lesson topics. Give a 
brief justification for each topic's allocation.

Output Format:
```markdown
## Slide Budget Allocation Summary
- **Total Content Slides**: {slide_budget}
- **Number of Topics/Sections**: X

## Detailed Allocation

### Topic 1: [Topic Name] — **X slides**

**Justification**:
[Why this topic gets X slides - reference complexity, teaching approach, cognitive load]
---

### Topic 2: [Topic Name] — **Y slides**
[...repeat for each topic...]

---

## Balance Check
- Total allocated: [sum of all allocations]
- Target: {slide_budget}
- All sections meet minimum (6 slides): ✓/✗
- Distribution reflects pedagogical needs: ✓/✗
```

Important:
- Make sure the total adds up to exactly {slide_budget}
- If topics naturally need more/fewer slides, adjust intelligently
- Prioritize pedagogical effectiveness over equal distribution
- Be specific about what each group of slides will accomplish

Begin your allocation:
