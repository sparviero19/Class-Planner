Subject: Slide Budget Allocation

Context:
You need to allocate a total of {slide_budget} slides for lesson {lesson_num}.
The lesson topics are: {topics}

Pedagogical Analysis (provided below):
{pedagogical_analysis}

Constraints:
1. **Total slides**: Exactly {slide_budget} slides (excluding title/section divider slides)
2. **Minimum per section**: At least 6 content slides per major topic/section
3. **No fluff**: No bibliography slides, exercise slides, or "thank you" slides
4. **Balance**: Allocation should reflect pedagogical complexity and teaching needs

Task:
Create a detailed slide budget allocation that distributes the {slide_budget} slides across the lesson topics.

Your allocation should:
1. **Rank topics** by importance, complexity, and time needed
2. **Justify allocation** for each topic based on the pedagogical analysis
3. **Break down slide types** for each topic:
   - Introduction/motivation slides
   - Concept explanation slides (text-heavy)
   - Mathematical slides (equation-heavy)
   - Visual explanation slides (diagram-heavy)
   - Example/demonstration slides
   - Summary/transition slides

4. **Ensure balance**:
   - More complex topics get more slides
   - Topics requiring visual build-up get adequate slides for progressive revelation
   - Mathematical topics get enough slides to not cram too many equations
   - Practical topics get sufficient example slides

Output Format:
```markdown
## Slide Budget Allocation Summary
- **Total Content Slides**: {slide_budget}
- **Number of Topics/Sections**: X
- **Average slides per topic**: Y

## Topic Priority Ranking
1. Topic A (justification)
2. Topic B (justification)
3. Topic C (justification)

---

## Detailed Allocation

### Topic 1: [Topic Name] — **X slides**

**Justification**:
[Why this topic gets X slides - reference complexity, teaching approach, cognitive load]

**Slide Breakdown**:
- 1 section introduction
- X concept explanation slides
- Y mathematical derivation slides
- Z visual intuition slides
- N example/demo slides
- 1 summary/transition slide

**Content Notes**:
- [Key points to cover]
- [Pacing considerations]

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
