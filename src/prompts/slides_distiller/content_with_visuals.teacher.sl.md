Subject: Generate Slide Content with Visual Assignments

Context:
You are now creating the actual slide deck for lesson {lesson_num}.

You have the following resources:

**Slide Budget Allocation**:
{slide_budget_allocation}

**Visual Inventory**:
{visual_inventory}

**Handout Content**:
{handout}

**Pedagogical Analysis**:
{pedagogical_analysis}

**Reference Materials**:
{materials}

Task:
Generate complete slide content following the budget allocation. For EACH slide, you must provide:

1. **Slide number** (sequential)
2. **Section/Topic** (which topic this slide belongs to)
3. **Slide title** (concise, descriptive)
4. **Content**:
   - Full text (not just bullet points - include explanations where needed)
   - Equations (if mathematical slide)
   - Key points or bullet lists (where appropriate)
   - Progressive reveals or build-up steps (if applicable)

5. **Visual Assignment** (CRITICAL):
   - **Primary visual**: Reference specific visual from inventory using Visual ID
     - Format: `[Visual ID] - Brief description - Source`
     - Example: `[T1-D1] - Two-camera stereo configuration - Szelinski p.470`
   - **Secondary visuals** (if needed): Additional supporting visuals
   - **Visual placement**: Where on slide (main, inset, background, side-by-side)
   - **Annotations needed**: Any labels, highlights, or modifications
   - **If no inventory visual exists**: Describe the needed visual in detail or suggest external resource

6. **Teaching Notes**: How to present this slide (transitions, emphasis points, timing)

Guidelines:
- Follow the slide budget allocation strictly
- Assign visuals from inventory whenever possible
- For gaps in inventory, provide detailed visual specifications
- Balance text density - not too sparse, not too crowded
- Ensure mathematical slides are not overloaded with equations
- Use progressive build-up for complex concepts
- Create natural transitions between topics
- Keep pedagogical flow in mind

Output Format:

```markdown
# Lesson {lesson_num} Slide Deck

---

## Slide 1: [Title]
**Section**: Topic 1 - [Topic Name]

**Content**:
[Full slide text content]
- Key point 1
- Key point 2
- [Additional explanatory text as needed]

**Visuals**:
- **Primary**: [T1-D1] - Two-camera stereo configuration - Szelinski 467-481.pdf, p.470
  - Placement: Center, full-width
  - Annotations: None needed
- **Secondary**: None

**Teaching Notes**:
Introduce the geometric setup before diving into mathematics. Emphasize the baseline and how it relates to depth perception.

---

## Slide 2: [Title]
**Section**: Topic 1 - [Topic Name]

**Content**:
[Content for slide 2...]

**Visuals**:
- **Primary**: [T1-E1] - Essential matrix equation - Szelinski p.472
  - Placement: Top center
  - Annotations: Highlight the transpose operator
- **Secondary**: [T1-D2] - Epipolar lines diagram - 15_Stereo3D_21.pdf, slide 8
  - Placement: Below equation
  - Annotations: Add color coding for left/right images

**Teaching Notes**:
Connect the geometric intuition from previous slide to this mathematical formulation. Show how the equation enforces the geometric constraint.

---

[...continue for ALL slides according to budget allocation...]

---

## Slide N: [Final Topic Summary]
**Section**: Topic 3 - [Topic Name]

**Content**:
[Summary content...]

**Visuals**:
- **Primary**: [NEW VISUAL NEEDED]
  - Description: Comparison table showing advantages/disadvantages of different stereo algorithms
  - Suggested creation: Create simple 3-column table with rows for accuracy, speed, complexity
  - Alternative: Could adapt table from reference materials if found

**Teaching Notes**:
Wrap up the topic with comparative perspective. Prepare students for practical considerations.

---

## Slide Content Summary
- Total slides created: X
- Slides per topic:
  - Topic 1: Y slides
  - Topic 2: Z slides
  - [...]
- Visual assignments:
  - From inventory: X slides
  - New visuals needed: Y slides
  - Text-only: Z slides

## New Visuals Required
List all new visuals that need to be created:
1. [Visual description] - For Slide X
2. [Visual description] - For Slide Y
[...]
```

Important:
- **Every slide** must have specific content, not just an outline
- **Visual assignments** must be concrete (use Visual IDs from inventory)
- **Balance content types** across the deck
- **Follow the budget** - don't exceed allocated slides per topic
- **Maintain flow** - ensure logical progression between slides

Begin creating the slide deck:
