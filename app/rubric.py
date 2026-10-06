"""Rubric prompts and the JSON schema the judge must return.

Every rubric version is kept so benchmark results stay reproducible: a result
row records which version produced it, and `python -m app.cli sweep` can
compare versions on the same dataset. Add new versions; don't edit old ones.
"""

from __future__ import annotations

from typing import Optional

STATUS = ["pass", "fail", "unsure"]

_requirement_check = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": STATUS},
        "evidence": {"type": "string"},
    },
    "required": ["status", "evidence"],
    "additionalProperties": False,
}

# Field order matters: the model fills the schema top to bottom, so all the
# evidence is written before it commits to a verdict.
JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "originals_summary": {"type": "string"},
        "visible_differences": {"type": "array", "items": {"type": "string"}},
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "category": {"type": "string",
                                 "enum": ["change", "prompt", "preserve", "anatomy", "artifact", "cohesion",
                                          "quality"]},
                    "severity": {"type": "string", "enum": ["critical", "major", "minor"]},
                    "description": {"type": "string"},
                    "a": _requirement_check,
                    "b": _requirement_check,
                },
                "required": ["id", "category", "severity", "description", "a", "b"],
                "additionalProperties": False,
            },
        },
        "decisive_difference": {"type": "string"},
        "reasoning": {"type": "string"},
        "verdict": {"type": "string", "enum": ["A", "B"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": [
        "originals_summary",
        "visible_differences",
        "requirements",
        "decisive_difference",
        "reasoning",
        "verdict",
        "confidence",
    ],
    "additionalProperties": False,
}


_V1 = """\
You are an exacting judge of image-generation and image-editing results.

You will be given a PROMPT, one or more ORIGINAL images (the inputs the prompt
was applied to), and two candidate outputs, RESULT A and RESULT B. Exactly one
of the two results is the correct output for the prompt. Your job is to find
which one, by checking each result against concrete requirements rather than
by overall impression.

Work through these steps and record them in the JSON fields.

1. originals_summary - Describe each original precisely: subjects, count of
   key objects, colours, positions, any text (exact spelling), background,
   framing and aspect ratio. These are what "preserve" requirements refer to.

2. visible_differences - List the concrete visual differences between RESULT A
   and RESULT B (what is present, absent, a different colour, count, position,
   wording, crop, style). Be literal; do not judge yet.

3. requirements - Build a checklist, then mark every item for A and for B.
   - change: split the prompt into atomic requirements, one per instruction.
     "Make the two cats wear red hats and put them on a beach" is four:
     both cats have hats, hats are red, setting is a beach, there are still
     two cats. Include exact counts, colours, positions, sizes, text content
     and spelling, style, and which object an edit applies to. Include clear
     implicit requirements (e.g. "remove the man" implies the area behind him
     is plausibly filled in).
   - preserve: for edits, everything in the originals the prompt did not ask
     to change should stay the same - identity and faces, other objects,
     background, text, pose, composition, aspect ratio. When several originals
     are given, the elements the prompt refers to from each original must be
     used. For pure generation with no originals, skip this category.
   - quality: obvious defects that would make an output unacceptable -
     garbled or misspelled text, broken anatomy, duplicated or melted objects,
     visible seams, an edit that was not applied at all (output identical to
     an original).
   Severity: critical = the core instruction or something that changes what
   the image is; major = a clearly required detail; minor = a nitpick.
   Status: pass, fail, or unsure (genuinely cannot tell from the image).
   Evidence must name what you actually see in that specific image
   ("A: hat is orange-red, on the left cat only"), not restate the
   requirement.

4. decisive_difference - The single requirement where one result passes and
   the other fails that best explains which is correct, stated in one or two
   sentences that name both results.

5. reasoning - Brief: why the decisive difference outweighs anything that
   favours the other result.

6. verdict and confidence.
   - Instruction compliance beats aesthetics. A prettier, more detailed or
     more realistic image that misses a requirement loses to a plainer one
     that meets it.
   - Compare failures by severity first, then by count.
   - The wrong result is often wrong in one small, specific way: an off-by-one
     count, a misspelled word, the edit applied to the wrong object, an
     unrequested change to the subject's face, the original returned
     unchanged. Look closely for these before deciding.
   - Image order carries no information; A and B are equally likely.
   - confidence: high = a clear failure in one result that the other does not
     have; medium = the decisive difference is real but subtle; low = you
     could not find a difference that settles it, or you are relying on taste.
     Use low honestly - a low-confidence answer is sent to a human.
"""

# v2: identical to v1 but asks the model to zoom in on likely failure points
# and to re-verify the decisive difference before committing. Candidate only;
# keep it as the default only if the benchmark says it helps.
_V2 = _V1 + """
7. Before you finalise, re-examine the specific region of both images that the
   decisive difference is about and confirm your reading of it. If on a second
   look the difference is not there, or both results share the failure, choose
   a different decisive difference or lower your confidence.
"""

# v3: the project's own labelling guidelines (holistic weighing by severity and visibility, local vs
# global edits, anatomy, artifacts, cohesion/style match) plus the image wiki's diagnosis checks
# (how-to-look, diagnosing-images, stock-look). Comes with difference maps for local edits.
_V3 = """\
You judge image-generation and image-editing results the way an experienced human labeller does.

You get a PROMPT, zero to four ORIGINAL images (the inputs the prompt was applied to), and two
candidate outputs, RESULT A and RESULT B. One of them is the better answer. Decide which one the
person who sent the request would be happier to receive.

HOW TO WEIGH - the most important part
- There is no strict priority order: no criterion automatically beats a later one. Look at
  everything, then judge each image as a whole.
- Prompt compliance is where you start, not where you stop. Check it first but never decide on it
  alone. Whether both images follow the prompt, both fail it, or one is closer, still go through
  every other criterion before choosing.
- Weigh how bad each problem is, not which criterion it belongs to. A slightly wrong shade of blue
  and a hand with six fingers are both "failures" but they are not remotely equal. For every flaw
  ask: how serious is it, how visible is it, and how much would it bother the person who asked?
- Trust what catches your eye first. Look at both images the way an ordinary viewer would - a few
  seconds, no zooming. The problem you notice immediately is usually the one that decides the
  task. Inspect closely afterwards to confirm what you saw, not to hunt for flaws nobody would
  ever notice.
- A perfect match to the prompt does not rescue a broken image. Three legs, melted text, a face
  with two noses make an image unusable. If the other image misses a smaller part of the request
  but is otherwise clean, it is usually the better answer.
- The reverse is equally true: a flawless image that ignores the request is not a good answer. If
  one image misses the main point of what was asked, its sharpness and beauty do not save it.

Examples
- A does exactly what the prompt asked, but the character now has three legs. B made the
  background a slightly different green than requested and is otherwise clean. -> B. A wrong
  shade is a small, forgivable miss; three legs ruins the image.
- A is sharp, beautiful and anatomically perfect, but the prompt asked to remove the car and the
  car is still there. B removed the car, with slightly visible edges around the gap. -> B. It did
  the thing that was actually asked for.
- Both follow the prompt well and neither has anatomy problems. -> Keep going: cohesion,
  artifacts, and finally overall quality and style match.

THE CRITERIA
1. Prompt compliance - every requirement: objects, counts, colours, positions, sizes, text (exact
   spelling), style, and which object an edit applies to.
   - Local edits (remove / add / replace an element, recolour one object, retouch an area): the
     result should ideally change only what was specified. Everything else stays as it was - no
     quality "improvements", no added or removed elements, same identity, framing and aspect
     ratio. When DIFFERENCE MAPS are given (black = unchanged, brighter = more changed), use them
     to spot changes outside the requested area.
   - Global edits (restyle, change season or time of day, redraw in another medium, change the
     camera angle, generate a new image from a reference): the result is NOT supposed to match the
     source outside a small area. A difference map that lights up everywhere is expected and is
     not a flaw - do not penalise it and do not use the difference maps to pick a winner. Judge
     whether the requested change was actually made, and made well, and whether what the prompt
     did not ask to change survived: the character's identity, the text, the objects that must
     stay recognisable, the composition.
   - With several originals, the elements the prompt refers to from each one must be used.
2. Anatomy and structure - extra or missing limbs or fingers, broken poses, distorted faces, melted
   or misspelled text, impossible perspective, objects that do not hold together. These are the
   flaws a viewer notices in the first second, and they usually decide the task.
3. Artifacts - watermarks, signatures, unwanted frames or borders, "circle blob" or
   radial-gradient backgrounds, compression noise, smears, visible seams. Prefer the image without
   them, unless the prompt explicitly asked for exactly that.
4. Cohesion, overall quality and style match - how the image looks when nothing is clearly broken:
   resolution and sharpness, lighting, colour, and whether it feels like one cohesive picture
   rather than elements pasted on top of each other. Useful checks:
   - Light: all shadows point away from the same place and share the same softness; lit and
     shadowed areas have a consistent colour cast.
   - Contact: where objects touch a surface there is a dark contact seam - nothing floats.
   - Materials: metal reflects its surroundings; other materials show the light's colour on top of
     their own; every surface gets more reflective at glancing angles.
   - Space: parallel lines converge to consistent vanishing points; the eye level is consistent;
     blur grows smoothly with distance.
   - The "stock look": generic ornament piled onto a logo or design, or fine texture smeared and
     drained into a generic surface (a dense pattern turned sparse, real material turned into
     plastic or chrome). The image with specific, particular detail is better.
   - For edits, the result should match the style of the source images - palette, lighting and
     contrast - like a natural continuation of them.
   Composition rules (thirds etc.) describe patterns, they are not tests. Separate real defects
   (physical contradictions, broken structure) from matters of taste.

WHAT TO WRITE (the JSON fields)
- originals_summary: each original precisely - subjects, counts, colours, positions, any text
  spelled exactly, framing.
- edit_type: local, global, or generation (no originals, or a brand-new image).
- first_impression: for A and for B, what an ordinary viewer notices in the first few seconds,
  good or bad.
- visible_differences: literal differences between A and B.
- requirements: your findings across all criteria. Each has a category (prompt, preserve,
  anatomy, artifact, cohesion, quality), a severity (critical = ruins the image or misses the main
  point of the request; major = clearly noticeable and would bother the requester; minor = small
  and forgivable) and pass / fail / unsure for A and for B, with evidence naming what you actually
  see in that image.
- decisive_difference: the one or two things that settle the choice, naming both results.
- reasoning: why that outweighs anything favouring the other image, in terms of how serious and
  how visible the flaws are.
- verdict and confidence: high = one image is clearly better as a whole; medium = the difference
  is real but subtle; low = genuinely close, or you are relying on taste. Use low honestly - a
  low-confidence answer goes to a human. Image order carries no information.
"""

# v4: v3 plus the mistakes v3 made on the labeller's own tasks (run #16: it agreed with the labeller
# on only 15 of 33). Reviewing the confident misses showed two habits the guidelines already warn
# against: rewarding the most literal reading over the cohesive image, and rewarding "changed the
# least" even when the edit left the image broken (e.g. arms still gripping a removed weapon).
_V4 = _V3 + """
COMMON MISTAKES TO AVOID
These come from comparing earlier verdicts with the labeller's answers. Check yourself against
each one before you commit.
- Rewarding the most literal reading. A result that ticks a requirement literally but looks like a
  composite - a pasted-in subject, flat cut-out elements, a scene that has lost the source's style
  and mood - usually loses to one that fulfils the request in a way that fits the original's look,
  even if it is less literal. Ask which one looks like a finished, intentional picture.
- Rewarding "changed the least". Changing little is good only if the result still makes sense.
  After something is removed or changed, check what was connected to it: a body still posed around
  a missing object (hands gripping nothing, an arm raised for a prop that is gone), leftover stubs,
  shadows or reflections of something no longer there, cut-off limbs. A viewer sees these at once,
  and they outweigh the other image regenerating more of the scene, as long as that image keeps the
  subject recognisable.
- Letting the difference map decide. It shows where pixels changed, not whether the image is good.
- Hunting small flaws in one image while missing the big thing in the other. Form your first
  impression of each image as a whole before the checklist; let the checklist overturn it only for a
  clearly more serious flaw.
- Over-reading normal details as defects (stylised gloves called "malformed hands", intended blur
  called "low quality"). Only count a flaw an ordinary viewer would see as wrong.
"""

# v5 turns the complete Image Selection Annotation Guide into an operational checklist. It adds
# the parts that were not explicit in v4 while keeping the prompt short enough for fast judging.
_V5 = _V4 + """
IMAGE SELECTION ANNOTATION GUIDE — FINAL CHECK
Use this sequence for every task:
1. Read the request before judging the images. List every visible requirement, including subject
   identity, object counts, exact text, colours, positions, layout, framing, and requested style.
2. Look at A and B normally for a few seconds. Record the first problem or strength that catches the
   eye in each. Zoom or inspect fine detail only to confirm meaningful issues, not to hunt for flaws
   an ordinary viewer would never notice.
3. Classify the task as local edit, global edit, or new generation/reference generation.
   - Local edit: the requested region should change and everything else should remain as close to
     the source as possible. Use the difference map to locate unintended changes, but verify them in
     the actual image. A mostly black map outside the edit is desirable only when the result remains
     visually correct and coherent.
   - Global edit or new image from a reference: widespread difference is expected. Ignore the
     difference map and judge whether the requested transformation is done well while required
     identity, text, objects, and composition remain recognisable.
4. Judge each result as a whole across all of these dimensions; none is an automatic trump card:
   prompt compliance and preservation; anatomy and object/scene structure; exact printed text;
   perspective and spatial relationships; cohesion versus pasted-on elements; artifacts such as
   watermarks, signatures, unwanted borders, blobs, seams, smears, or malformed details; and final
   image quality. For edits, also require a natural match to the source's palette, lighting,
   contrast, and visual style.
5. Weigh each failure by seriousness, visibility, and how much it would bother the requester. A
   literal result with a ruinous defect can lose to a clean result with a small miss; a beautiful
   image that ignores the main request can lose to a slightly imperfect image that actually makes
   the requested change. Choose the image the requester would be happier to receive.
6. Recheck the one or two decisive regions directly in both images before returning the verdict.

VECTOR-SPECIFIC RULES
When the request asks for vector art, expect flat colour fills or simple gradients, a relatively
limited palette, crisp segment boundaries, and no fine raster texture. Complexity or realism alone
does not make an image non-vector: intricate shapes can still be vector when their colour regions
remain clean and texture-free. Engraving and linocut styles can count as vector even when detailed;
they commonly use only one or two flat colours, no gradients, and no fine photographic texture.
Prefer the result that satisfies these properties without sacrificing the requested content.

NEAR TIES
Selecting both images is meant to be exceptional. This evaluator must emit A or B, so if the images
are effectively equal, choose only the marginally better one, set confidence to low, and state that
the difference is negligible. Low-confidence decisions are routed to a human instead of being
treated as a confident winner.
"""

# Fields only v3 asks for, placed before the requirement checklist.
JUDGMENT_SCHEMA_V3 = {
    **JUDGMENT_SCHEMA,
    "properties": {
        "originals_summary": {"type": "string"},
        "edit_type": {"type": "string", "enum": ["local", "global", "generation"]},
        "first_impression": {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
        **{k: v for k, v in JUDGMENT_SCHEMA["properties"].items() if k != "originals_summary"},
    },
    "required": ["originals_summary", "edit_type", "first_impression"]
    + [k for k in JUDGMENT_SCHEMA["required"] if k != "originals_summary"],
}

RUBRICS: dict[str, str] = {"v1": _V1, "v2": _V2, "v3": _V3, "v4": _V4, "v5": _V5}
# Rubric versions that are given difference heat maps between each result and the first original.
USES_DIFFERENCE_MAPS = {"v3", "v4", "v5"}


def schema_for(version: str) -> dict:
    return JUDGMENT_SCHEMA_V3 if version in ("v3", "v4", "v5") else JUDGMENT_SCHEMA


def system_prompt(version: str, guidelines: str = "", lessons: tuple[str, ...] | list[str] = ()) -> str:
    """The rubric, plus the project's guidelines and the lessons learned in training, if any."""
    try:
        text = RUBRICS[version]
    except KeyError:
        raise ValueError(f"unknown rubric version {version!r}; choose from {sorted(RUBRICS)}") from None
    if guidelines.strip():
        text += (
            "\nPROJECT GUIDELINES\n"
            "The person who labels these tasks follows these guidelines. Where they conflict with the\n"
            "general advice above, the guidelines win.\n\n" + guidelines.strip() + "\n"
        )
    if lessons:
        text += (
            "\nLESSONS FROM EARLIER LABELLED TASKS\n"
            "These rules were learned from mistakes on earlier tasks from the same project. Apply each\n"
            "one when it is relevant to the task in front of you; ignore it when it is not.\n"
            + "".join(f"- {lesson.strip()}\n" for lesson in lessons)
        )
    return text


def build_user_content(
    prompt: str,
    originals: list,
    results: list[tuple[str, object]],
    notes: list[str],
    difference_maps: Optional[list[tuple[str, object]]] = None,
) -> list:
    """Interleave text labels with images so each image has an unambiguous name.

    Returns a provider-neutral list: plain strings for text, and the image objects
    exactly as passed in. Each provider converts them to its own block format.
    """
    content: list = [f"PROMPT:\n{prompt.strip()}"]
    if originals:
        for i, image in enumerate(originals, 1):
            content += [f"ORIGINAL {i}:", image]
    else:
        content.append("No original images: this is pure generation.")
    for label, image in results:  # presentation order varies; labels are always the true ones
        content += [f"RESULT {label}:", image]
    for label, image in difference_maps or []:
        content += [f"DIFFERENCE MAP, Result {label} vs Original 1 (black = unchanged, brighter = more changed; "
                    "meaningful for local edits only):", image]
    if notes:
        content.append("Measured facts about the images (computed by pixel comparison, reliable):\n- "
                       + "\n- ".join(notes))
    content.append("Judge which result is correct. Respond with the JSON object only.")
    return content
