# Copyright (c) 2026 Jakir Hossain, Ahmet Erdem Sariyuce.
# Licensed under the MIT License

"""Prompts for question generation (persona -> task -> question, following Edge et al.)."""

# =========================
# Few-shot examples - style guidance only
# =========================
FEWSHOT_STYLE_EXAMPLES = """
Examples (STYLE ONLY; from OTHER datasets):
Use these only to understand sentence length, phrasing, and global sensemaking style. 
Do NOT copy these questions, topics, facts, or wording; focus entirely on the given corpus.

Dataset: Podcast transcripts
User: A tech journalist looking for insights and trends in the tech industry
Task: Understanding how tech leaders view the role of policy and regulation
Questions (examples):
1. Which episodes deal primarily with tech policy and government regulation?
2. How do guests perceive the impact of privacy laws on technology development?
3. Do any guests discuss the balance between innovation and ethical considerations?
4. What are the suggested changes to current policies mentioned by the guests?
5. Are collaborations between tech companies and governments discussed and how?

Dataset: News articles
User: Educator incorporating current affairs into curricula
Task: Teaching about health and wellness
Questions (examples):
1. What current topics in health can be integrated into health education curricula?
2. How do news articles address the concepts of preventive medicine and wellness?
3. Are there examples of health articles that contradict each other, and if so, why?
4. What insights can be gleaned about public health priorities based on news coverage?
5. How can educators use the dataset to highlight the importance of health literacy?
""".strip()

# =========================
# Prompts (Algorithm 1)
# =========================
CORPUS_DESCRIPTION_SYSTEM = """
You write concise dataset descriptions for evaluation.
Return ONLY valid JSON. No markdown. No extra keys.
"""

CORPUS_DESCRIPTION_USER_TEMPLATE = """You are given a set of sample excerpts from a corpus.
Write a **high-level description of the entire corpus and its purposes**, suitable for guiding global sensemaking evaluation.

Requirements:
- Focus on the corpus as a whole, not individual communities, clusters, or entity graphs.
- Describe broadly what the corpus contains: domains, topics, events, actors, trends, and types of documents.
- Mention the intended uses or purposes of the corpus (e.g., evaluation, global sensemaking, policy analysis, scenario testing, research).
- Use concise, natural language.
- Do NOT copy long passages verbatim from the excerpts.
- Do NOT invent facts not supported by the excerpts.
- Keep the "description" field between 200–350 words.
- Include 8–15 high-level items in "key_themes".
- Include major actors or items of interest in "typical_entities".
- Return JSON exactly in this schema (no extra keys, no markdown):

{{
  "corpus_name": string,
  "domain": string,
  "document_types": [string],
  "time_span": string,
  "description": string,
  "purposes": string,
  "typical_entities": [string],
  "key_themes": [string],
  "intended_uses": [string]
}}

Sample excerpts from the corpus:
{snippets}
"""



PERSONAS_SYSTEM = f"""You generate evaluation personas.
Return ONLY valid JSON. No markdown. No extra keys.

{FEWSHOT_STYLE_EXAMPLES}
"""

PERSONAS_USER_TEMPLATE = """Given this corpus description, generate {K} distinct personas of potential users.

Constraints:
- Personas must be plausible users of this corpus.
- Each persona must include a short "goal" and "background".
- Avoid stereotypes; keep it professional.
- Do not mention GraphRAG, embeddings, vector DB, or "community reports".
- Do not reuse or paraphrase the style examples; generate new personas specific to this corpus.

Return JSON exactly:
{{
  "personas": [
    {{
      "persona_id": 1,
      "name": string,
      "role": string,
      "background": string,
      "goal": string
    }}
  ]
}}

Corpus description:
{corpus_description_json}
"""

TASKS_SYSTEM = f"""You generate user tasks for an evaluation.
Return ONLY valid JSON. No markdown. No extra keys.

{FEWSHOT_STYLE_EXAMPLES}
"""

TASKS_USER_TEMPLATE = """For each persona, generate exactly {N} tasks they would use a RAG system on this corpus to complete.

Constraints:
- Tasks must require global understanding/sensemaking (themes, narratives, shifts, actors, causal patterns).
- Tasks must NOT be "look up a specific fact" tasks.
- Tasks should be phrased as objectives.
- Each task MUST clearly align with the persona’s stated goal.
- Do not reuse or paraphrase the style examples; generate new tasks specific to this corpus.

Return JSON exactly:
{{
  "tasks": [
    {{
      "persona_id": 1,
      "task_id": 1,
      "task": string
    }}
  ]
}}

Personas:
{personas_json}

Corpus description:
{corpus_description_json}
"""

QUESTIONS_SYSTEM = f"""You generate global sensemaking questions for evaluation.
Return ONLY valid JSON. No markdown. No extra keys.

{FEWSHOT_STYLE_EXAMPLES}

Additional constraint:
- Questions for the same persona-task pair must focus on different aspects of the task.
"""

QUESTIONS_FOR_PAIR_USER_TEMPLATE = """For each (persona, task) pair, generate exactly {M} high-level questions.

Constraints for each question:
- Focus on the entire corpus (global themes, patterns, or narratives).
- Support the persona’s goal and stay within the task scope.
- Synthesize information across multiple documents, actors, or events.
- Do NOT ask for low-level facts (e.g., dates, quotes).
- Cover different aspects of the task and explore multiple perspectives (policy, operations, incentives, trends).
- Emphasize themes or angles not already addressed in other persona–task questions.
- Requires understanding and synthesizing information across the entire corpus.
- One sentence only, with at most one conjunction (“and” or “or”).
- Prefer minimal punctuation; at most one comma.
- Avoid filler phrases and private identifiers.
- Do not reuse or paraphrase the style examples; generate new, corpus-specific questions.


You MUST follow these formatting rules:
- Output JSON only.
- EXACTLY {M} questions.
- Each question MUST be 8–15 words.
- Each question MUST be one sentence.
- Each question MUST start with: What, How, Why, Which, Do, or Are.
- Do NOT use colons, semicolons, parentheses, or lists.

Return JSON exactly:
{{
  "questions": [
    {{
      "persona_id": 1,
      "task_id": 1,
      "question_id": 1,
      "question": string
    }}
  ]
}}

Personas:
{persona_json}

Tasks:
{task_json}

Corpus description:
{corpus_description_json}
"""
