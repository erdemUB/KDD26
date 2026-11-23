# Copyright (c) 2026 Jakir Hossain, Ahmet Erdem Sariyuce.
# Licensed under the MIT License

"""Head-to-head LLM judge (prompt, criteria and judge backends).

Credentials and endpoints come from the environment (never hard-code keys):
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT (default "gpt-5-mini"), AZURE_OPENAI_API_VERSION
    GOOGLE_CLOUD_PROJECT      GCP project for Gemini and Vertex AI MaaS models
                              (Qwen, DeepSeek); auth via `gcloud auth application-default login`.
"""

import os
import time

import requests

CRITERIA = {
    "comprehensiveness":
        "How much detail does the answer provide to cover all the aspects and details of the question?"
        "A comprehensive answer should be thorough and complete, without being redundant or irrelevant."
        "For example, if the question is ’What are the benefits and drawbacks of nuclear energy?’, a comprehensive answer would provide both the positive and negative aspects of nuclear energy, such as its efficiency, environmental impact, safety, cost, etc."
        "A comprehensive answer should not leave out any important points or provide irrelevant information. For example, an incomplete answer would only provide the benefits of nuclear energy without describing the drawbacks, or a redundant answer would repeat the same information multiple times.",
    "diversity":
        "How varied and rich is the answer in providing different perspectives and insights on the question? "
        "A diverse answer should be multi-faceted and multi-dimensional, offering different viewpoints and angles on the question."
        "For example, if the question is ’What are the causes and effects of climate change?’, a diverse answer would provide different causes and effects of climate change, such as greenhouse gas emissions, deforestation, natural disasters, biodiversity loss, etc."
        "A diverse answer should also provide different sources and evidence to support the answer. For example, a single-source answer would only cite one source or evidence, or a biased answer would only provide one perspective or opinion.",
    "empowerment":
        "How well does the answer help the reader understand and make informed judgements about the topic without being misled or making fallacious assumptions."
        "Evaluate each answer on the quality of answer as it relates to clearly explaining and providing reasoning and sources behind the claims in the answer.",
    "directness":
        "How specifically and clearly does the answer address the question? A direct answer should provide a clear and concise answer to the question."
        "For example, if the question is ’What is the capital of France?’, a direct answer would be ’Paris’. A direct answer should not provide any irrelevant or unnecessary information that does not answer the question."
        "For example, an indirect answer would be ’The capital of France is located on the river Seine’"
}

SYSTEM_INSTRUCTION = (
    "You are an impartial evaluator. "
    "Judge two answers strictly by the given criterion. "
    "Evaluate each answer independently, then compare. "
    "Ignore answer order, style, and verbos unless required by the criterion. "
    "Do not infer missing information. "
)

# Gemini region per model (others use "global").
model_location_map = {
    "gemini-3-pro-preview": "global",
    "gemini-2.5-pro": "us-east1",
    "gemini-3-pro": "us-central1",
}

_azure_client = None


def _gcp_project():
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError("Set GOOGLE_CLOUD_PROJECT for Gemini / Vertex AI judges.")
    return project


def call_azure_llm(prompt):
    """GPT-5-mini (or AZURE_OPENAI_DEPLOYMENT) via Azure OpenAI."""
    global _azure_client
    if _azure_client is None:
        from openai import AzureOpenAI

        _azure_client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
    response = _azure_client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def call_vertex_maas_llm(full_prompt, model_backend):
    """Qwen / DeepSeek models via Vertex AI OpenAPI chat completions."""
    import google.auth
    from google.auth.transport.requests import Request

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(Request())

    url = (
        "https://aiplatform.googleapis.com/v1/projects/"
        f"{_gcp_project()}/locations/global/endpoints/openapi/chat/completions"
    )
    payload = {
        "model": model_backend,  # e.g. "qwen/qwen3-next-80b-a3b-instruct-maas"
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": 0,
    }
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {creds.token}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def call_gemini_llm(full_prompt, model_backend):
    """Gemini via google.genai on Vertex AI."""
    from google import genai
    from google.genai import types

    client = genai.Client(
        vertexai=True,
        project=_gcp_project(),
        location=model_location_map.get(model_backend.lower(), "global"),
    )
    response = client.models.generate_content(
        model=model_backend,
        contents=[full_prompt],
        config=types.GenerateContentConfig(temperature=0),
    )
    return response.text


def build_llm_prompt(question, answer1, answer2, metric, metric_descr):
    """Constructs the user prompt using the paper's template.

    Answer order is randomized by the caller (judge.judge_multiple_runs).
    """
    prompt = f"""
    ---Role---
    You are a helpful assistant responsible for grading two answers to a question that are provided by two different people.

    ---Goal---
    Given a Question and two answers (Answer 1 and Answer 2), assess which answer is better according to the following measure:

    {metric_descr}


    ---Question---
    {question}

    ---Answer 1---
    {answer1}

    ---Answer 2---
    {answer2}

    Your assessment should include two parts:
    - Winner: either 1 (if Answer 1 is better) and 2 (if Answer 2 is better) or 0 if they are fundamentally similar and the differences are immaterial.
    - Reasoning: Concise justification (≤20 words) referencing only the evaluation criterion.

    ---Output Format---
    Return ONLY a valid JSON object in the following format:

    {{
    "winner": <1, 2, or 0>,
    "reasoning": "Answer X is better because <your reasoning>."
    }}

    Do not include any additional text.

"""
    return prompt


def llm_judge(question, answer1, answer2, metric, model_backend="azure"):
    """Head-to-head LLM judge for two answers. Returns the raw LLM reply text."""
    prompt = build_llm_prompt(question, answer1, answer2, metric, CRITERIA[metric])
    full_prompt_with_markers = (
        f"---SYSTEM INSTRUCTION---\n{SYSTEM_INSTRUCTION}\n\n"
        f"---USER PROMPT---\n{prompt}"
    )
    backend = model_backend.lower()

    retries, max_retries, delay = 0, 10, 1
    while retries < max_retries:
        try:
            if backend == "azure":
                return call_azure_llm(prompt)
            if "gemini" in backend:
                return call_gemini_llm(full_prompt_with_markers, model_backend)
            if "deepseek" in backend or "qwen" in backend:
                return call_vertex_maas_llm(full_prompt_with_markers, model_backend)
            raise ValueError(f"Unknown model_backend '{model_backend}'.")
        except ValueError:
            raise
        except Exception as e:
            print(f"{model_backend} rate limit or error: {e}. Retrying in {delay} sec...")
            time.sleep(delay)
            retries += 1
            delay *= 2  # exponential backoff

    raise Exception(f"Maximum retries exceeded for {model_backend}.")
