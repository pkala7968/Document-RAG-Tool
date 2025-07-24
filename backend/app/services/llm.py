from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from app.core.config import GOOGLE_API_KEY
import google.generativeai as genai
import json

genai.configure(api_key=GOOGLE_API_KEY)

def get_conversational_chain(retriever):

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "Answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {summaries}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.3,)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=model,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True 
    )

    return chain

import re

def extract_json(text: str) -> str:
    """
    Extracts the first valid JSON object from a given string.
    Useful when LLM responses contain prose or explanations around the JSON.
    """
    match = re.search(r'\{.*\}', text.strip(), re.DOTALL)
    if match:
        return match.group(0)
    raise ValueError("No valid JSON object found in the text.")

def evaluate_with_llm(context: str, answer: str):
    prompt = f"""
You are an expert evaluator.

Context:
\"\"\"{context}\"\"\"

Answer:
\"\"\"{answer}\"\"\"

Evaluate the answer based only on the context provided. Score from 1 to 5:

- factual_accuracy: (Is the answer supported by the context?)
- completeness: (Does the answer fully address the likely question?)
- hallucination: (Does it contain any unsupported information?)

Respond with ONLY a JSON object with these keys:
{{
  "factual_accuracy": "0-100%",
  "completeness": "0-100%",
  "hallucination": "0-100%",
  "comment": "..."
}}
"""

    model = genai.GenerativeModel("gemini-1.5-flash")

    try:
        response = model.generate_content(prompt)
        # Extract JSON from response text
        raw = response.text.strip()
        # Try to extract valid JSON substring from output
        json_str = extract_json(raw)
        return json.loads(json_str)
    except Exception as e:
        return {
            "factual_accuracy": "0%",
            "completeness": "0%",
            "hallucination": "100%",
            "comment": f"Evaluation failed: {str(e)}\nRaw output: {raw}"
        }