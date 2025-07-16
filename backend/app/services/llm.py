import google.generativeai as genai
from app.config import settings

# Configure Gemini API
genai.configure(api_key=settings.GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

def generate_answer_with_citations(question, retrieved_chunks):
    """
    Generate an answer from each document chunk using Gemini with citation info.
    """
    answers = []
    for chunk in retrieved_chunks:
        prompt = f"""
User Question:
"{question}"

Document Excerpt (Page {chunk['page']} of {chunk['doc_id']}):
\"\"\"
{chunk['text']}
\"\"\"

Answer the question using ONLY the document excerpt. If not answerable, say "Not mentioned."

Cite the document ID and page number in your answer.
"""
        try:
            response = model.generate_content(prompt)
            answer = response.text.strip()
            answers.append({
                "doc_id": chunk["doc_id"],
                "page": chunk["page"],
                "answer": answer
            })
        except Exception as e:
            answers.append({
                "doc_id": chunk["doc_id"],
                "page": chunk["page"],
                "answer": f"Error: {str(e)}"
            })
    return answers


def generate_themes(document_answers):
    """
    Use Gemini to summarize common themes from the document-level answers.
    """
    answer_block = "\n\n".join(
        f"[{a['doc_id']} - Page {a['page']}]: {a['answer']}" for a in document_answers
    )

    prompt = f"""
You're given multiple document-based answers to a user query:

{answer_block}

Identify the MAIN THEMES present across these answers.
Group ideas under theme titles, and support each theme with document references (doc_id + page).

Format example:
Theme 1 – Title:
DOC001 - Page 2: Summary...
DOC002 - Page 4: Summary...

Return 2–4 themes.
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating theme summary: {str(e)}"
