from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from app.core.config import GOOGLE_API_KEY
import google.generativeai as genai

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