
import argparse
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from transformers import pipeline
from langchain_community.vectorstores import Chroma
import re



PROMPT_TEMPLATE = """
Based on the provided context, answer the question accurately and concisely. Do not repeat sentences or introduce unrelated information.
Context:
{context}
---
Question: {question}
Answer:
"""

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
CHROMA_PATH = "chroma"

# Initialize the text-generation model based on preference
model = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

# Initialize FLAN-T5 model
#model = pipeline("text2text-generation", model="google/flan-t5-large", device=-1)  # Use device=-1 for CPU

def q_chroma(query_text):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
    results = db.similarity_search_with_relevance_scores(query_text, k=1)

    if not results:
        print("No relevant context found for the query.")
        return
    
    context_text = "\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    response = model(prompt, max_new_tokens=200, temperature=0.2, top_p=0.9, do_sample=True)[0]['generated_text']

    # Clean the response
    clean_response = re.sub(r"(Human:|Answer the question.*?---\s*)", "", response, flags=re.DOTALL).strip()
    clean_response = re.sub(rf"{re.escape(query_text)}", "", clean_response).strip()
    clean_response = re.sub(r"^Question:.*\n*", "", clean_response)
    clean_response = re.sub(r"^Answer:.*\n*", "", clean_response)

    
    # Format the final output
    formatted_response = f"Question: {query_text}\n\nAnswer:\n\n{clean_response}"
    print(formatted_response, end="\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the Chroma database with a question.")
    parser.add_argument("query_text", type=str, nargs='?', help="The question you want to ask.")
    args = parser.parse_args()
    
    if not args.query_text:
        args.query_text = input("Enter your question: ")
    
    q_chroma(args.query_text)
