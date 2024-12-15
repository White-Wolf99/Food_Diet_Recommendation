import os
import re
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# -----------------------
# Configurations and Paths
# -----------------------
CHROMA_PATH = "chroma"  # Path to store the Chroma vector database
PDF_DIRECTORY = "../PDFs"  # Directory containing PDF documents
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace embedding model name

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# -----------------------
# Document Loading and Chunking
# -----------------------
def load_pdfs_from_directory(dir_path):
    """Load all PDF documents from the specified directory."""
    loader = PyPDFDirectoryLoader(dir_path)
    docs = loader.load()  # Load all PDF files

    # Count the number of unique PDF files
    pdf_files = [f for f in os.listdir(dir_path) if f.endswith(".pdf")]
    print(f"Number of PDFs loaded: {len(pdf_files)} PDFs.")
    print(f"Number of pages loaded: {len(docs)} pages.")
    return docs


def split_docs_into_chunks(docs):
    """Split loaded documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=300,
        length_function=len,
        separators=["\n\n", "\n", " "]  # Split by paragraphs first
    )

    split_docs = []
    for doc in docs:
        split_text = text_splitter.split_text(doc.page_content)
        for chunk in split_text:
            split_docs.append(Document(page_content=chunk))

    print(f"Number of chunks created: {len(split_docs)}")
    return split_docs


def save_to_chroma(full_chunk):
    """Save chunks to a Chroma vector store."""
    # Clear out the database if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new database from the documents
    db = Chroma.from_documents(
        documents=full_chunk,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )

    try:
        db.persist()  # Save the database for future use
        print(f"Database successfully persisted at {CHROMA_PATH}.")
    except AttributeError:
        print("Warning: 'persist()' method not available. Database saved in memory only.")

    print(f"Saved {len(full_chunk)} chunks to {CHROMA_PATH}.")

# -----------------------
# Integration into RAG Pipeline
# -----------------------
class RAGPipeline:
    def __init__(self):
        """Initialize RAG pipeline components."""
        self.embedding_model = embedding_model
        self.chroma_path = CHROMA_PATH

        # Initialize or load vector database
        if os.path.exists(self.chroma_path):
            self.vector_store = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=self.embedding_model
            )
            print("Chroma vector database loaded.")
        else:
            print("No existing Chroma database found. Please preprocess documents first.")

    def preprocess_documents(self, directory_path):
        """Load PDFs, split into chunks, and store in Chroma."""
        print("Preprocessing documents...")
        docs = load_pdfs_from_directory(directory_path)
        chunks = split_docs_into_chunks(docs)
        save_to_chroma(chunks)

    def query_database(self, query: str, top_k: int = 2):
        """Query the Chroma vector database."""
        if not hasattr(self, "vector_store"):
            raise ValueError("Vector store is not initialized. Please preprocess documents first.")
        
        results = self.vector_store.similarity_search(query, k=top_k)
        return results

    def prompt_chain(self, prompt: str):  
        """Generate a response using LLM and the retrieved context."""
        # Retrieve relevant documents from Chroma
        relevant_docs = self.query_database(prompt)

        # Create prompt template
        system_prompt_text = """You are an expert dietitian also a great cook. Your task is to produce a diet plan 
according to the patient's requirement and give your recipe to cook your prescribed diet. Supplementary cookbook to generate recipes: {cookbook}.
Your generated output should include calorie content of each recipe."""
        user_prompt_text = "Generate diet plan with recipes for the requirement: {prompt}"
        
        # Combine prompts
        prompt_template = PromptTemplate(
            template=system_prompt_text + " " + user_prompt_text,
            input_variables=["prompt", "cookbook"]
        )

        # Invoke the chain
        chain = (
            prompt_template
            | self.embedding_model
            | StrOutputParser()
            | (lambda x: (x.split("<|assistant|>")[-1]).strip())
        )

        cookbook = [doc.page_content for doc in relevant_docs]
        return chain.invoke({"prompt": prompt, "cookbook": cookbook})


if __name__ == "__main__":
    rag = RAGPipeline()

    # Preprocess documents (if needed)
    if not os.path.exists(CHROMA_PATH):
        rag.preprocess_documents(PDF_DIRECTORY)

    # Example Query
    query = "What are some Bangladeshi recipes for diabetic patients?"
    response = rag.prompt_chain(query)
    print(response)
