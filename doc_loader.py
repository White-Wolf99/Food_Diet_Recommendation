from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  

from langchain.vectorstores import Chroma
import shutil
from langchain.schema import Document

import chromadb
from langchain_chroma import Chroma
import os

# embedding = HuggingfaceEmbedding(model="")
directory_path = "../PDFs"

def load_pdfs_from_directory(dir_path):
    # Load all PDF documents from the directory
    loader = PyPDFDirectoryLoader(dir_path)
    docs = loader.load()  # Load all PDF files
     # Get the number of unique PDF files (not pages)
    pdf_files = []
    for f in os.listdir(dir_path):
        if(f.endswith(".pdf")):
            pdf_files.append(f)
            
    
    # print(f"Number of PDFs loaded: {len(pdf_files)} PDFs.\n") 
    # print(f"Number of pages loaded: {len(docs)} pages.\n") #considers each page of total pdfs as idividual pdfs , hence 2108
    # print(f"Content of first document: {docs[1].page_content}")
    return docs


def split_docs_into_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000, 
        chunk_overlap = 300,
        length_function = len,
        separators=["\n\n", "\n", " "],  # Split by paragraphs first
    )
    
    split_docs = []
    for doc in docs:
        split_text = text_splitter.split_text(doc.page_content)  # Split the document text
        for chunk in split_text:
            split_docs.append(chunk)  # Store each chunk
    
    return split_docs



#docs = load_pdfs_from_directory(directory_path)
#chunks = split_docs_into_chunks(docs)

# print((len(chunks[0]))) ; prints the length of one chunk / number of characters in one chunk 
# print(len(chunks))      ; prints the length of the array / total created chunks
# print(chunks)           ; prints the array which holds all the chunks 
# word_count = len(chunks[0].split())        ; Split function returns a list filled with splitted words
# print(f"Number of words: {word_count}")    ; Printing the number of words in the first chunk



# Convert chunks to Document objects for compatibility with Chroma
#document_chunks = [Document(page_content=chunk) for chunk in chunks]

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
CHROMA_PATH = "chroma"
def save_to_chroma(full_chunk):

    # Clear out the database first
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    # Create a new database from the document
    db = Chroma.from_documents(
        documents=full_chunk, 
        embedding=embedding_model,  # Local embedding model instead of OpenAI
        persist_directory=CHROMA_PATH
    )
    #print(f"Saved {len(full_chunk)} chunks to {CHROMA_PATH}.", end="\n")
# save_to_chroma(document_chunks)

# print(len(document_chunks), end="\n")
# print(cleaned_chunks)
# print("test done")



def main():
    # Load, split, and clean the document chunks as you already do
    docs = load_pdfs_from_directory(directory_path)
    chunks = split_docs_into_chunks(docs)
    
    # Convert chunks to Document objects for Chroma
    document_chunks = [Document(page_content=chunk) for chunk in chunks]
    
    # Save to Chroma
    save_to_chroma(document_chunks)
    
    # Clean up newline characters
    cleaned_chunks = [Document(page_content=chunk.page_content.replace('\n', ' '), metadata=chunk.metadata) for chunk in document_chunks]
    print(len(cleaned_chunks))
    return cleaned_chunks


if __name__ == "__main__":
    main()
