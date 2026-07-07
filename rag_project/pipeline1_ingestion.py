from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import chromadb
import os

def add_chunks_to_db(collection, chunks, embeddings, source):
    """
    Adds chunks and their embeddings to the specified collection in the vector database.
    """
    # Store in ChromaDB
    collection.add(
        documents=chunks,                                 # original text
        embeddings=embeddings,                            # converted numbers
        ids=[f"chunk_{i}" for i in range(len(chunks))],  # unique id for each chunk
        metadatas=[{
            "source": source,
            "chunk_index": i
            } for i in range(len(chunks))]
    )

    print(f"{len(chunks)} chunks stored in Vector DB")

def create_collection(client, collection_name):
    """
    Creates a collection in the vector database.
    """
    return client.get_or_create_collection(name=collection_name)

def create_vector_db(DB_PATH):
    """
    Creates a persistent vector database using ChromaDB.
    """
    #client = chromadb.PersistentClient(path="rag_project/chroma_db")
    client = chromadb.PersistentClient(path=DB_PATH)
    print(f"DB path: {DB_PATH}")  # verify where data is saved
    return client

def create_embeddings(chunks):
    """
    Creates embeddings for the given chunks using a sentence transformer model.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks).tolist()
    return embeddings

def chunk_document(text, chunk_size=200, overlap=50):
    """
    Breaks text into chunks of approximately chunk_size characters.
    overlap: how many characters to repeat between chunks
    so we dont lose context at chunk boundaries.
    """
    chunks = []
    start = 0

    while start < len(text):
        end=start+chunk_size
        chunk=text[start:end]
        chunks.append(chunk)
        start=end-overlap  # move start forward by chunk_size - overlap

    return chunks

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def main():
    # Dynamic path — works everywhere
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PDF_PATH = os.path.join(BASE_DIR, "hr_policy.pdf") 
    print(f"Loading PDF from: {PDF_PATH}") 

    DB_PATH = os.path.join(BASE_DIR, "chroma_db")  # always relative to script

    text = extract_text_from_pdf(PDF_PATH)

    # Extract text from the PDF
    text = extract_text_from_pdf(PDF_PATH)
    # print(text)

    # Chunk the text into smaller pieces
    chunks = chunk_document(text, chunk_size=200, overlap=50)
    #print(chunks)
    # for i, chunk in enumerate(chunks):
    #     print(f"Chunk {i+1}:\n{chunk}\n{'-'*50}\n")

    # Embed the chunks using a sentence transformer model
    embeddings = create_embeddings(chunks)
    #print(embeddings)

    # Create vector database, collection, and add the chunks and embeddings to the collection
    client = create_vector_db(DB_PATH)
    print("Vector database created successfully.")
    collection = create_collection(client, "hr_policy_collection")
    print("Adding chunks to Vector DB...")
    add_chunks_to_db(collection, chunks, embeddings, source="hr_policy.pdf")
    print("Ingestion pipeline complete!")

if __name__ == "__main__":
    main()