from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()


def load_and_split_document(pdf_path):
    """
    Step 1 & 2: Load PDF and split into chunks.
    RecursiveCharacterTextSplitter splits by paragraph first,
    then sentence, then word — keeps meaning intact.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages from PDF")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks, db_path):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Delete existing DB to avoid duplicates
    import shutil
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        print("✅ Cleared existing vector store")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )
    print(f"✅ Vector store created with {vectordb._collection.count()} chunks")
    return vectordb

def format_docs(docs):
    """Combines retrieved chunks into one string."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(vectordb, llm):
    """
    Step 5 & 6: Create retriever and RAG chain.
    RunnablePassthrough passes the question through unchanged.
    """
    # Convert vector store to retriever
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # Prompt template for RAG
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful HR assistant.
Answer questions using ONLY the context provided below.
If the answer is not in the context, say 'I don't have information on this.'
Be concise and direct.

Context:
{context}"""),
        ("human", "{question}")
    ])

    # Build RAG chain
    # 1. Retrieve relevant chunks
    # 2. Format chunks into context
    # 3. Pass to prompt
    # 4. Send to LLM
    # 5. Parse output
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PDF_PATH = os.path.join(BASE_DIR, "hr_policy.pdf")
    DB_PATH = os.path.join(BASE_DIR, "chroma_db")

    llm = ChatGroq(model="llama-3.3-70b-versatile")

    print("Setting up RAG pipeline...\n")

    # Step 1-2: Load and split
    chunks = load_and_split_document(PDF_PATH)

    # Step 3-4: Create vector store
    vectordb = create_vector_store(chunks, DB_PATH)

    # Step 5-6: Create RAG chain
    rag_chain = create_rag_chain(vectordb, llm)

    print("\nRAG Pipeline ready!\n")
    print("=" * 50)

    # Test questions
    questions = [
        "How many sick days do I get?",
        "What is the maternity leave policy?",
        "Can I work from home?",
        "What is the company stock price?",
    ]

    for question in questions:
        print(f"Q: {question}")
        answer = rag_chain.invoke(question)
        print(f"A: {answer}")
        print()


if __name__ == "__main__":
    main()