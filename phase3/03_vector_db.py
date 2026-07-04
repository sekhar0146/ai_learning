import chromadb
from sentence_transformers import SentenceTransformer


def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def create_vector_db():
    """
    Creates a local ChromaDB instance.
    Data is stored in a folder called 'chroma_db' in your project.
    """
    client = chromadb.PersistentClient(path="phase3/chroma_db")
    return client


def create_collection(client, collection_name):
    """
    A collection is like a table in SQL — stores related chunks together.
    get_or_create: creates if not exists, loads if already exists.
    """
    collection = client.get_or_create_collection(name=collection_name)
    return collection


def add_chunks_to_db(collection, model, chunks):
    """
    Converts chunks to embeddings and stores them in Vector DB.
    """
    # Convert all chunks to embeddings
    embeddings = model.encode(chunks).tolist()

    # Store in ChromaDB
    collection.add(
        documents=chunks,                                 # original text
        embeddings=embeddings,                            # converted numbers
        ids=[f"chunk_{i}" for i in range(len(chunks))]  # unique id for each chunk
    )
    print(f"✅ {len(chunks)} chunks stored in Vector DB")


def search_vector_db(collection, model, question, top_k=3):
    """
    Converts question to embedding and finds most similar chunks.
    """
    # Convert question to embedding
    question_embedding = model.encode([question]).tolist()

    # Search Vector DB
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )

    return results["documents"][0]  # returns list of matching chunks


def main():
    # Sample HR policy chunks
    chunks = [
        "Sick leave entitlement is 12 days per year for all full time employees.",
        "Sick leave can be carried forward up to 5 days to the next calendar year.",
        "Employees must notify their manager before 9 AM on the day of absence.",
        "Medical certificate is required for sick leave exceeding 3 consecutive days.",
        "Maternity leave is 26 weeks as per company policy and government regulations.",
        "Paternity leave is 5 days and must be taken within 3 months of childbirth.",
        "Office working hours are 9 AM to 6 PM Monday to Friday.",
        "Work from home is allowed up to 2 days per week with manager approval.",
        "Annual leave entitlement is 20 days per year for all permanent employees.",
        "Annual leave must be applied 2 weeks in advance through the HR portal.",
    ]

    print("Setting up Vector DB...")
    model = load_embedding_model()
    client = create_vector_db() # A kind of a database
    collection = create_collection(client, "hr_policy") # It's a kind of a table in SQL

    print("Adding chunks to Vector DB...")
    add_chunks_to_db(collection, model, chunks)

    print("\n" + "="*50 + "\n")

    # Test with different questions
    questions = [
        "How many sick days do I get?",
        "Can I work from home?",
        "What is the maternity leave policy?"
    ]

    for question in questions:
        print(f"Question: {question}")
        print("Top matching chunks:")
        matches = search_vector_db(collection, model, question, top_k=2)
        for i, match in enumerate(matches):
            print(f"  {i+1}. {match}")
        print()


if __name__ == "__main__":
    main()