from xml.parsers.expat import model

import chromadb
from sentence_transformers import SentenceTransformer


def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def load_collection(collection_name):
    """
    Loads existing Vector DB and collection.
    No need to re-add chunks — data is already persisted from 03_vector_db.py
    """
    client = chromadb.PersistentClient(path="phase3/chroma_db")
    collection = client.get_or_create_collection(name=collection_name)
    return collection


def retrieve_chunks(collection, model, question, top_k=3, threshold=0.15):
    """
    Retrieves relevant chunks for a question.
    threshold: minimum similarity score — filters out irrelevant chunks.
    """
    # Convert question to embedding
    question_embedding = model.encode([question]).tolist()

    # Search Vector DB — also request similarity scores
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k,
        include=["documents", "distances"]
    )

    chunks = results["documents"][0]
    distances = results["distances"][0]

    # ChromaDB returns distance (lower = more similar)
    # Convert distance to similarity score (0 to 1)
    filtered_chunks = []
    for chunk, distance in zip(chunks, distances):
        similarity = 1 - distance
        if similarity >= threshold:
            filtered_chunks.append({
                "chunk": chunk,
                "similarity": round(similarity, 2)
            })

    return filtered_chunks


def main():
    model = load_embedding_model()
    collection = load_collection("hr_policy")

    questions = [
        "How many sick days do I get?",       # should match well
        "What is the company stock price?",    # should NOT match — not in our docs
        "Can I work from home?",               # should match well
    ]

    for question in questions:
        print(f"Question: {question}")
        chunks = retrieve_chunks(collection, model, question, top_k=3, threshold=0.15)

        if not chunks:
            print("  ⚠️ No relevant chunks found above threshold.")
            print("  AI will say: 'I don't have information on this.'")
        else:
            print(f"  Found {len(chunks)} relevant chunks:")
            for c in chunks:
                print(f"  Score {c['similarity']}: {c['chunk']}")
        print()

    # =====> Debug — see raw scores for work from home question
    # print("DEBUG - Raw scores for 'Can I work from home?'")
    # question = "Can I work from home?"
    # question_embedding = model.encode([question]).tolist()
    # results = collection.query(
    #     query_embeddings=question_embedding,
    #     n_results=5,
    #     include=["documents", "distances"]
    # )
    # for doc, dist in zip(results["documents"][0], results["distances"][0]):
    #     print(f"  Score {round(1-dist, 2)}: {doc}")

if __name__ == "__main__":
    main()