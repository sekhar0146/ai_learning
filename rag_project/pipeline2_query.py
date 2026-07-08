import chromadb
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from groq import Groq

def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def load_collection(DB_PATH, collection_name):
    client = chromadb.PersistentClient(path=DB_PATH)   
    collection = client.get_or_create_collection(name=collection_name)
    return collection

def retrieve_chunks(collection, model, question, top_k=3, threshold=0.1):
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
    #print(f"Chunks retrieved: {chunks}")
    print(f"Distances retrieved: {distances}")

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

def generate_answer(groq_client, question, chunks):
    """
    Sends retrieved chunks + question to AI and gets final answer.
    """
    context = "\n".join([f"- {c['chunk']}" for c in chunks])

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": """You are a helpful HR assistant.
Answer questions using ONLY the context provided.
Be concise and direct.
If answer is not in context, say 'I don't have information on this.'"""
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )
    return response.choices[0].message.content

def main():
    load_dotenv()
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(BASE_DIR, "chroma_db")

    model = load_embedding_model()
    #print(model)  # Verify model is loaded
    collection = load_collection(DB_PATH, "hr_policy_collection")
    #print(f"Loaded collection: {collection.name} with {collection.count()} chunks.")

    # Ask user for a query
    print("\n🤖 Ask anything about the data (type 'exit' to quit):\n")
    while True:
        question = input("Your question: ")
        #print(f"Query received: {question}")
        #chunks = retrieve_chunks(collection, model, question, top_k=3, threshold=0.15)
        if question.lower() == "exit":
            break

        else:
            print(f"Query received: {question}")
            chunks = retrieve_chunks(collection, model, question, top_k=3, threshold=0.1)
            if not chunks:
                print("No relevant chunks found. Try rephrasing your question.")
            else:
                print(f"  Found {len(chunks)} relevant chunks:")
                # for c in chunks:
                #     print(f"  Score {c['similarity']}: {c['chunk']}")
                
                answer = generate_answer(groq_client, question, chunks)
                print(f"Answer: {answer}\n")

if __name__ == "__main__":
    main()

