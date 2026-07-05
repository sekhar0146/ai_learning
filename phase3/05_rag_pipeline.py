import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()


def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def load_groq_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def load_collection(collection_name):
    client = chromadb.PersistentClient(path="phase3/chroma_db")
    return client.get_or_create_collection(name=collection_name)


def retrieve_chunks(collection, model, question, top_k=3, threshold=0.15):
    """
    Finds relevant chunks for the question from Vector DB.
    """
    question_embedding = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k,
        include=["documents", "distances"]
    )

    chunks = results["documents"][0]
    distances = results["distances"][0]

    filtered_chunks = []
    for chunk, distance in zip(chunks, distances):
        similarity = 1 - distance
        if similarity >= threshold:
            filtered_chunks.append(chunk)

    return filtered_chunks


def build_augmented_prompt(question, chunks):
    """
    Combines retrieved chunks + question into one prompt.
    This is the AUGMENTATION step in RAG.
    """
    context = "\n".join([f"- {chunk}" for chunk in chunks])

    prompt = f"""Use the following context to answer the question.
If the answer is not in the context, say 'I don't have information on this.'

Context:
{context}

Question: {question}"""

    return prompt


def generate_answer(groq_client, augmented_prompt):
    """
    Sends augmented prompt to AI and gets final answer.
    This is the GENERATION step in RAG.
    """
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
                "content": augmented_prompt
            }
        ]
    )
    return response.choices[0].message.content


def ask(question, collection, embedding_model, groq_client):
    """
    Full RAG pipeline in one function:
    Question → Retrieve → Augment → Generate → Answer
    """
    print(f"Question: {question}")

    # Step 1: Retrieve relevant chunks
    chunks = retrieve_chunks(collection, embedding_model, question)

    # Step 2: Check if any chunks found
    if not chunks:
        print("Answer: I don't have information on this.\n")
        return

    # Step 3: Build augmented prompt
    augmented_prompt = build_augmented_prompt(question, chunks)

    # Step 4: Generate answer
    answer = generate_answer(groq_client, augmented_prompt)

    print(f"Answer: {answer}\n")


def main():
    print("Initializing RAG pipeline...\n")

    embedding_model = load_embedding_model()
    groq_client = load_groq_client()
    collection = load_collection("hr_policy")

    print("="*50 + "\n")

    # Test questions
    questions = [
        "How many sick days do I get?",
        "Can I work from home?",
        "What is the maternity leave policy?",
        "What is the company stock price?",  # not in our docs
    ]

    for question in questions:
        ask(question, collection, embedding_model, groq_client)
        print("-"*50 + "\n")


if __name__ == "__main__":
    main()