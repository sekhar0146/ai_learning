from sentence_transformers import SentenceTransformer
import numpy as np


def load_embedding_model():
    """
    Loads a free, local embedding model.
    First run will download the model (~90MB) - one time only.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def get_embedding(model, text):
    """
    Converts text into a list of numbers (embedding/vector).
    """
    embedding = model.encode(text)
    return embedding


def cosine_similarity(vec1, vec2):
    """
    Measures how similar two embeddings are.
    Score closer to 1.0 = more similar meaning.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)


def main():
    print("Loading embedding model...")
    model = load_embedding_model()
    print("Model loaded!\n")

    # Three sentences — two similar, one different
    sentence1 = "How many sick days do I get?"
    sentence2 = "What is my sick leave entitlement?"
    sentence3 = "What are the office working hours?"

    # Convert all three to embeddings
    embedding1 = get_embedding(model, sentence1)
    embedding2 = get_embedding(model, sentence2)
    embedding3 = get_embedding(model, sentence3)

    print(f"Sentence 1: {sentence1}")
    print(f"Embedding size: {len(embedding1)} numbers\n")

    # Compare similarities
    score_1_2 = cosine_similarity(embedding1, embedding2)
    score_1_3 = cosine_similarity(embedding1, embedding3)

    print("Similarity Scores:")
    print(f"'{sentence1}' vs '{sentence2}': {score_1_2:.2f}")
    print(f"'{sentence1}' vs '{sentence3}': {score_1_3:.2f}")

    print("\nConclusion:")
    if score_1_2 > score_1_3:
        print("Sentence 1 and 2 are more similar in meaning ✅")
        print("Even though they use different words!")


if __name__ == "__main__":
    main()