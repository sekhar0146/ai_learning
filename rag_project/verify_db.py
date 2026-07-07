import chromadb
import os

def verify_db():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(BASE_DIR, "chroma_db")
    print(f"Looking for DB at: {DB_PATH}")  # add this to verify path

    client = chromadb.PersistentClient(path=DB_PATH)

    # Check all collections
    collections = client.list_collections()
    print(f"Total collections: {len(collections)}")
    print(f"Collections: {[c.name for c in collections]}")
    print("="*50)

    # Check specific collection
    collection = client.get_collection("hr_policy_collection")

    # Total chunks stored
    total = collection.count()
    print(f"Total chunks in collection: {total}")
    print("="*50)

    # Preview first 3 chunks with metadata
    results = collection.get(
        limit=3,
        include=["documents", "metadatas"]
    )

    print("Preview of first 3 chunks:")
    for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
        print(f"\nChunk {i+1}:")
        print(f"  Text: {doc[:100]}...")  # first 100 chars
        print(f"  Metadata: {meta}")

def main():
    verify_db()

if __name__ == "__main__":
    main()