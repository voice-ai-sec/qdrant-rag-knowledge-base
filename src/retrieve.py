import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "faq_knowledge_base",
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    cloud_inference=True,
)


def search(question, limit=3):

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=models.Document(
            text=question,
            model=EMBEDDING_MODEL,
        ),
        with_payload=True,
        limit=limit,
    )

    return results.points


if __name__ == "__main__":

    question = input("Ask a question: ")

    results = search(question)

    print()
    print("Retrieved knowledge:")
    print("=" * 60)

    for i, result in enumerate(results, start=1):

        payload = result.payload

        print(f"\n#{i}")
        print(f"Score: {result.score:.4f}")
        print(f"Category: {payload['category']}")
        print(f"Question: {payload['question']}")
        print(f"Answer: {payload['answer']}")