import json
import os
import hashlib

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "faq_knowledge_base")

DATA_FILE = "data/faq_example.jsonl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    cloud_inference=True,
)


def load_faqs():
    items = []

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            item = record["item"]

            items.append(item)

    return items


def create_collection():
    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection already exists: {COLLECTION_NAME}")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
        ),
    )

    print(f"Created collection: {COLLECTION_NAME}")


def make_id(question):
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


def upload_faqs(items):
    points = []

    for item in items:
        question = item["input"]
        answer = item["expected_answer"]
        category = item["expected_category"]
        expected_tool = item["expected_tool"]

        text = f"Question: {question}\nAnswer: {answer}"

        points.append(
            models.PointStruct(
                id=make_id(question),
                vector=models.Document(
                    text=text,
                    model=EMBEDDING_MODEL,
                ),
                payload={
                    "question": question,
                    "answer": answer,
                    "category": category,
                    "expected_tool": expected_tool,
                },
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(f"Uploaded {len(points)} FAQ items.")


def main():
    print("Loading FAQs...")

    items = load_faqs()

    print(f"Loaded {len(items)} FAQs.")

    create_collection()

    upload_faqs(items)

    info = client.get_collection(COLLECTION_NAME)

    print()
    print("Indexing complete.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Points: {info.points_count}")


if __name__ == "__main__":
    main()