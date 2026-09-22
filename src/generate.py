from ollama import chat

from retrieve import search


MODEL = "llama3.2:3b"


def generate_answer(question):
    # 1. Retrieve relevant FAQs from Qdrant
    results = search(question, limit=3)

    # 2. Build context from retrieved FAQs
    context_parts = []

    for result in results:
        payload = result.payload

        context_parts.append(
            f"Question: {payload['question']}\n"
            f"Answer: {payload['answer']}"
        )

    context = "\n\n".join(context_parts)

    # 3. Build the prompt
    prompt = f""" You are a helpful customer support assistant.

Answer the user's question using ONLY the retrieved knowledge base.

The knowledge base contains the authoritative answer.
If the user's question matches the meaning of a retrieved FAQ,
use that FAQ's answer even if the wording is different.

Do not invent, add, or translate information from outside
the knowledge base.

Answer in English.

If the knowledge base does not contain enough information
to answer the question, say that you do not have enough
information.

Knowledge base:
{context}

User question:
{question}

Answer:
"""

    # 4. Generate answer using local Ollama model
    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response["message"]["content"]


if __name__ == "__main__":
    question = input("Ask a question: ")

    answer = generate_answer(question)

    print()
    print("Generated answer:")
    print("=" * 60)
    print(answer)