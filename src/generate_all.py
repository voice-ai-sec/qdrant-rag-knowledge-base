import json
from pathlib import Path

from generate import generate_answer


INPUT_FILE = Path("data/faq_example.jsonl")
OUTPUT_FILE = Path("data/rag_model_answer.jsonl")


def main():
    print(f"Reading questions from: {INPUT_FILE}")

    items = []

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            items.append(record["item"])

    print(f"Loaded {len(items)} questions.")

    # Check how many answers already exist
    existing_count = 0

    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_count += 1

    print(f"Existing generated answers: {existing_count}")

    if existing_count >= len(items):
        print("All answers already generated.")
        return

    print(f"Remaining questions: {len(items) - existing_count}")
    print()

    # Append new answers instead of overwriting existing ones
    with OUTPUT_FILE.open("a", encoding="utf-8") as output:

        for index in range(existing_count, len(items)):

            item = items[index]
            question = item["input"]

            print(f"[{index + 1}/{len(items)}] {question}")

            model_answer = generate_answer(question)

            result = {
                "item": {
                    "input": question,
                    "expected_answer": item["expected_answer"],
                    "expected_tool": item["expected_tool"],
                    "expected_category": item["expected_category"],
                    "model_answer": model_answer,
                }
            }

            output.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )

            output.flush()

    print()
    print("Evaluation dataset created.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Items: {len(items)}")


if __name__ == "__main__":
    main()