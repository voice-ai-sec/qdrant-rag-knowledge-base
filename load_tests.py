import json


def generate_tests(config):
    dataset = config["dataset"]

    test_cases = []

    with open(dataset, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            test_cases.append({
                "vars": {
                    "item": record["item"]
                }
            })

    return test_cases