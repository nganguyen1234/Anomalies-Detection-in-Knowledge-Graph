import json
import csv
import re
from openai import OpenAI
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

client = OpenAI(api_key="")
original_file = "pipeline/output/anomalies.json"
corrupted_file = "pipeline/output/corrupted_triples.json"
csv_output_file = "pipeline/output/anomaly_report_chatgpt.csv"
# Convert to format: s -> p -> o
def triple_to_str(triple):
    return f"{triple[0]} -> {triple[1]} -> {triple[2]}"

# Send triples to LLM
def send_in_chunks(triples, chunk_size=30, max_chunks=175):
    results = []
    for i in range(0, len(triples), chunk_size):
        if i // chunk_size >= max_chunks:
            break

        chunk = triples[i:i + chunk_size]
        query = (
            f"The following are triples from a knowledge graph:\n\n"
            f"{'; '.join(chunk)}\n\n"
            "You're an experienced data analyst. Identify anomalies, their anomaly types, and suggest how to correct them. "
            "Provide specific reasons for any invalid triples. "
            "Strictly follow this format only: 'Triple, Why it's anomaly (max 10 words), Suggestion'. "
            "Skip valid triples. Do not explain or summarize. Only list incorrect triples."
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": query}],
                max_tokens=1500
            )
            result_text = response.choices[0].message.content.strip()
            result_lines = result_text.split("\n")

            for line in result_lines:
                parts = line.split(",", 2)
                if len(parts) < 3:
                    continue

                triple = parts[0].strip()
                reason = parts[1].strip()
                suggestion = parts[2].strip()

                if "->" in triple:
                    results.append((triple, reason, suggestion))

            print(f"Processed chunk {i // chunk_size + 1}")
        except Exception as e:
            print(f"Error in chunk {i // chunk_size + 1}: {e}")
    return results

# Evaluation

def parse_gpt_triple(triple_str):
    parts = [p.strip() for p in triple_str.split("->")]
    return tuple(parts) if len(parts) == 3 else None

def load_gpt_anomalies(file_path):
    anomalies = set()
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            triple = parse_gpt_triple(row["Triple"])
            if triple:
                anomalies.add(triple)
    return anomalies

def load_corrupted_triples(file_path):
    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)
    corrupted = set()
    for s, p, o in data:
        corrupted.add((s.strip(), p.strip(), o.strip()))
    return corrupted



# Match triples exactly or fuzzily
def evaluate_predictions(gpt_anomalies, corrupted_triples):
    y_true, y_pred = [], []
    for triple in gpt_anomalies:
        is_anomaly = False
        if triple in corrupted_triples:
            is_anomaly = True
        y_pred.append(1)
        y_true.append(1 if is_anomaly else 0)

    # For corrupted triples not predicted by GPT
    missed = [1] * (len(corrupted_triples) - sum(y_true))
    y_true += missed
    y_pred += [0] * len(missed)

    return y_true, y_pred

if __name__ == "__main__":

# Load triples
    with open(original_file, "r", encoding="utf-8") as f:
        triples_raw = json.load(f)

    with open(corrupted_file, "r", encoding="utf-8") as f:
        corrupted_raw = json.load(f)

    triples = [triple_to_str(t) for t in triples_raw]
    corrupted_triples = set([triple_to_str(t) for t in corrupted_raw])

    results = send_in_chunks(triples, chunk_size=30, max_chunks=350)
    # Save to CSV
    with open(csv_output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Triple", "Anomaly Description", "Correction Suggestion"])
        for triple, reason, suggestion in results:
            writer.writerow([triple, reason, suggestion])
    print(f"Anomaly results saved to {csv_output_file}")
    gpt_anomalies = load_gpt_anomalies(csv_output_file)
    corrupted_triples = load_corrupted_triples(corrupted_file)
    print(f"GPT Anomalies: {len(gpt_anomalies)}")
    print(f"Corrupted Triples: {len(corrupted_triples)}")
    print("\nEvaluating...")
    y_true, y_pred = evaluate_predictions(gpt_anomalies, corrupted_triples)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    print("\nEvaluation Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
