import numpy as np
from collections import Counter
import json
import rdflib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from corrupt_triples import generate_corrupt_triples
# Load entity and label data
def load_data(file_path):
    training_entities = []
    training_labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            data_columns = line.strip().split('\t')
            if len(data_columns) < 4:
                continue
            entity = data_columns[1]
            etype = data_columns[3]
            training_entities.append(entity)
            training_labels.append(etype)
    return training_entities, training_labels

input_file = 'pipeline/qid_types.tsv'
training_entities, training_labels = load_data(input_file)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(training_labels)

# Sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
X = model.encode(training_entities)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=10, metric="cosine")
knn.fit(X_train, y_train)

# Parse Turtle triples
def get_local_name(iri):
    return iri.split("/")[-1] if "/" in iri else iri

def parse_turtle(file_path):
    g = rdflib.Graph()
    triples = []
    print(f"Loaded {len(triples)} triples")
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        print("Start parsing")
        for line in f:
            try:
                # Parse line by line
                g.parse(data=line, format='ttl')
                
                # Loop through the parsed triples
                for s, p, o in g:
                    s = get_local_name(s)
                    p = get_local_name(p)
                    o = get_local_name(o)
                    triples.append((str(s), str(p), str(o)))
                    count += 1
                    
                    # Stop after reaching the desired number of triples
                    if count >= 5000:
                        print(f"Loaded {len(triples)} triples")
                        return triples
                g.remove((None, None, None))  # Clear graph after processing each line
            except Exception:
                continue
    print(f"Loaded {len(triples)} triples")
    return triples

turtle_file = 'yago-1.0.0-turtle/yago-1.0.0-turtle.ttl'
subject_output_json = 'pipeline/subject_etype.json'
object_output_json = 'pipeline/object_etype.json'
corrupted_file = 'pipeline/corrupted_triples.json'

# === Load corrupted triples and inject them ===
# with open(corrupted_file, "r", encoding="utf-8") as f:
#     corrupted_triples = json.load(f)

original_triples = parse_turtle(turtle_file)
print(f'finish with {len(original_triples)}')
corrupted_triples = generate_corrupt_triples(original_triples, 'pipeline/corrupted_triples.json', corruption_ratio=0.1)
triples = corrupted_triples + original_triples
labels = [0] * len(original_triples) + [1] * len(corrupted_triples)
subject_to_classify = set([s for s, p, o in triples if p != 'rdf:type'])
subject_list = list(subject_to_classify)

# Encode new entities
X_subject = model.encode(subject_list)

# Get k nearest neighbors
s_distances, s_indices = knn.kneighbors(X_subject, n_neighbors=10)

# Predict top 3 most frequent labels among neighbors
subject_results = {}
for i, neighbor_idxs in enumerate(s_indices):
    s_neighbor_labels = y_train[neighbor_idxs]
    s_label_counts = Counter(s_neighbor_labels)
    top3 = [le.inverse_transform([label])[0] for label, _ in s_label_counts.most_common(3)]
    subject_results[subject_list[i]] = top3

# Save results
with open(subject_output_json, 'w', encoding='utf-8') as f:
    json.dump(subject_results, f, indent=2)
print(f"Predicted and saved top 3 entity types for {len(subject_results)} entities to {subject_output_json}")

object_to_classify = set([o for s, p, o in triples if p != 'rdf:type'])
object_list = list(object_to_classify)

# Encode new entities
X_object = model.encode(object_list)

# Get k nearest neighbors
o_distances, o_indices = knn.kneighbors(X_object, n_neighbors=10)

# Predict top 3 most frequent labels among neighbors
o_results = {}
for i, neighbor_idxs in enumerate(o_indices):
    o_neighbor_labels = y_train[neighbor_idxs]
    o_label_counts = Counter(o_neighbor_labels)
    top3 = [le.inverse_transform([label])[0] for label, _ in o_label_counts.most_common(3)]
    o_results[object_list[i]] = top3

# Save results
with open(object_output_json, 'w', encoding='utf-8') as f:
    json.dump(o_results, f, indent=2)

print(f"Predicted and saved top 3 entity types for {len(o_results)} entities to {object_output_json}")

from collections import defaultdict
MIN_PATTERN_SUPPORT = 5
ANOMALY_SCORE_THRESHOLD = 0.5
with open(subject_output_json,'r') as f:
    subj_types = json.load(f)

with open(object_output_json,'r') as f:
    obj_types = json.load(f)

pattern_counts = defaultdict(int)
pattern_examples = defaultdict(list)

for s,p,o in triples:
    if p == 'rdf:type':
        continue

    s_labels = subj_types.get(s,[])
    o_labels = obj_types.get(o,[])

    for stype in s_labels:
        for otype in o_labels:
            pattern = (stype,p,otype)
            pattern_counts[pattern] += 1
            if len(pattern_examples[pattern]) < 3:
                pattern_examples[pattern].append((s,p,o))

total_pattern_counts = sum(pattern_counts.values())
pattern_probs = {
    pattern: count/ total_pattern_counts
    for pattern, count in pattern_counts.items()
    # if count >= MIN_PATTERN_SUPPORT
}
anomalies = []
pattern_confidences = []
triple_conf_map = {}  # store confidence for each triple
debug_info = []
for s, p, o in triples:
    if p == 'rdf:type':
        continue

    s_labels = subj_types.get(s, [])
    o_labels = obj_types.get(o, [])
    pattern_scores = []

    for stype in s_labels:
        for otype in o_labels:
            pattern = (stype, p, otype)
            if pattern in pattern_probs:
                pattern_scores.append(pattern_probs[pattern])

    # Find the max confidence across all possible patterns
    max_confidence = max(pattern_scores) if pattern_scores else 0.0

    pattern_confidences.append(max_confidence)
    triple_conf_map[(s, p, o)] = {
        "confidence": max_confidence,
        "s_labels": s_labels,
        "o_labels": o_labels
    }

# Use percentile threshold to flag bottom 5% confidence patterns
threshold = np.percentile(pattern_confidences, 10)

for (s, p, o), info in triple_conf_map.items():
    if info["confidence"] < threshold:
        anomalies.append([s, p, o])
        debug_info.append({
            "triple": [s, p, o],
            "s_types": info["s_labels"],
            "o_types": info["o_labels"],
            "matched_confidence": info["confidence"],
            "reason": "All type-based patterns have low support/confidence"
        })
corrupted_set = set(tuple(t) for t in corrupted_triples)
# perfect recall but others low
# y_pred = [1] * len(anomalies)
# y_true = [1 if tuple(triple) in corrupted_set else 0 for triple in anomalies]

def normalize(triple):
    return tuple(part.strip().lower() for part in triple)

# Predict based on whether the normalized triple is in the anomaly set
# Accuracy : 0.8658 Precision: 0.2796 Recall   : 0.3020 F1 Score : 0.2904

anomaly_set = set(normalize(t) for t in anomalies)
all_triples = original_triples + corrupted_triples
y_true = [0] * len(original_triples) + [1] * len(corrupted_triples)
y_pred = [1 if normalize(triple) in anomaly_set else 0 for triple in all_triples]
precision = precision_score(y_true,y_pred)
recall = recall_score(y_true,y_pred)
f1 = f1_score(y_true,y_pred)
accuracy = accuracy_score(y_true,y_pred)


print("Evaluation Results:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")



with open("pipeline/anomalies.json","w") as f:
    json.dump(anomalies,f,indent=2)

with open("pipeline/debug_anomaly_info.json","w") as f:
    json.dump(debug_info,f,indent=2)

with open("pipeline/type_patterns.json","w") as f:
    json.dump({f"{st},{p},{ot}":count for (st,p,ot),count in pattern_counts.items()},f,indent=2)
    
print(f"Detected {len(anomalies)} anomalous triples based on pattern confidence.")
