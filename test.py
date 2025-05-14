import json
import numpy as np
from pathlib import Path
from data_loader import load_entity_types, parse_ttl_triples
from embedding import get_sentence_encoder, encode_entities, encode_labels
from type_prediction import train_knn_classifier, predict_top_k_types
from pattern_stats import extract_type_patterns, compute_confidences_combined
from anomaly_detector import detect_anomalies_with_isolation_forest
from evaluator import evaluate_predictions
from utils import save_json
from corrupt_triples import generate_corrupt_triples
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from scipy.stats import entropy

from scipy.stats import zscore
import matplotlib.pyplot as plt

def plot_confidence_distribution(confidences, threshold=0):
    plt.hist(confidences, bins=50, alpha=0.7)
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.3f}')
    plt.xlabel('Confidence Score')
    plt.ylabel('Triple Count')
    plt.legend()
    plt.title('Confidence Score Distribution')
    plt.show()

# Main function
def main(config):
    ttl_file = config['ttl_file']
    entity_file = config['entity_type_file']
    k_neighbors = config['k_neighbors']
    top_k_types = config['top_k_types']
    max_triples = config['max_triples']
    corruption_ratio = config['corruption_ratio']

    # Load and Embed Entity Types
    with open(config['corrupted_output'], 'r', encoding='utf-8') as f:
        corrupted_triples = json.load(f)

    original_triples = parse_ttl_triples(ttl_file, max_triples=max_triples)
    model = get_sentence_encoder()
    all_triples = original_triples + corrupted_triples
    all_labels = [0] * len(original_triples) + [1] * len(corrupted_triples)

    with open(config['subject_output'], 'r') as f:
        subj_types = json.load(f)

    with open(config['object_output'], 'r') as f:
        obj_types = json.load(f)
    predicates = {p for _, p, _ in all_triples}
    predicate_embeddings = {p: model.encode(p) for p in predicates}
    # Type Embedding
    all_types = set(
    t
    for ts in list(subj_types.values()) + list(obj_types.values())
    for t, _ in ts  # unpack (type, score)
)
    type_embeddings = {t: model.encode(t) for t in all_types}

    # Pattern Extraction and Confidence Scoring
    triple_confidences, triple_conf_map = compute_confidences_combined(all_triples,subj_types,obj_types,type_embeddings,predicate_embeddings)

    anomalies = detect_anomalies_with_isolation_forest(triple_conf_map)
    # Evaluation
    metrics = evaluate_predictions(anomalies, original_triples, corrupted_triples)

    save_json(subj_types, config['subject_output'])
    save_json(obj_types, config['object_output'])
    save_json(anomalies, config['anomalies_output'])

    print(f"Evaluation Metrics: {metrics}")
    print(f"Anomalies Detected: {len(anomalies)}")
    plot_confidence_distribution(triple_confidences)

if __name__ == "__main__":
    CONFIG = {
        'ttl_file': 'yago-1.0.0-turtle/yago-1.0.0-turtle.ttl',
        'entity_type_file': 'pipeline/input/qid_types.tsv',
        'k_neighbors': 5,
        'top_k_types': 3,
        'max_triples': 5000,
        'corruption_ratio': 0.1,
        'anomaly_threshold_percentile': 20,
        'corrupted_output': 'pipeline/output/corrupted_triples.json',
        'subject_output': 'pipeline/output/subject_etype.json',
        'object_output': 'pipeline/output/object_etype.json',
        'anomalies_output': 'pipeline/output/anomalies.json',
        'pattern_stats_output': 'pipeline/output/type_patterns.json'
    }
    main(CONFIG)


# def main(config):
#     ttl_file = config['ttl_file']
#     entity_file = config['entity_type_file']
#     k_neighbors = config['k_neighbors']
#     top_k_types = config['top_k_types']
#     max_triples = config['max_triples']
#     corruption_ratio = config['corruption_ratio']


#     #Load and Embed Entity Types
#     with open(config['corrupted_output'],'r',encoding='utf-8') as f:
#         corrupted_triples = json.load(f)
  
#     original_triples = parse_ttl_triples(ttl_file, max_triples=max_triples)
#     model = get_sentence_encoder()
#     all_triples = original_triples + corrupted_triples
#     all_labels = [0] * len(original_triples) + [1] * len(corrupted_triples)

    
#     with open(config['subject_output'],'r') as f:
#         subj_types = json.load(f)

#     with open(config['object_output'],'r') as f:
#         obj_types = json.load(f)

#     # Type Embedding
#     all_types = set(t for ts in list(subj_types.values()) + list(obj_types.values()) for t in ts)
#     type_embeddings = {t: model.encode(t) for t in all_types}

#     # Pattern Extraction and Confidence Scoring
#     # pattern_counts, pattern_examples = extract_type_patterns(all_triples, subj_types, obj_types)
#     # triple_confidences, triple_conf_map = compute_confidences(all_triples, subj_types, obj_types, pattern_counts)
#     triple_confidences, triple_conf_map = compute_confidences_consistency(all_triples, subj_types, obj_types,type_embeddings)

#     # Detect Anomalies
#     anomalies = detect_anomalies_with_isolation_forest(triple_conf_map, contamination=config.get('corruption_ratio', 0.1))

#     # Evaluation
#     metrics = evaluate_predictions(anomalies, original_triples, corrupted_triples)

#     save_json(subj_types, config['subject_output'])
#     save_json(obj_types, config['object_output'])
#     save_json(anomalies, config['anomalies_output'])

#     print(f"Evaluation Metrics: {metrics}")
#     print(f"Anomalies Detected: {len(anomalies)}")


# if __name__ == "__main__":
#     CONFIG = {
#         'ttl_file': 'yago-1.0.0-turtle/yago-1.0.0-turtle.ttl',
#         'entity_type_file': 'pipeline/input/qid_types.tsv',
#         'k_neighbors': 5,
#         'top_k_types': 3,
#         'max_triples': 5000,
#         'corruption_ratio': 0.1,
#         'anomaly_threshold_percentile': 20,
#         'corrupted_output': 'pipeline/output/corrupted_triples.json',
#         'subject_output': 'pipeline/output/subject_etype.json',
#         'object_output': 'pipeline/output/object_etype.json',
#         'anomalies_output': 'pipeline/output/anomalies.json',
#         'pattern_stats_output': 'pipeline/output/type_patterns.json'
#     }
#     main(CONFIG)
