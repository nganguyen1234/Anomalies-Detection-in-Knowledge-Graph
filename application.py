# main pipeline
import json
from pathlib import Path
from data_loader import load_entity_types, parse_ttl_triples, parse_yago5_ttl_triples
from embedding import get_sentence_encoder, encode_entities, encode_labels
from type_prediction import train_knn_classifier, predict_top_k_types, predict_top_k_types_weighted
from pattern_stats import extract_type_patterns, compute_confidences_cosine, compute_confidences_combined
from anomaly_detector import detect_anomalies_with_isolation_forest
from evaluator import evaluate_predictions, pattern_confidence_heatmap, plot_confidence_distribution, visualize_embeddings_pca
from utils import save_json
import random
from corrupt_triples import generate_corrupt_triples
from sklearn.metrics import classification_report



def main(config):
    ttl_file = config['ttl_file']
    entity_file = config['entity_type_file']
    k_neighbors = config['k_neighbors']
    top_k_types = config['top_k_types']
    max_etype_pairs = config['max_entity_type_pairs']
    max_triples = config['max_triples']
    corruption_ratio = config['corruption_ratio']

    #Load and Embed Entity Types
    entities, labels = load_entity_types(entity_file,max=max_etype_pairs)
    model = get_sentence_encoder()
    X = encode_entities(model, entities)


    # Train Classifier for Type Extract
    knn, label_encoder, X_train, y_train = train_knn_classifier(X, labels, k=k_neighbors)

    ## Parse Turtle RDF
    # Load yago1 dataset
    # original_triples = parse_ttl_triples(ttl_file, max_triples=max_triples)

    # Load yago5 dataset
    original_triples = parse_yago5_ttl_triples(ttl_file, max_triples=max_triples)

    #Generate Corrupted Triples
    corrupted_triples = generate_corrupt_triples(original_triples, config['corrupted_output'], corruption_ratio)
    all_triples = original_triples + corrupted_triples
    all_labels = [0] * len(original_triples) + [1] * len(corrupted_triples)

    # Predict Subject/Object Types
    subj_entities = list({s for s, p, o in all_triples if p != "rdf:type"})
    obj_entities = list({o for s, p, o in all_triples if p != "rdf:type"})
    subj_types = predict_top_k_types_weighted(knn, model, subj_entities, y_train, label_encoder, top_k=top_k_types)
    obj_types = predict_top_k_types_weighted(knn, model, obj_entities, y_train, label_encoder, top_k=top_k_types)


    # Create predicate embeddings
    predicates = {p for _, p, _ in all_triples}
    predicate_embeddings = {p: model.encode(p) for p in predicates}

    # Type Embedding
    all_types = set(t for ts in list(subj_types.values()) + list(obj_types.values()) for t in ts)
    type_embeddings = {t: model.encode(t) for t in all_types}

    # Compute the triple confidences based on pattern
    triple_confidences, triple_conf_map = compute_confidences_combined(
    all_triples, subj_types, obj_types, type_embeddings, predicate_embeddings
)
    print("finish compute triple confidence")


    # Detect Anomalies
    anomalies = detect_anomalies_with_isolation_forest(triple_conf_map,0.1)
    print("finish detect anomalies")

    # Evaluation
    metrics = evaluate_predictions(anomalies, original_triples, corrupted_triples)

    save_json(subj_types, config['subject_output'])
    save_json(obj_types, config['object_output'])
    save_json(anomalies, config['anomalies_output'])

    print(f"Evaluation Metrics: {metrics}")
    print(f"Anomalies Detected: {len(anomalies)}")
if __name__ == "__main__":
    CONFIG = {
        'ttl_file': 'yago-1.0.0-turtle/yago-1.0.0-turtle.ttl',
        # 'ttl_file': 'yago-4.5.0.1-tiny/yago-tiny.ttl',
        'entity_type_file': 'pipeline/input/qid_types.tsv',
        'k_neighbors': 5,
        'top_k_types': 1,
        'max_triples': 100000,  #Set to None to load entire file
        'max_entity_type_pairs':10000, #Set to None to load entire file
        'corruption_ratio': 0.05,
        'anomaly_threshold_percentile': 5,
        'corrupted_output': 'pipeline/output/corrupted_triples.json',
        'subject_output': 'pipeline/output/subject_etype.json',
        'object_output': 'pipeline/output/object_etype.json',
        'anomalies_output': 'pipeline/output/anomalies.json',
        'pattern_stats_output': 'pipeline/output/type_patterns.json'
    }
    main(CONFIG)