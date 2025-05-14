from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def evaluate_predictions(predicted_anomalies, original_triples, corrupted_triples):
    def normalize(triple):
        return tuple(part.strip().lower() for part in triple)
    # corrupted_triples = [tuple(triple) for triple in corrupted_triples]
    # predicted_anomalies = [tuple(d['triple']) for d in predicted_anomalies]
    anomaly_set = set(normalize(t) for t in predicted_anomalies)
    all_triples = original_triples + corrupted_triples
    y_true = [0] * len(original_triples) + [1] * len(corrupted_triples)
    y_pred = [1 if normalize(triple) in anomaly_set else 0 for triple in all_triples]
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }