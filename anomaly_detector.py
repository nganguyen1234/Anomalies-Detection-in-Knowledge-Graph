import numpy as np
from sklearn.ensemble import IsolationForest

def detect_anomalies(triple_conf_map, threshold_percentile=10):
    confidences = [info['confidence'] for info in triple_conf_map.values()]
    threshold = np.percentile(confidences, threshold_percentile)
    anomalies = []
    debug_info = []
    for (s, p, o), info in triple_conf_map.items():
        if info['confidence'] < threshold:
            anomalies.append([s, p, o])
            debug_info.append({
                'triple': [s, p, o],
                's_types': info['s_labels'],
                'o_types': info['o_labels'],
                'matched_confidence': info['confidence'],
                'reason': 'Low pattern confidence'
            })
    return anomalies, debug_info


def detect_anomalies_with_isolation_forest(triple_conf_map, contamination=0.1):
    triples = []
    features = []

    for (s, p, o), info in triple_conf_map.items():
        confidence = info.get("confidence", 0.0)
        s_type_count = len(info.get("s_labels", []))
        o_type_count = len(info.get("o_labels", []))
        entropy_s = info.get("entropy_s", 0.0)
        entropy_o = info.get("entropy_o", 0.0)
        pattern_freq = info.get("pattern_freq", 0.0)

        feature_vec = [confidence, s_type_count, o_type_count, entropy_s, entropy_o, pattern_freq]
        features.append(feature_vec)
        triples.append([s, p, o])

    features = np.array(features)

    clf = IsolationForest(contamination=contamination, random_state=42)
    preds = clf.fit_predict(features)

    anomalies = [triples[i] for i, pred in enumerate(preds) if pred <= 0]
    return anomalies
