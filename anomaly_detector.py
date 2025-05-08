import numpy as np

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

from sklearn.ensemble import IsolationForest
import numpy as np

def detect_anomalies_with_isolation_forest(triple_conf_map, contamination=0.1):
    triples = []
    features = []
    
    for (s, p, o), info in triple_conf_map.items():
        confidence = info.get("confidence", 0.0)
        s_type_count = len(info.get("s_labels", []))
        o_type_count = len(info.get("o_labels", []))
        triples.append([s, p, o])
        features.append([confidence, s_type_count, o_type_count])
    
    features = np.array(features)
    
    clf = IsolationForest(contamination=contamination, random_state=42)
    preds = clf.fit_predict(features)  # -1 = anomaly, 1 = normal

    anomalies = []
    debug_info = []
    for i, pred in enumerate(preds):
        if pred == -1:
            s, p, o = triples[i]
            info = triple_conf_map[(s, p, o)]
            anomalies.append([s, p, o])
            debug_info.append({
                "triple": [s, p, o],
                "s_types": info.get("s_labels", []),
                "o_types": info.get("o_labels", []),
                "matched_confidence": info.get("confidence", 0.0),
                "reason": "IsolationForest detected anomaly"
            })
    
    return anomalies, debug_info
