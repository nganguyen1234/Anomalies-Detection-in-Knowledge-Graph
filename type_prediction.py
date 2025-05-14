from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
import numpy as np
from embedding import encode_entities

def train_knn_classifier(X, y, k=10):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(X, y)
    return knn, label_encoder, X, y_encoded


def predict_top_k_types(knn, encoder, entities, y_train, label_encoder, top_k=3):
    X = encode_entities(encoder, entities)
    distances, indices = knn.kneighbors(X)
    results = {}
    for i, neighbor_idxs in enumerate(indices):
        neighbor_labels = y_train[neighbor_idxs]
        label_counts = Counter(neighbor_labels)
        top_k_labels = [label_encoder.inverse_transform([lbl])[0] for lbl, _ in label_counts.most_common(top_k)]
        results[entities[i]] = top_k_labels
    return results

from collections import defaultdict, Counter
import numpy as np

GENERIC_TYPES = {"Entity", "Thing", "Resource", "Object"}
def predict_top_k_types_weighted(knn, encoder, entities, y_train, label_encoder, top_k=3, confidence_threshold=0.3):

    X = encoder.encode(entities)
    distances, indices = knn.kneighbors(X)
    results = {}

    for i, (neighbor_idxs, dists) in enumerate(zip(indices, distances)):
        neighbor_labels = y_train[neighbor_idxs]
        weights = np.exp(-np.array(dists))  # inverse-distance weighting

        # Normalize weights
        weights /= np.sum(weights) + 1e-9

        label_scores = defaultdict(float)
        for lbl_idx, w in zip(neighbor_labels, weights):
            label = label_encoder.inverse_transform([lbl_idx])[0]

            # Skip generic types
            if label in GENERIC_TYPES:
                continue

            label_scores[label] += w

        # Sort and keep top-k
        sorted_labels = sorted(label_scores.items(), key=lambda x: -x[1])[:top_k]

        # Compute total confidence
        max_weight = sorted_labels[0][1] if sorted_labels else 0.0

        if max_weight >= confidence_threshold:
            results[entities[i]] = sorted_labels
        else:
            results[entities[i]] = []  

    return results
