from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import numpy as np
from embedding import encode_entities

def train_knn_classifier(X, y, k=10):
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(X, y)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
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