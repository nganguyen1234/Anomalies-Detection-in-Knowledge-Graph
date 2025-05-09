from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict

def extract_type_patterns(triples, subj_types, obj_types):
    pattern_counts = defaultdict(int)
    pattern_examples = defaultdict(list)
    for s, p, o in triples:
        if p == 'rdf:type':
            continue
        s_labels = subj_types.get(s, [])
        o_labels = obj_types.get(o, [])
        for stype in s_labels:
            for otype in o_labels:
                pattern = (stype, p, otype)
                pattern_counts[pattern] += 1
                if len(pattern_examples[pattern]) < 3:
                    pattern_examples[pattern].append((s, p, o))
    return pattern_counts, pattern_examples

def compute_confidences(triples, subj_types, obj_types, pattern_counts):
    total = sum(pattern_counts.values())
    pattern_probs = {pat: cnt / total for pat, cnt in pattern_counts.items()}
    triple_conf_map = {}
    confidences = []
    for s, p, o in triples:
        if p == 'rdf:type':
            continue
        s_labels = subj_types.get(s, [])
        o_labels = obj_types.get(o, [])
        scores = [pattern_probs.get((st, p, ot), 0) for st in s_labels for ot in o_labels]
        max_conf = max(scores) if scores else 0.0
        triple_conf_map[(s, p, o)] = {
            'confidence': max_conf,
            's_labels': s_labels,
            'o_labels': o_labels
        }
        confidences.append(max_conf)
    return confidences, triple_conf_map

def compute_confidences_cosine(triples, subj_types, obj_types, type_embeddings):
    triple_conf_map = {}
    confidences = []

    for s, p, o in triples:
        if p == 'rdf:type':
            continue

        s_labels = subj_types.get(s, [])
        o_labels = obj_types.get(o, [])
        best_score = 0.0

        for st in s_labels:
            for ot in o_labels:
                vec_st = type_embeddings.get(st)
                vec_ot = type_embeddings.get(ot)

                if vec_st is not None and vec_ot is not None:
                    score = cosine_similarity([vec_st], [vec_ot])[0][0]
                    best_score = max(best_score, score)

        triple_conf_map[(s, p, o)] = {
            'confidence': best_score,
            's_labels': s_labels,
            'o_labels': o_labels
        }
        confidences.append(best_score)

    return confidences, triple_conf_map
