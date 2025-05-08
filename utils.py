import json

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def normalize_triple(triple):
    return tuple(part.strip().lower() for part in triple)