import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, )):
            return int(obj)
        elif isinstance(obj, (np.floating, )):
            return float(obj)
        elif isinstance(obj, (np.ndarray, )):
            return obj.tolist()
        return super().default(obj)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2,cls=NumpyEncoder)

def normalize_triple(triple):
    return tuple(part.strip().lower() for part in triple)

