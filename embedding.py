from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder

def get_sentence_encoder(model_name='all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)

def encode_entities(encoder, entities):
    return encoder.encode(entities, show_progress_bar=True)

def encode_labels(labels):
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return y
