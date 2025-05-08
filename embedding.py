from sentence_transformers import SentenceTransformer

def get_sentence_encoder(model_name='all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)

def encode_entities(encoder, entities):
    return encoder.encode(entities, show_progress_bar=True)
