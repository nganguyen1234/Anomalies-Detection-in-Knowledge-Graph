import rdflib
import pandas as pd

def load_entity_types(file_path):
    df = pd.read_csv(file_path, sep='\t', names=['entity', 'type'])
    df.dropna(inplace=True)
    entities = df['entity'].tolist()
    types = df['type'].tolist()
    return entities, types

def get_local_name(iri):
    iri = str(iri)
    return iri.split('/')[-1] if '/' in iri else iri

def parse_ttl_triples(file_path, max_triples=5000):
    g = rdflib.Graph()
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                g.parse(data=line, format='ttl')
                for s, p, o in g:
                    s, p, o = map(get_local_name, [s, p, o])
                    triples.append((s, p, o))
                    if len(triples) >= max_triples:
                        return triples
                g.remove((None, None, None))
            except Exception:
                continue
    return triples