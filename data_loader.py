from rdflib import Graph
import pandas as pd


def load_entity_types(file_path,max=None):
    entities = []
    types = []
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        f.readline()  # Skip header
        for line in f:
            if max is not None and count >= max:
                return entities, types 
            data_columns = line.strip().split('\t')
            if len(data_columns) == 4:
                entity = data_columns[1]
                etype = data_columns[3]
                entities.append(entity)
                types.append(etype)
                count += 1
        return entities, types


def get_local_name(iri):
    iri = str(iri)
    return iri.split('/')[-1] if '/' in iri else iri

def parse_ttl_triples(file_path, max_triples=None):
    g = Graph()
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                g.parse(data=line, format='ttl')
                for s, p, o in g:
                    s, p, o = map(get_local_name, [s, p, o])
                    triples.append((s, p, o))
                    if max_triples is not None and len(triples) >= max_triples:
                        return triples
                g.remove((None, None, None))
            except Exception:
                continue
    return triples


# Define prefixes used in TTL file
PREFIXES = """
@prefix yago: <http://yago-knowledge.org/resource/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix ontolex: <http://www.w3.org/ns/lemon/ontolex#> .
@prefix dct: <http://purl.org/dc/terms/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix wikibase: <http://wikiba.se/ontology#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix schema: <http://schema.org/> .
@prefix cc: <http://creativecommons.org/ns#> .
@prefix geo: <http://www.opengis.net/ont/geosparql#> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix wd: <http://www.wikidata.org/entity/> .
@prefix data: <https://www.wikidata.org/wiki/Special:EntityData/> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix s: <http://www.wikidata.org/entity/statement/> .
@prefix ref: <http://www.wikidata.org/reference/> .
@prefix v: <http://www.wikidata.org/value/> .
@prefix wdt: <http://www.wikidata.org/prop/direct/> .
@prefix wpq: <http://www.wikidata.org/prop/quant/> .
@prefix wdtn: <http://www.wikidata.org/prop/direct-normalized/> .
@prefix p: <http://www.wikidata.org/prop/> .
@prefix ps: <http://www.wikidata.org/prop/statement/> .
@prefix psv: <http://www.wikidata.org/prop/statement/value/> .
@prefix psn: <http://www.wikidata.org/prop/statement/value-normalized/> .
@prefix pq: <http://www.wikidata.org/prop/qualifier/> .
@prefix pqv: <http://www.wikidata.org/prop/qualifier/value/> .
@prefix pqn: <http://www.wikidata.org/prop/qualifier/value-normalized/> .
@prefix pr: <http://www.wikidata.org/prop/reference/> .
@prefix prv: <http://www.wikidata.org/prop/reference/value/> .
@prefix prn: <http://www.wikidata.org/prop/reference/value-normalized/> .
@prefix wdno: <http://www.wikidata.org/prop/novalue/> .
@prefix ys: <http://yago-knowledge.org/schema#> .
"""

from rdflib import Graph, URIRef, BNode

def parse_yago5_ttl_triples(file_path, max_triples=None):
    triples = []
    current_block = ""

    with open(file_path, 'r', encoding='utf-8') as f:
        print("start parse ttl file")
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            current_block += line

            # Only parse when a block is likely finished
            if stripped.endswith('.'):
                try:
                    g = Graph()
                    g.parse(data=PREFIXES + current_block, format='turtle')
                    for s, p, o in g:
                        s_q = g.qname(s)
                        p_q = g.qname(p)
                        o_q = g.qname(o) if isinstance(o, (URIRef, BNode)) else str(o)
                        triples.append([s_q, p_q, o_q])
                        if max_triples and len(triples) >= max_triples:
                            print(f"Loaded {len(triples)} triples")
                            return triples
                except Exception as e:
                    print(f"Skipped block due to parse error: {e}")
                current_block = ""

    print(f"Total triples loaded: {len(triples)}")
    return triples
