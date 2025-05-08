import random
import json
import rdflib
turtle_file = 'yago-1.0.0-turtle/yago-1.0.0-turtle.ttl'

# Parse Turtle triples
def get_local_name(iri):
    return iri.split("/")[-1] if "/" in iri else iri

def parse_turtle(file_path):
    g = rdflib.Graph()
    triples = []
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        print("Start parsing")
        for line in f:
            try:
                g.parse(data=line, format='ttl')
                for s, p, o in g:
                    s = get_local_name(s)
                    p = get_local_name(p)
                    o = get_local_name(o)
                    triples.append((str(s), str(p), str(o)))
                    count += 1
                g.remove((None, None, None))
            except Exception:
                continue
            if count >= 10000:
                print(f'Loaded {len(triples)} triples')
                return triples[8000:10000]
            
# def corrupt_triples(triples, output_file, corruption_ratio=0.05):
#     triples_len = len(triples)
#     print(f"Total triples: {triples_len}")
#     target_corruption_count = int(triples_len * corruption_ratio)
#     print(f"Target corrupted triples: {target_corruption_count}")
    
#     corrupted_triples = []
#     corrupted_count = 0
#     random.shuffle(triples)

#     for triple in triples:
#         if corrupted_count >= target_corruption_count:
#             break

#         from_entity = triple[0]
#         relationship = triple[1]
#         to_entity = triple[2]

#         triple_obj = {
#             "from": from_entity,
#             "label": relationship,
#             "to": to_entity
#         }

#         if corrupted_count < target_corruption_count * 0.2:
#             triple_obj["from"] = to_entity
#             triple_obj["to"] = from_entity
#             triple_obj["anomaly_description"] = "swap s and o, preserve relationship"
#         elif corrupted_count < target_corruption_count * 0.4:
#             triple_obj["from"] = to_entity
#             triple_obj["to"] = from_entity
#             triple_obj["label"] = random.choice(triples)[1]
#             triple_obj["anomaly_description"] = "swap s with o, change relationship"
#         elif corrupted_count < target_corruption_count * 0.6:
#             triple_obj["from"] = random.choice(triples)[2]
#             triple_obj["label"] = random.choice(triples)[1]
#             triple_obj["anomaly_description"] = "change s and relationship"
#         elif corrupted_count < target_corruption_count * 0.8:
#             triple_obj["to"] = random.choice(triples)[2]
#             triple_obj["label"] = random.choice(triples)[1]
#             triple_obj["anomaly_description"] = "change o and relationship"
#         else:
#             triple_obj["from"] = random.choice(triples)[0]
#             triple_obj["to"] = random.choice(triples)[2]
#             triple_obj["anomaly_description"] = "change s and o"

#         corrupted_triples.append(triple_obj)
#         corrupted_count += 1

#     print(f"Total corrupted: {corrupted_count}")
#     with open(output_file, 'w', encoding='utf-8') as f_out:
#         json.dump({"metaedge_tuples": corrupted_triples}, f_out, ensure_ascii=False, indent=4)

#     print(f"Corruption completed. Corrupted triples saved to {output_file}")

def generate_corrupt_triples(triples, output_file, corruption_ratio=0.05):
    triples_len = len(triples)
    target_corruption_count = int(triples_len * corruption_ratio)

    corrupted_triples = []
    corrupted_count = 0
    random.shuffle(triples)

    for triple in triples:
        if corrupted_count >= target_corruption_count:
            break

        s, p, o = triple

        # Apply corruption
        if corrupted_count < target_corruption_count * 0.2:
            corrupted = [o, p, s]  # swap s and o
        elif corrupted_count < target_corruption_count * 0.4:
            corrupted = [o, random.choice(triples)[1], s]  # swap s and o, change p
        elif corrupted_count < target_corruption_count * 0.6:
            corrupted = [random.choice(triples)[2], random.choice(triples)[1], o]  # change s and p
        elif corrupted_count < target_corruption_count * 0.8:
            corrupted = [s, random.choice(triples)[1], random.choice(triples)[2]]  # change o and p
        else:
            corrupted = [random.choice(triples)[0], p, random.choice(triples)[2]]  # change s and o

        corrupted_triples.append(corrupted)
        corrupted_count += 1

    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(corrupted_triples, f_out, ensure_ascii=False, indent=2)

    print(f"Saved {len(corrupted_triples)} corrupted triples to {output_file}")
    return corrupted_triples

# Run
# triples = parse_turtle(turtle_file)
# generate_corrupt_triples(triples, 'pipeline/corrupted_triples.json', corruption_ratio=0.02)
