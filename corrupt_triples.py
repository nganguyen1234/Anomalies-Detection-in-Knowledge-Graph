import random
import json

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
