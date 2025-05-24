# Anomalies-Detection-in-Knowledge-Graph

## Work flow
Load entity types and encode them using a sentence-transformer.

Train a k-NN classifier to predict types based on the embeddings.

Parse RDF triples from a Turtle file

Generate corrupted triples for evaluation.

Predict subject/object types using the trained k-NN model.

Compute confidence scores using cosine similarity of type embeddings.

Detect anomalies using an Isolation Forest model.

Sent detected anomalies to LLM for validation and reasoning.

Evaluate performance using precision, recall, F1, and accuracy.

Save results to JSON for analysis and debugging.

## Requirements
Install the necessary libraries with: pip install -r requirements.txt

## Data Setup
Download the YAGO 1.0.0 and YAGO 4.5 Turtle format dataset.
* Make sure yago-1.0.0-turtle.ttl is inside the yago-1.0.0-turtle/ directory, yago-tiny.ttl is inside the  yago-4.5.0.1-tiny / directory  at the same level as the pipeline/ directory.

## Code Execution
- The file to be executed is application.py
- Have yago-1.0.0-turtle.ttl and yago-tiny.ttl downloaded and made available as per the data setup above.




