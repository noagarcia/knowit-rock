## Train the Knowledge Retrieval module

# Compute scores for each answer for sorting
python KnowledgeRetrieval/prior_scores.py

# Train the BertScoring network for retrieving instances from the KB
python KnowledgeRetrieval/process.py