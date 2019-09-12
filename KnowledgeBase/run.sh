## Create the Knowledge Base using the instances in the dataset
## Near duplicate instances are removed

# Compute bert embeddings for each knowledge instance in the dataset
python KnowledgeBase/reason_bert_embeddings.py

# Find near duplicates and create knowledge base
python KnowledgeBase/knowledge_clusters.py