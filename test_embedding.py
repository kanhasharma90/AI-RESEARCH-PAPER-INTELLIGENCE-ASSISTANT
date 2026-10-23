from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

text = """
The proposed Transformer model achieved 94.5% accuracy
on the benchmark dataset.
"""

embedding = model.encode(text)

print("Embedding generated successfully!")
print("Vector dimensions:", len(embedding))
print("First 10 values:")
print(embedding[:10])