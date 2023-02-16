import pandas as pd
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# load model
model_path = 'model/'
model = tf.saved_model.load(model_path)

db = pd.read_csv('data/db.csv')
text = db['Description']

query = 'IVORY EMBROIDERED QUILT'

items_vec = model(text)

db['embedding'] = [embedding for embedding in items_vec]

query_embedding = model([query])[0]


# Compute cosine similarity between input sentence and candidate sentences
similarities = tf.tensordot(query_embedding, items_vec, axes=[[0], [1]]).numpy()

# Find top 10 most similar sentences
top_indices = tf.argsort(similarities, direction='DESCENDING')[:10].numpy()
top_sentences = [text[i] for i in top_indices]

# Print top 10 most similar sentences
print('\n \n \n \n \n \n *-*-*-*-*-*-*-*-*-*-*-*')
print("Top 10 most similar sentences to '{}':".format(query))
print("*-*-*-*-*-*-*-*-*-*-*-*")
for i, sentence in enumerate(top_sentences):
    print("{}. {}".format(i+1, sentence))