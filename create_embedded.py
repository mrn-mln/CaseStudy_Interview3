import pandas as pd
import tensorflow as tf


# Load model
model_path = 'model/'
model = tf.saved_model.load(model_path)

# Read dataset
db = pd.read_csv('data/db.csv')

# Extract description
text = db['Description']

# Do inference for all description
items_vec = model(text)

# Save embedded vectors as a new column to the dataframe
db['embedding'] = [embedding.numpy() for embedding in items_vec]

# Export new dataset
db.to_csv('data/db_embedding.csv')