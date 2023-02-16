import pandas as pd
import tensorflow as tf
import mlflow
import mlflow.tensorflow


# Read dataset
db = pd.read_csv('data/db.csv')

# Set the active experiment
mlflow.set_experiment("Universe Sentence Encoder Model")

# Start a new run within the active experiment
with mlflow.start_run(run_name='first embedded creator on db.csv v1.0') as run:
    # Load model
    model_path = 'model/'
    model = tf.saved_model.load(model_path)

    # Log the model as an artifact to MLflow
    mlflow.log_artifact('model/', artifact_path="model/")

    # Extract description
    text = db['Description']

    # Do inference for all description
    items_vec = model(text)

    # Log the dataset name as a run parameter
    mlflow.log_param("dataset_name", "db.csv")
    mlflow.set_tags({"author": "Mehran", "dataset_version": "v1.0"})

    # Save embedded vectors as a new column to the dataframe
    db['embedding'] = [embedding.numpy() for embedding in items_vec]

    # Export new dataset
    db.to_csv('data/db_embedding.csv')
    mlflow.log_artifact('data', artifact_path="data/db_embedding.csv")

    mlflow.end_run

    print(f'Artifact uri = {mlflow.get_artifact_uri}')
    print(f'RunID = {run.info.run_uuid}')