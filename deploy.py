import boto3
import mlflow
import sagemaker


experiment_id = '748749698091262675'
run_id = '1985c1d06d774c049de0f38a73a60692'
region = 'us-east-1'
aws_id = '687137865617'
arn = 'arn:aws:iam::687137865617:role/aws-sagemaker-deploy-adidas-casestudy'
app_name = 'adidas_casestudy'
model_uri = f'mlruns/{experiment_id}/{run_id}/artifacts/model/model'
data_uri = f'data/db_embedding.csv'
tag_id = '2.1.1'

image_url = aws_id + '.dkr.ecr.' + region + '.amazonaws.com/mlflow-pyfunc:' + tag_id

# Set up a SageMaker session and S3 client
sagemaker_session = sagemaker.Session()
s3_client = boto3.client('s3')

# Create a SageMaker model object
model = sagemaker.model.Model(
    model_data=model_uri,
    image_uri=image_url,
    role=arn,
    sagemaker_session=sagemaker_session,
)


# Create an endpoint configuration using the SageMaker client
sagemaker_client = boto3.client('sagemaker', region_name=region)
endpoint_config_name = 'caseStudy_config'  # Updated endpoint configuration name
production_variants = [{
    'InstanceType': 'ml.m5.large',
    'InitialInstanceCount': 1,
    'ModelName': app_name,
    'VariantName': 'AllTraffic',
}]
create_endpoint_config_response = sagemaker_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=production_variants,
)

# Create a SageMaker endpoint object
endpoint_name = 'adidascasestudyendpoint'  # Updated endpoint name
create_endpoint_response = sagemaker_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name,
)

# Wait for the endpoint to be in service
waiter = sagemaker_client.get_waiter('endpoint_in_service')
waiter.wait(EndpointName=endpoint_name)