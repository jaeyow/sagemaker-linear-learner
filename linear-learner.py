import io
from metaflow import FlowSpec, step
import numpy as np
import os

try:
    from dotenv import load_dotenv
    load_dotenv(verbose=True, dotenv_path='.env')
except:
    print("No dotenv package")

class SageMakerLinearLearnerPipeline(FlowSpec):
    """
    SageMakerLinearLearnerPipeline is an end-to-end flow for SageMaker Linear Learner Built-in Algorithm
    """

    @step
    def start(self):
        """
        Initialization, place everything init related here, check that everything is
        in order like environment variables, connection strings, etc, and if there are
        any issues, fail fast here, now.
        """
        import sagemaker
        import boto3

        assert os.environ['SAGEMAKER_EXECUTION_ROLE']

        sess = sagemaker.Session()

        self.region = boto3.Session().region_name

        # S3 bucket where the original mnist data is downloaded and stored.
        self.downloaded_data_bucket = f"sagemaker-sample-files"
        self.downloaded_data_prefix = "datasets/image/MNIST"

        # S3 bucket for saving code and model artifacts.
        # Feel free to specify a different bucket and prefix
        self.bucket = sess.default_bucket()
        self.prefix = "sagemaker/DEMO-linear-mnist"

        # Define IAM role
        self.role = os.environ['SAGEMAKER_EXECUTION_ROLE']
        
        self.next(self.data_ingestion)

    @step
    def data_ingestion(self):
        """
        Data Ingestion
        - download MNIST dataset from S3
        - use the pickle library to load into memory
        - for smallish datasets like this, we can easily load it in memory
        """
        import boto3
        import pickle, gzip

        # Load the dataset
        s3 = boto3.client("s3")
        s3.download_file(self.downloaded_data_bucket, f"{self.downloaded_data_prefix}/mnist.pkl.gz", "mnist.pkl.gz")
        with gzip.open("mnist.pkl.gz", "rb") as f:
            self.train_set, self.valid_set, self.test_set = pickle.load(f, encoding="latin1")

        self.next(self.data_inspection)

    @step
    def data_inspection(self):
        """
        Data Inspection
        - very brief EDA, but in this case we just want to see what our data looks like
        - just save to a png file, since we are not running this in a Jupyter notebook, not able to display the image
        """
        import matplotlib.pyplot as plt

        plt.rcParams["figure.figsize"] = (2, 10)

        def show_digit(img, caption="", subplot=None):
            if subplot is None:
                _, (subplot) = plt.subplots(1, 1)
            imgr = img.reshape((28, 28))
            subplot.axis("off")
            subplot.imshow(imgr, cmap="gray")
            plt.title(caption)
            plt.savefig('sample-digit.png')

        show_digit(self.train_set[0][30], f"This is a {self.train_set[1][30]}")
        self.next(self.data_conversion)

    @step
    def data_conversion(self):
        """
        Data Conversion
        - converts training data, both vectors and labels into RecordIO format
        - ready for training in SageMaker
        - but have to upload to S3 in the next step
        """
        import io
        import numpy as np
        import sagemaker.amazon.common as smac

        vectors = np.array([t.tolist() for t in self.train_set[0]]).astype("float32")
        labels = np.where(np.array([t.tolist() for t in self.train_set[1]]) == 0, 1, 0).astype("float32")

        self.buf = io.BytesIO()
        smac.write_numpy_to_dense_tensor(self.buf, vectors, labels)
        self.buf.seek(0)

        self.next(self.upload_training_data)

    @step
    def upload_training_data(self):
        """
        Upload Training Data
        - upload RecordIO formatted trining data to S3
        """
        import boto3
        import os

        key = "recordio-pb-data"
        boto3.resource("s3").Bucket(self.bucket).Object(os.path.join(self.prefix, "train", key)).upload_fileobj(self.buf)
        self.s3_train_data = f"s3://{self.bucket}/{self.prefix}/train/{key}"
        print(f"uploaded training data location: {self.s3_train_data}")
        self.next(self.model_training)            

    @step
    def model_training(self):
        """
        Model training
        - now training starts, first we specify the Docker image for the required algorithm, in this case linear learner
        - create an estimator with the specified parameters, 
        - set the static hyperparamters, and SageMaker will automatically calculate those set as 'auto'
        - calling fit() starts the training process, upto the specified number of epochs
        - the save the model name and location for the next steps
        - take note that we have to specify an instance for trianing, which may be different from the endpoint instance
        """
        import boto3
        import sagemaker
        from sagemaker import image_uris
        image = image_uris.retrieve(region=boto3.Session().region_name, framework="linear-learner")

        self.output_location = f"s3://{self.bucket}/{self.prefix}/output"
        print(f"training artifacts will be uploaded to: {self.output_location}")

        sess = sagemaker.Session()

        linear = sagemaker.estimator.Estimator(
            image,
            self.role,
            instance_count=1,
            instance_type="ml.c4.xlarge",
            output_path=self.output_location,
            sagemaker_session=sess,
        )
        linear.set_hyperparameters(
            epochs=10,
            feature_dim=784,
            predictor_type="binary_classifier",
            mini_batch_size=200)

        linear.fit({"train": self.s3_train_data})

        # after an Estimator fit, the model will have been persisted in the defined S3 output location base folder
        self.model_data = linear.model_data
        print(f'Estimator model data: {self.model_data}')

        self.next(self.create_sagemaker_model)    

    @step
    def create_sagemaker_model(self):
        """
        Create SageMaker Model
        - once model training has completed, a Model can now be created
        - this will be the basis for creating our endpoint in the next steps
        """
        import boto3
        from sagemaker import image_uris
        from time import gmtime, strftime

        image = image_uris.retrieve(region=boto3.Session().region_name, framework="linear-learner")
        client = boto3.client("sagemaker")
        
        primary_container = {
            "Image": image,
            "ModelDataUrl": self.model_data
        }

        self.model_name = "linear-learner-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        create_model_response = client.create_model(
            ModelName=self.model_name,
            ExecutionRoleArn=self.role,
            PrimaryContainer=primary_container
        )

        print(f"Model Arn: {create_model_response['ModelArn']}")

        self.next(self.create_sagemaker_endpoint_configuration)

    @step
    def create_sagemaker_endpoint_configuration(self):
        """
        Create SageMaker Endpoint Configuration
        - specifies the configuration for our endpoint in the next step
        - specify the instance type, traffic ratio for A/B testing, but we only have one instance here
        - using the model created in the previous step
        """
        import boto3
        from time import gmtime, strftime
        client = boto3.client("sagemaker")

        self.endpoint_config_name = "LinearLearnerEndpointConfig-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

        print(f"Endpoint Configuration name: {self.endpoint_config_name}")

        create_endpoint_config_response = client.create_endpoint_config(
            EndpointConfigName=self.endpoint_config_name,
            ProductionVariants=[
                {
                    "InstanceType": "ml.c4.xlarge",
                    "InitialInstanceCount": 1,
                    "InitialVariantWeight": 1,
                    "ModelName": self.model_name,
                    "VariantName": "AllTraffic",
                }
            ],
        )

        print("Endpoint Config Arn: " + create_endpoint_config_response["EndpointConfigArn"])
        
        self.next(self.create_sagemaker_endpoint)

    @step
    def create_sagemaker_endpoint(self):
        """
        Create SageMaker Endpoint
        - finally ready to create the endpoint using the container specified in the configuration and model
        - inference endpoints may have different instance requirements as the training instance
        - poll the endpoint creation until finished, this takes about 5 minutes
        """
        import time
        import boto3
        from time import gmtime, strftime
        client = boto3.client("sagemaker")

        self.endpoint_name = "LinearLearnerEndpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        print(f"Endpoint name: {self.endpoint_name}")

        create_endpoint_response = client.create_endpoint(
            EndpointName = self.endpoint_name,
            EndpointConfigName = self.endpoint_config_name
        )
        print(f"Endpoint Arn: {create_endpoint_response['EndpointArn']}")

        resp = client.describe_endpoint(EndpointName=self.endpoint_name)
        status = resp["EndpointStatus"]
        print(f"Status: {status}...")

        while status == "Creating":
            time.sleep(60)
            resp = client.describe_endpoint(EndpointName=self.endpoint_name)
            status = resp["EndpointStatus"]
            print(f"Status: {status}...")

        print(f"Arn: {resp['EndpointArn']}")
        print(f"Status: {status}...")
        
        self.next(self.perform_prediction)

    # Simple function to create a csv from our numpy array
    def np2csv(self, arr):
        csv = io.BytesIO()
        np.savetxt(csv, arr, delimiter=",", fmt="%g")
        return csv.getvalue().decode().rstrip()

    @step
    def perform_prediction(self):
        """
        Placeholder for performing prediction on the SageMaker Endpoint
        - perform one prediction to see if our endpoint works
        - this example will not push it to a Rest API, perhaps in the next exercise we'll spin up AWS API Gateway with the model endpoint
        """
        import boto3
        import json
        runtime_client = boto3.client("runtime.sagemaker")

        payload = self.np2csv(self.test_set[0][10:11])

        try:
            response = runtime_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="text/csv",
                Body=payload)

            result = response["Body"].read().decode("ascii")
            print(f"Response: {result}")

        except:
            print("Endpoint invocation exception occurred, deleting endpoint...")
        
        self.next(self.delete_sagemaker_endpoint)

    @step
    def delete_sagemaker_endpoint(self):
        """
        Delete SageMaker Endpoint - you don't want that AWS bill, do you?
        - after all that work, delete all to avoid a credit card bill :)
        """
        import boto3
        client = boto3.client("sagemaker")

        client.delete_endpoint(EndpointName=self.endpoint_name)
        print(f"Deleting endpoint: {self.endpoint_name}, coz' I don't want AWS bills...")
        
        self.next(self.end)        
                 
    @step   
    def end(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.
        """

if __name__ == "__main__":
    SageMakerLinearLearnerPipeline()