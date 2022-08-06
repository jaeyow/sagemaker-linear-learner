from metaflow import FlowSpec, step
import os

try:
    from dotenv import load_dotenv
    load_dotenv(verbose=True, dotenv_path='.env')
except:
    print("No dotenv package")

class LinearLearnerPipeline(FlowSpec):
    """
    F1PredictorPipeline is an end-to-end flow for F1 Predictor
    """

    @step
    def start(self):
        """
        Initialization, place everything init related here, check that everything is
        in order like environment variables, connection strings, etc, and if there are
        any issues, fail fast here, now.
        """
        # Permissions and environment variables
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
        
        self.next(
          self.data_ingestion)

    @step
    def data_ingestion(self):
        """
        Data Ingestion
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
        """
        import boto3
        import sagemaker
        from sagemaker import image_uris
        container = image_uris.retrieve(region=boto3.Session().region_name, framework="linear-learner")

        self.output_location = f"s3://{self.bucket}/{self.prefix}/output"
        print(f"training artifacts will be uploaded to: {self.output_location}")

        sess = sagemaker.Session()

        linear = sagemaker.estimator.Estimator(
            container,
            self.role,
            instance_count=1,
            instance_type="ml.c4.xlarge",
            output_path=self.output_location,
            sagemaker_session=sess,
        )
        linear.set_hyperparameters(feature_dim=784, predictor_type="binary_classifier", mini_batch_size=200)

        linear.fit({"train": self.s3_train_data})

        self.next(self.deploy_winning_model)    

    @step
    def deploy_winning_model(self):
        """
        Placeholder for deploying model
        """
        self.next(self.end)

    @step   
    def end(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.
        """

if __name__ == "__main__":
    LinearLearnerPipeline()