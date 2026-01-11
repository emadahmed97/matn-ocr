import logging
import logging.config
import sys
import time
from io import StringIO
from pathlib import Path
import boto3
import os
import pandas as pd
from metaflow import IncludeFile, current, Parameter

PYTHON = "3.12.8"

PACKAGES = {
    "keras": "3.8.0",
    "scikit-learn": "1.6.1",
    "mlflow": "2.20.2",
    "tensorflow": "2.18.0",
    "evidently": "0.7.4"
}


class Pipeline:
    """A base class for all pipelines."""

    def logger(self) -> logging.Logger:
        """Configure the logging handler and return a logger instance."""
        if Path("logging.conf").exists():
            logging.config.fileConfig("logging.conf")
        else:
            logging.basicConfig(
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[logging.StreamHandler(sys.stdout)],
                level=logging.INFO,
            )

        return logging.getLogger("mlschool")


class DatasetMixin:
    """A mixin for loading and preparing a dataset.

    This mixin is designed to be combined with any pipeline that requires accessing
    the dataset.
    """

    data_folder = Parameter(
        "data_folder",
        help="Dataset folder that will be used to train the model.",
        default="data",
    )

    s3_bucket = Parameter(
        "s3_bucket",
        help="The S3 bucket where the dataset is stored.",
        default="mlschool-data",
    )

    def load_dataset(self, logger=None):
        """Load and prepare the dataset."""
        if current.is_production and self.s3_bucket:
            if logger:
                logger.info("Loading dataset from S3 bucket: %s", self.s3_bucket)

            s3 = boto3.client("s3")
            response = s3.list_objects_v2(Bucket=self.s3_bucket)

            if "Contents" not in response:
                raise ValueError(f"No files found in S3 bucket: {self.s3_bucket}")

            data_frames = []
            for obj in response["Contents"]:
                if obj["Key"].endswith(".csv"):
                    csv_obj = s3.get_object(Bucket=self.s3_bucket, Key=obj["Key"])
                    csv_data = csv_obj["Body"].read().decode("utf-8")
                    data_frames.append(pd.read_csv(StringIO(csv_data)))

            data = pd.concat(data_frames, ignore_index=True)
        else:
            if logger:
                logger.info("Loading dataset from local file.")
            # read from dataset folder
            if not os.path.isdir(self.data_folder):
                print(f"Error: The path '{self.data_folder}' is not a valid directory.")
                return

            # List to hold each DataFrame
            all_dfs = []
            # Iterate through all files in the specified directory
            for file_name in os.listdir(self.data_folder):
                # Check if the file is a CSV file
                if file_name.endswith('.csv'):
                    file_path = os.path.join(self.data_folder, file_name)
                    # Use a try-except block to handle potential reading errors
                    try:
                        # Read the CSV file into a DataFrame
                        df = pd.read_csv(file_path)
                        # Append the DataFrame to our list
                        all_dfs.append(df)
                        print(f"Successfully read {file_name}")
                        print(f"Length of csv: {len(df)}")
                    except Exception as e:
                        print(f"Error reading {file_name}: {e}")
            # Check if any CSV files were found
            if not all_dfs:
                print(f"No CSV files found in the folder '{self.data_folder}'.")
                return

            # Concatenate all DataFrames in the list into a single DataFrame
            # ignore_index=True resets the index of the combined DataFrame
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df.to_csv("combined.csv", index=False)
            data = pd.read_csv("combined.csv")

        print(f"Length of combined csv: {len(data)}")

        # Clean and shuffle the dataset
        data["sex"] = data["sex"].replace(".", pd.NA)
        data = data.dropna()
        data = data.sample(frac=1, random_state=42 if not current.is_production else None)

        if logger:
            logger.info("Loaded dataset with %d samples", len(data))

        return data


def packages(*names: str):
    """Return a dictionary of the specified packages and their corresponding version.

    This function is useful to set up the different pipelines while keeping the
    package versions consistent and centralized in a single location.

    Any packages that should be locked to a specific version will be part of the
    `PACKAGES` dictionary. If a package is not present in the dictionary, it will be
    installed using the latest version available.
    """
    return {name: PACKAGES.get(name, "") for name in names}


def build_features_transformer():
    """Build a Scikit-Learn transformer to preprocess the feature columns."""
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
    )

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        # We can use the `handle_unknown="ignore"` parameter to ignore unseen categories
        # during inference. When encoding an unknown category, the transformer will
        # return an all-zero vector.
        OneHotEncoder(handle_unknown="ignore"),
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                # We'll apply the numeric transformer to all columns that are not
                # categorical (object).
                make_column_selector(dtype_exclude="object"),
            ),
            (
                "categorical",
                categorical_transformer,
                # We want to make sure we ignore the target column which is also a
                # categorical column. To accomplish this, we can specify the column
                # names we only want to encode.
                ["island", "sex"],
            ),
        ],
    )


def build_target_transformer():
    """Build a Scikit-Learn transformer to preprocess the target column."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder

    return ColumnTransformer(
        transformers=[("species", OrdinalEncoder(), ["species"])],
    )


def build_model(input_shape, learning_rate=0.01):
    """Build and compile the neural network to predict the species of a penguin."""
    from keras import Input, layers, models, optimizers

    model = models.Sequential(
        [
            Input(shape=(input_shape,)),
            layers.Dense(10, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(3, activation="softmax"),
        ],
    )

    model.compile(
        optimizer=optimizers.SGD(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
