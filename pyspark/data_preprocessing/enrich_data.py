import json

import fsspec
import pandas as pd
from pyspark.sql import SparkSession, DataFrame


class DataEnricher:
    def __init__(self, external_info_file: str):
        self.external_info_file = external_info_file
        self.spark = SparkSession.builder \
            .appName("DataEnricher") \
            .master("local[*]") \
            .getOrCreate()

    def load_external_data(self) -> DataFrame:
        """Loads and processes external aircraft information."""
        with fsspec.open(self.external_info_file, "r") as file:
            external_information = json.load(file)

        external_df = pd.DataFrame.from_dict(external_information, orient="index")
        external_df.reset_index(inplace=True)
        external_df.rename(columns={"index": "aircraft_type"}, inplace=True)

        # Convert pandas DataFrame to Spark DataFrame
        external_spark_df = self.spark.createDataFrame(external_df)
        return external_spark_df

    def enrich_data(self, train_df: DataFrame, test_df: DataFrame) -> (DataFrame, DataFrame):
        """Enriches training and test data with external information."""
        external_df = self.load_external_data()

        train_enriched = train_df.join(external_df, on="aircraft_type", how="left")
        test_enriched = test_df.join(external_df, on="aircraft_type", how="left")

        return train_enriched, test_enriched

    def stop(self):
        """Stop SparkSession"""
        if hasattr(self, 'spark'):
            self.spark.stop()
