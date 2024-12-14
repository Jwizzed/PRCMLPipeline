from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.sql import SparkSession, DataFrame


class FeatureEncoder:
    def __init__(self, preserve_columns: list):
        self.preserve_columns = preserve_columns
        self.categorical_col = [
            "adep",
            "country_code_adep",
            "ades",
            "country_code_ades",
            "aircraft_type",
            "airline",
        ]
        self.oneHot_col = ["wtc"]
        self.spark = SparkSession.builder \
            .appName("FeatureEncoder") \
            .master("local[*]") \
            .getOrCreate()

    def encode_features(self, df: DataFrame) -> DataFrame:
        stages = []

        # Label encoding
        for col_name in self.categorical_col:
            indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_encoded")
            stages.append(indexer)
            if col_name not in self.preserve_columns:
                df = df.drop(col_name)

        # One-hot encoding
        for col_name in self.oneHot_col:
            indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_index")
            encoder = OneHotEncoder(inputCol=col_name + "_index", outputCol=col_name + "_encoded")
            stages += [indexer, encoder]

        pipeline = Pipeline(stages=stages)
        model = pipeline.fit(df)
        df_encoded = model.transform(df)

        return df_encoded

    def stop(self):
        """Stop SparkSession"""
        if hasattr(self, 'spark'):
            self.spark.stop()
