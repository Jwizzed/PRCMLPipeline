from pyspark.ml.feature import UnivariateFeatureSelector
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession, DataFrame


class FeatureProcessor:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("FeatureProcessor") \
            .master("local[*]") \
            .getOrCreate()

    def drop_features(self, df: DataFrame, final_drop: bool = False) -> DataFrame:
        drop_cols = [
            "date",
            "callsign",
            "name_adep",
            "name_ades",
            "actual_offblock_time",
            "arrival_time",
        ]

        if final_drop:
            drop_cols.extend(["aircraft_type"])

        for col_name in drop_cols:
            if col_name in df.columns:
                df = df.drop(col_name)

        return df

    def select_features(self, df: DataFrame, target_col: str, k: int = 15) -> DataFrame:
        feature_cols = [col for col in df.columns if col != target_col]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        df = assembler.transform(df)

        # Using PySpark's UnivariateFeatureSelector instead of scikit-learn's SelectKBest
        selector = UnivariateFeatureSelector(
            featuresCol="features",
            outputCol="selectedFeatures",
            labelCol=target_col,
            selectionMode="numTopFeatures",
            selectionThreshold=k,
            featureType="continuous"
        )

        selector_model = selector.fit(df)
        selected_features = selector_model.transform(df)

        return selected_features

    def split_data(self, df: DataFrame, target_col: str, test_size: float = 0.2):
        train_df, test_df = df.randomSplit([1 - test_size, test_size], seed=42)
        return train_df, test_df

    def stop(self):
        if hasattr(self, 'spark'):
            self.spark.stop()
