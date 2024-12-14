from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col


class DataCleaner:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("DataCleaner") \
            .master("local[*]") \
            .getOrCreate()

    def clean_dataframe(self, df: DataFrame, contamination: float = 0.01) -> DataFrame:
        """
        Cleans a dataframe using Bisecting K-Means for outlier detection.

        Args:
            df: Input DataFrame
            contamination: Fraction of outliers to remove

        Returns:
            Cleaned DataFrame
        """
        df_cleaned = df.dropna()

        numeric_columns = [field.name for field in df_cleaned.schema.fields if
                           field.dataType.simpleString() == 'double']

        # Scale numeric columns
        assembler = VectorAssembler(inputCols=numeric_columns, outputCol="features")
        df_features = assembler.transform(df_cleaned)

        scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
        scaler_model = scaler.fit(df_features)
        df_scaled = scaler_model.transform(df_features)

        # Use Bisecting K-Means for outlier detection
        bkm = BisectingKMeans(k=int((1 - contamination) * df_scaled.count()), featuresCol="scaled_features")
        model = bkm.fit(df_scaled)
        predictions = model.transform(df_scaled)

        return predictions.filter(predictions.prediction == 0).drop("features", "scaled_features", "prediction")

    def clean_trajectory(self, df: DataFrame, contamination: float = 0.01) -> DataFrame:
        """
        Cleans a trajectory dataframe using Bisecting K-Means for outlier detection.

        Args:
            df: Input DataFrame
            contamination: Fraction of outliers to remove

        Returns:
            Cleaned DataFrame
        """
        df_cleaned = df.filter((col("altitude") >= 0) & (col("altitude") <= 47000)).dropna()

        numeric_columns = [field.name for field in df_cleaned.schema.fields if
                           field.dataType.simpleString() == 'double']

        # Scale numeric columns
        assembler = VectorAssembler(inputCols=numeric_columns, outputCol="features")
        df_features = assembler.transform(df_cleaned)

        scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
        scaler_model = scaler.fit(df_features)
        df_scaled = scaler_model.transform(df_features)

        # Use Bisecting K-Means for outlier detection
        bkm = BisectingKMeans(k=int((1 - contamination) * df_scaled.count()), featuresCol="scaled_features")
        model = bkm.fit(df_scaled)
        predictions = model.transform(df_scaled)

        return predictions.filter(predictions.prediction == 0).drop("features", "scaled_features", "prediction")

    def stop(self):
        """Stop SparkSession"""
        if hasattr(self, 'spark'):
            self.spark.stop()


if __name__ == "__main__":
    try:
        cleaner = DataCleaner()

        # Example usage
        # input_path = "path/to/your/file.csv"
        # df = cleaner.spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)
        # cleaned_df = cleaner.clean_dataframe(df)
        # cleaned_trajectory_df = cleaner.clean_trajectory(df)

        # Print sample results
        # print("\nCleaned DataFrame Schema:")
        # cleaned_df.printSchema()
        # print("\nSample Cleaned Data:")
        # cleaned_df.show(5)

        # print("\nCleaned Trajectory DataFrame Schema:")
        # cleaned_trajectory_df.printSchema()
        # print("\nSample Cleaned Trajectory Data:")
        # cleaned_trajectory_df.show(5)

    except Exception as e:
        print(f"Error in main: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if 'cleaner' in locals():
            cleaner.stop()
