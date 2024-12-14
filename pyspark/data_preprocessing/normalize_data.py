from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql.functions import col as pyspark_col
from pyspark.sql import SparkSession, DataFrame


class DataNormalizer:
    def __init__(self, exclude_columns: list, split_by_flown_distance: bool = False):
        self.exclude_columns = exclude_columns.copy()
        if split_by_flown_distance:
            self.exclude_columns.append("flown_distance")
        self.spark = SparkSession.builder \
            .appName("DataNormalizer") \
            .master("local[*]") \
            .getOrCreate()

    def normalize_data(self, df: DataFrame) -> DataFrame:
        numeric_columns = [field.name for field in df.schema.fields if field.dataType.simpleString() == 'double']
        columns_to_normalize = [col for col in numeric_columns if col not in self.exclude_columns]

        assembler = VectorAssembler(inputCols=columns_to_normalize, outputCol="features")
        df_features = assembler.transform(df)

        scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
        scaler_model = scaler.fit(df_features)
        df_scaled = scaler_model.transform(df_features)

        for i, col_name in enumerate(columns_to_normalize):
            df_scaled = df_scaled.withColumn(col_name, pyspark_col("scaled_features").getItem(i))

        df_scaled = df_scaled.drop("features", "scaled_features")
        return df_scaled

    def stop(self):
        if hasattr(self, 'spark'):
            self.spark.stop()