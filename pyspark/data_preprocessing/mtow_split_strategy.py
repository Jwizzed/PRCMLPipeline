from typing import Dict, Tuple, Union

from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, ntile


class MTOWSplitStrategy:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("MTOWSplitStrategy") \
            .master("local[*]") \
            .getOrCreate()

    def process_subset(
            self, subset_df: DataFrame, is_test: bool = False
    ) -> Union[Tuple[DataFrame, DataFrame], Tuple[DataFrame, None]]:
        # Create vector assembler for MTOW standardization
        assembler = VectorAssembler(
            inputCols=["MTOW(kg)"],
            outputCol="MTOW_vector"
        )
        subset_df = assembler.transform(subset_df)

        # Standardize MTOW
        scaler = StandardScaler(
            inputCol="MTOW_vector",
            outputCol="MTOW_scaled",
            withStd=True,
            withMean=True
        )
        scaler_model = scaler.fit(subset_df)
        subset_df = scaler_model.transform(subset_df)

        # Extract the scaled value from vector
        subset_df = subset_df.withColumn(
            "MTOW(kg)",
            col("MTOW_scaled").getItem(0)
        ).drop("MTOW_vector", "MTOW_scaled")

        if not is_test:
            X = subset_df.drop("tow", "MTOW_range")
            y = subset_df.select("flight_id", "tow").drop("flight_id")
            return X, y
        else:
            X = subset_df.drop("MTOW_range")
            return X, None

    def split(
            self, df: DataFrame, **kwargs
    ) -> Dict[str, Tuple[DataFrame, DataFrame]]:
        is_test = kwargs.get("is_test", False)

        # Split narrow and wide body aircraft
        narrow_body_df = df.filter(col("MTOW(kg)") <= 115680)
        wide_body_df = df.filter(col("MTOW(kg)") > 115680)

        # Create MTOW ranges for narrow body aircraft
        narrow_body_df = narrow_body_df.withColumn(
            "ntile",
            ntile(4).over(Window.orderBy("MTOW(kg)"))
        )

        narrow_body_df = narrow_body_df.withColumn(
            "MTOW_range",
            when(col("ntile") == 1, "Very Low")
            .when(col("ntile") == 2, "Low")
            .when(col("ntile") == 3, "Medium")
            .when(col("ntile") == 4, "High")
        ).drop("ntile")

        # Create MTOW ranges for wide body aircraft
        wide_body_df = wide_body_df.withColumn(
            "MTOW_range",
            when(col("aircraft_type") == "B77W", "B77W")
            .otherwise("NonB77W")
        )

        result = {
            "very_low": self.process_subset(
                narrow_body_df.filter(col("MTOW_range") == "Very Low"),
                is_test
            ),
            "low": self.process_subset(
                narrow_body_df.filter(col("MTOW_range") == "Low"),
                is_test
            ),
            "medium": self.process_subset(
                narrow_body_df.filter(col("MTOW_range") == "Medium"),
                is_test
            ),
            "high": self.process_subset(
                narrow_body_df.filter(col("MTOW_range") == "High"),
                is_test
            ),
            "non_b77w": self.process_subset(
                wide_body_df.filter(col("MTOW_range") == "NonB77W"),
                is_test
            ),
            "b77w": self.process_subset(
                wide_body_df.filter(col("MTOW_range") == "B77W"),
                is_test
            ),
        }

        return result

    def stop(self):
        if hasattr(self, 'spark'):
            self.spark.stop()


if __name__ == "__main__":
    try:
        # Initialize the strategy
        strategy = MTOWSplitStrategy()

        # Example usage
        # df = spark.read.parquet("my_data.parquet")
        # result = strategy.split(df, is_test=False)

        # Print sample results
        # for key, (X, y) in result.items():
        #     print(f"\nSubset: {key}")
        #     print("X Schema:")
        #     X.printSchema()
        #     print("X Sample:")
        #     X.show(5)
        #     if y is not None:
        #         print("y Schema:")
        #         y.printSchema()
        #         print("y Sample:")
        #         y.show(5)

    except Exception as e:
        print(f"Error in main: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if 'strategy' in locals():
            strategy.stop()
