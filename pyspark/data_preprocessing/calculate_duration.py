from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col,
    to_timestamp,
    unix_timestamp,
    round,
    when
)


class DurationCalculator:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("DurationCalculator") \
            .master("local[*]") \
            .getOrCreate()

    def get_duration(self, df: DataFrame) -> DataFrame:
        """
        Calculate duration in minutes between actual_offblock_time and arrival_time

        Args:
            df: Input DataFrame with actual_offblock_time and arrival_time columns

        Returns:
            DataFrame with additional duration column in minutes
        """
        return df.withColumn(
            "actual_offblock_time",
            to_timestamp(col("actual_offblock_time"))
        ).withColumn(
            "arrival_time",
            to_timestamp(col("arrival_time"))
        ).withColumn(
            "duration",
            round(
                (unix_timestamp(col("arrival_time")) -
                 unix_timestamp(col("actual_offblock_time"))) / 60,
                2
            )
        ).withColumn(
            "duration",
            when(col("duration") < 0, None)  # Handle invalid durations
            .otherwise(col("duration"))
        )

    def calculate_duration(self, input_path: str) -> DataFrame:
        """
        Read CSV file and calculate duration

        Args:
            input_path: Path to input CSV file

        Returns:
            DataFrame with calculated durations
        """
        df = self.spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .csv(input_path)

        return self.get_duration(df)

    def stop(self):
        """Stop SparkSession"""
        if hasattr(self, 'spark'):
            self.spark.stop()


if __name__ == "__main__":
    try:
        calculator = DurationCalculator()

        # Example usage
        # input_path = "path/to/your/file.csv"
        # result_df = calculator.calculate_duration(input_path)

        # Print sample results
        # print("\nResult DataFrame Schema:")
        # result_df.printSchema()
        # print("\nSample Results:")
        # result_df.show(5)

    except Exception as e:
        print(f"Error in main: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if 'calculator' in locals():
            calculator.stop()
