import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import date_format, to_timestamp, col


class DataIngestor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.spark = SparkSession.builder \
            .appName("DataIngestor") \
            .master("local[*]") \
            .getOrCreate()

    def load_and_process_data(self):
        try:
            # Define file paths
            train_path = os.path.join(self.data_path, "challenge_set.csv")
            test_path = os.path.join(self.data_path, "submission_set.csv")
            parquet_path = os.path.join(self.data_path, "2022-01-01.parquet")

            # Read train data
            print(f"Reading train data from: {train_path}")
            train_df = self.spark.read \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .csv(train_path)

            # Read test data
            print(f"Reading test data from: {test_path}")
            test_df = self.spark.read \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .csv(test_path)

            # Drop 'tow' column from test_df
            if "tow" in test_df.columns:
                test_df = test_df.drop("tow")

            # Format datetime columns
            datetime_columns = ["date", "actual_offblock_time", "arrival_time"]

            # Process datetime columns for train_df
            for column in datetime_columns:
                if column in train_df.columns:
                    train_df = train_df.withColumn(
                        column,
                        date_format(
                            to_timestamp(column),
                            "yyyy-MM-dd HH:mm:ss"
                        )
                    )

            # Process datetime columns for test_df
            for column in datetime_columns:
                if column in test_df.columns:
                    test_df = test_df.withColumn(
                        column,
                        date_format(
                            to_timestamp(column),
                            "yyyy-MM-dd HH:mm:ss"
                        )
                    )

            # Read trajectory data using pandas first
            print(f"Reading trajectory data from: {parquet_path}")
            pd_trajectory_df = pd.read_parquet(parquet_path)

            # Convert timestamp columns to string format in pandas
            timestamp_columns = pd_trajectory_df.select_dtypes(include=['datetime64[ns]']).columns
            for col in timestamp_columns:
                pd_trajectory_df[col] = pd_trajectory_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

            # Convert pandas DataFrame to Spark DataFrame
            trajectory_df = self.spark.createDataFrame(pd_trajectory_df)

            return train_df, test_df, trajectory_df

        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def stop(self):
        if hasattr(self, 'spark'):
            self.spark.stop()


if __name__ == "__main__":
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "data")

    print(f"Current directory: {current_dir}")
    print(f"Looking for data in: {data_dir}")

    if os.path.exists(data_dir):
        print("Files in data directory:")
        for file in os.listdir(data_dir):
            print(f"  - {file}")
    else:
        print("Data directory not found!")
        exit(1)

    try:
        ingestor = DataIngestor(data_dir)
        train_df, test_df, trajectory_df = ingestor.load_and_process_data()

        # Show sample data
        print("\nTrain DataFrame Schema:")
        train_df.printSchema()
        print("\nSample Train Data:")
        train_df.show(5)

        print("\nTest DataFrame Schema:")
        test_df.printSchema()
        print("\nSample Test Data:")
        test_df.show(5)

        print("\nTrajectory DataFrame Schema:")
        trajectory_df.printSchema()
        print("\nSample Trajectory Data:")
        trajectory_df.show(5)

    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        if 'ingestor' in locals():
            ingestor.stop()