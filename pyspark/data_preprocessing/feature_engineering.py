from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
import numpy as np
import math
from scipy import signal
from tqdm import tqdm

class FeatureEngineering:
    def __init__(self):
        self.ktstofts = 1.6878098571012
        self.spark = SparkSession.builder \
            .appName("FeatureEngineering") \
            .master("local[*]") \
            .getOrCreate()

    @staticmethod
    def isa(alt):
        T0 = 288.15
        p0 = 101325
        rho0 = 1.225
        a0 = 340.294
        k = 1.4
        R = 287.05287
        betabelow = -0.0065
        trop = 11000

        if alt < 0 or alt > 47000:
            print("Altitude must be in [0, 47000]")
            return None, None

        if alt == 0:
            return T0, rho0

        if 0 < alt <= trop:
            temperature = T0 + betabelow * alt
            pressure = p0 * (temperature / T0) ** ((-1) * 9.80665 / (betabelow * R))
        elif trop < alt < 47000:
            temperature = T0 + betabelow * trop
            pressure = (
                p0 * (temperature / T0) ** ((-1) * 9.80665 / (betabelow * R))
            ) * math.exp(-9.80665 * (alt - trop) / (R * temperature))

        density = pressure / (R * temperature)
        return temperature, density

    def calculate_true_airspeed(self, GS, track, u_wind, v_wind):
        track_rad = np.radians(track)
        V = (
            np.sqrt(
                (GS * np.sin(track_rad) - u_wind) ** 2 + (GS * np.cos(track_rad) - v_wind) ** 2
            )
            * self.ktstofts
        )
        return V

    def calculate_temp_deviation(self, altitude, temperature):
        isa_temp, _ = self.isa(altitude)
        return temperature - isa_temp

    def calculate_vertical_speed(self, vertical_rate):
        return vertical_rate / 60

    def calculate_horizontal_acceleration(self, df: DataFrame) -> DataFrame:
        df = df.sort("timestamp")
        udf_true_airspeed = udf(self.calculate_true_airspeed, DoubleType())
        df = df.withColumn("V", udf_true_airspeed(col("groundspeed"), col("track"), col("u_component_of_wind"), col("v_component_of_wind")))
        df = df.withColumn("dV_dt", (col("V").cast(DoubleType()).diff() / col("timestamp").cast(DoubleType()).diff()))
        return df

    def calculate_wind_acceleration(self, df: DataFrame) -> DataFrame:
        df = df.sort("timestamp")
        df = df.withColumn("W_long", (col("u_component_of_wind") * np.sin(np.radians(col("track"))) + col("v_component_of_wind") * np.cos(np.radians(col("track")))) * self.ktstofts)
        df = df.withColumn("dWi_dt", (col("W_long").cast(DoubleType()).diff() / col("timestamp").cast(DoubleType()).diff()))
        return df

    def identify_flight_phases(self, df: DataFrame, flight_phases_refinement=False) -> (DataFrame, DataFrame, DataFrame):
        df = df.sort("timestamp")
        df = df.withColumn("altitude_diff", col("altitude").diff())

        altitude_smooth = signal.savgol_filter(
            df.select("altitude").toPandas().values.flatten(),
            window_length=min(21, len(df) // 2 * 2 + 1),
            polyorder=3,
        )

        timestamp_seconds = df.select("timestamp").cast("int").toPandas().values.flatten()

        df = df.withColumn("ROC", np.gradient(altitude_smooth, timestamp_seconds))

        max_altitude = df.select("altitude").max()
        takeoff_end = df.filter(col("altitude") > df.select("altitude").quantile(0.1)).first()["index"]
        top_of_climb = df.filter(col("altitude") > max_altitude * 0.95).first()["index"]

        takeoff_phase = df.filter(df["index"] <= takeoff_end)
        initial_climb_phase = df.filter((df["index"] > takeoff_end) & (df["index"] <= top_of_climb))
        cruise_phase = df.filter(df["index"] > top_of_climb)

        if flight_phases_refinement:
            takeoff_phase = takeoff_phase.filter(col("ROC") > takeoff_phase.select("ROC").quantile(0.5))
            initial_climb_phase = initial_climb_phase.filter((col("ROC") > initial_climb_phase.select("ROC").quantile(0.25)) & (col("altitude") < max_altitude * 0.8))

        return takeoff_phase, initial_climb_phase, cruise_phase

    def calculate_thrust_minus_drag_for_phases(self, trajectory_df: DataFrame, flight_phases_refinement: bool) -> DataFrame:
        td_list = []

        for flight_id, group in tqdm(trajectory_df.groupby("flight_id").toPandas().iterrows(), desc="Calculating T-D for each flight"):
            takeoff_phase, initial_climb_phase, _ = self.identify_flight_phases(group, flight_phases_refinement)
            relevant_phase = takeoff_phase.union(initial_climb_phase)

            if relevant_phase.count() == 0:
                continue

            relevant_phase = relevant_phase.withColumn("V", udf(self.calculate_true_airspeed, DoubleType())(col("groundspeed"), col("track"), col("u_component_of_wind"), col("v_component_of_wind")))
            relevant_phase = relevant_phase.withColumn("dh_dt", udf(self.calculate_vertical_speed, DoubleType())(col("vertical_rate")))
            relevant_phase = relevant_phase.withColumn("delta_t", udf(self.calculate_temp_deviation, DoubleType())(col("altitude"), col("temperature")))

            relevant_phase = self.calculate_horizontal_acceleration(relevant_phase)
            relevant_phase = self.calculate_wind_acceleration(relevant_phase)

            relevant_phase = relevant_phase.withColumn("T_minus_D", (
                32.17405 * col("dh_dt") / col("V") * (col("temperature") / (col("temperature") - col("delta_t"))) + col("dV_dt") + col("dWi_dt")
            ))

            relevant_phase = relevant_phase.withColumn("flight_id", col("flight_id"))
            td_list.append(relevant_phase)

        td_df = self.spark.createDataFrame(td_list)
        return td_df.filter(col("T_minus_D").isNotNull())

    def aggregate_features(self, td_df: DataFrame) -> DataFrame:
        numerical_cols = [field.name for field in td_df.schema.fields if field.dataType == DoubleType()]
        if "flight_id" in numerical_cols:
            numerical_cols.remove("flight_id")
        agg_funcs = {col: ["mean", "max", "stddev"] for col in numerical_cols}
        aggregated_features = td_df.groupBy("flight_id").agg(agg_funcs)
        return aggregated_features

    def process_data(self, trajectory_df: DataFrame, train_df: DataFrame, use_trajectory: bool, flight_phases_refinement: bool) -> DataFrame:
        if use_trajectory:
            td_df = self.calculate_thrust_minus_drag_for_phases(trajectory_df, flight_phases_refinement)
            aggregated_features = self.aggregate_features(td_df)
            merged_df = train_df.join(aggregated_features, on="flight_id", how="inner")
        else:
            merged_df = train_df
        return merged_df

    def stop(self):
        if hasattr(self, 'spark'):
            self.spark.stop()