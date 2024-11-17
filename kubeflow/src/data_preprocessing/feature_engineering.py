import math
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm


class FeatureEngineering:
    def __init__(self):
        self.ktstofts = 1.6878098571012

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

    def calculate_true_airspeed(self, row):
        GS = row["groundspeed"]
        track = np.radians(row["track"])
        u_wind = row["u_component_of_wind"]
        v_wind = row["v_component_of_wind"]

        V = (
            np.sqrt(
                (GS * np.sin(track) - u_wind) ** 2 + (GS * np.cos(track) - v_wind) ** 2
            )
            * self.ktstofts
        )
        return V

    @staticmethod
    def calculate_vertical_speed(row):
        return row["vertical_rate"] / 60

    def calculate_temp_deviation(self, row):
        isa_temp, _ = self.isa(row["altitude"])
        return row["temperature"] - isa_temp

    def calculate_horizontal_acceleration(self, group):
        group = group.sort_values("timestamp")
        group["V"] = group.apply(self.calculate_true_airspeed, axis=1)
        group["dV_dt"] = (
            group["V"].diff() / group["timestamp"].diff().dt.total_seconds()
        )
        return group

    def calculate_wind_acceleration(self, group):
        group = group.sort_values("timestamp")
        group["W_long"] = (
            group["u_component_of_wind"] * np.sin(np.radians(group["track"]))
            + group["v_component_of_wind"] * np.cos(np.radians(group["track"]))
        ) * self.ktstofts
        group["dWi_dt"] = (
            group["W_long"].diff() / group["timestamp"].diff().dt.total_seconds()
        )
        return group

    @staticmethod
    def identify_flight_phases(group, flight_phases_refinement=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        group["altitude_diff"] = group["altitude"].diff()

        altitude_smooth = signal.savgol_filter(
            group["altitude"],
            window_length=min(21, len(group) // 2 * 2 + 1),
            polyorder=3,
        )

        if isinstance(group["timestamp"].iloc[0], str):
            group["timestamp"] = pd.to_datetime(group["timestamp"])
        timestamp_seconds = group["timestamp"].astype("int64") // 10**9

        group["ROC"] = np.gradient(altitude_smooth, timestamp_seconds)

        max_altitude = group["altitude"].max()
        takeoff_end = group[group["altitude"] > group["altitude"].quantile(0.1)].index[
            0
        ]
        top_of_climb = group[group["altitude"] > max_altitude * 0.95].index[0]

        takeoff_phase = group.loc[:takeoff_end]
        initial_climb_phase = group.loc[takeoff_end:top_of_climb]
        cruise_phase = group.loc[top_of_climb:]

        if flight_phases_refinement:
            takeoff_phase = takeoff_phase[
                takeoff_phase["ROC"] > takeoff_phase["ROC"].quantile(0.5)
            ]
            initial_climb_phase = initial_climb_phase[
                (initial_climb_phase["ROC"] > initial_climb_phase["ROC"].quantile(0.25))
                & (initial_climb_phase["altitude"] < max_altitude * 0.8)
            ]

        return takeoff_phase, initial_climb_phase, cruise_phase

    def calculate_thrust_minus_drag_for_phases(
        self, trajectory_df, flight_phases_refinement
    ):
        td_list = []

        for flight_id, group in tqdm(
            trajectory_df.groupby("flight_id"), desc="Calculating T-D for each flight"
        ):
            takeoff_phase, initial_climb_phase, _ = self.identify_flight_phases(
                group, flight_phases_refinement
            )
            relevant_phase = pd.concat([takeoff_phase, initial_climb_phase])

            if relevant_phase.empty:
                continue

            relevant_phase["V"] = relevant_phase.apply(
                self.calculate_true_airspeed, axis=1
            )
            relevant_phase["dh_dt"] = relevant_phase.apply(
                self.calculate_vertical_speed, axis=1
            )
            relevant_phase["delta_t"] = relevant_phase.apply(
                self.calculate_temp_deviation, axis=1
            )

            relevant_phase = self.calculate_horizontal_acceleration(relevant_phase)
            relevant_phase = self.calculate_wind_acceleration(relevant_phase)

            relevant_phase["T_minus_D"] = (
                32.17405
                * relevant_phase["dh_dt"]
                / relevant_phase["V"]
                * (
                    relevant_phase["temperature"]
                    / (relevant_phase["temperature"] - relevant_phase["delta_t"])
                )
                + relevant_phase["dV_dt"]
                + relevant_phase["dWi_dt"]
            )

            relevant_phase["flight_id"] = flight_id
            td_list.append(relevant_phase)

        td_df = pd.concat(td_list, ignore_index=True)
        return td_df.dropna(subset=["T_minus_D"])

    @staticmethod
    def aggregate_features(td_df):
        numerical_cols = td_df.select_dtypes(include=np.number).columns.tolist()
        if "flight_id" in numerical_cols:
            numerical_cols.remove("flight_id")
        agg_funcs = {col: ["mean", "max", "std"] for col in numerical_cols}
        aggregated_features = td_df.groupby("flight_id").agg(agg_funcs).reset_index()
        aggregated_features.columns = [
            "_".join(col).rstrip("_") for col in aggregated_features.columns.values
        ]
        return aggregated_features

    def process_data(
        self, trajectory_df, train_df, use_trajectory, flight_phases_refinement
    ):
        if use_trajectory:
            td_df = self.calculate_thrust_minus_drag_for_phases(
                trajectory_df, flight_phases_refinement
            )
            aggregated_features = self.aggregate_features(td_df)
            merged_df = pd.merge(
                train_df,
                aggregated_features,
                left_on="flight_id",
                right_on="flight_id",
                how="inner",
            )
        else:
            merged_df = train_df.copy()
        return merged_df
