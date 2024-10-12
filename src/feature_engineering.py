import math

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class FeatureEngineering:
    def __init__(self):
        self.g = 32.17405  # ft/s^2
        self.kt_to_ft_per_sec = 1.6878098571012
        self.m_to_ft = 3.28084

    @staticmethod
    def isa(alt):
        # Standard atmosphere calculations
        T0 = 288.15
        p0 = 101325
        R = 287.05287
        g0 = 9.80665
        L = -0.0065
        h_trop = 11000

        if alt < 0 or alt > 47000:
            print("Altitude must be in [0, 47000]")
            return None, None

        if alt <= h_trop:
            T = T0 + L * alt
            p = p0 * (T / T0) ** (-g0 / (L * R))
        else:
            T = T0 + L * h_trop
            p = p0 * (T / T0) ** (-g0 / (L * R)) * math.exp(
                -g0 * (alt - h_trop) / (R * T))

        rho = p / (R * T)
        return T, rho

    def calculate_true_airspeed(self, row):
        GS = row['groundspeed']  # in knots
        track_rad = np.radians(row['track'])  # Convert track to radians
        u_wind = row['u_component_of_wind'] * self.m_to_ft  # m/s to ft/s
        v_wind = row['v_component_of_wind'] * self.m_to_ft  # m/s to ft/s

        V_ground = GS * self.kt_to_ft_per_sec

        V_ground_x = V_ground * np.sin(track_rad)
        V_ground_y = V_ground * np.cos(track_rad)

        V_air_x = V_ground_x - u_wind
        V_air_y = V_ground_y - v_wind

        V_true = np.sqrt(V_air_x ** 2 + V_air_y ** 2)
        return V_true

    @staticmethod
    def calculate_vertical_speed(row):
        return row['vertical_rate'] / 60  # ft/min to ft/s

    def calculate_temp_deviation(self, row):
        isa_temp, _ = self.isa(row['altitude'])
        return row['temperature'] - isa_temp

    def calculate_horizontal_acceleration(self, group):
        group = group.sort_values('timestamp')
        group['V'] = group.apply(self.calculate_true_airspeed, axis=1)
        group['dV_dt'] = group['V'].diff() / group[
            'timestamp'].diff().dt.total_seconds()
        return group

    def calculate_wind_acceleration(self, group):
        group = group.sort_values('timestamp')
        time_diff = group['timestamp'].diff().dt.total_seconds()

        track_rad = np.radians(group['track'])
        u_wind = group['u_component_of_wind'] * self.m_to_ft
        v_wind = group['v_component_of_wind'] * self.m_to_ft
        V_wind_along = u_wind * np.sin(track_rad) + v_wind * np.cos(track_rad)

        group['dWi_dt'] = V_wind_along.diff() / time_diff
        return group

    @staticmethod
    def identify_flight_phases(group):
        group = group.sort_values('timestamp').reset_index(drop=True)
        group['altitude_diff'] = group['altitude'].diff()

        # Smoothing altitude profile
        window_length = min(21, len(group) // 2 * 2 + 1)
        if window_length < 5:  # Need at least window length of 5
            window_length = 5
        altitude_smooth = signal.savgol_filter(group['altitude'],
                                               window_length=window_length,
                                               polyorder=3)

        group['ROC'] = np.gradient(altitude_smooth,
                                   group['timestamp'].astype(int) / 1e9)
        max_altitude = group['altitude'].max()
        takeoff_end_idx = \
            group[group['altitude'] > group['altitude'].quantile(0.1)].index[0]
        top_of_climb_idx = \
            group[group['altitude'] > max_altitude * 0.95].index[0]

        takeoff_phase = group.loc[:takeoff_end_idx]
        initial_climb_phase = group.loc[takeoff_end_idx:top_of_climb_idx]
        cruise_phase = group.loc[top_of_climb_idx:]

        return takeoff_phase, initial_climb_phase, cruise_phase

    def calculate_thrust_minus_drag_for_phases(self, trajectory_df):
        td_list = []
        g = self.g

        for flight_id, group in tqdm(trajectory_df.groupby('flight_id'),
                                     desc="Calculating T-D per flight"):
            takeoff_phase, initial_climb_phase, _ = self.identify_flight_phases(
                group)

            # Combine phases for calculations
            relevant_phase = pd.concat([takeoff_phase, initial_climb_phase])

            if relevant_phase.empty:
                continue

            relevant_phase['V'] = relevant_phase.apply(
                self.calculate_true_airspeed, axis=1)
            relevant_phase['dh_dt'] = relevant_phase.apply(
                self.calculate_vertical_speed, axis=1)
            relevant_phase['delta_t'] = relevant_phase.apply(
                self.calculate_temp_deviation, axis=1)

            relevant_phase = self.calculate_horizontal_acceleration(
                relevant_phase)
            relevant_phase = self.calculate_wind_acceleration(relevant_phase)

            # Calculate T - D
            relevant_phase['T_minus_D'] = (
                    g * relevant_phase['dh_dt'] / relevant_phase['V'] * (
                    relevant_phase['temperature'] / (
                    relevant_phase['temperature'] - relevant_phase[
                'delta_t']))
                    + relevant_phase['dV_dt'] + relevant_phase['dWi_dt']
            )

            relevant_phase['flight_id'] = flight_id
            td_list.append(relevant_phase)

        td_df = pd.concat(td_list, ignore_index=True)
        td_df = td_df.dropna(subset=['T_minus_D'])
        return td_df

    @staticmethod
    def aggregate_features(td_df):
        numerical_cols = td_df.select_dtypes(
            include=np.number).columns.tolist()
        if 'flight_id' in numerical_cols:
            numerical_cols.remove('flight_id')
        agg_funcs = {col: ['mean', 'max', 'std'] for col in numerical_cols}
        aggregated_features = td_df.groupby('flight_id').agg(
            agg_funcs).reset_index()
        aggregated_features.columns = ['_'.join(col).rstrip('_') for col in
                                       aggregated_features.columns.values]
        return aggregated_features

    @staticmethod
    def normalize_dataframe(df, exclude_columns=None):
        df_normalized = df.copy()

        if exclude_columns is None:
            exclude_columns = []

        columns_to_normalize = df.select_dtypes(
            include=[np.number]).columns.difference(exclude_columns)
        scaler = StandardScaler()
        df_normalized[columns_to_normalize] = scaler.fit_transform(
            df[columns_to_normalize])

        return df_normalized

    @staticmethod
    def encode_categorical_features(df):
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_features = encoder.fit_transform(df[['wtc']])
        feature_names = encoder.get_feature_names_out(['wtc'])
        encoded_df = pd.DataFrame(encoded_features, columns=feature_names,
                                  index=df.index)
        df_encoded = pd.concat([df, encoded_df], axis=1)
        df_encoded = df_encoded.drop(['wtc'], axis=1)
        return df_encoded

    @staticmethod
    def drop_unnecessary_features(df):
        drop_cols = ['flight_id', 'date', 'callsign', 'adep', 'ades', 'actual_offblock_time', 'arrival_time', 'aircraft_type', 'airline', 'name_adep', 'country_code_adep', 'name_ades', 'country_code_ades']
        df_dropped = df.drop(columns=drop_cols, errors='ignore')
        return df_dropped

    @staticmethod
    def get_external_information():
        return {
            "B738": {
                "MTOW(kg)": 70530,
                "passengers": 162,
                "ROC_Initial_Climb(ft/min)": 3000,
                "V2 (IAS)": 145,
            },
            "A333": {
                "MTOW(kg)": 230000,
                "passengers": 295,
                "ROC_Initial_Climb(ft/min)": 2000,
                "V2 (IAS)": 145,
            },
            "B77W": {
                "MTOW(kg)": 351500,
                "passengers": 365,
                "ROC_Initial_Climb(ft/min)": 2000,
                "V2 (IAS)": 149,
            },
            "B38M": {
                "MTOW(kg)": 82600,
                "passengers": 162,
                "ROC_Initial_Climb(ft/min)": 2500,
                "V2 (IAS)": 145,
            },
            "A320": {
                "MTOW(kg)": 73900,
                "passengers": 150,
                "ROC_Initial_Climb(ft/min)": 2500,
                "V2 (IAS)": 145,
            },
            "E190": {
                "MTOW(kg)": 45995,
                "passengers": 94,
                "ROC_Initial_Climb(ft/min)": 3400,
                "V2 (IAS)": 138,
            },
            "CRJ9": {
                "MTOW(kg)": 38330,
                "passengers": 80,
                "ROC_Initial_Climb(ft/min)": 2500,
                "V2 (IAS)": 140,
            },
            "A21N": {
                "MTOW(kg)": 97000,
                "passengers": 180,
                "ROC_Initial_Climb(ft/min)": 2000,
                "V2 (IAS)": 145,
            },
            "A20N": {
                "MTOW(kg)": 79000,
                "passengers": 150,
                "ROC_Initial_Climb(ft/min)": 2200,
                "V2 (IAS)": 145,
            },
            "B739": {
                "MTOW(kg)": 79015,
                "passengers": 177,
                "ROC_Initial_Climb(ft/min)": 3000,
                "V2 (IAS)": 149,
            },
            "BCS3": {
                "MTOW(kg)": 69900,
                "passengers": 120,
                "ROC_Initial_Climb(ft/min)": 3100,
                "V2 (IAS)": 165,
            },
            "E195": {
                "MTOW(kg)": 52290,
                "passengers": 100,
                "ROC_Initial_Climb(ft/min)": 3000,
                "V2 (IAS)": 140,
            },
            "A321": {
                "MTOW(kg)": 83000,
                "passengers": 185,
                "ROC_Initial_Climb(ft/min)": 2500,
                "V2 (IAS)": 145,
            },
            "A359": {
                "MTOW(kg)": 268000,
                "passengers": 314,
                "ROC_Initial_Climb(ft/min)": 3000,
                "V2 (IAS)": 150,
            },
            "A319": {
                "MTOW(kg)": 64000,
                "passengers": 124,
                "ROC_Initial_Climb(ft/min)": 2500,
                "V2 (IAS)": 135,
            },
            "A332": {
                "MTOW(kg)": 230000,
                "passengers": 253,
                "ROC_Initial_Climb(ft/min)": 2000,
                "V2 (IAS)": 145,
            },
            "B788": {
                "MTOW(kg)": 228000,
                "passengers": 210,
                "ROC_Initial_Climb(ft/min)": 2700,
                "V2 (IAS)": 165,
            },
            "B789": {
                "MTOW(kg)": 228000,
                "passengers": 210,
                "ROC_Initial_Climb(ft/min)": 2700,
                "V2 (IAS)": 165,
            },
            "BCS1": {
                "MTOW(kg)": 63100,
                "passengers": 100,
                "ROC_Initial_Climb(ft/min)": 3500,
                "V2 (IAS)": 140,
            },
            "B763": {
                "MTOW(kg)": 186880,
                "passengers": 269,
                "ROC_Initial_Climb(ft/min)": 3000,
                "V2 (IAS)": 160,
            },
            "AT76": {
                "MTOW(kg)": 23000,
                "passengers": 78,
                "ROC_Initial_Climb(ft/min)": 1350,
                "V2 (IAS)": 110,
            },
            "B772": {
                "MTOW(kg)": 247210,
                "passengers": 305,
                "ROC_Initial_Climb(ft/min)": 3000,
                "V2 (IAS)": 170,
            },
            "B737": {
                "MTOW(kg)": 66320,
                "passengers": 128,
                "ROC_Initial_Climb(ft/min)": 3000,
                "V2 (IAS)": 150,
            },
            "A343": {
                "MTOW(kg)": 275000,
                "passengers": 295,
                "ROC_Initial_Climb(ft/min)": 1400,
                "V2 (IAS)": 145,
            },
            "B39M": {
                "MTOW(kg)": 88300,
                "passengers": 178,
                "ROC_Initial_Climb(ft/min)": 2300,
                "V2 (IAS)": 150,
            },
            "B752": {
                "MTOW(kg)": 115680,
                "passengers": 200,
                "ROC_Initial_Climb(ft/min)": 3500,
                "V2 (IAS)": 145,
            },
            "B773": {
                "MTOW(kg)": 299370,
                "passengers": 368,
                "ROC_Initial_Climb(ft/min)": 3000,
                "V2 (IAS)": 168,
            },
            "E290": {
                "MTOW(kg)": 45995,
                "passengers": 94,
                "ROC_Initial_Climb(ft/min)": 3400,
                "V2 (IAS)": 138,
            },
        }
