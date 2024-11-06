import json

import fsspec
import pandas as pd


class DataEnricher:
    def __init__(self, external_info_file: str):
        self.external_info_file = external_info_file

    def load_external_data(self):
        """Loads and processes external aircraft information."""
        with fsspec.open(self.external_info_file, "r") as file:
            external_information = json.load(file)

        external_df = pd.DataFrame.from_dict(external_information,
                                             orient="index")
        external_df.reset_index(inplace=True)
        external_df.rename(columns={"index": "aircraft_type"}, inplace=True)
        return external_df

    def enrich_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Enriches training and test data with external information."""
        external_df = self.load_external_data()

        train_enriched = pd.merge(train_df, external_df, on="aircraft_type",
                                  how="left")
        test_enriched = pd.merge(test_df, external_df, on="aircraft_type",
                                 how="left")

        return train_enriched, test_enriched
