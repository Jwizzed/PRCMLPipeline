import json
from kfp.v2.dsl import component, InputPath, OutputPath


@component(
    packages_to_install=['pandas']
)
def add_external_data(
        train_file: InputPath('CSV'),
        test_file: InputPath('CSV'),
        external_info_file: InputPath('JSON'),
        train_enriched_file: OutputPath('CSV'),
        test_enriched_file: OutputPath('CSV')
):
    """Adds external aircraft information."""
    import pandas as pd

    with open(external_info_file, 'r') as file:
        external_information = json.load(file)

    external_df = pd.DataFrame.from_dict(external_information, orient='index')
    external_df.reset_index(inplace=True)
    external_df.rename(columns={'index': 'aircraft_type'}, inplace=True)

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    train_enriched = pd.merge(train_df, external_df, on='aircraft_type',
                              how='left')
    test_enriched = pd.merge(test_df, external_df, on='aircraft_type',
                             how='left')

    train_enriched.to_csv(train_enriched_file, index=False)
    test_enriched.to_csv(test_enriched_file, index=False)
