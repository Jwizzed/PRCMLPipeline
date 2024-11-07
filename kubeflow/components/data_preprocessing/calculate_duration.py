from kfp.v2.dsl import component, InputPath, OutputPath


@component(
    base_image='gcr.io/prc-data-pipeline/ml-image',
    packages_to_install=["pandas"])
def calculate_flight_duration(input_df: InputPath("CSV"),
                              output_df: OutputPath("CSV")):
    """Calculates flight duration in minutes."""
    from kubeflow.src.data_preprocessing.calculate_duration import DurationCalculator

    calculator = DurationCalculator()
    df_with_duration = calculator.calculate_duration(input_df)
    df_with_duration.to_csv(output_df, index=False)
