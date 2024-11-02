from kfp.v2 import dsl
from kubeflow.components.data_ingestion.ingest_data import load_data
from kubeflow.components.data_preprocessing.get_duration import calculate_flight_duration
from kubeflow.components.data_preprocessing.get_external_data import add_external_data
from kubeflow.components.data_preprocessing.clean_df_with_isolation_forest import clean_dataframe_with_isolation_forest, clean_trajectory_with_isolation_forest
from kubeflow.components.data_preprocessing.find_thrus_minus_drag import calculate_and_aggregate_features
from kubeflow.components.data_preprocessing.normalize_data import normalize_dataframe
from kubeflow.components.data_preprocessing.encode_category_data import encode_categorical_features


@dsl.pipeline(
    name="Flight TOW Prediction Pipeline",
    description="Pipeline for predicting Take-Off Weight (TOW)",
)
def flight_tow_pipeline(data_path: str, external_data_path: str):
    load_data_task = load_data(data_path=data_path)

    duration_task = calculate_flight_duration(
        input_df=load_data_task.outputs["train_file"]
    )

    external_data_task = add_external_data(
        train_file=duration_task.outputs["output_df"],
        test_file=load_data_task.outputs["test_file"],
        external_info_file=external_data_path
    )

    clean_data_task = clean_dataframe_with_isolation_forest(
        input_df=external_data_task.outputs["train_enriched_file"],
        contamination=0.01
    )

    clean_trajectory_task = clean_trajectory_with_isolation_forest(
        input_df=load_data_task.outputs["trajectory_file"],
        contamination=0.01
    )

    thrust_drag_task = calculate_and_aggregate_features(
        trajectory_df_path=clean_trajectory_task.outputs["output_df"],
        train_df_path=clean_data_task.outputs["output_df"],
        use_trajectory=True,
        flight_phases_refinement=True
    )

    normalize_task = normalize_dataframe(
        input_file=thrust_drag_task.outputs["aggregated_features_path"],
        exclude_columns=["flight_id", "tow"],
        split_by_flown_distance=True
    )

    encode_task = encode_categorical_features(
        input_file=normalize_task.outputs["output_file"],
        preserve_columns=["aircraft_type"]
    )

    # # 9. Drop unnecessary features
    # drop_features_task = drop_features_component(
    #     input_file=encode_task.outputs["output_file"],
    #     final_drop=True
    # )
    #
    # # 10. Split data by WTC
    # split_wtc_task = split_wtc_component(
    #     input_file=drop_features_task.outputs["output_file"]
    # )
    #
    # # 11. Train-test split for each WTC category
    # train_test_split_M = train_test_split_component(
    #     input_file=split_wtc_task.outputs["X_wtc_M_file"],
    #     test_size=0.2
    # )
    #
    # train_test_split_H = train_test_split_component(
    #     input_file=split_wtc_task.outputs["X_wtc_H_file"],
    #     test_size=0.2
    # )
    #
    # # 12. Feature selection for each WTC category
    # feature_selection_M = feature_selection_component(
    #     X_train_file=train_test_split_M.outputs["X_train_file"],
    #     y_train_file=train_test_split_M.outputs["y_train_file"],
    #     X_test_file=train_test_split_M.outputs["X_test_file"],
    #     k=15
    # )
    #
    # feature_selection_H = feature_selection_component(
    #     X_train_file=train_test_split_H.outputs["X_train_file"],
    #     y_train_file=train_test_split_H.outputs["y_train_file"],
    #     X_test_file=train_test_split_H.outputs["X_test_file"],
    #     k=15
    # )
    #
    # # 13. Train ensemble models for each WTC category
    # ensemble_M = ensemble_models(
    #     x_train_file=feature_selection_M.outputs["X_train_selected_file"],
    #     y_train_file=train_test_split_M.outputs["y_train_file"],
    #     x_test_file=feature_selection_M.outputs["X_test_selected_file"],
    #     y_test_file=train_test_split_M.outputs["y_test_file"],
    #     model_name="WTC_M_Model",
    #     find_best_parameters=False
    # )
    #
    # ensemble_H = ensemble_models(
    #     x_train_file=feature_selection_H.outputs["X_train_selected_file"],
    #     y_train_file=train_test_split_H.outputs["y_train_file"],
    #     x_test_file=feature_selection_H.outputs["X_test_selected_file"],
    #     y_test_file=train_test_split_H.outputs["y_test_file"],
    #     model_name="WTC_H_Model",
    #     find_best_parameters=False
    # )
