from kfp.v2 import dsl

from kubeflow.components.data_ingestion.ingest_data import load_data
from kubeflow.components.data_preprocessing.clean_data import (
    clean_dataframe_with_isolation_forest,
    clean_trajectory_with_isolation_forest,
)
from kubeflow.components.data_preprocessing.encode_feature import (
    encode_categorical_features,
)
from kubeflow.components.data_preprocessing.aggregate_feature import (
    calculate_and_aggregate_features,
)
from kubeflow.components.data_preprocessing.calculate_duration import (
    calculate_flight_duration,
)
from kubeflow.components.data_preprocessing.enrich_data import add_external_data
from kubeflow.components.data_preprocessing.normalize_data import normalize_dataframe
from kubeflow.components.data_preprocessing.select_features import (
    feature_selection,
    drop_features,
    process_category_split,
)
from kubeflow.components.data_preprocessing.split_data.split_by_mtow import (
    split_by_mtow,
)
from kubeflow.components.modelling.train_ensemble_model import train_ensemble_model
from kubeflow.components.deployment.deploy_model import deploy_model


@dsl.pipeline(
    name="Flight TOW Prediction Pipeline",
    description="Pipeline for predicting Take-Off Weight (TOW)",
)
def flight_tow_pipeline(data_path: str, external_data_path: str, deployment_path: str):
    load_data_task = load_data(data_path=data_path)
    duration_task = calculate_flight_duration(
        input_df=load_data_task.outputs["train_file"]
    )
    external_data_task = add_external_data(
        train_file=duration_task.outputs["output_df"],
        test_file=load_data_task.outputs["test_file"],
        external_info_file=external_data_path,
    )
    clean_data_task = clean_dataframe_with_isolation_forest(
        input_df=external_data_task.outputs["train_enriched_file"], contamination=0.01
    )
    clean_trajectory_task = clean_trajectory_with_isolation_forest(
        input_df=load_data_task.outputs["trajectory_file"], contamination=0.01
    )
    thrust_drag_task = calculate_and_aggregate_features(
        trajectory_df_path=clean_trajectory_task.outputs["output_df"],
        train_df_path=clean_data_task.outputs["output_df"],
        use_trajectory=True,
        flight_phases_refinement=True,
    )

    exclude_cols = [
        "MTOW(kg)",
        "flight_id",
        "tow",
        "date",
        "callsign",
        "adep",
        "ades",
        "actual_offblock_time",
        "arrival_time",
        "aircraft_type",
        "wtc",
        "airline",
    ]

    normalize_task = normalize_dataframe(
        input_file=thrust_drag_task.outputs["aggregated_features_path"],
        exclude_columns=exclude_cols,
        split_by_flown_distance=True,
    )

    encode_task = encode_categorical_features(
        input_file=normalize_task.outputs["output_file"],
        preserve_columns=["aircraft_type"],
    )

    initial_drop_task = drop_features(
        input_file=encode_task.outputs["output_file"], final_drop=False
    )

    split_mtow_task = split_by_mtow(input_file=initial_drop_task.outputs["output_file"])

    categories = ["very_low", "low", "medium", "high", "non_b77w", "b77w"]

    for category in categories:
        category_drop_task = drop_features(
            input_file=split_mtow_task.outputs[f"X_{category}_output"], final_drop=True
        )

        category_split_task = process_category_split(
            X_file=category_drop_task.outputs["output_file"],
            y_file=split_mtow_task.outputs[f"y_{category}_output"],
        )

        category_feature_selection = feature_selection(
            X_train_file=category_split_task.outputs["X_train_output"],
            y_train_file=category_split_task.outputs["y_train_output"],
            X_test_file=category_split_task.outputs["X_test_output"],
        )

        ensemble_task = train_ensemble_model(
            x_train_file=category_feature_selection.outputs["X_train_selected_file"],
            y_train_file=category_split_task.outputs["y_train_output"],
            x_test_file=category_feature_selection.outputs["X_test_selected_file"],
            y_test_file=category_split_task.outputs["y_test_output"],
            model_name=f"{category}_ensemble",
            find_best_parameters=False,
        )
        deploy_catboost_task = deploy_model(
            model_file=ensemble_task.outputs["catboost_model_output"],
            model_name=f"{category}_catboost",
            deployment_path=deployment_path,
        )

        deploy_xgboost_task = deploy_model(
            model_file=ensemble_task.outputs["xgboost_model_output"],
            model_name=f"{category}_xgboost",
            deployment_path=deployment_path,
        )
