import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split


class FeatureProcessor:
    @staticmethod
    def drop_features(df: pd.DataFrame,
                      final_drop: bool = False) -> pd.DataFrame:
        drop_cols = [
            "date",
            "callsign",
            "name_adep",
            "name_ades",
            "actual_offblock_time",
            "arrival_time",
        ]

        if final_drop:
            drop_cols.extend(["aircraft_type"])

        return df.drop(columns=drop_cols, errors="ignore")

    @staticmethod
    def select_features(X_train: pd.DataFrame, y_train: pd.DataFrame,
                        X_test: pd.DataFrame, k: int = 15):
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X_train, y_train)

        selected_features = X_train.columns[selector.get_support()]

        selected_features_df = pd.DataFrame(
            selected_features, columns=["Selected Features"]
        )

        feature_scores_df = pd.DataFrame({
            "Feature": X_train.columns,
            "Score": selector.scores_,
            "p-value": selector.pvalues_,
        })

        X_train_selected = pd.DataFrame(
            selector.transform(X_train),
            columns=selected_features,
            index=X_train.index
        )

        X_test_selected = pd.DataFrame(
            selector.transform(X_test),
            columns=selected_features,
            index=X_test.index
        )

        return (X_train_selected, X_test_selected,
                selected_features_df, feature_scores_df)

    @staticmethod
    def split_data(X: pd.DataFrame, y: pd.DataFrame):
        return train_test_split(X, y, test_size=0.2, random_state=42)