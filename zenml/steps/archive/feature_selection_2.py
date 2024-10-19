from typing import Tuple

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from typing_extensions import Annotated
from zenml import step

from config import Config


# TODO: feature selection strategy
@step
def feature_selection(
        config: Config,
        X_train: Annotated[pd.DataFrame, "Train features"],
        y_train: Annotated[pd.Series, "Train labels"],
        X_test: Annotated[pd.DataFrame, "Test features"],
) -> Tuple[
    Annotated[pd.DataFrame, "Selected Train features"],
    Annotated[pd.DataFrame, "Selected Test features"],
]:
    """Selects the best features using RFE and applies them to the test data."""
    estimator = LinearRegression()
    selector = RFE(estimator, n_features_to_select=config.NUM_FEATURES, step=1)
    selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    selected_features = X_train.columns[selector.get_support()]
    X_train_selected = pd.DataFrame(
        X_train_selected, columns=selected_features, index=X_train.index
    )
    X_test_selected = pd.DataFrame(
        X_test_selected, columns=selected_features, index=X_test.index
    )

    return X_train_selected, X_test_selected
