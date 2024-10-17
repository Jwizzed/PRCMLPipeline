from typing import Tuple

import pandas as pd
from typing_extensions import Annotated
from zenml import step
from sklearn.model_selection import train_test_split


@step
def split_data(
    X: Annotated[pd.DataFrame, "Features"], y: Annotated[pd.DataFrame, "Target"]
) -> Tuple[
    Annotated[pd.DataFrame, "Train features"],
    Annotated[pd.DataFrame, "Test features"],
    Annotated[pd.DataFrame, "Train labels"],
    Annotated[pd.DataFrame, "Test labels"],
]:
    """Splits data into training and testing sets."""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
