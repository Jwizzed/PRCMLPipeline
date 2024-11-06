from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Tuple, Union


class SplitDataStrategy(ABC):
    """
    Abstract base class for data splitting strategies.

    This class defines the interface for different data splitting strategies.
    Each concrete strategy should implement the split method according to its specific logic.
    """

    @abstractmethod
    def split(self, df: pd.DataFrame, **kwargs) -> Dict[
        str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split the input DataFrame according to the strategy's logic.

        :param df: Input DataFrame to be split
        :type df: pd.DataFrame
        :param kwargs: Additional keyword arguments that might be needed by specific strategies
        :return: Dictionary containing the split results with their corresponding labels
        :rtype: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]
        """
        pass

    def process_subset(self, subset_df: pd.DataFrame, is_test: bool = False) -> \
            Union[
                Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, None]]:
        """
        Process a subset of the data according to the strategy's requirements.

        :param subset_df: Subset of data to be processed
        :type subset_df: pd.DataFrame
        :param is_test: Flag indicating if this is test data processing
        :type is_test: bool
        :return: Tuple of processed X and y DataFrames, or (X, None) if is_test is True
        :rtype: Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, None]]
        """
        pass
