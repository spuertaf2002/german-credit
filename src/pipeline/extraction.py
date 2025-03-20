from dataclasses import dataclass

from pandas import DataFrame, read_csv

from src.config.config_schema import ExtractionStep
from src.pipeline.base_step import BaseStep


@dataclass
class Extraction(BaseStep):
    config: ExtractionStep
    name: str = "extraction"

    __MISSING_FEATURES_ERROR: str = "Some required features are missing from the dataset."

    # @override
    def run(self, _: None = None) -> DataFrame:
        """Executes the extraction step by loading data from the specified source.

        Reads a CSV file from the path defined in the extraction configuration,
        verifies that all required features are present in the dataset, and
        returns a DataFrame containing only the selected features.

        Returns:
            DataFrame: A pandas DataFrame containing only the configured features.

        Raises:
            AssertionError: If any of the required features are missing from the dataset.
        """
        df: DataFrame = read_csv(self.config["source"], low_memory=False)
        assert set(self.config["features"]).issubset(set(df.columns)), (
            f"{self.__MISSING_FEATURES_ERROR}\
                Missing: {set(self.config['features']) - set(df.columns)}"
        )
        return df[self.config["features"]]
