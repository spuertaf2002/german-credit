from dataclasses import dataclass
from typing import Any

from numpy import nan
from pandas import Categorical, DataFrame, Series
from typing_extensions import override

from src.config.config_schema import TransformationStep
from src.pipeline.base_step import BaseStep


@dataclass
class Transformation(BaseStep):
    config: TransformationStep
    name: str = "transformation"

    __MISSING_FEATURES_ERROR: str = "Transformed DataFrame contains unexpected or missing features."

    def __process_duplicates(self, input_data: DataFrame) -> DataFrame:
        """Removes duplicate rows from the DataFrame based on the configuration.

        Args:
            data (DataFrame): Input DataFrame to process for duplicates.

        Returns:
            DataFrame: DataFrame with duplicates removed and optional subset columns dropped.
        """
        if not self.config["drop_duplicates"]["enabled"]:
            return input_data

        data: DataFrame = input_data.drop_duplicates(
            subset=self.config["drop_duplicates"]["subset"],
            keep=self.config["drop_duplicates"]["keep"],
        )
        return (
            data.drop(columns=self.config["drop_duplicates"]["subset"])
            if self.config["drop_duplicates"]["drop_subset_columns"]
            else data
        )

    def __convert_to_categorical(self, data: DataFrame) -> DataFrame:
        """Converts specified columns in the DataFrame to categorical data types.

        If categories are defined for a column, it sets the category order.

        Args:
            data (DataFrame): Input DataFrame to convert categorical columns.

        Returns:
            DataFrame: DataFrame with categorical columns converted.
        """
        data[list(self.config["categorical_features"].keys())] = data[
            list(self.config["categorical_features"].keys())
        ].astype("category")

        for col, categories in self.config["categorical_features"].items():
            if categories:
                data[col] = Categorical(data[col], categories=categories, ordered=True)

        return data

    def __convert_to_numeric(self, input_data: DataFrame) -> DataFrame:
        """Cleans and converts specified columns in the DataFrame to numeric data types.

        Invalid numeric values are replaced with NaN based on regex validation.

        Args:
            data (DataFrame): Input DataFrame to process numeric columns.

        Returns:
            DataFrame: DataFrame with numeric columns cleaned and converted.
        """

        def replace_invalid_numerics(data: DataFrame) -> DataFrame:
            """Replaces invalid numeric values in the DataFrame with NaN.

            Args:
                data (DataFrame): DataFrame containing numeric columns as strings.

            Returns:
                DataFrame: DataFrame with invalid numeric entries replaced by NaN.
            """
            check: DataFrame = data.copy()
            for col in self.config["numeric_features"]:
                check[col] = check[col].astype(str)  # convert to string temporarily for checking
                mask: Series[bool] = ~check[col].str.match(
                    r"^-?\d+\.?\d*$", na=True
                )  # detecting weird values inside numerical columns
                data.loc[mask, col] = nan
            return data

        data: DataFrame = replace_invalid_numerics(input_data)
        return data.astype(
            {col: numeric_type for col, numeric_type in self.config["numeric_features"].items()}
        )

    @override
    def run(
        self, data: Any | None = None
    ) -> DataFrame:  # All this steps execution could be improved by using builder pattern
        """Executes the transformation pipeline on the input DataFrame.

        The transformation includes:
        - Dropping duplicates if configured.
        - Converting categorical columns.
        - Cleaning and converting numeric columns.

        Args:
            data (DataFrame): Input DataFrame to transform.

        Returns:
            DataFrame: Transformed DataFrame ready for downstream processing.
        """
        assert isinstance(data, DataFrame), "Provided data must be DataFrame"

        temp_data: DataFrame = data.copy()
        features: list = list(self.config["numeric_features"].keys()) + list(
            self.config["categorical_features"].keys()
        )

        temp_data = self.__process_duplicates(temp_data)
        temp_data = self.__convert_to_categorical(temp_data)
        temp_data = self.__convert_to_numeric(temp_data)

        assert all(feature in features for feature in temp_data.columns), (
            f"{self.__MISSING_FEATURES_ERROR} Unexpected: {set(temp_data.columns) - set(features)}"
        )

        return temp_data
