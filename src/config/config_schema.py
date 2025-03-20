from typing import Literal

from pydantic import BaseModel, ValidationInfo, field_validator


class ExtractionStep(BaseModel):
    """Configuration schema for the extraction step.

    Attributes:
        source (str): Path to the data source (currently supports only CSV).
        features (list[str]): List of feature names to extract from the data.
    """

    source: str
    features: list[str]
    # for now only CSV is supported


class DropDuplicates(BaseModel):
    """Configuration schema for handling duplicate records during transformation.

    Attributes:
        enabled (bool): Indicates if duplicate removal is enabled.
        subset (Optional[list[str]]): List of columns to consider when identifying duplicates.
        keep (Literal): Which duplicate to keep ('first', 'last', or 'none').
        drop_subset_columns (bool): If True, drops the subset columns after removing duplicates.

    Raises:
        ValueError: If 'drop_subset_columns' is True and 'subset' is not provided or empty.
    """

    __DROP_SUBSET_ERROR_MSG = (
        "When 'drop_subset_columns' is True, 'subset' must be provided and not empty."
    )

    enabled: bool
    subset: list[str] | None = None
    keep: Literal["first", "last", "none"] = "first"
    drop_subset_columns: bool

    @field_validator("drop_subset_columns")
    def validate_subset_if_drop_columns(
        cls, drop_subset_columns: bool, info: ValidationInfo
    ) -> bool:
        """Validates that 'subset' is provided when 'drop_subset_columns' is True.

        Args:
            drop_subset_columns (bool): Flag indicating whether to drop subset columns.
            info (ValidationInfo): Contains the field values of the model.

        Raises:
            ValueError: If 'drop_subset_columns' is True and 'subset' is not set or is empty.

        Returns:
            bool: The validated 'drop_subset_columns' value.
        """
        subset_columns: list[str] | None = info.data.get("subset")
        if drop_subset_columns and (not subset_columns or len(subset_columns) == 0):
            raise ValueError(cls.__DROP_SUBSET_ERROR_MSG)
        return drop_subset_columns


class TransformationStep(BaseModel):
    """Configuration schema for the transformation step.

    Attributes:
        numeric_features (dict[str, str]): Dictionary mapping numeric feature names to their types.
        categorical_features (dict[str, list[str] | None]): Dictionary mapping categorical
            feature names to their possible categories. If None, categories will not be enforced.
        drop_duplicates (DropDuplicates): Configuration for handling duplicate removal.
    """

    numeric_features: dict[str, str]
    categorical_features: dict[str, list[str] | None]
    drop_duplicates: DropDuplicates


class ConfigSchema(BaseModel):
    """Main configuration schema combining extraction and transformation steps.

    Attributes:
        extraction (ExtractionStep): Configuration for the extraction step.
        transformation (TransformationStep): Configuration for the transformation step.
    """

    extraction: ExtractionStep
    transformation: TransformationStep
