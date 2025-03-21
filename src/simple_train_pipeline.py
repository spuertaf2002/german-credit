from collections.abc import Callable
from typing import Any

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from joblib import dump
from numpy import nan, ndarray
from pandas import Categorical, DataFrame, Series, read_csv
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder


def get_features_names(_: Any, feature_names: ndarray) -> ndarray:
    """
    Returns the provided feature names.

    Args:
        _ (Any): Placeholder argument, not used.
        feature_names (ndarray): Array of feature names.

    Returns:
        ndarray: The input array of feature names.
    """
    return feature_names


def clean_features(
    X: DataFrame, values: dict[str, list] | None = None, expected: bool = True
) -> DataFrame:
    """
    Cleans the input DataFrame by filtering values in specific categorical columns.

    If `expected` is True, it keeps only the expected values defined in the `values` dict.
    If `expected` is False, it removes the values defined in `values`.

    Args:
        X (DataFrame): Input DataFrame to be cleaned.
        values (dict[str, list[str]] | None, optional):
            Dictionary mapping column names to lists of allowed (or disallowed) values.
            If None and `expected` is True, a default dictionary is used. Defaults to None.
        expected (bool, optional):
            If True, keeps only the expected values in each column.
            If False, removes those values. Defaults to True.

    Returns:
        DataFrame: The cleaned DataFrame with filtered categorical values.
    """
    assert isinstance(X, DataFrame)
    assert isinstance(values, dict) or values is None

    values = (
        values
        if values
        else {
            "Housing": ["own", "rent", "free"],
            "Sex": ["male", "female"],
            "Purpose": [
                "car",
                "radio/TV",
                "furniture/equipment",
                "business",
                "education",
                "repairs",
                "domestic appliances",
                "vacation/others",
            ],
        }
    )

    categories_to_review: list[str] = list(set(X.columns).intersection(set(values.keys())))
    if len(categories_to_review) == 0:
        return X

    X[categories_to_review] = (
        X[categories_to_review].apply(lambda x: x.where(x.isin(values[x.name])))
        if expected
        else X[categories_to_review].apply(lambda x: x.where(~x.isin(values[x.name])))
    )
    return X


def remove_outliers(X: DataFrame, threshold: float = 1.5) -> DataFrame:
    """
    Replace outliers from numeric columns given a certain treshold with NAN.
    Args:
        threshold (float, optional): Treshold for removing minor (1.5) or extreme outliers (3.0).
        Defaults to 1.5.
    """
    MIN_OUTLIER_THRESHOLD: float = 1.5
    MAX_OUTLIER_THRESHOLD: float = 3.0

    assert isinstance(X, DataFrame)
    assert MIN_OUTLIER_THRESHOLD <= threshold <= MAX_OUTLIER_THRESHOLD
    Q1, Q3 = X.quantile(0.25), X.quantile(0.75)
    IQR: float = Q3 - Q1
    mask: Callable = ~(((Q1 - threshold * IQR) <= X) & ((Q3 + threshold * IQR) >= X)).all(axis=1)
    X.loc[mask] = nan
    return X


DATA_URL: str = "https://github.com/JoseRZapata/Data_analysis_notebooks/raw/refs/heads/main/data/datasets/datos_credito_alemania_data.csv"
data: DataFrame = read_csv(DATA_URL, low_memory=False)

selected_features = [
    "Credit amount",
    "Purpose",
    "Job",
    "Sex",
    "Saving accounts",
    "Housing",
    "Risk",
    "Age",
    "Duration",
    "Unnamed: 0",  # this column contains indexes for duplicated data
]

dataset: DataFrame = data[selected_features]

dataset = dataset.drop_duplicates(subset=["Unnamed: 0"])

dataset = dataset.drop(columns=["Unnamed: 0"])

categorical_cols = {
    "Purpose": None,
    "Job": [0, 1, 2, 3],
    "Sex": None,
    "Saving accounts": ["little", "quite rich", "rich"],
    "Housing": None,
    "Risk": ["good", "bad"],
}

dataset[list(categorical_cols.keys())] = dataset[list(categorical_cols.keys())].astype("category")

for col, categories in categorical_cols.items():
    if categories:
        dataset[col] = Categorical(dataset[col], categories=categories, ordered=True)


numeric_cols = {
    "Duration": "Int64",  # has to be base 64 so it can be nullable
    "Credit amount": "Int64",
    "Age": "Int64",
}

checking_df = dataset.copy()
for col in numeric_cols:
    checking_df[col] = checking_df[col].astype(str)  # convert to string temporarily for checking
    mask = ~checking_df[col].str.match(
        r"^-?\d+\.?\d*$", na=True
    )  # detecting weird values inside numerical columns
    print(f"Invalid values found for column {col}:", dataset[mask][col].unique())
    dataset.loc[mask, col] = nan

dataset = dataset.astype({col: num_type for col, num_type in numeric_cols.items()})


num_cols = ["Credit amount", "Age"]
cat_cols = ["Purpose", "Housing", "Sex"]
cat_ord_cols = ["Saving accounts", "Job"]

target = "Risk"

dataset.dropna(subset=[target], inplace=True)

dataset[target] = dataset[target].map({"good": 1, "bad": 0}).astype("int8")

X_features: DataFrame = dataset.drop(target, axis="columns")
Y_target: Series = dataset[target]

x_train, x_test, y_train, y_test = train_test_split(
    X_features, Y_target, stratify=Y_target, test_size=0.2, random_state=42
)

numeric_pipe = Pipeline(
    steps=[
        (
            "outlier removal",
            FunctionTransformer(
                remove_outliers,
                kw_args={"threshold": 3.0},
                feature_names_out=get_features_names,
            ),
        ),
        ("imputer", KNNImputer(n_neighbors=5)),
    ]
)

cat_pipe = Pipeline(
    steps=[
        (
            "clean_categories",
            FunctionTransformer(clean_features, feature_names_out=get_features_names),
        ),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first")),
    ]
)

cat_ord_pipe = Pipeline(
    steps=[
        (
            "clean_categories",
            FunctionTransformer(clean_features, feature_names_out=get_features_names),
        ),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OrdinalEncoder()),
    ]
)

ordinal_transformers = [(f"{col}_ordinal", cat_ord_pipe, [col]) for col in cat_ord_cols]


preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipe, num_cols),
        ("categoric", cat_pipe, cat_cols),
        *ordinal_transformers,
    ]
)

data_model_pipeline = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("undersampling", RandomUnderSampler(random_state=42)),
        ("model", LogisticRegression(solver="liblinear")),
    ]
)

score: str = "precision"

hyperparams: dict = {
    "model__penalty": ["l1", "l2"],
    "model__C": [0.1, 0.4, 0.8, 1, 2, 5],
}

grid_search = GridSearchCV(
    data_model_pipeline,
    hyperparams,
    cv=5,
    scoring=score,
    n_jobs=8,
)
grid_search.fit(x_train, y_train)

best_data_model_pipeline: Pipeline = grid_search.best_estimator_

y_pred = best_data_model_pipeline.predict(x_test)
metric_result = precision_score(y_test, y_pred)
print(f"evaluation metric: {metric_result}")

BASELINE_SCORE = 0.57

if metric_result > BASELINE_SCORE:
    print("Model validation passed")
else:
    print(f"Model validation failed: score {metric_result} below baseline {BASELINE_SCORE}")
    raise ValueError()

dump(
    best_data_model_pipeline,
    r"models/first_basic_model.joblib",
)
