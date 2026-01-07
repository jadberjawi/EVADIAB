from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def build_logreg_pipeline(continuous_cols: list, categorical_cols: list, params: dict) -> Pipeline:
    num_cols = list(continuous_cols) + list(categorical_cols)

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipe, num_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = LogisticRegression(**params)

    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])


def build_rf_pipeline(continuous_cols: list, categorical_cols: list, params: dict) -> Pipeline:
    num_cols = list(continuous_cols) + list(categorical_cols)

    # RF does NOT need scaling; only imputation
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipe, num_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = RandomForestClassifier(**params)

    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])


def build_xgb_pipeline(continuous_cols: list, categorical_cols: list, params: dict) -> Pipeline:
    num_cols = list(continuous_cols) + list(categorical_cols)

    # XGBoost needs imputation but not scaling
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipe, num_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = XGBClassifier(**params)

    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])


def build_model_pipeline(model_type: str, continuous_cols: list, categorical_cols: list, params: dict) -> Pipeline:
    if model_type == "logistic_regression":
        return build_logreg_pipeline(continuous_cols, categorical_cols, params)
    if model_type == "random_forest":
        return build_rf_pipeline(continuous_cols, categorical_cols, params)
    if model_type == "xgboost":
        return build_xgb_pipeline(continuous_cols, categorical_cols, params)
    raise ValueError(f"Unknown model type: {model_type}")

