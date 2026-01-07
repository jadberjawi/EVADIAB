from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


def build_logreg_pipeline(continuous_cols: list, categorical_cols: list, params: dict) -> Pipeline:
    # In your case, categorical are already numeric (0/1/2), so we treat them as numeric.
    # If later you want one-hot for Smoke, we can change it.
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

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])
    return pipe
