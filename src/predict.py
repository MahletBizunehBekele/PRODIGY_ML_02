import pandas as pd
from sklearn.linear_model import LinearRegression
from .data_utils import impute_missing

def train_full_and_predict(train_df, test_df, features, target):
    X_full = impute_missing(train_df[features])
    y_full = train_df[target]
    model = LinearRegression().fit(X_full, y_full)

    X_test = impute_missing(test_df[features])
    preds = model.predict(X_test)

    submission = pd.DataFrame({
        "Id": test_df["Id"],
        "SalePrice": preds
    })
    return submission
