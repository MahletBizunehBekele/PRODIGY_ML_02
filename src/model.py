from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
import numpy as np
import pandas as pd


def train_validate(X, y, test_size=0.2, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    linreg = LinearRegression()
    linreg.fit(X_train, y_train)

    y_val_pred = linreg.predict(X_val)
    rmse = np.sqrt(root_mean_squared_error(y_val, y_val_pred))
    r2 = r2_score(y_val, y_val_pred)

    return linreg, rmse, r2, (X_train, X_val, y_train, y_val, y_val_pred)
