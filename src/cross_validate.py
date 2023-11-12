from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


def train_model_cv(X_train, y_train, X_test, y_test):
    # GroupKFold
    group_kfold = GroupKFold(n_splits=8)
    groups = X_train["YEAR"]

    model = XGBRegressor(random_state=42, n_estimators=500)

    val_losses = []
    test_losses = []

    for idx, (train_index, test_index) in enumerate(
        group_kfold.split(X_train, y_train, groups)
    ):
        X_train_group, X_val_group = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_group, y_val_group = y_train.iloc[train_index], y_train.iloc[test_index]

        model.fit(X_train_group, y_train_group)

        y_val_pred = model.predict(X_val_group)
        val_loss = mean_squared_error(y_val_group, y_val_pred, squared=False)
        val_losses.append(val_loss)

        y_test_pred = model.predict(X_test)
        test_loss = mean_squared_error(y_test, y_test_pred, squared=False)
        test_losses.append(test_loss)

    return np.mean(val_losses), np.mean(test_losses)
