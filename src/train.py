import sys
import warnings
import re

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    d2_tweedie_score,
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder

from category_encoders import LeaveOneOutEncoder, TargetEncoder

from tqdm import tqdm

DATA_PATH = "../data"


# Load excel file
df = pd.read_excel(DATA_PATH + "/consumo_material_clean.xlsx")


# Separate code into two columns
new_columns = df["CODIGO"].str.extract(r"([a-zA-Z]+)([0-9]+)", expand=False)
df["CODIGO_CLASS"] = new_columns[0]
df["CODIGO_NUM"] = new_columns[1]
df.drop(columns=["CODIGO"], inplace=True)


# FECHAPEDIDO to datetime in day/month/year format
df["FECHAPEDIDO"] = pd.to_datetime(df["FECHAPEDIDO"], dayfirst=True)
df.sort_values(by=["FECHAPEDIDO"], inplace=True)
df.reset_index(drop=True, inplace=True)

# separate ORIGEN in three columns by '-'
origin_separated_columns = df["ORIGEN"].str.split("-", expand=True)
df["PURCHASING_HOSPITAL"] = origin_separated_columns[1]
df["PURCHASING_DEPARTMENT"] = origin_separated_columns[2]
df.drop(columns=["ORIGEN"], inplace=True)


# drop duplicates
df.drop_duplicates(inplace=True)

# dictionary CODIGO_NUM to PRECIO
product_num_to_price_per_unit = (
    df.groupby("CODIGO_NUM")["PRECIO"].max()
    / df.groupby("CODIGO_NUM")["UNIDADESCONSUMOCONTENIDAS"].max()
).to_dict()


# basic date features
def generate_date_features(df):
    df["YEAR"] = df["FECHAPEDIDO"].dt.year
    df["MONTH"] = np.sin(2 * np.pi * df["FECHAPEDIDO"].dt.month / 12)
    df["DAYOFMONTH"] = np.sin(2 * np.pi * df["FECHAPEDIDO"].dt.day / 31)
    df["DAYOFYEAR"] = np.sin(2 * np.pi * df["FECHAPEDIDO"].dt.dayofyear / 365)
    return df


def add_timeseries_features(df):
    # MEANS
    df["ROLLING_MEAN_3M"] = df["CANTIDADCOMPRA"].rolling(90).mean()
    df["ROLLING_MEAN_1Y"] = df["CANTIDADCOMPRA"].rolling(365).mean()

    # WEIGHTED MEANS
    df["WEIGHTED_MEAN_3M"] = (
        df["CANTIDADCOMPRA"]
        .rolling(90)
        .apply(lambda x: np.average(x, weights=range(1, len(x) + 1)))
    )
    df["WEIGHTED_MEAN_1Y"] = (
        df["CANTIDADCOMPRA"]
        .rolling(365)
        .apply(lambda x: np.average(x, weights=range(1, len(x) + 1)))
    )

    # EXPONENTIAL WEIGHTED MEANS
    df["EWMA_1W"] = df["CANTIDADCOMPRA"].ewm(span=7).mean()
    df["EWMA_1M"] = df["CANTIDADCOMPRA"].ewm(span=30).mean()
    df["EWMA_3M"] = df["CANTIDADCOMPRA"].ewm(span=90).mean()
    df["EWMA_1Y"] = df["CANTIDADCOMPRA"].ewm(span=365).mean()

    # LAGS
    df["SHIFT_1W"] = df["CANTIDADCOMPRA"].shift(7)
    df["SHIFT_1M"] = df["CANTIDADCOMPRA"].shift(30)
    df["SHIFT_3M"] = df["CANTIDADCOMPRA"].shift(90)
    df["SHIFT_1Y"] = df["CANTIDADCOMPRA"].shift(365)

    # DIFFS
    df["DIFF_1W"] = df["CANTIDADCOMPRA"].diff(7)
    df["DIFF_1M"] = df["CANTIDADCOMPRA"].diff(30)
    df["DIFF_3M"] = df["CANTIDADCOMPRA"].diff(90)
    df["DIFF_1Y"] = df["CANTIDADCOMPRA"].diff(365)

    return df


def generate_train_test_df(full_df):
    # Get train and test sets
    train = full_df[full_df["YEAR"] < 2023]
    X_train = train.drop(columns=["CANTIDADCOMPRA", "FECHAPEDIDO"])
    y_train = train["CANTIDADCOMPRA"]

    test = full_df[full_df["YEAR"] == 2023]
    X_test = test.drop(columns=["CANTIDADCOMPRA", "FECHAPEDIDO"])
    y_test = test["CANTIDADCOMPRA"]

    return train, X_train, y_train, test, X_test, y_test


def smape_score(A, F):
    return 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


def train_model_eval(X_train, y_train, X_test, y_test, product):
    model_list = [
        LGBMRegressor(random_state=42, n_estimators=1000, verbose=-1),
        XGBRegressor(random_state=42, n_estimators=1000),
    ]
    preds = []

    for model in model_list:
        model.fit(X_train, y_train)
        y_test_par_pred = model.predict(X_test)
        y_test_par_pred = np.maximum(y_test_par_pred, 0)
        preds.append(y_test_par_pred)

    y_test_pred = np.mean(preds, axis=0)

    # LOSSES
    test_loss = mean_squared_error(y_test, y_test_pred, squared=False)
    mape_error = mean_absolute_percentage_error(y_test, y_test_pred)
    tweedie = d2_tweedie_score(y_test, y_test_pred)
    smape_err = smape_score(y_test, y_test_pred)

    forecasted_expenses = y_test_pred.sum() * product_num_to_price_per_unit[product]
    real_expenses = y_test.sum() * product_num_to_price_per_unit[product]
    mape_expenses = np.abs(forecasted_expenses - real_expenses) / real_expenses

    return_test_preds = pd.concat([X_test, y_test], axis=1)
    return return_test_preds, test_loss, mape_error, smape_err, tweedie, mape_expenses


if __name__ == "__main__":
    columns = [
        "FECHAPEDIDO",
        "CANTIDADCOMPRA",
        "PURCHASING_HOSPITAL",
        "PURCHASING_DEPARTMENT",
    ]

    product_losses = pd.DataFrame(
        columns=["PRODUCT", "Tweedie", "MSE", "SMAPE", "EXPENSE_ERROR"]
    )
    for product in tqdm(df["CODIGO_NUM"].unique()):
        partial_df = df[df["CODIGO_NUM"] == product]
        partial_df = partial_df.groupby(columns).sum().reset_index()

        loo = LeaveOneOutEncoder()
        partial_df["PURCHASING_HOSPITAL"] = loo.fit_transform(
            partial_df["PURCHASING_HOSPITAL"], partial_df["CANTIDADCOMPRA"]
        )

        loo = TargetEncoder()
        partial_df["PURCHASING_DEPARTMENT"] = loo.fit_transform(
            partial_df["PURCHASING_DEPARTMENT"], partial_df["CANTIDADCOMPRA"]
        )

        partial_df = partial_df[columns]
        partial_df = generate_date_features(partial_df)
        partial_df = add_timeseries_features(partial_df)

        is_2023_in_df = 2023 in partial_df["YEAR"].unique()
        product_blacklist = [
            "85758",
            "73753",
            "65007",
            "66071",
            "64544",
        ]  # stops selling in 2023, treated separately
        if not is_2023_in_df or product in product_blacklist:
            continue

        train, X_train, y_train, test, X_test, y_test = generate_train_test_df(
            partial_df
        )
        (
            y_test_preds,
            test_loss,
            mape_error,
            smape,
            tweedie,
            mae_expenses,
        ) = train_model_eval(X_train, y_train, X_test, y_test, product)

        product_losses = pd.concat(
            [
                product_losses,
                pd.DataFrame(
                    [[product, tweedie, test_loss, smape, mae_expenses]],
                    columns=["PRODUCT", "Tweedie", "MSE", "SMAPE", "EXPENSE_ERROR"],
                ),
            ]
        )

    mean_smape = product_losses["SMAPE"].mean()
    mean_mse = product_losses["MSE"].mean()
    mean_tweedie = product_losses[product_losses["Tweedie"] != -np.inf][
        "Tweedie"
    ].mean()
    expense_mape = product_losses["EXPENSE_ERROR"].mean()

    print(f"{mean_smape=}")
    print(f"{mean_tweedie=}")
    print(f"{expense_mape=}")
