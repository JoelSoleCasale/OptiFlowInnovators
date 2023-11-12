import sys
import warnings
import re

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    d2_tweedie_score,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from category_encoders import LeaveOneOutEncoder, TargetEncoder

from tqdm import tqdm

from cross_validate import train_model_cv
from preprocessing import preprocessing
from feature_engineering import generate_date_features, add_timeseries_features
from dataloaders import generate_train_test_df
from utils import get_product_price_dict, smape_score


def train_model_eval(
    X_train, y_train, X_test, y_test, product, product_num_to_price_per_unit
):
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

    # Cross-validation gives boltzmann weights close to 0.5,
    # we change it to the mean to avoid overfitting
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


def single_product_train(df, product, columns, product_num_to_price_per_unit):
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
        return -np.inf, 0, 0, 0

    train, X_train, y_train, test, X_test, y_test = generate_train_test_df(partial_df)
    (
        y_test_preds,
        test_loss,
        mape_error,
        smape,
        tweedie,
        mae_expenses,
    ) = train_model_eval(
        X_train, y_train, X_test, y_test, product, product_num_to_price_per_unit
    )

    return tweedie, test_loss, smape, mae_expenses


def train_boltzmann_ensemble(data_path):
    df = pd.read_excel(data_path + "/consumo_material_clean.xlsx")
    df = preprocessing(df)

    product_num_to_price_per_unit = get_product_price_dict(df)

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
        tweedie, test_loss, smape, mae_expenses = single_product_train(
            df, product, columns, product_num_to_price_per_unit
        )
        if tweedie == -np.inf:
            continue
        prod_df = pd.DataFrame(
            [[product, tweedie, test_loss, smape, mae_expenses]],
            columns=["PRODUCT", "Tweedie", "MSE", "SMAPE", "EXPENSE_ERROR"],
        )
        product_losses = pd.concat([product_losses, prod_df])

    mean_smape = product_losses["SMAPE"].mean()
    mean_mse = product_losses["MSE"].mean()
    mean_tweedie = product_losses[product_losses["Tweedie"] != -np.inf][
        "Tweedie"
    ].mean()
    expense_mape = product_losses["EXPENSE_ERROR"].mean()

    print(f"{mean_smape=}")
    print(f"{mean_tweedie=}")
    print(f"{expense_mape=}")
