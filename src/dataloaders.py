import pandas as pd


def generate_train_test_df(full_df):
    # Get train and test sets
    train = full_df[full_df["YEAR"] < 2023]
    X_train = train.drop(columns=["CANTIDADCOMPRA", "FECHAPEDIDO"])
    y_train = train["CANTIDADCOMPRA"]

    test = full_df[full_df["YEAR"] == 2023]
    X_test = test.drop(columns=["CANTIDADCOMPRA", "FECHAPEDIDO"])
    y_test = test["CANTIDADCOMPRA"]

    return train, X_train, y_train, test, X_test, y_test
