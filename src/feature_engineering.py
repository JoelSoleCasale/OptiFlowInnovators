import numpy as np


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
