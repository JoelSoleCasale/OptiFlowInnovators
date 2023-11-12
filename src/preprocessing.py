import pandas as pd


def preprocessing(df):
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

    return df
