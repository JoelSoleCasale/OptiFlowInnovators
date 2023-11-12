import numpy as np
import pandas as pd


# dictionary CODIGO_NUM to PRECIO
def get_product_price_dict(df):
    return (
        df.groupby("CODIGO_NUM")["PRECIO"].max()
        / df.groupby("CODIGO_NUM")["UNIDADESCONSUMOCONTENIDAS"].max()
    ).to_dict()


def smape_score(A, F):
    return 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
