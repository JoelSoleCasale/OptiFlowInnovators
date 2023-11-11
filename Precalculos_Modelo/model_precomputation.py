import sys
import warnings
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

DATA_PATH = "data"


# basic date features
def generate_date_features(df):
    df["YEAR"] = df["FECHAPEDIDO"].dt.year
    df["MONTH"] = df["FECHAPEDIDO"].dt.month
    return df


# import dataframe
df = pd.read_excel(DATA_PATH + "/consumo_material_clean.xlsx")

# FECHAPEDIDO to datetime in day/month/year format
df["FECHAPEDIDO"] = pd.to_datetime(df["FECHAPEDIDO"], dayfirst=True)

# Select useful columns
df = generate_date_features(df)
df = df[["CODIGO", "YEAR", "MONTH", "CANTIDADCOMPRA"]]

# Select dates 2023
df = df.loc[df["YEAR"] == 2023]

# Create dataframe with xi values
df_xi_velocity = df.groupby(["CODIGO", "MONTH"]).CANTIDADCOMPRA.sum().reset_index()

unique_codes = df_xi_velocity["CODIGO"].unique()
for un in unique_codes:
    for m in range(1,13):
        if(df_xi_velocity.loc[(df_xi_velocity["CODIGO"] == un) & (df_xi_velocity["MONTH"] == m)].empty):
            df_xi_velocity.loc[len(df_xi_velocity)] = [un, m, 0]

df_xi_velocity = df_xi_velocity.sort_values(["CODIGO", "MONTH"], ascending=[True, True])
print(df_xi_velocity.head(13))

# Compute velocity of consumption for each product at each time
df_xi_velocity["VELOCITY"] = 0
for i in range(0, len(df_xi_velocity)):
    if (i + 1 < len(df_xi_velocity)) and (
        df_xi_velocity.iloc[i, 0] == df_xi_velocity.iloc[i + 1, 0]
    ):
        df_xi_velocity.iloc[i, 3] = df_xi_velocity.iloc[i, 2] / (
            df_xi_velocity.iloc[i + 1, 1] - df_xi_velocity.iloc[i, 1]
        )
    else:
        df_xi_velocity.iloc[i, 3] = df_xi_velocity.iloc[i, 2] / (
            13 - df_xi_velocity.iloc[i, 1]
        )

unique_codes = df_xi_velocity["CODIGO"].unique()
for un in unique_codes:
    for m in range(1,13):
        if(df_xi_velocity.loc[(df_xi_velocity["CODIGO"] == un) & (df_xi_velocity["MONTH"] == m) & (df_xi_velocity["CANTIDADCOMPRA"] != 0)].empty):
            if(m != 1): 
                df_xi_velocity.loc[(df_xi_velocity["CODIGO"] == un) & (df_xi_velocity["MONTH"] == m)] = [un, m, 0, df_xi_velocity.loc[(df_xi_velocity["CODIGO"] == un) & (df_xi_velocity["MONTH"] == m-1)].iloc[0,3]]
            else:
                df_xi_velocity.loc[(df_xi_velocity["CODIGO"] == un) & (df_xi_velocity["MONTH"] == m)] = [un, m, 0, 0]

df_xi_velocity = df_xi_velocity.sort_values(["CODIGO", "MONTH"], ascending=[True, True])
print(df_xi_velocity.head(25))
