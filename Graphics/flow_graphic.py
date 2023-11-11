import sys
import warnings
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

DATA_PATH = "data"

'''
# basic date features
def generate_date_features(df):
    df["YEAR"] = df["FECHAPEDIDO"].dt.year
    df["MONTH"] = df["FECHAPEDIDO"].dt.month
    return df

#import dataframe
df = pd.read_excel(DATA_PATH + "/consumo_material_clean.xlsx")

# FECHAPEDIDO to datetime in day/month/year format
df["FECHAPEDIDO"] = pd.to_datetime(df["FECHAPEDIDO"], dayfirst=True)

#Select useful columns
df = generate_date_features(df)
df = df[["CODIGO", "YEAR", "MONTH", "CANTIDADCOMPRA"]]

#Select dates 2023
df = df.loc[df["YEAR"] == 2023]

#Create dataframe with xi values
df_xi_velocity = df.groupby(["CODIGO","MONTH"]).CANTIDADCOMPRA.sum().reset_index()
df_xi_velocity = df_xi_velocity.sort_values(["CODIGO", "MONTH"], ascending=[True, True])

#Compute velocity of consumption for each product at each time
df_xi_velocity["VELOCITY"] = 0
for i in range(0, len(df_xi_velocity)):
    if((i+1 < len(df_xi_velocity)) and (df_xi_velocity.iloc[i,0] == df_xi_velocity.iloc[i+1,0])): 
        df_xi_velocity.iloc[i,3] = df_xi_velocity.iloc[i,2]/(df_xi_velocity.iloc[i+1,1]-df_xi_velocity.iloc[i,1])
    else:
        df_xi_velocity.iloc[i,3] = df_xi_velocity.iloc[i,2]/(13-df_xi_velocity.iloc[i,1])

#Print results
#print(df_xi_velocity.head())
'''








# Importing the pygame module
import pygame
import numpy as np
from pygame.locals import *
 
# Import randint method random module
from random import randint
 
# Initiate pygame and give permission
# to use pygame's functionality
pygame.init()
 
# Create a display surface object
# of specific dimension
anchura = 800
altura = 600
window = pygame.display.set_mode((anchura, altura))
 
# Creating a new clock object to
# track the amount of time
clock = pygame.time.Clock()
 
 
# Creating a boolean variable that
# we will use to run the while loop
run = True
 
pos_hosp = np.linspace(100,500,5)
pos_distrib = np.linspace(100,500,3)
colors_distrib = [(252,169,133), (251,182,209), (191,228,118)]
colors_lines = [(253,202,162), (253,222,238), (224,243,176)]

# Creating an infinite loop
# to run our game
while run:
    
    # Watch for keyboard and mouse events.
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    # Setting the framerate to 60fps
    clock.tick(60)

    

    for i in range(0,5):
        pygame.draw.line(window, (204,236,239), (700, pos_hosp[i]), (400,300), width = 15)
        pygame.draw.circle(window, (111,183,214), (700, pos_hosp[i]), 25)
    
    for i in range(0,3):
        pygame.draw.line(window, colors_lines[i], (100, pos_distrib[i]), (400,300), width = 15)
        pygame.draw.circle(window, colors_distrib[i], (100, pos_distrib[i]), 40)
 
    pygame.draw.circle(window, (165,137,193), (400,300), 60)
    
 
    # Updating the display surface
    pygame.display.update()
 
    # Filling the window with white color
    window.fill((255, 255, 255))