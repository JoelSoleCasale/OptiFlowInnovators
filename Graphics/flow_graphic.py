import sys
import warnings
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

DATA_PATH = "data"
GRAPHICS_PATH = "Graphics"

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
import math
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
 

#load images
hosp_img = pygame.image.load(GRAPHICS_PATH + '/hospital_logo.png')
hosp_img = pygame.transform.scale(hosp_img, (50,50))
storage_img = pygame.image.load(GRAPHICS_PATH + '/warehouse_logo.png')
storage_img = pygame.transform.scale(storage_img, (75,75))
products_imgs = [pygame.image.load(GRAPHICS_PATH + '/cleaning_logo.png'),
                 pygame.image.load(GRAPHICS_PATH + '/vaccine_logo.png'),
                 pygame.image.load(GRAPHICS_PATH + '/bandages_logo.png')]
products_imgs[0] = pygame.transform.scale(products_imgs[0], (60,60))
products_imgs[1] = pygame.transform.scale(products_imgs[1], (60,60))
products_imgs[2] = pygame.transform.scale(products_imgs[2], (60,60))

# Creating a boolean variable that
# we will use to run the while loop
run = True
 
#Constants positon / colors
pos_hosp = np.linspace(100,500,5)
pos_distrib = np.linspace(100,500,3)
colors_distrib = [(252,169,133), (251,182,209), (191,228,118)]
colors_lines = [(253,202,162), (253,222,238), (224,243,176)]

#Variables model
list_compras_x_prod = np.array([[40,50,30,40,50,60,70,80,90,100,110,120],
                       [50,110,100,90,80,70,60,50,40,30,20,10],
                       [70,70,90,100,110,120,10,20,30,40,50,60]])

list_compras_x_prod = list_compras_x_prod*(40/max([sublist[-1] for sublist in list_compras_x_prod]))

list_deltas_x_prod = [[1,1,1,1,1,1,1,1,1,1,1,1],
                      [1,0,1,0,1,0,1,0,1,0,1,0],
                      [1,1,0,1,0,1,0,1,0,1,0,1]]

list_velocity_x_prod = [[5,5,5,5,5,5,5,5,5,5,5,5],
                        [1,1,1,1,1,1,1,1,1,1,1,1],
                        [3,3,3,3,3,3,3,3,3,3,3,3]]

hospitals_ask_for_prod_in_month = [ [[1,1,1,1,1,1,1,1,1,1,1,1], [0,1,0,1,0,1,0,1,0,1,0,1], [1,0,1,0,1,0,1,0,1,0,1,0]],
                                    [[0,0,1,0,1,0,1,0,0,0,0,1], [0,0,0,0,0,0,0,0,0,0,0,0], [1,0,1,0,0,0,0,0,0,0,1,0]],
                                    [[1,0,1,0,1,0,1,0,1,0,1,0], [1,1,1,1,1,1,1,1,1,1,1,1], [0,1,0,1,0,1,0,1,0,1,0,1]],
                                    [[1,1,1,1,1,1,1,1,1,1,1,1], [0,1,0,1,0,1,0,1,0,1,0,1], [1,0,1,0,1,0,1,0,1,0,1,0]],
                                    [[1,1,1,1,1,1,1,1,1,1,1,1], [0,1,0,1,0,1,0,1,0,1,0,1], [1,0,1,0,1,0,1,0,1,0,1,0]]]

#Create time variables
cont = 0
day = 0
month = 0
limit_cont = 50
cont2 = 0

#variable to pause simulation
pause = False

def calcular_pos(origen, desti, cont):
    cont = cont % float(limit_cont)
    d = math.dist(origen, desti)
    ret = origen + (desti-origen)/np.linalg.norm((desti-origen))*d*cont/float(limit_cont)
    return ret

# Creating an infinite loop
# to run our game
while run and month < 12:
    
    # Watch for keyboard and mouse events.
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        elif event.type == pygame.KEYDOWN:
            pause = not(pause)

    # Setting the framerate to 60fps
    clock.tick(60)

    # Update time variables
    if(cont == limit_cont):
        day += 1
        cont = 0
        if(day == 30/3):    #acelerar simulación
            day = 0
            month += 1
            

    #Draw hospitals
    for i in range(0,5):
        cont_intern = 0
        pygame.draw.line(window, (204,236,239), (700, pos_hosp[i]), (400,300), width = 15)
        #Draw products from storage to hospitals
        for j in range(0,3):
            for k in range(0,list_velocity_x_prod[j][month]):
                if(hospitals_ask_for_prod_in_month[i][j][month]):
                    pygame.draw.circle(window, 0.9*np.array(colors_distrib[j]), 
                                        calcular_pos(np.array([400,300]),
                                            np.array([700,pos_hosp[i]]),
                                            (cont+5*cont_intern))
                                        , 5)
                    cont_intern += 1
        pygame.draw.circle(window, (111,183,214), (700, pos_hosp[i]), 25)
        window.blit(hosp_img, (700-25, pos_hosp[i]-25))
    

    #Draw providers
    for i in range(0,3):
        pygame.draw.line(window, colors_lines[i], (100, pos_distrib[i]), (400,300), width = 15)
        #Draw products from providers to storage
        if(day == 0 and list_deltas_x_prod[i][month]):
            pygame.draw.circle(window, 0.9*np.array(colors_distrib[i]), calcular_pos(np.array([100,pos_distrib[i]]),np.array([400,300]),cont), list_compras_x_prod[i][month])
        pygame.draw.circle(window, colors_distrib[i], (100, pos_distrib[i]), 40)
        window.blit(products_imgs[i], (100-30, pos_distrib[i]-30))


    #Draw storage center
    pygame.draw.circle(window, (165,137,193), (400,300), 60)
    window.blit(storage_img, (400-75/2,300-75/2))
 
    # Updating the display surface
    pygame.display.update()
 
    # Filling the window with white color
    window.fill((255, 255, 255))

    if(not pause): cont += 1

