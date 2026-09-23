# -*- coding: utf-8 -*-
"""
Curso Integral de Econometría 
Modulo: Econometría espacial
Tema: Introducción a la econometría espacial
Sesión: 01 
Fecha: 22/09/2026
Docente/Asesor: Alexis Adonai Morales Alberto
"""

# Instalación de modulos 

pip install contextily rioxarray 

# Modulos a cargar 

import contextily
import geopandas as gpd 
import rioxarray as rx 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

# Clases o submodulos de modulos generales 

from shapely.geometry import Polygon

# Revisión de directorio de trabajo

print(os.getcwd())

# os.chdir("E:/Econometrics Data Lab/Curso integral de econometría/Series de tiempo/ST_SEP26")

# Cargar mapa 

Mex = (
       gpd.read_file("Mapas\\Mexico_ent\\00ent.shp")
       .to_crs(epsg = 6372)
       )

Mex

# Plotear mapa

fig, ax = plt.subplots(figsize=(12, 8), dpi=500)
fig.patch.set_facecolor("#F7F7F5")
ax.set_facecolor("#F7F7F5")

Mex.plot(
    ax=ax,
    color="#F5B07C",
    edgecolor="#FFFFFF",
    linewidth=0.7
)

Mex.boundary.plot(
    ax=ax,
    color="#84422D",
    linewidth=0.35
)

# Dejar espacio para los textos
fig.subplots_adjust(top=0.82, bottom=0.13, left=0.04, right=0.96)

fig.text(
    0.06, 0.94,
    "Entidades federativas de México",
    fontsize=20, fontweight="bold",
    color="#242424", ha="left"
)

fig.text(
    0.06, 0.89,
    "División político-administrativa de las 32 entidades federativas",
    fontsize=11,
    color="#666666", ha="left"
)

fig.text(
    0.06, 0.055,
    "Fuente: INEGI. Marco geoestadístico de México, 2024.",
    fontsize=8.5,
    color="#666666", ha="left"
)

ax.set_axis_off()
ax.set_aspect("equal")
plt.show()

