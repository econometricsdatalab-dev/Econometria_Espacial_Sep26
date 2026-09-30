# -*- coding: utf-8 -*-
"""
Curso Integral de Econometría 
Modulo: Econometría espacial
Tema: Introducción a la econometría espacial
Sesión: 02 
Fecha: 24/09/2026
Docente/Asesor: Alexis Adonai Morales Alberto
"""

# Instalación de libpysal

pip install libpysal

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
from matplotlib.patches import Patch
from libpysal.weights import Queen

# 1) Cargar datos de pobreza 

pobreza = pd.read_csv(
    "Datos\\Pobreza_rel_2024.csv",
    encoding="utf-8-sig",
    dtype = {"CVE_ENT": str}
    )

# 2) Limpieza general de los datos 

pobreza.columns
pobreza.columns = pobreza.columns.str.strip() # Quita espacios en los nombres de las columnas

## Garantizar que el formato de CVE_ENT sea de dos dígitos

pobreza["CVE_ENT"] = pobreza["CVE_ENT"].str.zfill(2)

## Convertir variable "pobreza en numérico" 

pobreza["Pobreza"] = pd.to_numeric(pobreza["Pobreza"], errors = "raise")

pobreza.dtypes

# 3) Construcción de quintiles para la represtación visual (mapa coroplético)

pobreza["Quintil"] = pd.qcut(
    pobreza["Pobreza"],
    q = 5,
    labels = False
    ) + 1

# 4) Cargar mapa 

Mex = (
       gpd.read_file("Mapas\\Mexico_ent\\00ent.shp")
       .to_crs(epsg = 6372)
       )

Mex.dtypes

Mex["CVE_ENT"] = Mex["CVE_ENT"].str.zfill(2)

# 5) Unión de los datos en el mapa 

Mex = Mex.merge(
    pobreza[["CVE_ENT", "Pobreza", "Quintil"]],
    on = 'CVE_ENT',
    how = "left",
    validate="one_to_one"
    )


# 6) Construcción del mapa coroplético

## a) Definición de paletas de colores (naranja)

colores = {
    1: "#FBE4D4",
    2: "#F5B07C",
    3: "#F79A5A",
    4: "#F46F2C",
    5: "#C93612"
    }

fondo = "#FAF9F7"
texto = "#202020"
secundario = "#666666"

## b) Programación del mapa con matplotlib

fig, ax = plt.subplots(figsize = (13,9), dpi = 300)
fig.patch.set_facecolor(fondo)
ax.set_facecolor(fondo)

Mex.plot(
    ax = ax,
    color = Mex["Quintil"].map(colores),
    edgecolor = "#FFFFFF",
    linewidth = 0.85)

Mex.boundary.plot(
    ax = ax,
    color = "#8C513D",
    linewidth = 0.25)

ax.set_axis_off()
fig.subplots_adjust(left = 0.04,
                    right = 0.96,
                    top = 0.81,
                    bottom = 0.26) 

fig.text(
    0.06, 0.94,
    "Distribución de la pobreza en México por entidad federativa",
    ha = 'left', va = 'top',
    fontsize = 22, fontweight = "bold", color = texto)

fig.text(
    0.06, 0.885,
    "Porcentaje de población en situación de pobreza\n2024\n(Por quintiles)",
    ha = "left", va = "top",
    fontsize = 11.5, color = secundario)

etiquetas = []
for q in range(1,6):
    valores = pobreza.loc[pobreza["Quintil"] == q, "Pobreza"]
    nivel = "Menor" if q == 1 else "Mayor" if q == 5 else ""
    etiquetas.append(
        f"Q{q} {nivel}\n{valores.min():.1f}-{valores.max():.1f} %"
        )

elementos = [
    Patch(facecolor = colores[q], edgecolor="none", label = etiquetas[q-1])
    for q in range(1,6)
    ]

fig.legend(
    handles = elementos,
    loc = "lower center",
    bbox_to_anchor = (0.5, 0.115),
    ncol = 5,
    frameon = False,
    fontsize = 9.5,
    handlelength = 1.7,
    handleheight = 1.3,
    columnspacing = 2.1
    )

fig.text(
    0.06,
    0.055,
    "Fuente. INEGI. Medición de la Pobreza, 2024."
    "Los rangos indican el mínimo y el máximo observado en cada quintil",
    ha = "left",
    va = "bottom",
    fontsize = 8.5,
    color = secundario
    )

fig.savefig(
    "Evidencias\\mapa_pobreza_quintiles_2024.png",
    dpi = 500,
    facecolor = fig.get_facecolor(),
    bbox_inches = "tight"
    )

plt.show()


# Construcción de la matriz de peso espacial

## Método: Contigüedad - Reina 

w = Queen.from_dataframe(
    Mex,
    ids = "CVE_ENT",
    use_index = False
    )

### Matriz binaria 

w.transform = "B"
valores, claves = w.full()
W_binaria = pd.DataFrame(
    valores.astype(int),
    index = claves,
    columns = claves
    )