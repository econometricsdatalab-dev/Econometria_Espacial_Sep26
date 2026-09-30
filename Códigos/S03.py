# -*- coding: utf-8 -*-
"""
Curso Integral de Econometría 
Modulo: Econometría espacial
Tema: Matrices de pesos espaciales
Sesión: 03 
Fecha: 29/09/2026
Docente/Asesor: Alexis Adonai Morales Alberto
"""

# Módulos a utilizar

## Módulos completos

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

## Clases y funciones específicas

from pathlib import Path
from io import StringIO
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from libpysal.weights import Queen, Rook, KNN, W, higher_order, lag_spatial

# Configuración del proyecto 

## Rutas y opciones de accesos 

BASE = Path.cwd()
RUTA_MAPA = BASE / "Mapas" / "Mexico_ent" / "00ent.shp"
RUTA_POBREZA = BASE / "Datos" / "Pobreza_rel_2024.csv"
SALIDA = BASE / "Evidencias" / "Pesos_espaciales"
EPSG_METRICO = 6372
ENTIDAD_EJEMPLO = "01"
MOSTRAR_FIGURAS = True
DPI = 500
UMBRAL_KM = None

# Sistema visual de Econometrics Data Lab 

NARANJA = "#F04A1D"
NARANJA_CLARO = "#F5B07C"
AZUL = "#003057"
GRIS_OSCURO = "#1F2933"
GRIS = "#4D565E"
GRIS_CLARO = "#D9DEE3"
FONDO = "#F7F7F5"
BLANCO = "#FFFFFF"

# Configuración visual de mapas basado en sistema visual

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Bahnschrift", "Noto Sans", "DejaVu Sans"],
    "axes.edgecolor": GRIS_CLARO,
    "axes.labelcolor": GRIS,
    "xtick.color": GRIS,
    "ytick.color": GRIS,
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.65,
    "figure.facecolor": FONDO,
    "axes.facecolor": FONDO,
    "axes.prop_cycle": plt.cycler(color=[NARANJA, AZUL, NARANJA_CLARO]),
    "font.size": 10
})

PALETA_POBREZA = LinearSegmentedColormap.from_list(
    "EDL_pobreza", [BLANCO, NARANJA_CLARO, NARANJA, AZUL]
)

PALETA_PESOS = LinearSegmentedColormap.from_list(
    "EDL_pesos", [FONDO, NARANJA_CLARO, NARANJA, AZUL]
)


# Ciudades principales 

CAPITALES = """CVE_ENT,Ciudad,Longitud,Latitud,GeoNames_ID
01,Aguascalientes,-102.28430,21.88262,4019233
02,Mexicali,-115.45446,32.62781,3996069
03,La Paz,-110.31316,24.14231,4000900
04,Campeche,-90.51676,19.84073,3531732
05,Saltillo,-100.97963,25.42595,3988086
06,Colima,-103.71270,19.24473,4013516
07,Tuxtla Gutiérrez,-93.11578,16.75357,3515001
08,Chihuahua,-106.08889,28.63528,4014338
09,Ciudad de México,-99.12766,19.42847,3530597
10,Victoria de Durango,-104.65756,24.02032,4011743
11,Guanajuato,-101.25910,21.01858,4005270
12,Chilpancingo,-99.50124,17.55228,3530870
13,Pachuca de Soto,-98.73329,20.11697,3522210
14,Guadalajara,-103.34749,20.67738,4005539
15,Toluca,-99.65324,19.28786,3515302
16,Morelia,-101.18443,19.70078,3995402
17,Cuernavaca,-99.23075,18.92610,3529947
18,Tepic,-104.89332,21.50733,3981941
19,Monterrey,-100.31721,25.68435,3995465
20,Oaxaca de Juárez,-96.72544,17.06025,3522507
21,Puebla,-98.20723,19.04778,3521081
22,Santiago de Querétaro,-100.38806,20.58806,3991164
23,Chetumal,-88.30397,18.51957,3531023
24,San Luis Potosí,-100.97135,22.15234,3985606
25,Culiacán,-107.39421,24.80209,4012176
26,Hermosillo,-110.96677,29.08874,4004898
27,Villahermosa,-92.93928,17.98625,3514670
28,Ciudad Victoria,-99.14364,23.74061,3530580
29,Tlaxcala,-98.23846,19.31777,3815415
30,Xalapa de Enríquez,-96.91589,19.53124,3526617
31,Mérida,-89.62318,20.96700,3523349
32,Zacatecas,-102.58141,22.76843,3979844
"""

# Carpetas para guardar las evidencias del proceso 

SALIDA.mkdir(parents = True, exist_ok = True)
(SALIDA / "Matrices").mkdir(exist_ok = True)
(SALIDA / "Figuras").mkdir(exist_ok = True)
figuras = SALIDA / "Figuras"

## Compatibilidad con rutas del ejercicio anterior 

ruta_mapa = RUTA_MAPA
ruta_pobreza = RUTA_POBREZA
salida = SALIDA

if not ruta_mapa.exists() and ruta_mapa.name == "ent00.shp":
    anterior = ruta_mapa.with_name("00ent.shp")
    if anterior.exists():
        ruta_mapa = anterior
if not ruta_pobreza.exists() and (BASE / ruta_pobreza.name).exists():
    ruta_pobreza = BASE / ruta_pobreza.name
if not ruta_mapa.exists() or not ruta_pobreza.exists():
    raise FileNotFoundError(
        f"Revisa las rutas de entrada:\nMapa: {ruta_mapa}\nPobreza: {ruta_pobreza}"
    )

# Lectura del mapa 

mapa = gpd.read_file(ruta_mapa)
mapa
serie = mapa["CVE_ENT"]

## Reconfiguración de proyección cartográfica 

mapa = mapa.to_crs(epsg = EPSG_METRICO) 

# Lectura de datos 

pobreza = pd.read_csv(
    ruta_pobreza,
    encoding="utf-8-sig",
    dtype = {"CVE_ENT": str}
    )

# Limpieza general de los datos 

pobreza.columns
pobreza.columns = pobreza.columns.str.strip()

# Unión de datos de pobreza en el mapa 

Mex = mapa.merge(
    pobreza[["CVE_ENT", "Pobreza"]],
    on = 'CVE_ENT',
    how = "left",
    validate="one_to_one"
    )

# Convertir el texto de ciudades en un geopandas 

ciudades = pd.read_csv(StringIO(CAPITALES), dtype = {"CVE_ENT": str})
ciudades.columns = ciudades.columns.str.strip()
ciudades = gpd.GeoDataFrame(
    ciudades,
    geometry= gpd.points_from_xy(ciudades.Longitud, ciudades.Latitud),
    crs = "EPSG:4326"
    ).to_crs(Mex.crs)
ciudades

# Obtener elementos clave 

ids = Mex.CVE_ENT.tolist()
y = Mex.Pobreza.to_numpy(dtype = float)

# Centroides geométricos y ciudades 

centroides = Mex.geometry.centroid

xy_centroides = np.column_stack([centroides.x, centroides.y])
xy_ciudades = np.column_stack([ciudades.geometry.x,
                               ciudades.geometry.y])


# Matrices de contiguedad de primer orden 

## Reina: contacto en un bordo o vértice 

w_reina = Queen.from_dataframe(Mex,
                               ids = ids,
                               silence_warnings=True)

w_reina.id_order = ids

B_reina = w_reina.full()[0]

W_reina_binaria = pd.DataFrame(
    B_reina.astype(int),
    index = ids,
    columns = ids
    )

## Torre: Contacto por un borde 

w_torre = Rook.from_dataframe(Mex,
                               ids = ids,
                               silence_warnings=True)

w_torre.id_order = ids

B_torre = w_torre.full()[0]

W_torre_binaria = pd.DataFrame(
    B_torre.astype(int),
    index = ids,
    columns = ids
    )

## Alfil: Contacto exclusivamente con vértices 

if np.any(B_torre > B_reina):
    raise ValueError("La vecindad torre debe estar contenida en la vecindad reina.")
    
B_alfil = ((B_reina > 0) & (B_torre == 0)).astype(float)

W_alfil_binaria = pd.DataFrame(B_alfil.astype(int), index = ids, columns = ids)


# Gráficos de contiguedades 

figuras = salida / "Figuras"
carpeta_figuras = figuras


if ENTIDAD_EJEMPLO not in ids:
    raise ValueError("ENTIDAD_EJEMPLO debe ser una clave de dos dígitos, de 01 a 32.")
foco = ids.index(ENTIDAD_EJEMPLO)

from shapely.geometry import box

malla = gpd.GeoDataFrame(
    {"clave": [str(i) for i in range(9)]},
    geometry=[box(x, y, x + 1, y + 1) for y in range(3) for x in range(3)]
)
reina = Queen.from_dataframe(malla, ids=malla.clave.tolist(), silence_warnings=True).full()[0]
torre = Rook.from_dataframe(malla, ids=malla.clave.tolist(), silence_warnings=True).full()[0]
matrices = [reina, torre, reina - torre]
fig, ejes = plt.subplots(1, 3, figsize=(14, 6), dpi = DPI)
for ax, matriz, nombre in zip(ejes, matrices, ["Reina: borde o vértice", "Torre: borde", "Alfil: solo vértice"]):
    malla.plot(ax=ax, color=GRIS_CLARO, edgecolor=BLANCO, linewidth=3)
    malla.iloc[np.flatnonzero(matriz[4])].plot(ax=ax, color=NARANJA_CLARO, edgecolor=BLANCO, linewidth=3)
    malla.iloc[[4]].plot(ax=ax, color=AZUL, edgecolor=BLANCO, linewidth=3)
    ax.set_title(nombre, fontsize=12, color=AZUL, weight="bold")
    ax.set_axis_off()
titulo = 'Contigüidad: reina, torre y alfil'
subtitulo = 'Ejemplo didáctico: vecinos de la celda central en una malla regular'
pie = 'Reina: 8 vecinos; torre: 4; alfil: 4.\nEn polígonos irregulares, alfil significa contacto exclusivamente puntual: reina menos torre.'

fig.subplots_adjust(left=0.055, right=0.96, top=0.80, bottom=0.15, wspace=0.20)
fig.text(0.055, 0.955, "ECONOMETRICS DATA LAB", color=NARANJA,
         fontsize=9, weight="bold", ha="left")
fig.text(0.055, 0.905, titulo, fontsize=20, weight="bold", color=AZUL, ha="left")
fig.text(0.055, 0.855, subtitulo, fontsize=10.5, color=GRIS, ha="left")
fig.text(0.055, 0.075, pie, fontsize=8, color=GRIS, ha="left", va="top")
nombre = '01_ejemplo_contiguidad'

fig.savefig(Path(carpeta_figuras) / f"{nombre}.png", dpi=DPI, facecolor=FONDO)
if MOSTRAR_FIGURAS:
    plt.show()
plt.close(fig)
matrices = [B_reina, B_torre, B_alfil]
nombres = ['Reina', 'Torre', 'Alfil']
etiquetas = ids
titulo = 'Matrices de contigüidad'
subtitulo = 'Pesos binarios: 1 = vecino; 0 = sin relación'
archivo = '02_matrices_contiguidad'
carpeta_figuras = figuras

fig, ejes = plt.subplots(1, len(matrices), figsize=(6 * len(matrices), 7), dpi = DPI)
for ax, matriz, nombre in zip(np.atleast_1d(ejes), matrices, nombres):
    sns.heatmap(
        matriz, ax=ax, cmap=PALETA_PESOS, vmin=0,
        vmax=max(float(np.max(matriz)), 1e-12), square=True,
        xticklabels=etiquetas, yticklabels=etiquetas,
        cbar_kws={"shrink": 0.65}, linewidths=0.15, linecolor=BLANCO
    )
    ax.set_title(nombre, fontsize=12, weight="bold", color=AZUL, pad=12)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_xlabel("Entidad vecina j")
    ax.set_ylabel("Entidad de origen i")
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)
pie = 'El color muestra el peso de cada relación; la diagonal es cero.\nLas claves de los ejes se identifican en catalogo_entidades.csv.'

fig.subplots_adjust(left=0.055, right=0.96, top=0.80, bottom=0.15, wspace=0.20)
fig.text(0.055, 0.955, "ECONOMETRICS DATA LAB", color=NARANJA,
         fontsize=9, weight="bold", ha="left")
fig.text(0.055, 0.905, titulo, fontsize=20, weight="bold", color=AZUL, ha="left")
fig.text(0.055, 0.855, subtitulo, fontsize=10.5, color=GRIS, ha="left")
fig.text(0.055, 0.075, pie, fontsize=8, color=GRIS, ha="left", va="top")
nombre = archivo

fig.savefig(Path(carpeta_figuras) / f"{nombre}.png", dpi=DPI, facecolor=FONDO)
if MOSTRAR_FIGURAS:
    plt.show()
plt.close(fig)



puntos = xy_centroides
matrices = [B_reina, B_torre, B_alfil]
nombres = ['Reina', 'Torre', 'Alfil']
titulo = 'Vecindad por contigüidad'
subtitulo = 'Comparación de las reglas para una entidad de origen'
archivo = '03_mapas_contiguidad'
carpeta_figuras = figuras
dirigidas = False
ponderadas = False

foco = 18

fig, ejes = plt.subplots(1, len(matrices), figsize=(6 * len(matrices), 7))
for ax, matriz, nombre in zip(np.atleast_1d(ejes), matrices, nombres):
    vecinos = np.flatnonzero(matriz[foco] > 0)
    Mex.plot(ax=ax, color=GRIS_CLARO, edgecolor=BLANCO, linewidth=0.45)
    if len(vecinos):
        Mex.iloc[vecinos].plot(ax=ax, color=NARANJA_CLARO, edgecolor=BLANCO, linewidth=0.5)
    Mex.iloc[[foco]].plot(ax=ax, color=AZUL, edgecolor=BLANCO, linewidth=0.6)
    for j in vecinos:
        relativo = matriz[foco, j] / max(matriz[foco].max(), 1e-12)
        ancho = 0.35 + 2.3 * relativo if ponderadas else 1.0
        opacidad = 0.15 + 0.70 * relativo if ponderadas else 0.7
        if dirigidas:
            ax.annotate("", xy=puntos[j], xytext=puntos[foco],
                        arrowprops={"arrowstyle": "->", "color": NARANJA,
                                    "lw": ancho, "alpha": opacidad})
        else:
            ax.plot(puntos[[foco, j], 0], puntos[[foco, j], 1],
                    color=NARANJA, lw=ancho, alpha=opacidad, zorder=3)
        ax.text(*puntos[j], Mex.CVE_ENT.iloc[j], fontsize=8, color=AZUL,
                bbox={"facecolor": FONDO, "alpha": 0.8, "edgecolor": "none", "pad": 1})
    ax.scatter(puntos[:, 0], puntos[:, 1], s=10, color=GRIS, zorder=4)
    ax.scatter(*puntos[foco], s=120, marker="*", color=NARANJA, edgecolor=BLANCO, zorder=5)
    ax.set_title(f"{nombre} | {len(vecinos)} vecinos", fontsize=12, color=AZUL, weight="bold")
    ax.set_axis_off()
pie = f'Entidad de origen: {Mex.NOMGEO.iloc[foco]} ({Mex.CVE_ENT.iloc[foco]}). Azul: origen; naranja claro: vecinos.\nLas líneas representan relaciones entre entidades, no carreteras. En KNN las flechas muestran i → j.'

fig.subplots_adjust(left=0.055, right=0.96, top=0.80, bottom=0.15, wspace=0.20)
fig.text(0.055, 0.955, "ECONOMETRICS DATA LAB", color=NARANJA,
         fontsize=9, weight="bold", ha="left")
fig.text(0.055, 0.905, titulo, fontsize=20, weight="bold", color=AZUL, ha="left")
fig.text(0.055, 0.855, subtitulo, fontsize=10.5, color=GRIS, ha="left")
fig.text(0.055, 0.075, pie, fontsize=8, color=GRIS, ha="left", va="top")
nombre = archivo

fig.savefig(Path(carpeta_figuras) / f"{nombre}.png", dpi=DPI, facecolor=FONDO)
if MOSTRAR_FIGURAS:
    plt.show()
plt.close(fig)