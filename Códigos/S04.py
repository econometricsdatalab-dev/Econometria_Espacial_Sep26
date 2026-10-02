# -*- coding: utf-8 -*-
"""
Curso Integral de Econometría 
Modulo: Econometría espacial
Tema: Matrices de pesos espaciales pt2
Sesión: 04 
Fecha: 01/10/2026
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
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
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

# Todas las relaciones de contigüidad: una figura por criterio.
# En matrices simétricas basta tomar el triángulo superior: no se dibuja
# dos veces la misma relación i-j. El alfil puede no tener vínculos.

contiguidades = [
    ("Reina", B_reina, "03_reina_todas_las_relaciones"),
    ("Torre", B_torre, "04_torre_todas_las_relaciones"),
    ("Alfil", B_alfil, "05_alfil_todas_las_relaciones"),
]

for metodo, matriz, archivo in contiguidades:
    origenes, destinos = np.where(np.triu(matriz > 0, k=1))
    tramos = np.stack(
        [xy_centroides[origenes], xy_centroides[destinos]], axis=1
    ) if len(origenes) else np.empty((0, 2, 2))

    fig, ax = plt.subplots(figsize=(12, 8), dpi=DPI)
    Mex.plot(ax=ax, color=GRIS_CLARO, edgecolor=BLANCO, linewidth=0.7)
    if len(tramos):
        ax.add_collection(LineCollection(
            tramos, colors=NARANJA, linewidths=1.15, alpha=0.75, zorder=3
        ))
    ax.scatter(xy_centroides[:, 0], xy_centroides[:, 1],
               s=18, color=AZUL, edgecolor=BLANCO, linewidth=0.4, zorder=4)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.subplots_adjust(left=0.035, right=0.965, top=0.81, bottom=0.16)
    fig.text(0.055, 0.955, "ECONOMETRICS DATA LAB",
             color=NARANJA, fontsize=9, weight="bold")
    fig.text(0.055, 0.905, f"Contigüidad tipo {metodo.lower()}",
             fontsize=20, weight="bold", color=AZUL)
    fig.text(0.055, 0.855,
             f"Red completa · {len(origenes)} relaciones entre entidades",
             fontsize=10.5, color=GRIS)
    fig.text(0.055, 0.075,
             "Cada línea une dos entidades vecinas y aparece una sola vez. "
             "Los puntos representan centroides geométricos.\n"
             "Alfil = contacto solo en vértice (reina menos torre); "
             "la red depende de la precisión del shapefile.",
             fontsize=8, color=GRIS, va="top")
    fig.savefig(figuras / f"{archivo}.png", dpi=DPI, facecolor=FONDO)
    if MOSTRAR_FIGURAS:
        plt.show()
    plt.close(fig)


# Matrices de pesos espaciales mediante k-vecinos -----

# NOTA: k vecinos más cercanos: 2, 3 y 4 desde los centroides y desde ciudades
# la relación i - j puede no ser recíproca. Se conserva las flechas. 

for sitio, puntos in (("centroides", xy_centroides),
                      ("ciudades", xy_ciudades)):
    for k in (2,3,4):
        w_knn = KNN.from_array(puntos, k=k, ids=ids,
                               silence_warnings= True)
        w_knn.id_order = ids
        matriz_knn = w_knn.full()[0].astype(int)
        matriz_knn_df = pd.DataFrame(matriz_knn, index = ids, columns = ids)
        matriz_knn_df.to_csv(
            SALIDA / "Matrices" / f"W_knn_{sitio}_{k}_binaria.csv",
            encoding= 'utf-8-sig')
        print(f"{sitio}, k = {k}: {int(matriz_knn.sum())} relaciones dirigidas")
        
        fig, ax = plt.subplots(figsize=(12, 8), dpi=DPI)
        Mex.plot(ax=ax, color=GRIS_CLARO, edgecolor=BLANCO, linewidth=0.7)
        origenes, destinos = np.where(matriz_knn > 0)
        for i, j in zip(origenes, destinos):
            flecha = FancyArrowPatch(
                puntos[i], puntos[j], arrowstyle="-|>",
                mutation_scale=7, linewidth=0.7, color=NARANJA,
                alpha=0.54, shrinkA=3, shrinkB=4, zorder=3
            )
            ax.add_patch(flecha)
        ax.scatter(puntos[:, 0], puntos[:, 1],
                   s=18, color=AZUL, edgecolor=BLANCO,
                   linewidth=0.4, zorder=4)
        ax.set_aspect("equal")
        ax.set_axis_off()
        fig.subplots_adjust(left=0.035, right=0.965, top=0.81, bottom=0.16)
        fig.text(0.055, 0.955, "ECONOMETRICS DATA LAB",
                 color=NARANJA, fontsize=9, weight="bold")
        fig.text(0.055, 0.905, f"Vecindad por {k} vecinos cercanos",
                 fontsize=20, weight="bold", color=AZUL)
        fig.text(0.055, 0.855,
                 f"{sitio.capitalize()} · {len(origenes)} relaciones dirigidas",
                 fontsize=10.5, color=GRIS)
        fig.text(0.055, 0.075,
                 "Cada flecha parte de una entidad hacia uno de sus k vecinos "
                 "más cercanos. Una conexión puede aparecer en ambos sentidos.\n"
                 "Distancias euclidianas en la proyección EPSG:6372.",
                 fontsize=8, color=GRIS, va="top")
        fig.savefig(
            figuras / f"06_knn_{sitio}_{k}_todas_las_relaciones.png",
            dpi=DPI, facecolor=FONDO
        )
        if MOSTRAR_FIGURAS:
            plt.show()
        plt.close(fig)

# Método de distancia inversa 

for sitio, puntos in (("centroides", xy_centroides),
                      ("ciudades", xy_ciudades)):
    distancias_km = cdist(puntos, puntos, metric="euclidean") / 1000
    np.fill_diagonal(distancias_km, np.inf)
    if np.any(distancias_km[np.isfinite(distancias_km)] == 0):
        raise ValueError(f"Hay dos {sitio} con coordenadas idénticas.")

    pesos_inversos = 1 / distancias_km
    np.fill_diagonal(pesos_inversos, 0)
    pesos_df = pd.DataFrame(pesos_inversos, index=ids, columns=ids)
    pesos_df.to_csv(SALIDA / "Matrices" / f"W_inversa_{sitio}.csv",
                    encoding="utf-8-sig")

    
    origenes, destinos = np.triu_indices(len(ids), k=1)
    intensidades = pesos_inversos[origenes, destinos]
    maximo = intensidades.max()
    orden_dibujo = np.argsort(intensidades)
    origenes, destinos = origenes[orden_dibujo], destinos[orden_dibujo]
    relativos = intensidades[orden_dibujo] / maximo
    tramos = np.stack([puntos[origenes], puntos[destinos]], axis=1)
    rgba = np.tile(to_rgba(NARANJA), (len(tramos), 1))
    rgba[:, 3] = 0.035 + 0.75 * np.sqrt(relativos)
    anchos = 0.15 + 2.15 * np.sqrt(relativos)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=DPI)
    Mex.plot(ax=ax, color=GRIS_CLARO, edgecolor=BLANCO, linewidth=0.7)
    ax.add_collection(LineCollection(
        tramos, colors=rgba, linewidths=anchos, zorder=3
    ))
    ax.scatter(puntos[:, 0], puntos[:, 1], s=20, color=AZUL,
               edgecolor=BLANCO, linewidth=0.4, zorder=4)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.subplots_adjust(left=0.035, right=0.965, top=0.81, bottom=0.16)
    fig.text(0.055, 0.955, "ECONOMETRICS DATA LAB",
             color=NARANJA, fontsize=9, weight="bold")
    fig.text(0.055, 0.905, f"Distancia inversa · {sitio}",
             fontsize=20, weight="bold", color=AZUL)
    fig.text(0.055, 0.855,
             f"Red completa · {len(tramos)} pares · peso = 1 / distancia (km)",
             fontsize=10.5, color=GRIS)
    fig.text(0.055, 0.075,
             "Se muestran todos los pares de entidades. "
             "Las líneas más marcadas representan un peso espacial mayor.\n"
             "Distancia euclidiana en EPSG:6372; "
             "los puntos son centroides o ciudades según el mapa.",
             fontsize=8, color=GRIS, va="top")
    fig.savefig(figuras / f"07_inversa_{sitio}_todas_las_relaciones.png",
                dpi=DPI, facecolor=FONDO)
    if MOSTRAR_FIGURAS:
        plt.show()
    plt.close(fig)

# Matrices de peso espacial de segundo orden y tercer orden

## Segundo orden: Vecinos de los vecinos 
## Tercer orden: Tercer salto de distancia en la red espacial 

### Método de contiguedad 

ordenes_superiores = [
    ("reina", B_reina),
    ("torre", B_torre),
    ("alfil", B_alfil),
]

matrices_segundo_orden = {}
matrices_tercer_orden = {}

for metodo, matriz_base in ordenes_superiores:

    # Número mínimo de pasos entre cada par de entidades
    
    distancias_pasos = shortest_path(
        csr_matrix(matriz_base),
        directed=False,
        unweighted=True
    )

    # Matrices binarias de orden exacto
    
    B_orden2 = (distancias_pasos == 2).astype(int)
    B_orden3 = (distancias_pasos == 3).astype(int)

    np.fill_diagonal(B_orden2, 0)
    np.fill_diagonal(B_orden3, 0)

    matrices_segundo_orden[metodo] = B_orden2
    matrices_tercer_orden[metodo] = B_orden3

    for orden, matriz_orden in ((2, B_orden2), (3, B_orden3)):

        # Guardar matriz binaria
        
        tabla_binaria = pd.DataFrame(
            matriz_orden,
            index=ids,
            columns=ids
        )

        tabla_binaria.to_csv(
            SALIDA / "Matrices" / f"W_{metodo}_orden{orden}_binaria.csv",
            encoding="utf-8-sig"
        )

        # Estandarizar por filas y guardar
        
        total_fila = matriz_orden.sum(axis=1, keepdims=True)

        matriz_filas = np.divide(
            matriz_orden.astype(float),
            total_fila,
            out=np.zeros_like(matriz_orden, dtype=float),
            where=total_fila > 0
        )

        pd.DataFrame(
            matriz_filas,
            index=ids,
            columns=ids
        ).to_csv(
            SALIDA / "Matrices" / f"W_{metodo}_orden{orden}_filas.csv",
            encoding="utf-8-sig"
        )

        # Dibujar todas las relaciones, una sola vez por par
        
        origenes, destinos = np.where(
            np.triu(matriz_orden > 0, k=1)
        )

        tramos = (
            np.stack(
                [xy_centroides[origenes], xy_centroides[destinos]],
                axis=1
            )
            if len(origenes)
            else np.empty((0, 2, 2))
        )

        fig, ax = plt.subplots(figsize=(12, 8), dpi=DPI)

        Mex.plot(
            ax=ax,
            color=GRIS_CLARO,
            edgecolor=BLANCO,
            linewidth=0.7
        )

        if len(tramos):
            ax.add_collection(
                LineCollection(
                    tramos,
                    colors=NARANJA,
                    linewidths=0.95,
                    alpha=0.6,
                    zorder=3
                )
            )

        ax.scatter(
            xy_centroides[:, 0],
            xy_centroides[:, 1],
            s=18,
            color=AZUL,
            edgecolor=BLANCO,
            linewidth=0.4,
            zorder=4
        )

        ax.set_aspect("equal")
        ax.set_axis_off()

        fig.subplots_adjust(
            left=0.035,
            right=0.965,
            top=0.81,
            bottom=0.16
        )

        fig.text(
            0.055, 0.955,
            "ECONOMETRICS DATA LAB",
            color=NARANJA,
            fontsize=9,
            weight="bold"
        )

        fig.text(
            0.055, 0.905,
            f"Contigüidad {metodo} · orden {orden}",
            fontsize=20,
            weight="bold",
            color=AZUL
        )

        fig.text(
            0.055, 0.855,
            f"Red completa · {len(origenes)} pares a {orden} pasos",
            fontsize=10.5,
            color=GRIS
        )

        fig.text(
            0.055, 0.075,
            f"Dos entidades son vecinas de orden {orden} si el camino "
            f"más corto en la red {metodo} tiene exactamente {orden} enlaces.\n"
            "Cada par aparece una sola vez; una fila sin relaciones "
            "permanece en cero al estandarizar.",
            fontsize=8,
            color=GRIS,
            va="top"
        )

        fig.savefig(
            figuras / f"08_{metodo}_orden{orden}_todas_las_relaciones.png",
            dpi=DPI,
            facecolor=FONDO
        )

        if MOSTRAR_FIGURAS:
            plt.show()

        plt.close(fig)

        print(f"{metodo}, orden {orden}: {len(origenes)} pares")

