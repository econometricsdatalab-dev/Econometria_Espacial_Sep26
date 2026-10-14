# -*- coding: utf-8 -*-
"""
Curso Integral de Econometría 
Modulo: Econometría espacial
Tema: Autocorrelación espacial local
Sesión: 06 
Fecha: 08/10/2026
Docente/Asesor: Alexis Adonai Morales Alberto
"""

# Modulos a utilizar 

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from copy import deepcopy
from esda import Moran_Local, Geary_Local, G_Local
from libpysal.weights import Queen, lag_spatial
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from statsmodels.stats.multitest import multipletests

# Configuraciones iniciales 

BASE = Path(
    "E:\Econometrics Data Lab\Curso integral de econometría\Econometría espacial\E_SP_SEP26"
    )
RUTA_MAPA = BASE / "Mapas" / "Mexico_pobreza" / "Pobreza_ent.shp"
SALIDA = BASE / "Resultados" / "Autocorrelacion_local"
SALIDA.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
PERMUTACIONES = 999
SEMILLA = 2026
MOSTRAR_MAPAS = True
DPI = 500

AZUL = "#003057"
NARANJA = "#F04A1D"
TURQUESA = "#08989C"
MORADO = "#9F2578"
GRIS = "#4D565E"
GRIS_CLARO = "#DFE4E8"
FONDO = "#F7F7F5"
BLANCO = "#FFFFFF"

COLORES_LISA = {
    "Alto–Alto": NARANJA,
    "Bajo–Bajo": AZUL,
    "Alto–Bajo": TURQUESA,
    "Bajo–Alto": MORADO,
    "No significativo": GRIS_CLARO,
}
COLORES_GEARY = {
    "Agrupamiento": TURQUESA,
    "Valor atípico": NARANJA,
    "Otro": "#F5B07C",
    "No significativo": GRIS_CLARO,
}
COLORES_G = {
    "Hotspot": NARANJA,
    "Coldspot": AZUL,
    "No significativo": GRIS_CLARO,
}
PALETA = LinearSegmentedColormap.from_list(
    "EDL_Pobreza", ["#FBE4D4", "#F5B07C", "#F46F2C", NARANJA]
)
PALETA_DIVERGENTE = LinearSegmentedColormap.from_list(
    "EDL_Divergente", [AZUL, BLANCO, NARANJA]
)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": FONDO,
    "axes.facecolor": FONDO,
    "axes.edgecolor": GRIS_CLARO,
    "text.color": AZUL,
})

# Lectura del mapa 

Mex = gpd.read_file(RUTA_MAPA)
ids = Mex.CVE_ENT.tolist()
y = Mex["Pobreza"].to_numpy(dtype=float)

# Matriz de pesos espaciales: Contigüedad - Reina 

w_r = Queen.from_dataframe(Mex, use_index=False)
w_r.transform = "R" 
w_b = deepcopy(w_r)
w_b.transform = "B"

# Rezago espacial y estandarización 

z = (y - y.mean()) / y.std(ddof=0)
wz = lag_spatial(w_r, z)
Mex["Pobreza_vec"] = lag_spatial(w_r, y)
Mex["Z_Pobreza"] = z
Mex["WZ_Pobreza"] = wz

# Indice de Moran Local 

moran_local = Moran_Local(
    y, w_r, transformation="r", permutations=PERMUTACIONES,
    n_jobs=1, seed=SEMILLA
)
Mex["I_Moran"] = moran_local.Is
Mex["p_Moran"] = moran_local.p_sim
Mex["Q_Moran"] = moran_local.q

# Convención PySAL: 1 HH, 2 LH, 3 LL, 4 HL.
cuadrantes = {1: "Alto–Alto", 2: "Bajo–Alto", 
              3: "Bajo–Bajo", 4: "Alto–Bajo"}

Mex["LISA_base"] = pd.Series(moran_local.q, 
                             index=Mex.index).map(cuadrantes)

Mex["LISA_05"] = np.where(Mex["p_Moran"] < ALPHA, 
                          Mex["LISA_base"], "No significativo")
                          
# Geary Local 

geary_local = Geary_Local(
    connectivity=w_r, labels=True, sig=ALPHA,
    permutations=PERMUTACIONES, n_jobs=1, seed=SEMILLA
).fit(y)
Mex["C_Geary"] = geary_local.localG
Mex["p_Geary"] = geary_local.p_sim
Mex["Q_Geary"] = geary_local.labs
categorias_geary = {1: "Valor atípico", 2: "Agrupamiento", 
                    3: "Otro", 4: "No significativo"}
Mex["Geary_05"] = pd.Series(geary_local.labs, 
                            index=Mex.index).map(categorias_geary)

# Getis-Ord Gi 

gi_local = G_Local(
    y, deepcopy(w_b), transform="B", star=False,
    permutations=PERMUTACIONES, n_jobs=1, seed=SEMILLA
)
Mex["Gi"] = gi_local.Gs
Mex["Z_Gi"] = gi_local.Zs
Mex["p_Gi"] = gi_local.p_sim
Mex["Gi_05"] = np.select(
    [(Mex["p_Gi"] < ALPHA) & (Mex["Z_Gi"] > 0),
     (Mex["p_Gi"] < ALPHA) & (Mex["Z_Gi"] < 0)],
    ["Hotspot", "Coldspot"], default="No significativo"
)

# Getis Ord Gi* 

gistar_local = G_Local(
    y, deepcopy(w_b), transform="B", star=True,
    permutations=PERMUTACIONES, n_jobs=1, seed=SEMILLA
)
Mex["Gi_star"] = gistar_local.Gs
Mex["Z_Gistar"] = gistar_local.Zs
Mex["p_Gistar"] = gistar_local.p_sim
Mex["Gistar_05"] = np.select(
    [(Mex["p_Gistar"] < ALPHA) & (Mex["Z_Gistar"] > 0),
     (Mex["p_Gistar"] < ALPHA) & (Mex["Z_Gistar"] < 0)],
    ["Hotspot", "Coldspot"], default="No significativo"
)

# Corrección por comparaciones múltiples 

for prefijo, columna in [
    ("Moran", "p_Moran"), ("Geary", "p_Geary"),
    ("Gi", "p_Gi"), ("Gistar", "p_Gistar")
]:
    rechaza, p_ajustado, _, _ = multipletests(
        Mex[columna].to_numpy(), alpha=ALPHA, method="fdr_bh"
    )
    Mex[f"FDR_{prefijo}"] = rechaza
    Mex[f"q_{prefijo}"] = p_ajustado

Mex["LISA_FDR"] = np.where(Mex["FDR_Moran"],
                           Mex["LISA_base"], "No significativo")
Mex["Geary_FDR"] = np.where(Mex["FDR_Geary"], 
                            Mex["Geary_05"], "No significativo")
Mex["Gi_FDR"] = np.where(Mex["FDR_Gi"], 
                         Mex["Gi_05"], "No significativo")
Mex["Gistar_FDR"] = np.where(Mex["FDR_Gistar"], 
                             Mex["Gistar_05"], "No significativo")
                             
# Tabla de resultados

columnas_tabla = [
    "CVE_ENT", "Pobreza", "Pobreza_vec", "I_Moran", "p_Moran", "q_Moran",
    "LISA_05", "LISA_FDR", "C_Geary", "p_Geary", "q_Geary",
    "Geary_05", "Geary_FDR", "Gi", "Z_Gi", "p_Gi", "q_Gi", "Gi_FDR",
    "Gi_star", "Z_Gistar", "p_Gistar", "q_Gistar", "Gistar_FDR"
]
if "ENTIDAD" in Mex.columns:
    columnas_tabla.insert(1, "ENTIDAD")
resultados = Mex[columnas_tabla].copy()
print("\nRESULTADOS LOCALES POR ENTIDAD")
print(resultados.round(4).to_string(index=False))
print("\nENTIDADES CON SIGNIFICANCIA FDR (por prueba):")
for col in ["LISA_FDR", "Geary_FDR", "Gi_FDR", "Gistar_FDR"]:
    print(col + ":", (Mex[col] != "No significativo").sum())

resultados.to_csv(SALIDA / "Resultados_LISA.csv", index=False, encoding="utf-8-sig")

# Mapa coropletico base: pobreza y promedio de vecinos 

fig, ejes = plt.subplots(1, 2, figsize=(16, 8), facecolor=FONDO)
minimo = min(Mex["Pobreza"].min(), Mex["Pobreza_vec"].min())
maximo = max(Mex["Pobreza"].max(), Mex["Pobreza_vec"].max())
for ax, col, subtitulo in zip(
    ejes, ["Pobreza", "Pobreza_vec"],
    ["Pobreza observada", "Promedio de entidades vecinas"]
):
    Mex.plot(column=col, ax=ax, cmap=PALETA, vmin=minimo, vmax=maximo,
             edgecolor=BLANCO, linewidth=0.75)
    ax.set_title(subtitulo, fontsize=12, color=AZUL, fontweight="bold", pad=8)
    ax.set_axis_off()
fig.subplots_adjust(left=0.04, right=0.96, top=0.77, bottom=0.19, wspace=0.08)
fig.text(0.05, 0.955, "ECONOMETRICS DATA LAB", fontsize=10, color=NARANJA, weight="bold")
fig.text(0.05, 0.905, "Pobreza y entorno espacial", fontsize=20, color=AZUL, weight="bold")
fig.text(0.05, 0.855, "México, 2024 · Porcentaje por entidad y promedio por contigüidad de reina", fontsize=10, color=GRIS)
cax = fig.add_axes([0.32, 0.135, 0.36, 0.018])
fig.colorbar(ScalarMappable(norm=Normalize(minimo, maximo), cmap=PALETA), cax=cax, orientation="horizontal").set_label("Porcentaje (%)", color=GRIS)
fig.text(0.05, 0.045, "Fuente: Pobreza_rel_2024.csv y 00ent.shp · W de reina, estandarizada por filas. EDL.", fontsize=8, color=GRIS)
fig.savefig(SALIDA / "01_pobreza_y_vecinas.png", dpi=DPI, facecolor=FONDO)
fig.savefig(SALIDA / "01_pobreza_y_vecinas.svg", facecolor=FONDO)
if MOSTRAR_MAPAS:
    plt.show()
plt.close(fig)


# Mapa LISA Moran: Nominal y corregido por FDR 

fig, ejes = plt.subplots(1, 2, figsize=(16, 8), facecolor=FONDO)
for ax, col, subtitulo in zip(
    ejes, ["LISA_05", "LISA_FDR"],
    ["Significancia nominal (p < 0.05)", "Significancia ajustada (FDR < 0.05)"]
):
    Mex.plot(ax=ax, color=Mex[col].map(COLORES_LISA), edgecolor=BLANCO, linewidth=0.7)
    ax.set_title(subtitulo, fontsize=12, color=AZUL, fontweight="bold")
    ax.set_axis_off()
fig.subplots_adjust(left=0.03, right=0.97, top=0.79, bottom=0.23, wspace=0.08)
fig.text(0.05, 0.955, "ECONOMETRICS DATA LAB", fontsize=10, color=NARANJA, weight="bold")
fig.text(0.05, 0.905, "Moran local · Mapa de agrupamientos LISA", fontsize=20, color=AZUL, weight="bold")
fig.text(0.05, 0.855, f"México, 2024 · W reina · {PERMUTACIONES:,} permutaciones condicionales", fontsize=10, color=GRIS)
fig.legend(handles=[Patch(facecolor=v, label=k) for k, v in COLORES_LISA.items()],
           loc="lower center", bbox_to_anchor=(0.5, 0.155), ncol=5, frameon=False, fontsize=9)
fig.text(0.05, 0.065, "H0: ausencia de asociación espacial local. HH/LL: agrupamientos; HL/LH: valores atípicos.\nFuente: Pobreza_rel_2024.csv y 00ent.shp · FDR de Benjamini-Hochberg aplicado a 32 contrastes.", fontsize=8, color=GRIS)
fig.savefig(SALIDA / "02_moran_lisa.png", dpi=DPI, facecolor=FONDO)
fig.savefig(SALIDA / "02_moran_lisa.svg", facecolor=FONDO)
if MOSTRAR_MAPAS:
    plt.show()
plt.close(fig)


# Geary local 

fig, ejes = plt.subplots(1, 2, figsize=(16, 8), facecolor=FONDO)
Mex.plot(column="C_Geary", ax=ejes[0], cmap=PALETA, edgecolor=BLANCO,
         linewidth=0.75, legend=True,
         legend_kwds={"shrink": 0.55, "label": "Cᵢ local"})
ejes[1].set_axis_off()
Mex.plot(ax=ejes[1], color=Mex["Geary_FDR"].map(COLORES_GEARY), edgecolor=BLANCO, linewidth=0.7)
for ax, titulo in zip(ejes, ["Geary local · intensidad", "Geary local · categorías FDR"]):
    ax.set_title(titulo, fontsize=12, fontweight="bold", color=AZUL)
    ax.set_axis_off()
fig.subplots_adjust(left=0.03, right=0.97, top=0.79, bottom=0.23, wspace=0.13)
fig.text(0.05, 0.955, "ECONOMETRICS DATA LAB", fontsize=10, color=NARANJA, weight="bold")
fig.text(0.05, 0.905, "Geary local · Similitud y valores atípicos", fontsize=20, color=AZUL, weight="bold")
fig.text(0.05, 0.855, f"México, 2024 · Contrastes locales · {PERMUTACIONES:,} permutaciones", fontsize=10, color=GRIS)
fig.legend(handles=[Patch(facecolor=v, label=k) for k, v in COLORES_GEARY.items()],
           loc="lower center", bbox_to_anchor=(0.5, 0.155), ncol=4, frameon=False)
fig.text(0.05, 0.065, "Un Cᵢ pequeño indica mayor semejanza con sus vecinas; Cᵢ grande, mayor disimilitud.\nFuente: Pobreza_rel_2024.csv y 00ent.shp · Significancia de categorías ajustada por FDR.", fontsize=8, color=GRIS)
fig.savefig(SALIDA / "04_geary_local.png", dpi=DPI, facecolor=FONDO)
fig.savefig(SALIDA / "04_geary_local.svg", facecolor=FONDO)
if MOSTRAR_MAPAS:
    plt.show()
plt.close(fig)

# Getis Ord (Gi y Gi*)

fig, ejes = plt.subplots(1, 2, figsize=(16, 8), facecolor=FONDO)
for ax, col, subtitulo in zip(
    ejes, ["Gi_FDR", "Gistar_FDR"],
    ["Gi · no incluye la entidad focal", "Gi* · incluye la entidad focal"]
):
    Mex.plot(ax=ax, color=Mex[col].map(COLORES_G), edgecolor=BLANCO, linewidth=0.7)
    ax.set_title(subtitulo, fontsize=12, color=AZUL, fontweight="bold")
    ax.set_axis_off()
fig.subplots_adjust(left=0.03, right=0.97, top=0.79, bottom=0.23, wspace=0.08)
fig.text(0.05, 0.955, "ECONOMETRICS DATA LAB", fontsize=10, color=NARANJA, weight="bold")
fig.text(0.05, 0.905, "Getis-Ord local · Concentraciones de pobreza", fontsize=20, color=AZUL, weight="bold")
fig.text(0.05, 0.855, f"México, 2024 · W de reina binaria · {PERMUTACIONES:,} permutaciones · FDR", fontsize=10, color=GRIS)
fig.legend(handles=[Patch(facecolor=v, label=k) for k, v in COLORES_G.items()],
           loc="lower center", bbox_to_anchor=(0.5, 0.155), ncol=3, frameon=False)
fig.text(0.05, 0.065, "Rojo: concentración local de pobreza alta; azul: concentración local de pobreza baja.\nFuente: Pobreza_rel_2024.csv y 00ent.shp · Dirección según Z; significancia mediante permutaciones y FDR.", fontsize=8, color=GRIS)
fig.savefig(SALIDA / "05_getis_gi_y_gistar.png", dpi=DPI, facecolor=FONDO)
fig.savefig(SALIDA / "05_getis_gi_y_gistar.svg", facecolor=FONDO)
if MOSTRAR_MAPAS:
    plt.show()
plt.close(fig)

# Intensidad de Getis-Ord Gi*
limz = max(1e-6, float(np.nanmax(np.abs(Mex["Z_Gistar"]))))
fig, ax = plt.subplots(figsize=(12, 8), facecolor=FONDO)
Mex.plot(column="Z_Gistar", ax=ax, cmap=PALETA_DIVERGENTE,
         vmin=-limz, vmax=limz, edgecolor=BLANCO, linewidth=0.75)
ax.set_axis_off()
fig.subplots_adjust(left=0.04, right=0.96, top=0.80, bottom=0.18)
fig.text(0.05, 0.955, "ECONOMETRICS DATA LAB", fontsize=10, color=NARANJA, weight="bold")
fig.text(0.05, 0.905, "Intensidad de puntos calientes y fríos", fontsize=20, color=AZUL, weight="bold")
fig.text(0.05, 0.85, "México, 2024 · Z de Getis-Ord Gi* (para la prueba de significancia, revisar p y FDR)", fontsize=10, color=GRIS)
cax = fig.add_axes([0.31, 0.125, 0.38, 0.02])
fig.colorbar(ScalarMappable(norm=Normalize(-limz, limz), cmap=PALETA_DIVERGENTE), cax=cax, orientation="horizontal").set_label("Z de Gi*", color=GRIS)
fig.text(0.05, 0.05, "Fuente: Pobreza_rel_2024.csv y 00ent.shp · Z analítico; significancia en prueba por permutaciones.", fontsize=8, color=GRIS)
fig.savefig(SALIDA / "06_getis_gistar_z.png", dpi=DPI, facecolor=FONDO)
fig.savefig(SALIDA / "06_getis_gistar_z.svg", facecolor=FONDO)
if MOSTRAR_MAPAS:
    plt.show()
plt.close(fig)

