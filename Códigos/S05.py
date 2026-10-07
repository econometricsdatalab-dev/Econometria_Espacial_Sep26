# -*- coding: utf-8 -*-
"""
Curso Integral de Econometría 
Modulo: Econometría espacial
Tema: Autocorrelación espacial
Sesión: 05 
Fecha: 06/10/2026
Docente/Asesor: Alexis Adonai Morales Alberto
"""
# Instalar esda 

pip install esda

# Modulos a implementar 

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

# Clases directas 

from pathlib import Path
from esda import G, Geary, Moran
from libpysal.weights import Queen, lag_spatial
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle 

# Mapa con datos de pobreza 

base = Path.cwd()
Mex = gpd.read_file(base / "Mapas" / "Mexico_ent" / "00ent.shp")
pobreza = pd.read_csv(
    base / "Datos" / "Pobreza_rel_2024.csv",
    encoding="utf-8-sig",
    dtype={"CVE_ENT": str},
)
pobreza.columns = pobreza.columns.str.strip()  # El CSV tiene "Pobreza "

Mex["CVE_ENT"] = Mex["CVE_ENT"].astype(str).str.zfill(2)
pobreza["CVE_ENT"] = pobreza["CVE_ENT"].str.zfill(2)
pobreza["Pobreza"] = pd.to_numeric(pobreza["Pobreza"], errors="raise")

Mex = Mex.merge(
    pobreza[["CVE_ENT", "Pobreza"]],
    on="CVE_ENT",
    how="left",
    validate="one_to_one",
)
Mex = Mex.sort_values("CVE_ENT").reset_index(drop=True)

if Mex["Pobreza"].isna().any():
    raise ValueError("Hay entidades del mapa sin dato de pobreza.")
if (Mex["Pobreza"] <= 0).any():
    raise ValueError("Getis-Ord G requiere valores de pobreza positivos.")

# Exportar mapa para futuros (sin unir datos) ----

ruta_shp = base / "Mapas" / "Mexico_pobreza" / "Pobreza_ent.shp"

Mex.to_file(
    ruta_shp,
    driver = "ESRI Shapefile",
    encoding = "UTF-8",
    index = False
    )

Mex = gpd.read_file(base / "Mapas" / "Mexico_pobreza" / "Pobreza_ent.shp")

Mex

# Matriz de pesos espaciales de contigüedad del método de Reina. 

w = Queen.from_dataframe(Mex, use_index=True)
if w.islands:
    raise ValueError(f"Entidades sin vecinas: {w.islands}")

y = Mex["Pobreza"].to_numpy()
print("Entidades analizadas:", len(y))

# Prueba de Moran 

w.transform = "R"
moran = Moran(y, w, permutations=999)
p_moran = (np.count_nonzero(moran.sim >= moran.I) + 1) / 1000


# Prueba de Geary 

geary = Geary(y, w, permutations=999)
p_geary = (np.count_nonzero(geary.sim <= geary.C) + 1) / 1000


# Prueba de Getis Ord 

getis = G(y, w, permutations=999)
p_getis = (np.count_nonzero(getis.sim >= getis.G) + 1) / 1000


# Compactar resultados 

resultados = pd.DataFrame({
    "Prueba": ["Moran I", "Geary C", "Getis-Ord G"],
    "Estadístico": [moran.I, geary.C, getis.G],
    "Esperado bajo H0": [moran.EI, geary.EC, getis.EG],
    "p unilateral": [p_moran, p_geary, p_getis],
})

resultados

# Mapas 

salida = base / "Evidencias" / "Autocorrelacion_global"
salida.mkdir(parents=True, exist_ok=True)

# Matriz binaria para visualizar los términos que intervienen en G y Geary
w.transform = "B"
B = w.full()[0]
vecinos = B.sum(axis=1)

# Aporte descriptivo de cada entidad al numerador del G global (%).
# Depende tanto del porcentaje de pobreza como del número de vecinos.
aporte_G = y * (B @ y)
Mex["Aporte_G"] = 100 * aporte_G / aporte_G.sum()

# Diferencia cuadrática media con entidades contiguas: contraste local
# que alimenta el numerador de Geary C. No es una prueba local.
Mex["Contraste_Geary"] = (
    B * (y[:, None] - y[None, :]) ** 2
).sum(axis=1) / vecinos

w.transform = "R"
Mex["Pobreza_vecinas"] = lag_spatial(w, y)

fondo = "#F7F7F5"
azul = "#003057"
naranja = "#F04A1D"
gris = "#4D565E"
paleta = LinearSegmentedColormap.from_list(
    "EDL_naranja", ["#FBE4D4", "#F5B07C", "#F46F2C", "#C93612"]
)
paleta_contraste = LinearSegmentedColormap.from_list(
    "EDL_contraste", ["#F0F3F5", "#F5B07C", "#F04A1D", azul]
)

escala_pobreza = max(Mex["Pobreza"].max(), Mex["Pobreza_vecinas"].max())
mapas = [
    ("Pobreza", "01_pobreza", "Pobreza observada",
     "Porcentaje de población en situación de pobreza por entidad · 2024",
     "Porcentaje (%)", escala_pobreza, paleta),
    ("Pobreza_vecinas", "02_pobreza_vecinas", "Pobreza de las entidades vecinas",
     "Promedio de las entidades contiguas · pesos de reina estandarizados",
     "Porcentaje (%)", escala_pobreza, paleta),
    ("Contraste_Geary", "03_contraste_geary", "Contraste con las entidades vecinas",
     "Diferencia cuadrática media de pobreza · componente descriptivo de Geary C",
     "Puntos porcentuales²", Mex["Contraste_Geary"].max(), paleta_contraste),
    ("Aporte_G", "04_aporte_getis", "Contribución al Getis–Ord G global",
     "Participación de cada entidad en el numerador del estadístico",
     "Porcentaje del numerador (%)", Mex["Aporte_G"].max(), paleta),
]

# Un mapa independiente por variable
for columna, archivo, titulo, subtitulo, unidad, maximo, colores in mapas:
    fig = plt.figure(figsize=(12, 8), facecolor=fondo)
    ax = fig.add_axes([0.04, 0.22, 0.92, 0.58])
    ax.set_facecolor(fondo)

    Mex.plot(
        column=columna, ax=ax, cmap=colores, vmin=0, vmax=maximo,
        edgecolor="white", linewidth=0.7
    )
    Mex.boundary.plot(ax=ax, color="#8C513D", linewidth=0.2)
    ax.set_aspect("equal")
    ax.set_axis_off()

    fig.text(0.06, 0.955, "ECONOMETRICS DATA LAB",
             color=naranja, fontsize=10, weight="bold")
    fig.text(0.06, 0.90, titulo, color=azul,
             fontsize=21, weight="bold")
    fig.text(0.06, 0.85, subtitulo, color=gris, fontsize=11)

    cax = fig.add_axes([0.58, 0.155, 0.35, 0.025])
    barra = fig.colorbar(
        ScalarMappable(norm=Normalize(0, maximo), cmap=colores),
        cax=cax, orientation="horizontal"
    )
    barra.outline.set_visible(False)
    barra.ax.tick_params(labelsize=9, colors=gris, length=0)
    fig.text(0.58, 0.19, unidad, color=gris, fontsize=9)
    fig.text(0.06, 0.075,
             "Fuente: Pobreza_rel_2024.csv; cartografía: 00ent.shp. "
             "Relaciones espaciales: contigüidad de reina.\n"
             "Los contrastes y aportes son descriptivos; "
             "la inferencia corresponde a las pruebas globales.",
             color=gris, fontsize=8, va="top")

    fig.savefig(salida / f"{archivo}.png", dpi=500, facecolor=fondo)
    plt.close(fig)


# Gráfico de dispersión de Moran
# Eje X: pobreza estandarizada; eje Y: promedio estandarizado de vecinas.
z = (y - y.mean()) / y.std(ddof=0)
wz = lag_spatial(w, z)
pendiente, intercepto = np.polyfit(z, wz, 1)

cuadrantes = np.select(
    [(z >= 0) & (wz >= 0), (z < 0) & (wz < 0),
     (z >= 0) & (wz < 0), (z < 0) & (wz >= 0)],
    ["Alto–Alto", "Bajo–Bajo", "Alto–Bajo", "Bajo–Alto"],
    default="Sin clasificar"
)
colores_cuadrantes = {
    "Alto–Alto": naranja,
    "Bajo–Bajo": azul,
    "Alto–Bajo": "#08989C",
    "Bajo–Alto": "#9F2578",
}

fig = plt.figure(figsize=(12, 8), facecolor=fondo, dpi = 500)
ax = fig.add_axes([0.11, 0.23, 0.78, 0.55])
ax.set_facecolor("#FFFFFF")

lim_x = max(abs(z.min()), abs(z.max())) * 1.20
lim_y = max(abs(wz.min()), abs(wz.max())) * 1.25
ax.set_xlim(-lim_x, lim_x)
ax.set_ylim(-lim_y, lim_y)

# Fondos sutiles para distinguir los cuatro patrones espaciales
for x0, y0, color in [
    (0, 0, "#FFF0E6"), (-lim_x, -lim_y, "#EAF0F4"),
    (0, -lim_y, "#E8F4F3"), (-lim_x, 0, "#F6EAF2")
]:
    ax.add_patch(Rectangle(
        (x0, y0), lim_x, lim_y,
        facecolor=color, edgecolor="none", alpha=0.70, zorder=0
    ))

ax.axhline(0, color=gris, linewidth=0.9, zorder=1)
ax.axvline(0, color=gris, linewidth=0.9, zorder=1)
xx = np.linspace(-lim_x, lim_x, 100)
ax.plot(xx, intercepto + pendiente * xx,
        color=naranja, linewidth=2.5, zorder=2)

for grupo, color in colores_cuadrantes.items():
    sel = cuadrantes == grupo
    ax.scatter(z[sel], wz[sel], s=72, color=color,
               edgecolor="white", linewidth=1.1,
               alpha=0.95, label=f"{grupo} ({sel.sum()})", zorder=3)

# Identificar algunas entidades extremas sin cubrir los 32 puntos
destacadas = np.argsort(z**2 + wz**2)[-4:]
for i in destacadas:
    ax.annotate(Mex.loc[i, "CVE_ENT"], (z[i], wz[i]),
                xytext=(6, 6), textcoords="offset points",
                fontsize=9, color=azul, weight="bold")

ax.set_xlabel("Pobreza estandarizada de la entidad", color=gris, fontsize=11)
ax.set_ylabel("Pobreza estandarizada promedio de sus vecinas",
              color=gris, fontsize=11)
ax.grid(color="#C6CDD2", linestyle=":", linewidth=0.7, alpha=0.6)
ax.set_axisbelow(True)
ax.tick_params(colors=gris)
for borde in ax.spines.values():
    borde.set_visible(False)
ax.legend(loc="lower right", frameon=False, fontsize=9)

fig.text(0.06, 0.955, "ECONOMETRICS DATA LAB",
         color=naranja, fontsize=10, weight="bold")
fig.text(0.06, 0.90, "Dispersión espacial de Moran",
         color=azul, fontsize=21, weight="bold")
fig.text(0.06, 0.85,
         f"Pobreza estatal, 2024  |  I = {moran.I:.3f}  |  "
         f"p unilateral = {p_moran:.3f}", color=gris, fontsize=11)
fig.text(0.11, 0.11,
         "Cada punto es una entidad; las etiquetas muestran la clave de "
         "cuatro casos extremos. La pendiente de la recta equivale al I de Moran.\n"
         "Fuente: Pobreza_rel_2024.csv; cartografía: 00ent.shp. "
         "Pesos de reina estandarizados por filas.",
         color=gris, fontsize=8, va="top")

fig.savefig(salida / "05_dispersion_moran.png", dpi=500, facecolor=fondo)
plt.close(fig)
print(f"\nGráficos guardados en: {salida}")



