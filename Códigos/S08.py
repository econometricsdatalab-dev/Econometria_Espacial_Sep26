# -*- coding: utf-8 -*-
"""
Curso Integral de Econometría 
Modulo: Econometría espacial
Tema: Modelo econométricos espaciales (SAR y SDM)
Sesión: 08
Fecha: 15/10/2026
Docente/Asesor: Alexis Adonai Morales Alberto
"""

# Modulos necesarios 

## Instalación de modulos 

pip install numpy pandas scipy geopandas matplotlib statsmodels libpysal esda spreg

## Carga de modulos completos 

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

## Clases o submodulos directos del modulo general 

from pathlib import Path
from scipy import stats
from scipy.sparse.csgraph import shortest_path
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from libpysal.weights import Queen, Rook, KNN
from esda.moran import Moran
from spreg import OLS, ML_Lag, LMtests
from spreg.diagnostics import likratiotest

# Configuración inicial 

BASE = Path(__name__).resolve().parent
BASE = Path(
    "E:\Econometrics Data Lab\Curso integral de econometría\Econometría espacial\E_SP_SEP26"
    )
DATOS = BASE / 'Datos' / 'Delincuencia_Mexico_2024.csv'
if not DATOS.exists():
    DATOS = BASE / 'Delincuencia_Mexico_2024.csv'

MAPA = BASE / 'Mapas' / 'Mexico_ent' / '00ent.shp'
if not MAPA.exists():
    alternativa = MAPA.with_name('ent00.shp')
    if alternativa.exists():
        MAPA = alternativa

SALIDA = BASE / 'Resultados' / 'Sesion08_SAR_SDM'
FIGURAS = SALIDA / 'Figuras'
FIGURAS.mkdir(parents=True, exist_ok=True)

METODO_PESOS = 'reina'
K_VECINOS = 4
ALFA = 0.05
PERMUTACIONES = 9999
SIMULACIONES_IMPACTOS = 1000
SEMILLA = 2026
DPI = 500
MOSTRAR = True

ESPECIFICACION = 'gasto'

FONDO = '#F7F7F5'
AZUL = '#003057'
NARANJA = '#F04A1D'
TURQUESA = '#08989C'
VIOLETA = '#9F2578'
GRIS = '#4D565E'
GRIS_CLARO = '#DFE4E7'
BLANCO = '#FFFFFF'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Bahnschrift', 'Noto Sans', 'DejaVu Sans'],
    'figure.facecolor': FONDO,
    'axes.facecolor': FONDO,
    'axes.edgecolor': GRIS_CLARO,
    'axes.labelcolor': GRIS,
    'xtick.color': GRIS,
    'ytick.color': GRIS,
    'font.size': 10,
    'savefig.facecolor': FONDO,
    'axes.unicode_minus': False,
})

if not DATOS.exists():
    raise FileNotFoundError(f'Falta CSV: {DATOS}')
if not MAPA.exists():
    raise FileNotFoundError(
        'Falta cartografía oficial para construir W. Coloca 00ent.shp, .shx, '
        f'.dbf y .prj en {MAPA.parent}. No se estiman modelos con W inventada.'
    )


# Lectura de datos 

datos = pd.read_csv(DATOS, dtype={'CVE_ENT': str}, encoding='utf-8-sig')
datos.columns = datos.columns.str.strip()
datos['CVE_ENT'] = datos['CVE_ENT'].astype(str).str.strip().str.zfill(2)

requeridas = [
    'CVE_ENT', 'ENTIDAD', 'Tasa_Incidencia_ENVIPE_2024',
    'Pobreza_2024', 'Ingreso_Hogar_2024', 'Gasto_Hogar_2024',
    'Desocupacion_2024T2'
]
if not set(requeridas).issubset(datos.columns):
    raise ValueError(f'Faltan variables: {sorted(set(requeridas) - set(datos.columns))}')
if len(datos) != 32 or datos.CVE_ENT.duplicated().any():
    raise ValueError('Se requieren las 32 entidades, una fila por CVE_ENT.')
if set(datos.CVE_ENT) != {f'{i:02d}' for i in range(1, 33)}:
    raise ValueError('CVE_ENT debe contener las claves 01 a 32.')
for columna in requeridas[2:]:
    datos[columna] = pd.to_numeric(datos[columna], errors='raise')
if datos[requeridas].isna().any().any():
    raise ValueError('Hay datos faltantes en las variables utilizadas.')
if (datos[['Tasa_Incidencia_ENVIPE_2024', 'Ingreso_Hogar_2024',
           'Gasto_Hogar_2024']] <= 0).any().any():
    raise ValueError('Los indicadores usados con logaritmos / ingreso deben ser positivos.')

datos['Ln_Incidencia'] = np.log(datos['Tasa_Incidencia_ENVIPE_2024'])
datos['Ln_Ingreso'] = np.log(datos['Ingreso_Hogar_2024'])
datos['Gasto_Miles'] = datos['Gasto_Hogar_2024'] / 1000.0

if ESPECIFICACION == 'gasto':
    VARIABLES = ['Pobreza_2024', 'Gasto_Miles', 'Desocupacion_2024T2']
elif ESPECIFICACION == 'ingreso':
    VARIABLES = ['Pobreza_2024', 'Ln_Ingreso', 'Desocupacion_2024T2']
else:
    raise ValueError("ESPECIFICACION debe ser 'gasto' o 'ingreso'.")

carto = gpd.read_file(MAPA)
carto.columns = carto.columns.str.strip()
if carto.crs is None:
    raise ValueError('No se identificó proyección geográfica. Revisa .prj.')
if 'CVE_ENT' not in carto.columns:
    raise ValueError('El shapefile debe tener la variable CVE_ENT.')
carto['CVE_ENT'] = carto['CVE_ENT'].astype(str).str.strip().str.zfill(2)
if carto.geometry.is_empty.any() or carto.geometry.isna().any():
    raise ValueError('Existen geometrías vacías en la cartografía.')
if not carto.geometry.is_valid.all():
    raise ValueError('Hay geometrías inválidas; corrige el shapefile.')
# Disolver fragmentos, si una entidad aparece distribuida en varias filas.
carto = carto[['CVE_ENT', 'geometry']].dissolve(by='CVE_ENT', as_index=False)
carto = carto.to_crs(epsg=6372)

Mex = carto.merge(datos, on='CVE_ENT', how='inner', validate='one_to_one')
Mex = Mex.sort_values('CVE_ENT').reset_index(drop=True)
if len(Mex) != 32:
    raise ValueError('No se recuperaron las 32 entidades al unir SHP con CSV.')

n = len(Mex)
identificadores = Mex.CVE_ENT.tolist()
y = Mex['Ln_Incidencia'].to_numpy(dtype=float).reshape(-1, 1)
X = Mex[VARIABLES].to_numpy(dtype=float)

# Matriz de pesos espaciales 

if METODO_PESOS == 'reina':
    w = Queen.from_dataframe(Mex, use_index=True)
elif METODO_PESOS == 'torre':
    w = Rook.from_dataframe(Mex, use_index=True)
elif METODO_PESOS == 'knn':
    puntos = Mex.geometry.centroid
    coordenadas = np.column_stack([puntos.x.to_numpy(), puntos.y.to_numpy()])
    w = KNN.from_array(coordenadas, k=K_VECINOS)
else:
    raise ValueError('METODO_PESOS debe ser reina, torre o knn.')

if w.islands:
    raise ValueError(
        f'Entidades sin vecinos con {METODO_PESOS}: {w.islands}. '
        'Revisa topología o usa KNN como análisis alternativo claramente documentado.'
    )
w.transform = 'R'
W = w.full()[0].astype(float)
I = np.eye(n)
if not np.allclose(W.sum(axis=1), 1):
    raise ValueError('La matriz W no tiene todas las filas estandarizadas a 1.')
if not np.allclose(np.diag(W), 0):
    raise ValueError('La matriz W debe tener diagonal cero.')

B = (W > 0).astype(float)
# shortest_path orientado a contactos (no a pesos estandarizados).
dist_orden = shortest_path(B, directed=False, unweighted=True)
vecinas_1 = (dist_orden == 1)
vecinas_2mas = np.isfinite(dist_orden) & (dist_orden >= 2)

print('\nMatriz:', METODO_PESOS, '| vecinos mínimo:', w.min_neighbors,
      '| máximo:', w.max_neighbors, '| componentes:', w.n_components)
pd.DataFrame(W, index=identificadores, columns=identificadores).to_csv(
    SALIDA / f'01_matriz_W_{METODO_PESOS}_estandarizada.csv', encoding='utf-8-sig')

# Modelo OLS tradicional y SLX; prueba de LMtest 

ols = OLS(
    y, X, w=w, spat_diag=True, moran=True, nonspat_diag=True,
    white_test=True, vif=True, name_y='Ln_Incidencia',
    name_x=VARIABLES, name_w=METODO_PESOS,
    name_ds='Incidencia estatal México 2024'
)

slx = OLS(
    y, X, w=w, slx_lags=1, spat_diag=True, moran=True,
    nonspat_diag=True, name_y='Ln_Incidencia', name_x=VARIABLES,
    name_w=METODO_PESOS
)

lms = LMtests(ols, w, tests=['all'])
pruebas_lm = [
    ('LM-Lag', 'H0: rho = 0 frente a SAR', 'lml'),
    ('LM-Error', 'H0: lambda = 0 frente a SEM', 'lme'),
    ('Robust LM-Lag', 'Lag ajustado ante alternativa error', 'rlml'),
    ('Robust LM-Error', 'Error ajustado ante alternativa lag', 'rlme'),
    ('LM-SARMA', 'Sin dependencia conjunta lag + error', 'sarma'),
    ('LM-WX', 'H0: theta = 0 (solo WX)', 'lmwx'),
    ('Robust LM-WX', 'WX ajustado ante lag', 'rlmwx'),
    ('Robust LM-Lag SDM', 'Lag ajustado frente a WX', 'rlmdurlag'),
    ('LM-SDM conjunto', 'H0: rho = theta = 0', 'lmspdurbin')
]
filas_lm = []
for nombre, hipotesis, atributo in pruebas_lm:
    if hasattr(lms, atributo):
        resultado = getattr(lms, atributo)
        filas_lm.append([nombre, hipotesis, float(resultado[0]), float(resultado[1])])
    else:
        print(f'AVISO: {atributo} no está disponible en esta versión de spreg.')
LM = pd.DataFrame(filas_lm, columns=['Prueba', 'H0', 'Estadistico', 'p_valor'])
LM['Decision_5pct'] = np.where(LM.p_valor < ALFA, 'Rechazar H0', 'No rechazar H0')
LM.to_csv(SALIDA / '02_pruebas_LM.csv', index=False, encoding='utf-8-sig')
print('\n\nPRUEBAS LM (basadas en OLS)')
print(LM.to_string(index=False, float_format=lambda v: f'{v:.5f}'))

# Estimación SAR, SDM (MLE) en toda la muestra

argumentos = dict(
    y=y, x=X, w=w, method='full', spat_impacts=['full'],
    name_y='Ln_Incidencia', name_x=VARIABLES,
    name_w=METODO_PESOS, name_ds='México 2024', vm=True
)
sar = ML_Lag(**argumentos, slx_lags=0, spat_diag=True)
sdm = ML_Lag(**argumentos, slx_lags=1, spat_diag=True)

(SALIDA / '03_resultado_OLS.txt').write_text(ols.summary, encoding='utf-8')
(SALIDA / '04_resultado_SLX.txt').write_text(slx.summary, encoding='utf-8')
(SALIDA / '05_resultado_SAR.txt').write_text(sar.summary, encoding='utf-8')
(SALIDA / '06_resultado_SDM.txt').write_text(sdm.summary, encoding='utf-8')
print('\n\nMODELO SAR | ML')
print(sar.summary)
print('\n\nMODELO SPATIAL DURBIN | ML')
print(sdm.summary)

modelos = [('OLS', ols), ('SLX', slx), ('SAR', sar), ('SDM', sdm)]
comparacion = pd.DataFrame([
    dict(Modelo=nombre, LogLik=float(reg.logll), AIC=float(reg.aic),
         BIC=float(reg.schwarz), R2_o_Pseudo_R2=float(getattr(reg, 'r2', getattr(reg, 'pr2', np.nan))),
         Parametros=len(np.asarray(reg.betas).ravel()))
    for nombre, reg in modelos
])
comparacion.to_csv(SALIDA / '07_comparacion_modelos.csv', index=False, encoding='utf-8-sig')
print('\n\nCOMPARACIÓN DE MODELOS (misma muestra y dependiente)')
print(comparacion.round(4).to_string(index=False))

# Coeficientes estimados con prueba asintótica z en SAR y SDM.
filas_coefs = []
for nombre, reg in [('SAR', sar), ('SDM', sdm)]:
    valores = np.asarray(reg.betas).ravel()
    ee = np.asarray(reg.std_err).ravel()
    nombres = list(reg.name_x)
    if len(valores) != len(nombres):
        raise ValueError(f'Error de longitud de coeficientes / nombres: {nombre}')
    for i, variable in enumerate(nombres):
        z, p = reg.z_stat[i]
        filas_coefs.append([nombre, variable, valores[i], ee[i], z, p])
coeficientes = pd.DataFrame(filas_coefs, columns=[
    'Modelo', 'Variable', 'Coeficiente', 'Error_estandar', 'z', 'p_valor'])
coeficientes.to_csv(SALIDA / '08_coeficientes.csv', index=False, encoding='utf-8-sig')

# Pruebas LR (estimaciones anidadas), hipótesis del factor común
pares_lr = [
    ('OLS vs SAR', ols, sar, 'H0: rho = 0'),
    ('OLS vs SDM', ols, sdm, 'H0: rho = theta_1 = ... = theta_k = 0'),
    ('SAR vs SDM', sar, sdm, 'H0: todos theta_k = 0'),
    ('SLX vs SDM', slx, sdm, 'H0: rho = 0'),
]
filas_lr = []
for nombre, restringido, irrestricto, hipotesis in pares_lr:
    lr = likratiotest(restringido, irrestricto)
    filas_lr.append([nombre, hipotesis,
                     float(lr['likr']), int(lr['df']), float(lr['p-value'])])
LR = pd.DataFrame(filas_lr, columns=['Contraste', 'H0', 'LR', 'gl', 'p_valor'])
LR['Decision_5pct'] = np.where(LR.p_valor < ALFA, 'Rechazar H0', 'No rechazar H0')
LR.to_csv(SALIDA / '09_pruebas_LR.csv', index=False, encoding='utf-8-sig')
print('\n\nPRUEBAS DE RAZÓN DE VEROSIMILITUD')
print(LR.round(5).to_string(index=False))

if getattr(sdm, 'cfh_test', None) is not None:
    cf = getattr(sdm, 'cfh_test')
    print('\nTEST DE FACTOR COMÚN (H0: theta = -rho*beta):', cf)
    pd.DataFrame([{'Estadistico': float(cf[0]), 'p_valor': float(cf[1])}]).to_csv(
        SALIDA / '10_factor_comun.csv', index=False, encoding='utf-8-sig')

# Diagnóstico residual: Moran permutacional exploratorio, JB y BP
residuos = []
for nombre, reg in modelos:
    e = np.asarray(reg.u).ravel()
    np.random.seed(SEMILLA)
    m = Moran(e, w, permutations=PERMUTACIONES)
    # Monte Carlo de dos colas centrado en la distribución permutada,
    # solo como diagnóstico descriptivo (no es prueba exacta posestimación).
    centro = float(np.mean(m.sim))
    p_dos_colas = (np.count_nonzero(np.abs(m.sim-centro) >= abs(m.I-centro))
                   + 1) / (PERMUTACIONES + 1)
    jb = stats.jarque_bera(e)
    # Comprobación descriptiva de heterocedasticidad por X (no LM formal de MLE).
    from statsmodels.stats.diagnostic import het_breuschpagan
    x_const = np.column_stack([np.ones(n), X])
    bp = het_breuschpagan(e, x_const)
    residuos.append([nombre, m.I, p_dos_colas, jb.statistic, jb.pvalue, bp[0], bp[1]])
    Mex[f'Resid_{nombre}'] = e
    Mex[f'Pred_{nombre}'] = np.asarray(reg.predy).ravel()
    # predy: predicción estructural condicional (contiene rho*Wy en SAR/SDM).

Diagnosticos = pd.DataFrame(residuos, columns=[
    'Modelo', 'Moran_I_residual', 'p_permutaciones_aprox',
    'Jarque_Bera', 'p_JB', 'Breusch_Pagan_descriptivo', 'p_BP_descriptivo'
])
Diagnosticos.to_csv(SALIDA / '11_diagnostico_residuos.csv',
                   index=False, encoding='utf-8-sig')
print('\n\nDIAGNÓSTICOS DE RESIDUOS')
print(Diagnosticos.round(5).to_string(index=False))

# Impactos SAR / SDM: matrices marginales exactas para W elegida

k = len(VARIABLES)

def calcular_impactos(coefs, nombre_modelo):
    """Devuelve impactos por variable, n x n. Sin incertidumbre inferencial."""
    coef = np.asarray(coefs, dtype=float).ravel()
    rho = float(coef[-1])
    if not np.isfinite(rho) or abs(rho) >= 1:
        raise ValueError(f'Rho no admisible para inversión estable: {rho}')
    A = np.linalg.solve(I - rho * W, I)
    matrices = {}
    tabla = []
    for j, v in enumerate(VARIABLES):
        beta = float(coef[1+j])
        theta = float(coef[1+k+j]) if nombre_modelo == 'SDM' else 0.0
        # B_j = beta I + theta W; theta=0 en el SAR.
        B_j = beta * I + theta * W
        M = A @ B_j
        matrices[v] = M
        directos = np.diag(M)
        indirectos_rec = M.sum(axis=1) - directos
        totales = M.sum(axis=1)
        indirectos_gen = M.sum(axis=0) - directos
        directo_wx = theta * W
        feedback = (A - I) @ B_j
        # Efectos en vecinos inmediatos y no inmediatos: clasificación
        # por distancia topológica final, no por longitud de la ruta causal.
        directa_1orden = (M * vecinas_1).sum(axis=1)
        distante_2mas = (M * vecinas_2mas).sum(axis=1)
        tabla.append(dict(
            Modelo=nombre_modelo, Variable=v, beta=beta, theta=theta, rho=rho,
            ADI=float(directos.mean()), AII=float(indirectos_rec.mean()),
            ATI=float(totales.mean()),
            Indirecto_1orden=float(directa_1orden.mean()),
            Indirecto_2mas=float(distante_2mas.mean()),
            WX_explicit_1orden=float((directo_wx * vecinas_1).sum(axis=1).mean()),
            Feedback_indirecto=float((feedback * (1-I)).sum(axis=1).mean())
        ))
        Mex[f'{nombre_modelo}_Rec_{j+1}'] = indirectos_rec
        Mex[f'{nombre_modelo}_Gen_{j+1}'] = indirectos_gen
        Mex[f'{nombre_modelo}_Dir_{j+1}'] = directos
        Mex[f'{nombre_modelo}_Tot_{j+1}'] = totales
        Mex[f'{nombre_modelo}_O1_{j+1}'] = directa_1orden
        Mex[f'{nombre_modelo}_O2_{j+1}'] = distante_2mas
        pd.DataFrame(M, index=identificadores, columns=identificadores).to_csv(
            SALIDA / f'Impactos_matriz_{nombre_modelo}_{j+1}.csv',
            encoding='utf-8-sig')
    return matrices, pd.DataFrame(tabla)

matrices_sar, impacto_sar = calcular_impactos(sar.betas, 'SAR')
matrices_sdm, impacto_sdm = calcular_impactos(sdm.betas, 'SDM')
Impactos = pd.concat([impacto_sar, impacto_sdm], ignore_index=True)
if not np.allclose(Impactos.ADI + Impactos.AII, Impactos.ATI, atol=1e-9):
    raise AssertionError('Falló identidad: ADI + AII = ATI.')
if not np.allclose(Impactos.Indirecto_1orden + Impactos.Indirecto_2mas,
                   Impactos.AII, atol=1e-9):
    raise AssertionError('Los spillovers 1 orden y 2+ orden no suman AII.')
Impactos.to_csv(SALIDA / '12_impactos_ADI_AII_ATI.csv',
                index=False, encoding='utf-8-sig')
print('\n\nIMPACTOS MEDIOS (coeficientes por unidad de regresor)')
print(Impactos.round(5).to_string(index=False))

# Intervalos Monte Carlo: incertidumbre asintótica de coeficientes MLE

filas_mc = []
rng = np.random.default_rng(SEMILLA)
for nombre, reg in [('SAR', sar), ('SDM', sdm)]:
    mu = np.asarray(reg.betas).ravel()
    cov = np.asarray(reg.vm, dtype=float)
    cov = (cov + cov.T) / 2
    if cov.shape != (len(mu), len(mu)):
        raise ValueError(f'Covarianza de {nombre} no coincide con coeficientes.')
    val, vect = np.linalg.eigh(cov)
    if val.min() < -1e-6:
        raise ValueError(f'Covarianza del modelo {nombre} no PSD: {val.min()}')
    raiz = vect @ np.diag(np.sqrt(np.clip(val, 0, None)))
    extracciones = mu + rng.standard_normal((SIMULACIONES_IMPACTOS, len(mu))) @ raiz.T
    muestras = {v: [] for v in VARIABLES}
    descartadas = 0
    for b in extracciones:
        rho = float(b[-1])
        if not np.isfinite(rho) or abs(rho) >= 0.999:
            descartadas += 1
            continue
        A = np.linalg.solve(I - rho * W, I)
        for j, variable in enumerate(VARIABLES):
            beta = b[1+j]
            theta = b[1+k+j] if nombre == 'SDM' else 0.0
            M = A @ (beta * I + theta * W)
            adi = float(np.trace(M) / n)
            ati = float(M.sum() / n)
            muestras[variable].append([adi, ati-adi, ati])
    if descartadas > 0:
        print(f'{nombre}: descartadas {descartadas} simulaciones con rho fuera de (-1,1).')
    for variable, arr in muestras.items():
        arr = np.asarray(arr)
        if len(arr) < SIMULACIONES_IMPACTOS // 2:
            raise ValueError(f'Pocas simulaciones estables para {nombre}/{variable}.')
        for j, efecto in enumerate(['ADI', 'AII', 'ATI']):
            lo, hi = np.quantile(arr[:, j], [0.025, 0.975])
            filas_mc.append([nombre, variable, efecto, float(lo), float(hi), len(arr)])
MC = pd.DataFrame(filas_mc, columns=[
    'Modelo', 'Variable', 'Efecto', 'IC95_inf', 'IC95_sup', 'Simulaciones_validas'
])
MC.to_csv(SALIDA / '13_IC_impactos_MonteCarlo.csv',
          index=False, encoding='utf-8-sig')

# Ejercicio contrafactual local: subir pobreza +1pp en una entidad

ORIGEN = '32'  
idx_origen = identificadores.index(ORIGEN)
filas_shock = []
for nombre, matrices in [('SAR', matrices_sar), ('SDM', matrices_sdm)]:
    respuesta = matrices['Pobreza_2024'][:, idx_origen]
    for i in range(n):
        filas_shock.append([
            nombre, ORIGEN, identificadores[i], Mex.ENTIDAD.iloc[i],
            float(respuesta[i]), float(100*np.expm1(respuesta[i])),
            'directo' if i == idx_origen else ('vecino_1' if vecinas_1[i, idx_origen]
                                                else 'distancia_2mas')
        ])
Shock = pd.DataFrame(filas_shock, columns=[
    'Modelo', 'CVE_origen', 'CVE_destino', 'Entidad_destino',
    'Delta_ln_tasa', 'Delta_tasa_pct', 'Clase_vecindad'
])
Shock.to_csv(SALIDA / '14_escenario_pobreza_1pp_origen_01.csv',
             index=False, encoding='utf-8-sig')

# Exportaciones territoriales
columnas_ent = ['CVE_ENT', 'ENTIDAD', 'Ln_Incidencia'] + VARIABLES
columnas_extra = [c for c in Mex if c.startswith(('Resid_', 'Pred_', 'SAR_', 'SDM_'))]
Mex[columnas_ent + columnas_extra].to_csv(
    SALIDA / '15_indicadores_estatales_spillovers.csv',
    index=False, encoding='utf-8-sig')
Mex[['CVE_ENT', 'ENTIDAD', 'Ln_Incidencia'] + columnas_extra + ['geometry']].to_file(
    SALIDA / 'Mexico_SAR_SDM_2024.gpkg', driver='GPKG', index=False
)

def guardar(fig, nombre):
    for extension in ('png', 'svg'):
        fig.savefig(FIGURAS / f'{nombre}.{extension}', dpi=DPI,
                    bbox_inches='tight', facecolor=FONDO)
    if MOSTRAR:
        plt.show()
    plt.close(fig)

# A. Tres mapas de los residuos
fig, ejes = plt.subplots(1, 3, figsize=(17, 7))
res = Mex[['Resid_OLS', 'Resid_SAR', 'Resid_SDM']].to_numpy()
absmax = max(float(np.abs(res).max()), 1e-9)
for ax, mod in zip(ejes, ['OLS', 'SAR', 'SDM']):
    Mex.plot(column=f'Resid_{mod}', ax=ax, cmap='RdBu_r',
             norm=TwoSlopeNorm(vcenter=0, vmin=-absmax, vmax=absmax),
             linewidth=0.5, edgecolor=BLANCO)
    ax.set_title(mod, fontsize=13, color=AZUL, weight='bold')
    ax.set_axis_off()
fig.subplots_adjust(top=.75, bottom=.19, left=.02, right=.97, wspace=.02)
fig.text(.04, .96, 'ECONOMETRICS DATA LAB', color=NARANJA, fontsize=10, weight='bold')
fig.text(.04, .90, 'Residuos por entidad federativa', color=AZUL, fontsize=19, weight='bold')
fig.text(.04, .84, 'OLS frente a SAR y SDM | tasa ENVIPE 2024 (log)', color=GRIS)
fig.text(.04, .08, 'Fuente: ENVIPE 2025, INEGI y cartografía estatal. Pesos: '
         f'{METODO_PESOS} estandarizados. Escala común en los tres mapas.', fontsize=8, color=GRIS)
cax = fig.add_axes([.37,.15,.27,.018])
sm = plt.cm.ScalarMappable(norm=TwoSlopeNorm(vcenter=0,vmin=-absmax,vmax=absmax), cmap='RdBu_r')
fig.colorbar(sm, cax=cax, orientation='horizontal', label='Residuo (log)')
guardar(fig, '01_residuos_OLS_SAR_SDM')

# B. Mapas efectos recibidos vs generados, pobreza, SDM
colrec, colgen = 'SDM_Rec_1', 'SDM_Gen_1'
amp = max(float(np.abs(Mex[[colrec,colgen]].to_numpy()).max()), 1e-9)
fig, ejes = plt.subplots(1, 2, figsize=(15, 7))
for ax, col, titulo in zip(ejes, [colrec,colgen],
                            ['Spillover recibido', 'Spillover generado']):
    Mex.plot(column=col, ax=ax, cmap='RdBu_r',
             norm=TwoSlopeNorm(vcenter=0,vmin=-amp,vmax=amp),
             linewidth=.5, edgecolor=BLANCO)
    ax.set_title(titulo, color=AZUL, fontsize=12, weight='bold')
    ax.set_axis_off()
fig.subplots_adjust(top=.76,bottom=.22,left=.035,right=.96,wspace=.05)
fig.text(.05,.95,'ECONOMETRICS DATA LAB',color=NARANJA,fontsize=10,weight='bold')
fig.text(.05,.89,'Derrames espaciales de la pobreza',color=AZUL,fontsize=19,weight='bold')
fig.text(.05,.835,'SDM | efectos marginales indirectos por origen y destino',color=GRIS)
cax=fig.add_axes([.36,.175,.29,.022])
fig.colorbar(plt.cm.ScalarMappable(
    norm=TwoSlopeNorm(vcenter=0,vmin=-amp,vmax=amp),cmap='RdBu_r'),
    cax=cax,orientation='horizontal',label='Cambio en ln(incidencia) por 1 pp de pobreza')
fig.text(.05,.07, 'Recibido: suma por fila sin diagonal; generado: suma por columna sin diagonal. '
         f'W={METODO_PESOS}. No implican causalidad.',color=GRIS,fontsize=8)
guardar(fig, '02_SDM_spillovers_recibidos_generados')

# C. Mapa de impactos SDM sobre vecinos inmediatos y entidades más lejanas
cols=['SDM_O1_1','SDM_O2_1']
maxabs=max(float(np.abs(Mex[cols].to_numpy()).max()),1e-9)
fig,ejes=plt.subplots(1,2,figsize=(15,7))
for ax,col,titulo in zip(ejes,cols,['Vecinos inmediatos (1 salto)',
                                   'Entidades con 2 o más saltos']):
    Mex.plot(column=col,ax=ax,cmap='RdBu_r',
             norm=TwoSlopeNorm(vcenter=0,vmin=-maxabs,vmax=maxabs),
             linewidth=.5,edgecolor=BLANCO)
    ax.set_title(titulo,color=AZUL,fontsize=12,weight='bold')
    ax.set_axis_off()
fig.subplots_adjust(top=.77,bottom=.20,left=.035,right=.96,wspace=.05)
fig.text(.05,.95,'ECONOMETRICS DATA LAB',color=NARANJA,fontsize=10,weight='bold')
fig.text(.05,.89,'Alcance de los efectos indirectos',color=AZUL,fontsize=19,weight='bold')
fig.text(.05,.83,'SDM | pobreza, 2024 | distancia topológica en la red de reina',color=GRIS)
cax=fig.add_axes([.37,.145,.27,.020])
fig.colorbar(plt.cm.ScalarMappable(
    norm=TwoSlopeNorm(vcenter=0,vmin=-maxabs,vmax=maxabs),cmap='RdBu_r'),
    cax=cax,orientation='horizontal',label='Efecto indirecto por 1 pp')
fig.text(.05,.07,'Las distancias topológicas clasifican DESTINOS, no pasos causales exclusivos.',
         color=GRIS,fontsize=8)
guardar(fig,'03_SDM_efectos_vecinos_1_y_2mas')

# D. Efectos promedio de SAR y SDM por variable
fig,ejes=plt.subplots(1,k,figsize=(15,5),sharey=False)
for ax,variable in zip(ejes,VARIABLES):
    temp=Impactos.loc[Impactos.Variable==variable]
    indices=np.arange(3)
    colores=[AZUL,TURQUESA,NARANJA]
    for offset,mod in [(-.17,'SAR'),(.17,'SDM')]:
        vals=temp.loc[temp.Modelo==mod,['ADI','AII','ATI']].to_numpy().ravel()
        ax.bar(indices+offset,vals,width=.31,label=mod,color=AZUL if mod=='SAR' else NARANJA,
               alpha=.8)
    ax.axhline(0,color=GRIS,lw=.8)
    ax.set_xticks(indices,['Directo','Indirecto','Total'])
    ax.set_title(variable.replace('_',' '),fontsize=10,color=AZUL,weight='bold')
    ax.grid(axis='y',alpha=.25)
    ax.set_axisbelow(True)
handles=[Patch(facecolor=AZUL,label='SAR'),Patch(facecolor=NARANJA,label='SDM')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.13),ncol=2,frameon=False)
fig.subplots_adjust(left=.065,right=.98,top=.72,bottom=.27,wspace=.32)
fig.text(.065,.96,'ECONOMETRICS DATA LAB',color=NARANJA,weight='bold',fontsize=10)
fig.text(.065,.89,'Impactos directos e indirectos promedio',color=AZUL,fontsize=18,weight='bold')
fig.text(.065,.82,'SAR vs Spatial Durbin | efectos marginales agregados',color=GRIS)
fig.text(.065,.065,'Fuente: ENVIPE 2025 / INEGI. Resultados condicionales a W y a los regresores.',color=GRIS,fontsize=8)
guardar(fig,'04_comparacion_impactos_SAR_SDM')

# E. Mapa matriz completa: cambios de pobreza en origen j sobre destino i
M = matrices_sdm['Pobreza_2024']
fig,ax=plt.subplots(figsize=(12,10))
lim=max(float(np.max(np.abs(M))),1e-9)
im=ax.imshow(M,cmap='RdBu_r',norm=TwoSlopeNorm(vcenter=0,vmin=-lim,vmax=lim),aspect='equal')
ax.set_xticks(range(n),identificadores,rotation=90,fontsize=7)
ax.set_yticks(range(n),identificadores,fontsize=7)
ax.set_xlabel('Entidad ORIGEN del cambio en pobreza',color=GRIS)
ax.set_ylabel('Entidad DESTINO en incidencia',color=GRIS)
fig.subplots_adjust(left=.16,right=.92,top=.77,bottom=.14)
fig.text(.11,.96,'ECONOMETRICS DATA LAB',color=NARANJA,fontsize=10,weight='bold')
fig.text(.11,.90,'Matriz completa de spillovers',color=AZUL,fontsize=20,weight='bold')
fig.text(.11,.84,'SDM | pobreza (1 punto porcentual) | 32 x 32 entidades',color=GRIS)
fig.colorbar(im,ax=ax,fraction=.031,pad=.015,label='Derivada de ln(incidencia)')
fig.text(.11,.045,'Cada columna representa un shock local en una entidad. Diagonal: efectos directos.',fontsize=8,color=GRIS)
guardar(fig,'05_matriz_impactos_SDM_pobreza')

# F. Ejemplo shock de pobreza de la entidad de origen
fig,ax=plt.subplots(figsize=(13,6.8))
for mod,color,marker in [('SAR',AZUL,'o'),('SDM',NARANJA,'s')]:
    tmp=Shock.loc[Shock.Modelo==mod]
    ax.scatter(range(n),tmp.Delta_tasa_pct,s=45,color=color,marker=marker,label=mod,alpha=.85)
ax.axhline(0,color=GRIS,lw=.8)
ax.axvline(idx_origen,color=TURQUESA,lw=1,ls='--',label='Origen')
ax.set_xticks(range(n),identificadores,fontsize=8)
ax.set_ylabel('Cambio porcentual aproximado/exacto en tasa (%)',color=GRIS)
ax.set_xlabel('CVE_ENT de destino',color=GRIS)
ax.legend(frameon=False,ncol=3)
ax.grid(axis='y',alpha=.23)
fig.subplots_adjust(left=.10,right=.97,top=.73,bottom=.20)
fig.text(.085,.95,'ECONOMETRICS DATA LAB',color=NARANJA,fontsize=10,weight='bold')
fig.text(.085,.89,'Respuesta territorial ante un cambio localizado',color=AZUL,fontsize=17,weight='bold')
fig.text(.085,.81,f'Pobreza +1 pp solo en CVE_ENT={ORIGEN} | derivadas de SAR vs SDM',color=GRIS)
fig.text(.085,.06,'Escenario algebraico, no efecto causal identificado. % = 100×(exp(delta log)-1).',color=GRIS,fontsize=8)
guardar(fig,'06_shock_pobreza_en_una_entidad')

print('\nARCHIVOS GENERADOS EN:', SALIDA)
print('Matríz W, LM, LR, comparación, coeficientes, impactos, intervalos MC,')
print('impactos por entidad, matrices 32x32, escenarios, GeoPackage y 6 figuras PNG/SVG.')
