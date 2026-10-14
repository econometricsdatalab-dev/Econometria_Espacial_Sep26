# -*- coding: utf-8 -*-
"""
Curso Integral de Econometría 
Modulo: Econometría espacial
Tema: De OLS a modelo espacial
Sesión: 07
Fecha: 13/10/2026
Docente/Asesor: Alexis Adonai Morales Alberto
"""

# Modulos a cargar 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import statsmodels.api as sm 

# Clases directas 

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests
from pathlib import Path

# Rutas, parámetros y estilo visual 

# BASE = "/Users/nombredeusuario/Documentos/Proyectos/Trabajo/"
# También se puede designar manualmente

BASE = Path(__name__).resolve().parent if '__file__' in globals() else Path.cwd()
RUTA_DATOS = BASE / 'Datos' / 'Delincuencia_Mexico_2024.csv'
if not RUTA_DATOS.exists():
    RUTA_DATOS = BASE / 'Delincuencia_Mexico_2024.csv'

RUTA_MAPA = BASE / 'Mapas' / 'Mexico_ent' / '00ent.shp'
if not RUTA_MAPA.exists() and (BASE / 'Mapas' / 'Mexico_ent' / 'ent00.shp').exists():
    RUTA_MAPA = BASE / 'Mapas' / 'Mexico_ent' / 'ent00.shp'

SALIDA = BASE / 'Resultados' / 'Sesion07_OLS_Espacial'
FIGURAS = SALIDA / 'Figuras'
FIGURAS.mkdir(parents=True, exist_ok=True)

ALFA = 0.05
PERMUTACIONES = 9999
SEMILLA = 2026
DPI = 500
MOSTRAR = True  # Cambia a True en un IDE si deseas desplegar cada gráfico.

FONDO = '#F7F7F5'
BLANCO = '#FFFFFF'
AZUL = '#003057'
NARANJA = '#F04A1D'
TURQUESA = '#08989C'
GRIS = '#4D565E'
GRIS_CLARO = '#DEE4E7'
VIOLETA = '#9F2578'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Bahnschrift', 'Noto Sans', 'DejaVu Sans'],
    'figure.facecolor': FONDO,
    'axes.facecolor': FONDO,
    'axes.labelcolor': GRIS,
    'xtick.color': GRIS,
    'ytick.color': GRIS,
    'font.size': 10,
    'savefig.facecolor': FONDO,
    'axes.unicode_minus': False,
})


# Carga de datos 

Datos = pd.read_csv(RUTA_DATOS, 
                    dtype={'CVE_ENT': str}, 
                    encoding='utf-8-sig')
Datos.columns = Datos.columns.str.strip()

Datos['CVE_ENT'] = Datos['CVE_ENT'].str.strip().str.zfill(2)

Datos = Datos.sort_values('CVE_ENT').reset_index(drop=True)


# Evaluación de variables requeridas para el modelo 

requeridas = [
    'CVE_ENT', 'ENTIDAD', 'Tasa_Incidencia_ENVIPE_2024', 'Pobreza_2024',
    'Ingreso_Hogar_2024', 'Gasto_Hogar_2024', 'Desocupacion_2024T2'
]
if not set(requeridas).issubset(Datos.columns):
    raise ValueError('Faltan columnas: ' + str(set(requeridas)-set(Datos.columns)))
if len(Datos) != 32 or Datos['CVE_ENT'].duplicated().any():
    raise ValueError('Se esperaban exactamente 32 CVE_ENT distintas.')
if set(Datos['CVE_ENT']) != {str(i).zfill(2) for i in range(1, 33)}:
    raise ValueError('Las claves deben corresponder a 01, 02, ..., 32.')

for col in requeridas[2:]:
    Datos[col] = pd.to_numeric(Datos[col], errors='raise')
if Datos[requeridas].isna().any().any():
    raise ValueError('Existen valores faltantes en las variables del ejercicio.')
if (Datos['Tasa_Incidencia_ENVIPE_2024'] <= 0).any():
    raise ValueError('Las tasas deben ser positivas para aplicar logaritmos.')
if (Datos['Ingreso_Hogar_2024'] <= 0).any():
    raise ValueError('El ingreso debe ser positivo para aplicar logaritmos.')

print('\n[2] Datos y estadísticos descriptivos:')
print(Datos[requeridas].head().to_string(index=False))
print(Datos[requeridas[2:]].describe().round(2).to_string())


# Transformaciones y correlaciones 

'Datos['Ln_Incidencia'] = np.log(Datos['Tasa_Incidencia_ENVIPE_2024'])
Datos['Ln_Ingreso'] = np.log(Datos['Ingreso_Hogar_2024'])
Datos['Gasto_Miles'] = Datos['Gasto_Hogar_2024'] / 1000

variables_exploratorias = [
    'Ln_Incidencia', 'Pobreza_2024', 'Ln_Ingreso',
    'Gasto_Miles', 'Desocupacion_2024T2'
]
correlaciones = Datos[variables_exploratorias].corr()'
correlaciones.to_csv(SALIDA / '03_correlaciones.csv', encoding='utf-8-sig')
print('\n[3] Correlaciones:')
print(correlaciones.round(3).to_string())

# Especificación del modelo 

## A: Ln incidencia = pobreza + ln ingreso + desocupación

Y = Datos['Ln_Incidencia'].astype(float)
variables_A = ['Pobreza_2024', 'Ln_Ingreso', 'Desocupacion_2024T2']
X_A = sm.add_constant(Datos[variables_A].astype(float), has_constant='add')
OLS_A = sm.OLS(Y, X_A).fit()
OLS_A_HC3 = OLS_A.get_robustcov_results(cov_type='HC3')
print('\n[4] OLS modelo A: ingreso')
print(OLS_A.summary())
print('\n[4] Errores estándar HC3 (sensibilidad):')
print(OLS_A_HC3.summary())

## B: Sustituir ingreso por gasto promedio en miles 

variables_B = ['Pobreza_2024', 'Gasto_Miles', 'Desocupacion_2024T2']
X_B = sm.add_constant(Datos[variables_B].astype(float), has_constant='add')
OLS_B = sm.OLS(Y, X_B).fit()
OLS_B_HC3 = OLS_B.get_robustcov_results(cov_type='HC3')
print('\n[5] OLS modelo B: gasto')
print(OLS_B.summary())

modelos = pd.DataFrame({
    'Modelo': ['A (ingreso)', 'B (gasto)'],
    'N': [OLS_A.nobs, OLS_B.nobs],
    'R2': [OLS_A.rsquared, OLS_B.rsquared],
    'R2_ajustado': [OLS_A.rsquared_adj, OLS_B.rsquared_adj],
    'AIC': [OLS_A.aic, OLS_B.aic],
    'BIC': [OLS_A.bic, OLS_B.bic],
    'F': [OLS_A.fvalue, OLS_B.fvalue],
    'p_F': [OLS_A.f_pvalue, OLS_B.f_pvalue]
})
print('\n[5] Comparación:')
print(modelos.round(5).to_string(index=False))
modelos.to_csv(SALIDA / '01_comparacion_OLS.csv', index=False, encoding='utf-8-sig')

coeficientes = pd.concat([
    pd.DataFrame({'Modelo':'A (ingreso)', 'Variable':OLS_A.params.index,
                  'Coeficiente':OLS_A.params.values, 'EE_OLS':OLS_A.bse.values,
                  'p_OLS':OLS_A.pvalues.values, 'EE_HC3':OLS_A_HC3.bse,
                  'p_HC3':OLS_A_HC3.pvalues}),
    pd.DataFrame({'Modelo':'B (gasto)', 'Variable':OLS_B.params.index,
                  'Coeficiente':OLS_B.params.values, 'EE_OLS':OLS_B.bse.values,
                  'p_OLS':OLS_B.pvalues.values, 'EE_HC3':OLS_B_HC3.bse,
                  'p_HC3':OLS_B_HC3.pvalues})
], ignore_index=True)
coeficientes.to_csv(SALIDA / '02_coeficientes_OLS.csv', index=False, encoding='utf-8-sig')

# Diagnostico del modelo 

diagnosticos = []
vif_tablas = []

for nombre, modelo, X in [('A (ingreso)', OLS_A, X_A), ('B (gasto)', OLS_B, X_B)]:
    bp = het_breuschpagan(modelo.resid, X)
    white = het_white(modelo.resid, X)
    jb = stats.jarque_bera(modelo.resid)
    reset = linear_reset(modelo, power=2, use_f=True)

    diagnosticos.extend([
        (nombre, 'Breusch-Pagan: H0 homocedasticidad', bp[0], bp[1]),
        (nombre, 'White: H0 homocedasticidad', white[0], white[1]),
        (nombre, 'Jarque-Bera: H0 normalidad', jb.statistic, jb.pvalue),
        (nombre, 'RESET: H0 especificacion funcional correcta', float(reset.fvalue), float(reset.pvalue)),
    ])
    for j, variable in enumerate(X.columns):
        if variable != 'const':
            vif_tablas.append((nombre, variable, variance_inflation_factor(X.values, j)))

Diagnosticos = pd.DataFrame(diagnosticos, columns=['Modelo','Prueba_H0','Estadistico','p_valor'])
Diagnosticos['Decision_5pct'] = np.where(Diagnosticos['p_valor'] < ALFA, 'Rechazar H0', 'No rechazar H0')
VIF = pd.DataFrame(vif_tablas, columns=['Modelo', 'Variable', 'VIF'])
print('\n[6] Pruebas clásicas:')
print(Diagnosticos.round(4).to_string(index=False))
print('\n[6] Multicolinealidad:')
print(VIF.round(3).to_string(index=False))
Diagnosticos.to_csv(SALIDA / '04_diagnosticos_clasicos.csv', index=False, encoding='utf-8-sig')
VIF.to_csv(SALIDA / '05_VIF.csv', index=False, encoding='utf-8-sig')

# Residuos OLS y observaciones influyentes 

for sufijo, modelo in [('A', OLS_A), ('B', OLS_B)]:
    Datos['Pred_Ln_'+sufijo] = np.asarray(modelo.fittedvalues)
    Datos['Pred_Tasa_'+sufijo] = np.exp(np.asarray(modelo.fittedvalues))
    Datos['Resid_'+sufijo] = np.asarray(modelo.resid)
    Datos['Resid_Std_'+sufijo] = np.asarray(modelo.get_influence().resid_studentized_internal)
    Datos['Cook_'+sufijo] = np.asarray(modelo.get_influence().cooks_distance[0])

Datos.to_csv(SALIDA / '06_base_estimaciones_residuos.csv', index=False, encoding='utf-8-sig')
print('\n[7] Mayores distancias de Cook, modelo B:')
print(Datos[['CVE_ENT', 'ENTIDAD', 'Cook_B']].sort_values('Cook_B', ascending=False).head(5).to_string(index=False))

# Análisis de los residuales (errores) de los modelos 

fig, ax = plt.subplots(figsize=(11, 7.5))
fig.subplots_adjust(left=0.11, right=0.95, top=0.78, bottom=0.24)
ax.scatter(Y, OLS_A.fittedvalues, color=NARANJA, s=78, edgecolor=BLANCO,
           linewidth=1, alpha=0.88, label='A · Ingreso')
ax.scatter(Y, OLS_B.fittedvalues, color=TURQUESA, s=78, marker='s',
           edgecolor=BLANCO, linewidth=1, alpha=0.82, label='B · Gasto')
lim_inf = min(Y.min(), OLS_A.fittedvalues.min(), OLS_B.fittedvalues.min())-0.08
lim_sup = max(Y.max(), OLS_A.fittedvalues.max(), OLS_B.fittedvalues.max())+0.08
ax.plot([lim_inf, lim_sup], [lim_inf, lim_sup], ls='--', color=GRIS, linewidth=1.1)
ax.set_xlim(lim_inf, lim_sup)
ax.set_ylim(lim_inf, lim_sup)
ax.set_xlabel('Ln(incidencia) observada')
ax.set_ylabel('Ln(incidencia) estimada')
ax.legend(frameon=False)
ax.grid(alpha=.20)
fig.text(.08, .96, 'ECONOMETRICS DATA LAB', color=NARANJA, weight='bold', size=9)
fig.text(.08, .90, 'Incidencia observada y predicción OLS', color=AZUL, weight='bold', size=19)
fig.text(.08, .84, 'ENVIPE 2024 · 32 entidades · comparación de dos especificaciones', color=GRIS, size=10)
fig.text(.08, .10, 'La diagonal indica coincidencia perfecta. Los valores son logaritmos de tasas de incidencia.\n'
                 'Fuentes: INEGI (ENVIPE 2025, ENIGH 2024 y pobreza 2024).', color=GRIS, size=8)
fig.savefig(FIGURAS / '01_ajuste_OLS.png', dpi=DPI)
fig.savefig(FIGURAS / '01_ajuste_OLS.svg')
if MOSTRAR: plt.show()
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.subplots_adjust(left=.08, right=.95, top=.79, bottom=.16, hspace=.41, wspace=.23)
res = np.asarray(OLS_B.resid)
fit = np.asarray(OLS_B.fittedvalues)
axes[0,0].scatter(fit, res, color=TURQUESA, s=64, edgecolor=BLANCO)
axes[0,0].axhline(0, color=NARANJA, lw=1.25, ls='--')
axes[0,0].set_title('Residuos frente a valores ajustados', color=AZUL, weight='bold')
axes[0,0].set_xlabel('Valor ajustado')
axes[0,0].set_ylabel('Residuo')

stats.probplot(res, dist='norm', plot=axes[0,1])
axes[0,1].get_lines()[0].set_markerfacecolor(TURQUESA)
axes[0,1].get_lines()[0].set_markeredgecolor(AZUL)
axes[0,1].get_lines()[1].set_color(NARANJA)
axes[0,1].set_title('Gráfico Q–Q de residuos', color=AZUL, weight='bold')
axes[0,1].set_xlabel('Cuantil teórico')
axes[0,1].set_ylabel('Residuo ordenado')

axes[1,0].hist(res, bins=9, color=NARANJA, edgecolor=BLANCO, alpha=.82)
axes[1,0].set_title('Distribución residual', color=AZUL, weight='bold')
axes[1,0].set_xlabel('Residuo OLS')
axes[1,0].set_ylabel('Entidades')

orden = np.argsort(Datos['Cook_B'].to_numpy())[::-1][:7]
axes[1,1].barh(Datos.loc[orden, 'CVE_ENT'].iloc[::-1], Datos.loc[orden,'Cook_B'].iloc[::-1], color=AZUL)
axes[1,1].axvline(4/len(Datos), color=NARANJA, ls='--', lw=1.25, label='Referencia 4/n')
axes[1,1].set_title('Entidades con mayor distancia de Cook', color=AZUL, weight='bold')
axes[1,1].set_xlabel('Distancia de Cook')
axes[1,1].legend(frameon=False, fontsize=8)
for ax in axes.flat:
    ax.grid(alpha=.15)
    for spine in ax.spines.values(): spine.set_visible(False)
fig.text(.08, .96, 'ECONOMETRICS DATA LAB', color=NARANJA, weight='bold', size=9)
fig.text(.08, .91, 'Diagnóstico de regresión OLS', color=AZUL, weight='bold', size=19)
fig.text(.08, .85, 'Modelo B · ln incidencia ~ pobreza + gasto del hogar + desocupación', color=GRIS)
fig.text(.08, .065, 'La referencia 4/n para Cook es exploratoria. El valor Durbin–Watson no es una prueba de dependencia espacial.\n'
                 'Fuente: INEGI, indicadores estatales de 2024. Elaboración: Econometrics Data Lab.', color=GRIS, size=8)
fig.savefig(FIGURAS / '02_diagnosticos_OLS.png', dpi=DPI)
fig.savefig(FIGURAS / '02_diagnosticos_OLS.svg')
if MOSTRAR: plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(9.5, 8))
fig.subplots_adjust(left=.23, right=.94, top=.78, bottom=.20)
cor_X = Datos[['Pobreza_2024','Ln_Ingreso','Gasto_Miles','Desocupacion_2024T2']].corr()
visual = ax.imshow(cor_X, cmap=LinearSegmentedColormap.from_list('edl_div', [AZUL, BLANCO, NARANJA]), vmin=-1, vmax=1)
labels=['Pobreza', 'Ln ingreso', 'Gasto (miles)', 'Desocupación']
ax.set_xticks(range(4), labels=labels, rotation=30, ha='right')
ax.set_yticks(range(4), labels=labels)
for i in range(4):
    for j in range(4):
        ax.text(j,i,f'{cor_X.iloc[i,j]:.2f}',ha='center',va='center',color=BLANCO if abs(cor_X.iloc[i,j])>.65 else GRIS,weight='bold',fontsize=10)
for spine in ax.spines.values():spine.set_visible(False)
fig.text(.08,.96,'ECONOMETRICS DATA LAB',color=NARANJA,weight='bold',size=9)
fig.text(.08,.90,'Colinealidad entre variables económicas',color=AZUL,weight='bold',size=17)
fig.text(.08,.84,'Correlación de Pearson · 32 entidades federativas · 2024',color=GRIS,size=10)
fig.text(.08,.07,'Una correlación elevada entre regresores puede dificultar la interpretación parcial de los coeficientes.',color=GRIS,size=8)
fig.savefig(FIGURAS / '03_correlacion_regresores.png', dpi=DPI)
fig.savefig(FIGURAS / '03_correlacion_regresores.svg')
if MOSTRAR: plt.show()
plt.close(fig)

# Componente espacial 

pip install geopandas libpysal esda spreg

if RUTA_MAPA.exists():
    try:
        import geopandas as gpd
        from libpysal.weights import Queen, lag_spatial
        from esda.moran import Moran_Local
        from spreg import OLS as OLS_espacial
    except ImportError as e:
        print('\n[9] Faltan dependencias espaciales:', e)
        print('Instala: python -m pip install geopandas libpysal esda spreg')
    else:
        Mapa = gpd.read_file(RUTA_MAPA)
        Mapa.columns = Mapa.columns.str.strip()
        Mapa['CVE_ENT'] = Mapa['CVE_ENT'].astype(str).str.strip().str.zfill(2)
        if Mapa.crs is None:
            raise ValueError('El SHP carece de CRS. Verifica 00ent.prj.')
        if Mapa.geometry.isna().any() or Mapa.geometry.is_empty.any():
            raise ValueError('Existen geometrías faltantes o vacías.')
        # Si el SHP está desagregado por varias partes, se unifica por clave.
        Mapa = Mapa[['CVE_ENT','geometry']].dissolve(by='CVE_ENT').reset_index()
        Mex = Mapa.merge(Datos, on='CVE_ENT', how='inner', validate='one_to_one')
        Mex = Mex.sort_values('CVE_ENT').reset_index(drop=True)
        if len(Mex) != 32:
            raise ValueError('La unión cartográfica no conservó las 32 entidades.')
        Mex = Mex.to_crs(epsg=6372)

        # Matriz reina, con filas alineadas a la base 01..32.
        w = Queen.from_dataframe(Mex, use_index=True)
        if w.islands:
            raise ValueError('Entidades sin vecinos por reina: ' + str(w.islands))
        w.transform = 'R'
        print('\n[9] Matriz reina:', w.n, 'entidades; enlaces:', w.s0)

        # Estadísticos OLS espaciales: NO son coeficientes de un modelo SAR/SEM.
        # Se analiza el mismo modelo de referencia A y la alternativa B.
        PruebasEspaciales = []
        for nombre, exogenas in [('A (ingreso)',variables_A),('B (gasto)',variables_B)]:
            reg_esp = OLS_espacial(
                y=Mex[['Ln_Incidencia']].to_numpy(),
                x=Mex[exogenas].to_numpy(),
                w=w,
                name_y='Ln_Incidencia',
                name_x=exogenas,
                name_w='Reina_R',
                name_ds='Mexico_2024',
                spat_diag=True,
                moran=True,
                nonspat_diag=True,
                white_test=True,
                vif=True
            )
            (SALIDA / ('07_resumen_spreg_'+nombre[0]+'.txt')).write_text(reg_esp.summary,encoding='utf-8')
            pruebas = [
                ('Moran I residual ajustado (asintótico)', reg_esp.moran_res[0], reg_esp.moran_res[2], 'I de Moran; p asintótico'),
                ('LM-Lag',*reg_esp.lm_lag,'chi2(1)'),
                ('LM-Error',*reg_esp.lm_error,'chi2(1)'),
                ('Robust LM-Lag',*reg_esp.rlm_lag,'chi2(1)'),
                ('Robust LM-Error',*reg_esp.rlm_error,'chi2(1)'),
                ('LM-SARMA',*reg_esp.lm_sarma,'chi2(2)')
            ]
            for etiqueta,estad,p,ref in pruebas:
                PruebasEspaciales.append((nombre,etiqueta,float(estad),float(p),ref))
            print('\n[9] Modelo',nombre)
            print(reg_esp.summary)

        PruebasEspaciales = pd.DataFrame(PruebasEspaciales,columns=['Modelo','Prueba','Estadistico','p_valor','Referencia'])
        PruebasEspaciales['Decision_5pct'] = np.where(PruebasEspaciales.p_valor<ALFA,'Rechazar H0','No rechazar H0')
        PruebasEspaciales.to_csv(SALIDA/'08_pruebas_espaciales.csv',index=False,encoding='utf-8-sig')
        print('\n[9] Resumen de diagnósticos espaciales:')
        print(PruebasEspaciales.round(5).to_string(index=False))

        # Mapa del error estandarizado: Modelo B 

        lim = max(1, np.abs(Mex['Resid_Std_B']).max())
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
        mapa_colores = LinearSegmentedColormap.from_list('residuos_edl',[AZUL,BLANCO,NARANJA])
        fig,ax = plt.subplots(figsize=(13,9))
        fig.subplots_adjust(left=.05,right=.93,top=.81,bottom=.17)
        Mex.plot(column='Resid_Std_B',cmap=mapa_colores,norm=norm,ax=ax,
                 linewidth=.6,edgecolor=BLANCO,legend=False)
        Mex.boundary.plot(ax=ax,color=GRIS,lw=.22,alpha=.5)
        ax.axis('off')
        cax=fig.add_axes([.55,.155,.34,.023])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=mapa_colores),cax=cax,
                     orientation='horizontal').set_label('Residuo estandarizado · OLS B',color=GRIS)
        fig.text(.07,.955,'ECONOMETRICS DATA LAB',color=NARANJA,weight='bold')
        fig.text(.07,.905,'Distribución espacial de los residuos OLS',color=AZUL,weight='bold',fontsize=20)
        fig.text(.07,.855,'Incidencia delictiva estatal · México, 2024 · modelo B',color=GRIS,fontsize=11)
        fig.text(.07,.075,'Naranja: incidencia observada mayor a la estimada. Azul: menor.\n'
                 'Fuente: INEGI. Matriz W: contigüidad de reina para el diagnóstico inferencial.',color=GRIS,size=8)
        fig.savefig(FIGURAS/'04_mapa_residuos_OLS.png',dpi=DPI)
        fig.savefig(FIGURAS/'04_mapa_residuos_OLS.svg')
        if MOSTRAR:plt.show()
        plt.close(fig)
        
        # Dispersión de Moran de los residuos B 
        
        r = np.asarray(Mex['Resid_B'])
        z = (r-r.mean())/r.std(ddof=0)
        wz = lag_spatial(w,z)
        pendiente = float((z*wz).sum()/(z*z).sum())

        fig, ax = plt.subplots(figsize=(11,7.5))
        fig.subplots_adjust(left=.12,right=.95,top=.79,bottom=.22)
        ax.scatter(z,wz,color=TURQUESA,s=80,edgecolor=BLANCO,lw=1.0)
        ax.axhline(0,color=GRIS,lw=1)
        ax.axvline(0,color=GRIS,lw=1)
        xx=np.linspace(z.min()-0.2,z.max()+0.2,80)
        ax.plot(xx,pendiente*xx,color=NARANJA,lw=2)
        for idx in np.argsort(z*z+wz*wz)[-5:]:
            ax.annotate(Mex.loc[idx,'CVE_ENT'],(z[idx],wz[idx]),xytext=(5,5),
                        textcoords='offset points',color=AZUL,fontsize=9)
        ax.set_xlabel('Residuo OLS centrado y estandarizado')
        ax.set_ylabel('Rezago espacial del residuo estandarizado (Wz)')
        ax.grid(alpha=.15)
        fig.text(.08,.955,'ECONOMETRICS DATA LAB',color=NARANJA,weight='bold')
        fig.text(.08,.900,'Diagrama de dispersión de Moran: residuos',color=AZUL,weight='bold',fontsize=19)
        fig.text(.08,.845,f'Modelo B · matriz reina estandarizada · pendiente = {pendiente:.3f}',color=GRIS)
        fig.text(.08,.080,'La pendiente es un estadístico descriptivo; para la inferencia sobre los residuos OLS,\n'
                 'consulta el Moran residual ajustado de spreg en 08_pruebas_espaciales.csv.',color=GRIS,size=8)
        fig.savefig(FIGURAS/'05_dispersion_moran_residuos.png',dpi=DPI)
        fig.savefig(FIGURAS/'05_dispersion_moran_residuos.svg')
        if MOSTRAR:plt.show()
        plt.close(fig)


        # Correlación 
        
        
        lisa = Moran_Local(r, w, permutations=PERMUTACIONES, seed=SEMILLA, n_jobs=1, geoda_quads=True)
        p_ajustados = multipletests(lisa.p_sim, alpha=ALFA, method='fdr_bh')[1]
        etiquetas = np.array(['Alto–Alto','Bajo–Bajo','Bajo–Alto','Alto–Bajo'],dtype=object)
        colores_lisa = {
            'No significativo': GRIS_CLARO,
            'Alto–Alto': NARANJA, 'Bajo–Bajo': AZUL,
            'Bajo–Alto': VIOLETA, 'Alto–Bajo': TURQUESA
        }
        Mex['I_LISA_resid'] = lisa.Is
        Mex['p_LISA_resid'] = lisa.p_sim
        Mex['p_FDR_resid'] = p_ajustados
        Mex['Grupo_LISA_resid'] = np.where(p_ajustados<ALFA,etiquetas[lisa.q-1],'No significativo')

        fig,ax = plt.subplots(figsize=(13,9))
        fig.subplots_adjust(left=.05,right=.94,top=.81,bottom=.22)
        Mex.plot(ax=ax,color=Mex['Grupo_LISA_resid'].map(colores_lisa),
                 linewidth=.65,edgecolor=BLANCO)
        Mex.boundary.plot(ax=ax,color=GRIS,lw=.24,alpha=.5)
        ax.axis('off')
        leyenda=[Patch(facecolor=colores_lisa[k],label=k) for k in colores_lisa]
        fig.legend(handles=leyenda,loc='lower center',ncol=3,frameon=False,
                   bbox_to_anchor=(.5,.15),fontsize=9)
        fig.text(.07,.955,'ECONOMETRICS DATA LAB',color=NARANJA,weight='bold')
        fig.text(.07,.905,'LISA exploratorio de los residuos',color=AZUL,weight='bold',fontsize=20)
        fig.text(.07,.855,'OLS modelo B · Moran local con FDR-BH · México 2024',color=GRIS)
        fig.text(.07,.072,'Exploratorio: p por permutación de residuos no equivale a una inferencia ajustada por la estimación OLS.\n'
                 'Los clusters no deben interpretarse como efectos causales.',color=GRIS,size=8)
        fig.savefig(FIGURAS/'06_LISA_residuos_exploratorio.png',dpi=DPI)
        fig.savefig(FIGURAS/'06_LISA_residuos_exploratorio.svg')
        if MOSTRAR:plt.show()
        plt.close(fig)
        Mex[['CVE_ENT','ENTIDAD','Pobreza_2024','Ln_Incidencia',
             'Resid_A','Resid_B','Resid_Std_B','I_LISA_resid',
             'p_LISA_resid','p_FDR_resid','Grupo_LISA_resid','geometry']].to_file(
                 SALIDA/'09_residuos_y_LISA_Mexico_2024.gpkg',driver='GPKG')
        Mex.drop(columns='geometry').to_csv(SALIDA/'09_residuos_y_LISA_tabla.csv',index=False,encoding='utf-8-sig')

else:
    print('\n[9] FALTA CARTOGRAFÍA:',RUTA_MAPA)
    print('Los resultados OLS sí se calcularon. Coloca los archivos .shp/.shx/.dbf/.prj')
    print('en Mapas/Mexico_ent/ y vuelve a ejecutar para Moran residual y LM.')

print('\nResultados guardados en:',SALIDA)





