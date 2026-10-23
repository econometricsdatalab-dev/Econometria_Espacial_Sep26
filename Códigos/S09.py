# -*- coding: utf-8 -*-
"""
Curso Integral de Econometría 
Modulo: Econometría espacial
Tema: Modelo econométricos espaciales (SEM, SDEM y SARAR/SAC)
Sesión: 09
Fecha: 22/10/2026
Docente/Asesor: Alexis Adonai Morales Alberto
"""

# Modulos a cargar (completos)

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Clases directas de modulos 

from pathlib import Path
from matplotlib.patches import Patch
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from libpysal.weights import Queen, Rook, KNN
from esda.moran import Moran
from spreg import OLS, ML_Error, ML_Lag, GM_Combo, GM_Error_Het, LMtests


# Configuración inicial 

BASE = Path(
    "E:\Econometrics Data Lab\Curso integral de econometría\Econometría espacial\E_SP_SEP26"
    )

DATOS = BASE / 'Datos' / 'Delincuencia_Mexico_2024.csv'
if not DATOS.exists():
    DATOS = BASE / 'Delincuencia_Mexico_2024.csv'
MAPA = BASE / 'Mapas' / 'Mexico_ent' / '00ent.shp'
if not MAPA.exists() and MAPA.with_name('ent00.shp').exists():
    MAPA = MAPA.with_name('ent00.shp')
SALIDA = BASE / 'Resultados' / 'S09_SEM_SDEM_SARAR'
FIGURAS = SALIDA / 'Figuras'
FIGURAS.mkdir(parents=True, exist_ok=True)
METODO_W = 'reina'  # 'reina', 'torre', 'knn'
K_VECINOS = 4
PERMUTACIONES = 9999
SEMILLA = 2026
DPI = 500
MOSTRAR = False
NARANJA, AZUL, TURQUESA = '#F04A1D', '#003057', '#08989C'
FONDO, GRIS, GRIS_CLARO = '#F7F7F5', '#4D565E', '#DDE2E5'
plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':FONDO,
                     'axes.facecolor':FONDO,'savefig.facecolor':FONDO,
                     'font.size':10,'axes.unicode_minus':False})

# Lectura y validación

if not DATOS.exists():
    raise FileNotFoundError(f'Falta el archivo CSV: {DATOS}')
if not MAPA.exists():
    raise FileNotFoundError(f'Falta la cartografía {MAPA}. Incluye .shp/.shx/.dbf/.prj. No se construirá una matriz artificial.')
D = pd.read_csv(DATOS, encoding='utf-8-sig',dtype={'CVE_ENT':str})
D.columns = D.columns.str.strip()
D['CVE_ENT'] = D['CVE_ENT'].str.strip().str.zfill(2)
required = ['CVE_ENT','ENTIDAD','Tasa_Incidencia_ENVIPE_2024','Pobreza_2024',
            'Gasto_Hogar_2024','Desocupacion_2024T2']
missing = set(required) - set(D)
if missing:
    raise ValueError(f'Variables faltantes: {sorted(missing)}')
if len(D)!=32 or D.CVE_ENT.duplicated().any() or set(D.CVE_ENT)!={f'{i:02d}' for i in range(1,33)}:
    raise ValueError('Se requieren 32 claves CVE_ENT únicas: 01,...,32')
for col in required[2:]:
    D[col] = pd.to_numeric(D[col],errors='raise')
if D[required].isna().any().any() or (D['Tasa_Incidencia_ENVIPE_2024']<=0).any():
    raise ValueError('Valores faltantes o incidencia no positiva')
D['Ln_Incidencia'] = np.log(D['Tasa_Incidencia_ENVIPE_2024'])
D['Gasto_Miles'] = D['Gasto_Hogar_2024']/1000
X_VARS = ['Pobreza_2024','Gasto_Miles','Desocupacion_2024T2']

G = gpd.read_file(MAPA)
G.columns = G.columns.str.strip()
if G.crs is None:
    raise ValueError('El shapefile requiere CRS (archivo .prj)')
G['CVE_ENT'] = G.CVE_ENT.astype(str).str.strip().str.zfill(2)
if G.geometry.isna().any() or G.geometry.is_empty.any():
    raise ValueError('Geometrías vacías en mapa')
if not G.geometry.is_valid.all():
    raise ValueError('Corregir geometrías inválidas antes de construir W')
G = G[['CVE_ENT','geometry']].dissolve(by='CVE_ENT',as_index=False)
Mex = G.merge(D,on='CVE_ENT',how='inner',validate='one_to_one').sort_values('CVE_ENT').reset_index(drop=True)
if len(Mex)!=32:
    raise ValueError('No se empataron las 32 entidades')
Mex = Mex.to_crs(epsg=6372)
y = Mex[['Ln_Incidencia']].to_numpy(dtype=float)
X = Mex[X_VARS].to_numpy(dtype=float)

# Matrices de peso espacial 

if METODO_W=='reina':
    w = Queen.from_dataframe(Mex,use_index=True)
elif METODO_W=='torre':
    w = Rook.from_dataframe(Mex,use_index=True)
elif METODO_W=='knn':
    w = KNN.from_dataframe(Mex,k=K_VECINOS)
else:
    raise ValueError('METODO_W debe ser reina/torre/knn')
if w.islands:
    raise ValueError(f'Unidades sin vecinos: {w.islands}. Probar KNN o revisar geometría.')
w.transform = 'R'
W = w.full()[0]
np.testing.assert_allclose(W.sum(axis=1),1,atol=1e-10)
pd.DataFrame(W,index=Mex.CVE_ENT,columns=Mex.CVE_ENT).to_csv(SALIDA/'W_estandarizada.csv')

# OLS, LM y VIF 

ols = OLS(y,X,w=w,name_y='Ln_Incidencia',name_x=X_VARS,
          spat_diag=True,moran=True,nonspat_diag=True,white_test=True,vif=True)
print('\n'+'='*65+'\nOLS Y DIAGNÓSTICOS\n'+'='*65)
print(ols.summary)
lm = LMtests(ols,w,tests=['all'])
for name,attr in [('LM-Lag','lml'),('LM-Error','lme'),('Robust LM-Lag','rlml'),
                  ('Robust LM-Error','rlme'),('LM-WX','lmwx'),('LM conjunto Durbin','lmspdurbin')]:
    result = getattr(lm,attr,None)
    print(name,result)
    
    
# Estimación principal SEM y SDEM mediante ML 

sem = ML_Error(y,X,w=w,method='full',slx_lags=0,name_y='Ln_Incidencia',name_x=X_VARS)
sdem = ML_Error(y,X,w=w,method='full',slx_lags=1,name_y='Ln_Incidencia',name_x=X_VARS)
# SAR y SDM como referencias para comparación, estimados por ML
sar = ML_Lag(y,X,w=w,method='full',slx_lags=0,name_y='Ln_Incidencia',name_x=X_VARS)
sdm = ML_Lag(y,X,w=w,method='full',slx_lags=1,name_y='Ln_Incidencia',name_x=X_VARS)
# SARAR / SAC mediante método generalizado de momentos
# No comparar su AIC con ML sin justificación: estimadores no equivalentes.
sarar = GM_Combo(y,X,w=w,w_lags=2,name_y='Ln_Incidencia',name_x=X_VARS)
# SDEM-GMM con robustez frente a heterocedasticidad (comprobación complementaria)
sdem_het = GM_Error_Het(y,X,w=w,slx_lags=1,name_y='Ln_Incidencia',name_x=X_VARS)
modelos = {'OLS':ols,'SEM_ML':sem,'SDEM_ML':sdem,'SAR_ML':sar,
           'SDM_ML':sdm,'SARAR_GMM':sarar,'SDEM_GMM_HET':sdem_het}
for name,model in modelos.items():
    print('\n'+'='*65+'\n'+name+'\n'+'='*65)
    print(model.summary)
    (SALIDA/f'{name}_resumen.txt').write_text(str(model.summary),encoding='utf-8')

# Test LR 

LR = 2*(float(sdem.logll)-float(sem.logll))
LR = max(LR,0.)
df_lr = len(X_VARS)
p_lr = stats.chi2.sf(LR,df_lr)
comparacion = pd.DataFrame([{'Prueba':'LR SDEM vs SEM','H0':'Theta=0',
                             'Estadistico':LR,'gl':df_lr,'p_valor':p_lr}])
comparacion.to_csv(SALIDA/'Prueba_LR_SDEM_SEM.csv',index=False,encoding='utf-8-sig')
print('\nLR SDEM vs SEM:',comparacion.to_string(index=False))

# Diagnostico de residuos crudos e innovaciones filtradas 

results=[]
for name,m in modelos.items():
    u=np.asarray(m.u,dtype=float).reshape(-1)
    filtered=getattr(m,'e_filtered',None)
    e=np.asarray(filtered,dtype=float).reshape(-1) if filtered is not None else None
    # En SAR y SDM `u` es el error estructural después de incorporar Wy.
    if e is None: e=u.copy()
    for tipo,arr in [('residuo',u),('innovacion_filtrada',e)]:
        np.random.seed(SEMILLA)
        moran=Moran(arr,w,permutations=PERMUTACIONES,two_tailed=True)
        results.append({'Modelo':name,'Tipo':tipo,'Moran_I':moran.I,
                        'p_perm_bilateral_aprox':min(1.,2*float(moran.p_sim)),
                        'p_normal_bilateral':moran.p_norm})
        Mex[f'{name}_{"res" if tipo=="residuo" else "inn"}']=arr
# Nota: p_sim es cola extrema por permutación; 2*p_sim es aproximación
# bilateral y NO equivale al contraste bilateral exacto de permutaciones.
diagnosticos=pd.DataFrame(results)
diagnosticos.to_csv(SALIDA/'Moran_residuos.csv',index=False,encoding='utf-8-sig')
print('\nMORAN EN ERRORES\n',diagnosticos.to_string(index=False))

# Resumen comparativo 

mls={'SEM_ML':sem,'SDEM_ML':sdem,'SAR_ML':sar,'SDM_ML':sdm}
comp=[]
for name,m in mls.items():
    comp.append({'Modelo':name,'LogLik':float(m.logll),
                 'AIC':float(m.aic),'SC_BIC':float(m.schwarz)})
pd.DataFrame(comp).to_csv(SALIDA/'Comparacion_ML.csv',index=False,encoding='utf-8-sig')

# Exportación de la capa espacial

cols=['CVE_ENT','ENTIDAD','Ln_Incidencia']+X_VARS+[c for c in Mex if c.startswith(('OLS_','SEM_ML_','SDEM_ML_','SAR_ML_','SDM_ML_','SARAR_GMM_','Recibido_','Generado_'))]
Mex[cols+['geometry']].to_file(SALIDA/'Mexico_S09.gpkg',driver='GPKG',layer='S09_espacial',index=False)
Mex.drop(columns='geometry').to_csv(SALIDA/'Resultados_por_entidad.csv',index=False,encoding='utf-8-sig')

# Spillovers 

k=len(X_VARS)
coef=np.asarray(sdem.betas,dtype=float).reshape(-1)
if len(coef) != 2*k+2:
    raise ValueError(f'Orden de coeficientes SDEM inesperado: {len(coef)}. Revisar versión spreg.')
impactos=[]
for j,variable in enumerate(X_VARS):
    beta=float(coef[1+j]); theta=float(coef[1+k+j])
    S=beta*np.eye(len(Mex))+theta*W
    ad=float(np.trace(S)/len(Mex))
    ai=float((S.sum()-np.trace(S))/len(Mex))
    at=ad+ai
    impactos.append({'Variable':variable,'Directo_ADI':ad,'Indirecto_AII':ai,
                     'Total_ATI':at,'beta':beta,'theta':theta})
    pd.DataFrame(S,index=Mex.CVE_ENT,columns=Mex.CVE_ENT).to_csv(
        SALIDA/f'Matriz_impactos_{variable}.csv')
    # Efectos recibidos por i: suma de fila; generados por j: suma de columna.
    Mex[f'Recibido_{variable}']=S.sum(axis=1)-np.diag(S)
    Mex[f'Generado_{variable}']=S.sum(axis=0)-np.diag(S)
impactos=pd.DataFrame(impactos)
impactos.to_csv(SALIDA/'Impactos_SDEM.csv',index=False,encoding='utf-8-sig')
np.testing.assert_allclose(impactos.Total_ATI,impactos.Directo_ADI+impactos.Indirecto_AII)
print('\nIMPACTOS SDEM\n',impactos.to_string(index=False))


# Mapas finales 

def mapa(variable,titulo,subtitulo,archivo,unidad='Unidades del indicador'):
    fig,ax=plt.subplots(figsize=(12,8),dpi=500)
    fig.subplots_adjust(left=.05,right=.95,top=.80,bottom=.16)
    vals=Mex[variable]
    vmax=float(max(abs(vals.min()),abs(vals.max())))
    if vmax==0: vmax=1
    Mex.plot(column=variable,ax=ax,cmap='RdBu_r',vmin=-vmax,vmax=vmax,
             edgecolor='white',linewidth=.6,legend=True,
             legend_kwds={'shrink':.60,'label':unidad})
    Mex.boundary.plot(ax=ax,color=AZUL,linewidth=.25)
    ax.set_axis_off()
    fig.text(.06,.955,'ECONOMETRICS DATA LAB',color=NARANJA,weight='bold',fontsize=10)
    fig.text(.06,.91,titulo,color=AZUL,weight='bold',fontsize=20)
    fig.text(.06,.85,subtitulo,color=GRIS,fontsize=10)
    fig.text(.06,.085,'Fuente: ENVIPE 2025 (año del delito 2024), ENIGH, INEGI y cartografía estatal.\n'
             f'W: {METODO_W}, filas estandarizadas. Resultados condicionales al modelo; no implican causalidad.',
             color=GRIS,fontsize=8,va='top')
    for ext in ['png','svg']:
        fig.savefig(FIGURAS/f'{archivo}.{ext}',dpi=DPI,bbox_inches='tight')
    if MOSTRAR: plt.show()
    plt.close(fig)

for nombre in ['OLS','SEM_ML','SDEM_ML','SAR_ML','SDM_ML','SARAR_GMM']:
    mapa(f'{nombre}_res',f'Residuos: {nombre}','Incidencia delictiva estatal, 2024',f'01_residuos_{nombre}')
mapa('Recibido_Pobreza_2024','Spillovers recibidos: pobreza','Modelo SDEM | contribución marginal desde otras entidades','02_recibidos_pobreza')
mapa('Generado_Pobreza_2024','Spillovers generados: pobreza','Modelo SDEM | propagación marginal hacia otras entidades','03_generados_pobreza')
print(f'\nResultados guardados en: {SALIDA}')
