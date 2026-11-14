# -*- coding: utf-8 -*-
"""
Curso Integral de Econometría 
Modulo: Econometría espacial
Tema: Modelos espaciales combinados (SARMA/SARAR, GNS y criterios AIC-BIC)
Sesión: 10
Fecha: 27/10/2026
Docente/Asesor: Alexis Adonai Morales Alberto
"""

# Modulos a cargar (completos)

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Clases directas de modulos 

from pathlib import Path
from matplotlib.patches import Patch
from scipy.stats import chi2
from libpysal.weights import Queen, Rook, KNN
from esda.moran import Moran
from spreg import OLS, ML_Lag, ML_Error, GM_Combo_Het, LMtests
from statsmodels.api import OLS as OLS_SM, add_constant


# Configuración inicial 

BASE = Path(__file__).resolve().parent
RUTA_CSV = BASE / 'Delincuencia_Mexico_2024.csv'
if not RUTA_CSV.exists():
    RUTA_CSV = BASE.parent / 'Delincuencia_Mexico_2024.csv'
RUTA_MAPA = BASE / 'Mapas' / 'Mexico_ent' / '00ent.shp'
if not RUTA_MAPA.exists():
    RUTA_MAPA = BASE.parent / 'Mapas' / 'Mexico_ent' / '00ent.shp'
SALIDA = BASE / 'Resultados_S10'
SALIDA.mkdir(exist_ok=True)
SEMILLA, PERMUTACIONES = 2026, 9999
COLORES = {'OLS':'#4D565E','SAR':'#F04A1D','SEM':'#003057',
           'SDM':'#08989C','SDEM':'#9F2578','SARAR':'#B97918','GNS':'#303030'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})

# Lectura y validación de la base de datos
D = pd.read_csv(RUTA_CSV, encoding='utf-8-sig', dtype={'CVE_ENT':str})
D.columns = D.columns.str.strip()
D['CVE_ENT'] = D.CVE_ENT.str.zfill(2)
cols = ['Tasa_Incidencia_ENVIPE_2024','Pobreza_2024','Gasto_Hogar_2024','Desocupacion_2024T2']
assert len(D) == 32 and D.CVE_ENT.is_unique and set(D.CVE_ENT) == {f'{i:02d}' for i in range(1,33)}
for c in cols: D[c] = pd.to_numeric(D[c],errors='raise')
assert D[cols].notna().all().all() and (D['Tasa_Incidencia_ENVIPE_2024'] > 0).all()
D['Ln_Incidencia'] = np.log(D['Tasa_Incidencia_ENVIPE_2024'])
D['Gasto_Miles'] = D['Gasto_Hogar_2024'] / 1000
X_VARS = ['Pobreza_2024','Gasto_Miles','Desocupacion_2024T2']
D = D.sort_values('CVE_ENT').reset_index(drop=True)
print('\nDescriptivos (n=32):\n', D[['Ln_Incidencia']+X_VARS].describe().round(3))

# Estimación OLS como modelo de referencia
ols_sm = OLS_SM(D['Ln_Incidencia'],add_constant(D[X_VARS])).fit()
print('\nOLS NO ESPACIAL\n',ols_sm.summary())
D['Residuo_OLS'] = ols_sm.resid
D.to_csv(SALIDA/'01_base_y_residuos_OLS.csv',index=False,encoding='utf-8-sig')

# Lectura y validación de la cartografía
if not RUTA_MAPA.exists():
    raise FileNotFoundError(
        'No se encontró el shapefile oficial. Copia 00ent.shp/.shx/.dbf/.prj en: '
        + str(RUTA_MAPA.parent) + '. El OLS ya quedó exportado; NO se inventará W.')
G = gpd.read_file(RUTA_MAPA)
if G.crs is None: raise ValueError('Falta proyección de la cartografía (.prj).')
G['CVE_ENT'] = G['CVE_ENT'].astype(str).str.strip().str.zfill(2)
if G.geometry.isna().any() or G.geometry.is_empty.any() or not G.geometry.is_valid.all():
    raise ValueError('Revisar geometrías vacías o inválidas antes de estimar.')
G = G[['CVE_ENT','geometry']].dissolve(by='CVE_ENT',as_index=False)
Mex = G.merge(D,on='CVE_ENT',how='inner',validate='one_to_one').sort_values('CVE_ENT').reset_index(drop=True)
assert len(Mex)==32 and (Mex.CVE_ENT.values==D.CVE_ENT.values).all()
Mex = Mex.to_crs(epsg=6372)
y = Mex[['Ln_Incidencia']].to_numpy()
X = Mex[X_VARS].to_numpy()

# Matrices de peso espacial
w = Queen.from_dataframe(Mex,use_index=True)
if w.islands:
    raise ValueError(f'Reina tiene unidades sin vecinos: {w.islands}. No se deben ocultar.')
w.transform='R'
W=w.full()[0]
np.testing.assert_allclose(W.sum(axis=1), np.ones(len(Mex)), atol=1e-12)
pd.DataFrame(W,index=Mex.CVE_ENT,columns=Mex.CVE_ENT).to_csv(SALIDA/'02_W_reina.csv')
print('Vecinos mínimos/máximos:',w.min_neighbors,w.max_neighbors)

# OLS espacial y pruebas LM
ols = OLS(y,X,w=w,name_y='Ln_Incidencia',name_x=X_VARS,
          nonspat_diag=True,spat_diag=True,moran=True,white_test=True,vif=True)
print(ols.summary)
lm = LMtests(ols,w,tests=['all'])
pruebas_lm=[]
for nombre,atributo in [('LM-Error','lme'),('LM-Lag','lml'),
                         ('Robust LM-Error','rlme'),('Robust LM-Lag','rlml'),
                         ('LM-SARMA conjunto','sarma'),('LM-WX','lmwx'),
                         ('LM-SDM conjunto','lmspdurbin')]:
    v=getattr(lm,atributo,None)
    if v is not None:
        pruebas_lm.append({'Prueba':nombre,'Estadistico':float(v[0]),'p_valor':float(v[1])})
LM_TABLA=pd.DataFrame(pruebas_lm)
print('\nPruebas LM:\n', LM_TABLA.to_string(index=False))
LM_TABLA.to_csv(SALIDA/'03_pruebas_LM.csv',index=False,encoding='utf-8-sig')

# Estimación de modelos SAR, SEM, SDM y SDEM mediante ML
# Misma Y, regresores básicos, W y n. OLS SM incluye intercepto y sigma² en llf.
sar = ML_Lag(y,X,w=w,method='full',slx_lags=0,name_x=X_VARS)
sem = ML_Error(y,X,w=w,method='full',slx_lags=0,name_x=X_VARS)
sdm = ML_Lag(y,X,w=w,method='full',slx_lags=1,spat_impacts=['full'],name_x=X_VARS)
sdem = ML_Error(y,X,w=w,method='full',slx_lags=1,name_x=X_VARS)
modelos_ml={'OLS':ols_sm,'SAR':sar,'SEM':sem,'SDM':sdm,'SDEM':sdem}
for nombre,m in modelos_ml.items():
    if nombre!='OLS':
        print('\n',nombre,'\n',m.summary)
        (SALIDA/f'04_resumen_{nombre}.txt').write_text(str(m.summary),encoding='utf-8')

# Estimación de SARAR/SAC mediante GMM
sarar = GM_Combo_Het(y,X,w=w,w_lags=2,step1c=True,max_iter=2,
                     name_y='Ln_Incidencia',name_x=X_VARS)
print('\nSARAR / SAC (GMM robusto)\n',sarar.summary)
(SALIDA/'05_resumen_SARAR_GMM.txt').write_text(str(sarar.summary),encoding='utf-8')

# Estimación del modelo espacial general GNS
# Alto riesgo de sobreparametrización con n=32: estimación exploratoria.
try:
    gns = GM_Combo_Het(y,X,w=w,w_lags=2,slx_lags=1,step1c=True,max_iter=2,
                       name_y='Ln_Incidencia',name_x=X_VARS)
    print('\nGNS (GMM exploratorio)\n',gns.summary)
    (SALIDA/'06_resumen_GNS_GMM.txt').write_text(str(gns.summary),encoding='utf-8')
except (ValueError, np.linalg.LinAlgError) as exc:
    gns=None
    print('GNS no estimable de manera estable con esta muestra:',exc)

# Comparación de especificaciones: AIC y BIC
# k = cantidad de coeficientes de regresión + parámetros espaciales + varianza sigma².
comparacion=[]
for nombre,m in modelos_ml.items():
    loglik=float(ols_sm.llf if nombre=='OLS' else m.logll)
    k = int(len(ols_sm.params)+1 if nombre=='OLS' else len(m.betas)+1)
    aic=-2*loglik+2*k
    bic=-2*loglik+np.log(len(y))*k
    comparacion.append({'Modelo':nombre,'n':len(y),'k_incluye_sigma2':k,
                        'LogLik':loglik,'AIC':aic,'BIC':bic})
CRITERIOS=pd.DataFrame(comparacion).sort_values('AIC').reset_index(drop=True)
CRITERIOS['Delta_AIC']=CRITERIOS.AIC-CRITERIOS.AIC.min()
CRITERIOS['Delta_BIC']=CRITERIOS.BIC-CRITERIOS.BIC.min()
print('\nCOMPARACIÓN ML (AIC/BIC)\n',CRITERIOS.round(4).to_string(index=False))
CRITERIOS.to_csv(SALIDA/'07_comparacion_AIC_BIC_ML.csv',index=False,encoding='utf-8-sig')

# Pruebas de razón de verosimilitudes (LR)
# No usar LR entre modelos no anidados ni GMM vs ML.
pruebas_lr=[]
for nombre0,nombre1,df in [('OLS','SAR',1),('OLS','SEM',1),('SAR','SDM',len(X_VARS)),
                            ('SEM','SDEM',len(X_VARS))]:
    l0=float(ols_sm.llf if nombre0=='OLS' else modelos_ml[nombre0].logll)
    l1=float(modelos_ml[nombre1].logll)
    LR=max(0.0,2*(l1-l0))
    pruebas_lr.append({'Modelo_restringido':nombre0,'Modelo_general':nombre1,
                       'LR':LR,'gl':df,'p_valor_asintotico':chi2.sf(LR,df)})
LR_TABLA=pd.DataFrame(pruebas_lr)
print('\nLR anidados:\n',LR_TABLA.to_string(index=False))
LR_TABLA.to_csv(SALIDA/'08_LR_anidados.csv',index=False,encoding='utf-8-sig')

# Diagnóstico de autocorrelación espacial en residuos
# GMM: errores y_filtered / e_filtered difieren por especificación y versión.
residuos=[]
for nombre,m in [('OLS',ols),('SAR',sar),('SEM',sem),('SDM',sdm),('SDEM',sdem),('SARAR',sarar)]:
    for tipo,arr in [('u',getattr(m,'u',None)),('innovacion',getattr(m,'e_filtered',None))]:
        if arr is None: continue
        arr=np.asarray(arr).ravel()
        if len(arr)!=len(y):continue
        np.random.seed(SEMILLA)
        mor=Moran(arr,w,permutations=PERMUTACIONES,two_tailed=True)
        residuos.append({'Modelo':nombre,'Tipo':tipo,'I':mor.I,
                         'p_sim_unilateral_cola_extrema':mor.p_sim,
                         'p_normal_aprox_2colas':mor.p_norm})
        if tipo=='u': Mex[f'Res_{nombre}']=arr
MORAN=pd.DataFrame(residuos)
print('\nMORAN RESIDUAL:\n',MORAN.to_string(index=False))
MORAN.to_csv(SALIDA/'09_moran_residuos.csv',index=False,encoding='utf-8-sig')

# Efectos espaciales: impactos directos e indirectos
# En extensiones GMM, comprobar nombres y orden de los coeficientes.
# Para SAR y SDM de ML_Lag, rho y coeficientes tienen atributos definidos.
I=np.eye(len(y))
for nombre,m,con_wx in [('SAR',sar,False),('SDM',sdm,True)]:
    rho=float(m.rho)
    S=np.linalg.solve(I-rho*W,I)
    b=np.asarray(m.betas).ravel()
    kx=len(X_VARS)
    for j,var in enumerate(X_VARS):
        beta=float(b[1+j])
        theta=float(b[1+kx+j]) if con_wx else 0.0
        M=S@(beta*I+theta*W)
        ad=float(np.trace(M)/len(y))
        at=float(np.mean(M.sum(axis=1)))
        ai=at-ad
        print(f'{nombre}: {var}: directo={ad:.5f}, indirecto={ai:.5f}, total={at:.5f}')
        pd.DataFrame(M,index=Mex.CVE_ENT,columns=Mex.CVE_ENT).to_csv(
            SALIDA/f'10_impactos_{nombre}_{var}.csv')

# Mapas y gráficos comparativos
fig,axs=plt.subplots(2,3,figsize=(15,9),facecolor='#F7F7F5')
for ax,nombre in zip(axs.flat,['OLS','SAR','SEM','SDM','SDEM','SARAR']):
    values=Mex[f'Res_{nombre}']
    limite=max(abs(values.min()),abs(values.max()))
    Mex.plot(column=f'Res_{nombre}',ax=ax,cmap='RdBu_r',vmin=-limite,vmax=limite,
             edgecolor='white',linewidth=.5)
    ax.set_title(nombre,color='#003057',fontweight='bold')
    ax.axis('off')
fig.subplots_adjust(top=.82,bottom=.11,hspace=.13,wspace=.07)
fig.text(.055,.95,'ECONOMETRICS DATA LAB',color='#F04A1D',fontsize=10,weight='bold')
fig.text(.055,.905,'Residuos por especificación espacial',fontsize=20,color='#003057',weight='bold')
fig.text(.055,.86,'México · incidencia delictiva 2024 · modelo con tres regresores',fontsize=10,color='#4D565E')
fig.text(.055,.035,'Fuente: ENVIPE/INEGI y base integrada EDL. Pesos: reina. Escala individual por panel; comparar con cautela.',fontsize=8,color='#4D565E')
fig.savefig(SALIDA/'11_mapas_residuos.png',dpi=300,bbox_inches='tight')
fig.savefig(SALIDA/'11_mapas_residuos.svg',bbox_inches='tight')
plt.close(fig)

fig,ax=plt.subplots(figsize=(10,5),facecolor='#F7F7F5')
orden=CRITERIOS.sort_values('AIC',ascending=False)
ax.barh(orden.Modelo,orden.Delta_AIC,color=[COLORES[v] for v in orden.Modelo])
ax.set_xlabel('Diferencia respecto al AIC mínimo (menor es mejor)')
ax.set_title('Comparación penalizada de modelos ML',color='#003057',weight='bold')
ax.spines[['top','right']].set_visible(False)
fig.text(.10,.01,'No incluye SARAR ni GNS estimados por GMM.',fontsize=9,color='#4D565E')
fig.savefig(SALIDA/'12_delta_AIC.png',dpi=300,bbox_inches='tight')
fig.savefig(SALIDA/'12_delta_AIC.svg',bbox_inches='tight')
plt.close(fig)

# Exportación de resultados cartográficos y cierre
Mex.to_file(SALIDA/'13_mapa_modelos.gpkg',driver='GPKG')
print('\nResultados exportados a:',SALIDA.resolve())
