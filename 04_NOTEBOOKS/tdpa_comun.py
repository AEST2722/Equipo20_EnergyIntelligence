"""Funciones compartidas del proyecto TDPA.

Extraídas del notebook Avance3.20(3).ipynb para evitar duplicar
el preprocesamiento y la validación en los nuevos experimentos.

Poner en la carpeta 04_NOTEBOOKS junto al notebook de comparación.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_regression
from sklearn.metrics.pairwise import haversine_distances
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler, RobustScaler

SEED = 20
BUFFER_KM = 40.0
TARGET = "log_tdpa"
CAT_COLS = ["road_tipo_vial", "road_circula", "road_cond_pav", "road_recubri", "road_administra", "road_nivel"]
ID_COLS = ["site_id", "state_file", "ruta", "carretera", "KM", "lugar_base", "cve_mun",
           "mun_assignment_method", "group_block25", "group_route", "group_state", "is_corridor_test"]
OBJETIVO = ["tdpa", "log_tdpa"]
SOLO_AFORO = ["pct_C", "pct_B", "n_te", "te_list", "n_calzadas", "red", "toll_road", "anio"]
COORDS = ["lat", "lon"]
CALIDAD = ["road_dist_m"]
NO_FEATURES = set(ID_COLS + OBJETIVO + SOLO_AFORO + COORDS + CALIDAD) | {"road_peaje", "fe_municipio_proxy"}


def encontrar_raiz():
    actual = Path.cwd().resolve()
    for carpeta in [actual, *actual.parents]:
        if (carpeta / "03_PROCESSED").is_dir() and (carpeta / "04_NOTEBOOKS").is_dir():
            return carpeta
    raise FileNotFoundError("Abre el repositorio que contiene 03_PROCESSED y 04_NOTEBOOKS")

def bool_seguro(s):
    valores = s.astype(str).str.strip().str.lower().map({
        "true": True, "false": False, "1": True, "0": False
    })
    if valores.isna().any():
        raise ValueError("is_corridor_test contiene valores no reconocidos")
    return valores.astype(bool)


def purgar_espacial(entrenamiento, validacion, buffer_km=BUFFER_KM):
    """Purge same 25-km block and nearby training sites (original Avance 2)."""
    conservar = ~entrenamiento.group_block25.isin(validacion.group_block25)
    if len(validacion) and buffer_km > 0:
        a = np.radians(entrenamiento[["lat", "lon"]].to_numpy(float))
        b = np.radians(validacion[["lat", "lon"]].to_numpy(float))
        dist = haversine_distances(a, b).min(axis=1) * 6371.0088
        conservar &= dist > buffer_km
    return entrenamiento.loc[conservar].copy()


def preparar_datos(archivo):
    t = pd.read_csv(archivo)
    necesarias = {"site_id", "is_corridor_test", "group_block25", "lat", "lon", "tdpa", TARGET}
    faltantes = necesarias.difference(t.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas indispensables: {sorted(faltantes)}")
    if t.site_id.isna().any() or not t.site_id.is_unique:
        raise ValueError("site_id debe ser único y no nulo")
    t["is_corridor_test"] = bool_seguro(t["is_corridor_test"])
    for nombre in ["lat", "lon", "tdpa", TARGET]:
        if not np.isfinite(t[nombre].to_numpy(float)).all():
            raise ValueError(f"Valores no finitos en {nombre}")
    if (t.tdpa <= 0).any():
        raise ValueError("El TDPA observado debe ser positivo")
    if t.group_block25.isna().any():
        raise ValueError("Faltan grupos espaciales")
    if not (t.lat.between(-90, 90) & t.lon.between(-180, 180)).all():
        raise ValueError("Coordenadas inválidas")
    global NO_FEATURES
    dominio = [c for c in t.columns if c.endswith("_min_month")]
    dominio += ["mun_agri_top_crop", "road_jurisdi"]
    NO_FEATURES = NO_FEATURES | set(dominio)
    test = t.loc[t.is_corridor_test].copy()
    pre = t.loc[~t.is_corridor_test].copy()
    if test.empty or pre.empty:
        raise ValueError("Se requieren entrenamiento y prueba externa")
    train = purgar_espacial(pre, test)
    if len(train) < 20:
        raise ValueError(f"Quedan sólo {len(train)} sitios de entrenamiento al purgar 40km")
    if train.group_block25.nunique() < 5:
        raise ValueError("Se requieren al menos 5 grupos espaciales distintos")
    print(f"Tabla maestra {t.shape} | train {len(train)} | test externo {len(test)} | purgados {len(pre)-len(train)}")
    return t, train, test


def detectar_log(df):
    """Inferir el inverso del objetivo usando sólo registros de DESARROLLO."""
    a = df[["tdpa", TARGET]].dropna()
    if a.empty or (a.tdpa <= 0).any():
        raise ValueError("tdpa debe ser positivo y conocido para invertir log_tdpa")
    y = a[TARGET].to_numpy(float)
    originals = a.tdpa.to_numpy(float)
    dif_log = np.max(np.abs(y - np.log(originals)))
    dif_log1p = np.max(np.abs(y - np.log1p(originals)))
    modo = "log" if dif_log <= dif_log1p else "log1p"
    error = min(dif_log, dif_log1p)
    if error > 0.025:
        raise ValueError(f"No se reconoce log_tdpa como log(tdpa) ni log1p(tdpa); diferencia={error:.5f}")
    return modo, float(error)


def invertir_log(x, modo):
    with np.errstate(over="raise", invalid="raise"):
        try:
            p = np.exp(x) if modo == "log" else np.expm1(x)
        except FloatingPointError as e:
            raise ValueError("Predicción log fuera de rango al regresar a vehículos/día") from e
    return np.asarray(p, dtype=float)

class PreparadorSeguro(BaseEstimator, TransformerMixin):
    """Todas las decisiones aprendidas viven en fit; transform nunca usa y."""
    def __init__(self, modo="Filtro + PCA", recortar=False, umbral_corr=0.85):
        self.modo=modo
        self.recortar=recortar
        self.umbral_corr=umbral_corr

    def _numericas(self,X):
        z=X.reindex(columns=self.num_entrada_).copy().replace([np.inf,-np.inf],np.nan)
        self_flags={}
        for c in self.continuas_:
            a=np.log1p(z[c]) if self.log_iqr_[c] else z[c]
            lo,hi=self.limites_iqr_[c]
            self_flags["atipico__"+c]=((a<lo)|(a>hi)).astype(float)
            if self.recortar and c in self.limites_recorte_:
                l,h=self.limites_recorte_[c]
                z[c]=z[c].clip(l,h)
        z=z.fillna(self.medianas_).fillna(0.0)
        return z,pd.DataFrame(self_flags,index=X.index)

    def fit(self,X,y):
        self.num_entrada_=[c for c in X.select_dtypes("number") if c not in NO_FEATURES and c not in CAT_COLS]
        raw=X[self.num_entrada_].replace([np.inf,-np.inf],np.nan)
        self.medianas_=raw.median().fillna(0.0)
        self.binarias_=[c for c in self.num_entrada_ if raw[c].dropna().nunique()<=2 and set(raw[c].dropna()).issubset({0,1})]
        # Los desconocidos binarios se imputan por moda del entrenamiento, nunca 0.5.
        for c in self.binarias_:
            moda=raw[c].mode()
            self.medianas_[c]=float(moda.iloc[0]) if len(moda) else 0.0
        self.continuas_=[c for c in self.num_entrada_ if c not in self.binarias_]
        self.log_iqr_={};self.limites_iqr_={};self.limites_recorte_={};aud=[]
        for c in self.continuas_:
            a=raw[c].dropna()
            self.log_iqr_[c]=len(a)>0 and a.min()>=0
            u=np.log1p(a) if self.log_iqr_[c] else a
            q1,q3=u.quantile([.25,.75]);iqr=q3-q1
            if np.isfinite(iqr) and iqr>0: lo,hi=q1-1.5*iqr,q3+1.5*iqr
            else: lo,hi=u.quantile([.005,.995])
            self.limites_iqr_[c]=(lo,hi)
            # No recortar proporciones, porcentajes, binarias o flags.
            protegida=("share" in c or c=="fe_concentracion_pob" or c.startswith("pct_") or c.startswith(("fe_invalido_","fe_estructural_","fe_desconocido_","fe_inconsistente_")))
            if not protegida and len(a): self.limites_recorte_[c]=tuple(a.quantile([.005,.995]))
            aud.append(dict(variable=c,n_observados=len(a),n_atipicos=int(((u<lo)|(u>hi)).sum()),
                escala_IQR="log1p" if self.log_iqr_[c] else "original",limite_IQR_inf=lo,limite_IQR_sup=hi,
                cuantile_inf=self.limites_recorte_.get(c,(np.nan,np.nan))[0],
                cuantile_sup=self.limites_recorte_.get(c,(np.nan,np.nan))[1],recortada=self.recortar and c in self.limites_recorte_))
        self.auditoria_atipicos_=pd.DataFrame(aud)
        z,flags=self._numericas(X)
        num=[c for c in self.num_entrada_ if z[c].nunique()>1 and not c.startswith("allsky_sfc_sw_dwn")]
        clima=[c for c in num if c.startswith(("prectotcorr","t2m","ws10m"))]
        self.pca_cols_=clima if self.modo=="Filtro + PCA" else []
        if self.pca_cols_: num=[c for c in num if c not in self.pca_cols_]
        yy=pd.Series(np.asarray(y),index=X.index)
        if self.modo!="Todas" and num:
            # Relevancia lineal dentro del fold (diagnóstico adicional MI en sección 4.3).
            _,p=f_regression(z[num],yy)
            relevantes=[c for c,pp in zip(num,p) if np.isfinite(pp) and pp<.05]
            if relevantes: num=relevantes
            rho=z[num].corrwith(yy,method="spearman").abs().fillna(0).sort_values(ascending=False,kind="stable")
            corr=z[num].corr(method="spearman").abs().fillna(0)
            elegidas=[]
            for c in rho.index:
                if not any(corr.loc[c,k]>=self.umbral_corr for k in elegidas): elegidas.append(c)
            num=elegidas
        self.num_seleccionadas_=num
        self.cat_seleccionadas_=[c for c in CAT_COLS if c in X and X[c].nunique()>1]
        if self.modo!="Todas":
            cats=[]
            for c in self.cat_seleccionadas_:
                gs=[yy.loc[g.index].to_numpy() for _,g in X.groupby(c) if len(g)>=5]
                if len(gs)>1:
                    _,p=stats.f_oneway(*gs)
                    if np.isfinite(p) and p<.05: cats.append(c)
            self.cat_seleccionadas_=cats
        self.flags_cols_=["atipico__"+c for c in self.continuas_ if c in num or c in self.pca_cols_]
        self.yj_={};self.scalers_={};self.fallos_yj_=[]
        for c in num:
            a=z[[c]].to_numpy(float)
            if c in self.binarias_: continue
            if abs(z[c].skew())>1:
                try:
                    pt=PowerTransformer(method="yeo-johnson",standardize=False).fit(a)
                    aa=pt.transform(a)
                    if not np.isfinite(aa).all(): raise ValueError("Yeo-Johnson no finito")
                    self.yj_[c]=pt;a=aa
                except (ValueError,RuntimeError,FloatingPointError,OverflowError): self.fallos_yj_.append(c)
            self.scalers_[c]=RobustScaler().fit(a)
        self.encoder_=None
        if self.cat_seleccionadas_:
            self.encoder_=OneHotEncoder(handle_unknown="infrequent_if_exist",min_frequency=10,sparse_output=False).fit(X[self.cat_seleccionadas_])
        self.pca_=None
        if self.pca_cols_:
            self.clima_scaler_=StandardScaler().fit(z[self.pca_cols_])
            self.pca_=PCA(n_components=.90,svd_solver="full").fit(self.clima_scaler_.transform(z[self.pca_cols_]))
        names=list(num)+self.flags_cols_
        if self.encoder_ is not None: names+=list(self.encoder_.get_feature_names_out(self.cat_seleccionadas_))
        if self.pca_ is not None: names += [f"clima_PC{i+1}" for i in range(self.pca_.n_components_)]
        self.feature_names_out_=np.asarray(names,dtype=object)
        if not len(names): raise ValueError("No hay características disponibles en el fold")
        return self

    def transform(self,X):
        requeridas=set(self.num_entrada_+self.cat_seleccionadas_)
        faltan=requeridas-set(X.columns)
        if faltan: raise ValueError(f"Faltan predictores en la tabla de entrada: {sorted(faltan)}")
        z,flags=self._numericas(X)
        partes=[]
        for c in self.num_seleccionadas_:
            a=z[[c]].to_numpy(float)
            if c in self.yj_: a=self.yj_[c].transform(a)
            if c in self.scalers_: a=self.scalers_[c].transform(a)
            partes.append(a)
        if self.flags_cols_: partes.append(flags[self.flags_cols_].to_numpy(float))
        if self.encoder_ is not None: partes.append(self.encoder_.transform(X[self.cat_seleccionadas_]))
        if self.pca_ is not None: partes.append(self.pca_.transform(self.clima_scaler_.transform(z[self.pca_cols_])))
        a=np.hstack(partes)
        if not np.isfinite(a).all(): raise ValueError("Valores no finitos después del preprocesamiento")
        return a

    def get_feature_names_out(self,input_features=None):
        return self.feature_names_out_

def construir_folds(datos):
    divisiones = GroupKFold(n_splits=5)
    folds = []
    for numero, (it, iv) in enumerate(divisiones.split(datos, datos[TARGET], datos["group_block25"]), 1):
        entrenamiento = purgar_espacial(datos.iloc[it], datos.iloc[iv])
        validacion = datos.iloc[iv]
        if len(entrenamiento) < 20:
            raise ValueError(f"Fold {numero}: quedan muy pocos sitios después de la purga")
        folds.append((numero, entrenamiento.index, validacion.index))
    return folds
