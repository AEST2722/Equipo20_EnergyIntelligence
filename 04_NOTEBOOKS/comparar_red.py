# Experimento: ¿mejora el modelo con las variables de red? Correr desde 04_NOTEBOOKS después de s09 y s10.
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import tdpa_comun as tc
from features_gravedad import agregar_variables_gravedad, NUEVAS as GRAV
from validacion_regiones import construir_folds_regiones

RED = ["red_flujo_paso", "red_grav_t2", "red_grav_t1", "red_tiempo_zmvm_min", "red_tiempo_zm500k_min"]
t, tr, te = tc.preparar_datos(tc.encontrar_raiz() / "03_PROCESSED" / "FE_TOP_Model_Table_v2.csv")
d = agregar_variables_gravedad(tr)
folds = construir_folds_regiones(d)
print("Spearman con log_tdpa:")
print(d[RED + GRAV + ["log_tdpa"]].corr(method="spearman")["log_tdpa"].drop("log_tdpa").round(3).to_string())

def cv(datos, est):
    e, o = [], pd.Series(np.nan, index=datos.index)
    for f, a, b in folds:
        p = Pipeline([("prep", tc.PreparadorSeguro()), ("m", est)]).fit(datos.loc[a], datos.loc[a, tc.TARGET])
        o[b] = p.predict(datos.loc[b])
        e.append(np.sqrt(mean_squared_error(datos.loc[b, tc.TARGET], o[b])))
    return round(np.mean(e), 4), round(np.std(e, ddof=1), 4), round(spearmanr(datos[tc.TARGET], o)[0], 3)

gb = lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.03, max_depth=4, min_samples_leaf=15,
                                           l2_regularization=1.0, early_stopping=False, random_state=20)
filas = []
for nombre, datos in [("Original", d.drop(columns=RED + GRAV)),
                      ("+ gravedad línea recta", d.drop(columns=RED)),
                      ("+ red vial", d.drop(columns=GRAV)),
                      ("+ gravedad + red vial", d)]:
    for mn, m in [("GBoost", gb), ("Ridge", lambda: Ridge(alpha=10))]:
        filas.append((mn, nombre, *cv(datos, m())))
print(pd.DataFrame(filas, columns=["modelo", "variables", "RMSE", "sd", "Spearman"]).to_string(index=False))
