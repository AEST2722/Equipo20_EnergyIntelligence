# Experimento: ¿cambia el modelo de tránsito con el uso de suelo? Correr desde 04_NOTEBOOKS después de s11 y s12.
# Nota: el uso de suelo es sobre todo para CARACTERIZAR los CTN (territorio); aquí sólo se revisa si además ayuda al tránsito.
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import tdpa_comun as tc
from validacion_regiones import construir_folds_regiones

t, tr, te = tc.preparar_datos(tc.encontrar_raiz() / "03_PROCESSED" / "FE_TOP_Model_Table_v3.csv")
USV = [c for c in tr.columns if c.startswith("usv_") and c != "usv_clase_sitio"]
folds = construir_folds_regiones(tr)
print("Spearman con log_tdpa (variables de uso de suelo):")
print(tr[USV + ["log_tdpa"]].corr(method="spearman")["log_tdpa"].drop("log_tdpa").sort_values().round(3).to_string())

def cv(datos, est):
    e, o = [], pd.Series(np.nan, index=datos.index)
    for f, a, b in folds:
        p = Pipeline([("prep", tc.PreparadorSeguro()), ("m", est)]).fit(datos.loc[a], datos.loc[a, tc.TARGET])
        o[b] = p.predict(datos.loc[b])
        e.append(np.sqrt(mean_squared_error(datos.loc[b, tc.TARGET], o[b])))
    return round(np.mean(e), 4), round(np.std(e, ddof=1), 4), round(spearmanr(datos[tc.TARGET], o)[0], 3)

gb = lambda: HistGradientBoostingRegressor(learning_rate=0.03, max_iter=150, min_samples_leaf=20, early_stopping=False, random_state=20)
filas = []
for nombre, datos in [("v2 (sin uso de suelo)", tr.drop(columns=USV)), ("v3 (con uso de suelo)", tr)]:
    for mn, m in [("GBoost", gb), ("Ridge", lambda: Ridge(alpha=20))]:
        filas.append((mn, nombre, *cv(datos, m())))
print(pd.DataFrame(filas, columns=["modelo", "variables", "RMSE", "sd", "Spearman"]).to_string(index=False))
