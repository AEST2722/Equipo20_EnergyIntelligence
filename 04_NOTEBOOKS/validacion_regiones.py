"""Validación espacial por regiones contiguas (propuesta para el Avance 5).

Problema de la validación anterior: los bloques de 25 km de cada pliegue quedaban dispersos
por toda la zona; al purgar 40 km alrededor de cada sitio de validación se eliminaba ~80% del
entrenamiento (101-176 de 703 sitios por pliegue) y en algunos pliegues no quedaba ningún
sitio de alto flujo. El modelo final se entrena con los 703 sitios, así que esa validación
evaluaba un modelo muy distinto (y peor) que el que se va a usar.

Solución: agrupar los sitios en regiones geográficas CONTIGUAS (k-means sobre coordenadas en
km) y asignar regiones completas a cada pliegue. Se conserva la misma purga de 40 km, así que
ningún sitio de entrenamiento queda a menos de 40 km de uno de validación (la autocorrelación
del TDPA desaparece hacia los 25 km, según el EDA).
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold

from tdpa_comun import purgar_espacial


def regiones_contiguas(datos, n_regiones=15, seed=20):
    """Etiqueta de región por sitio. Coordenadas aproximadas en km (centradas en la zona)."""
    xy = np.c_[(datos.lon - datos.lon.mean()) * 104.6, (datos.lat - datos.lat.mean()) * 111.0]
    return KMeans(n_clusters=n_regiones, n_init=10, random_state=seed).fit_predict(xy)


def construir_folds_regiones(datos, n_regiones=15, buffer_km=40.0, n_splits=5, seed=20):
    """Igual formato que tdpa_comun.construir_folds: lista de (numero, idx_train, idx_valid)."""
    reg = regiones_contiguas(datos, n_regiones, seed)
    folds = []
    for numero, (it, iv) in enumerate(GroupKFold(n_splits=n_splits).split(datos, groups=reg), 1):
        entrenamiento = purgar_espacial(datos.iloc[it], datos.iloc[iv], buffer_km=buffer_km)
        if len(entrenamiento) < 20:
            raise ValueError(f"Fold {numero}: quedan muy pocos sitios después de la purga")
        folds.append((numero, entrenamiento.index, datos.index[iv]))
    return folds
