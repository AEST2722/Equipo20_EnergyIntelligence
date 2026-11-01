"""Variables de gravedad: qué tanto "atraen" tráfico las ciudades cercanas a cada sitio.

Idea (modelo gravitacional de transporte): el tráfico en un punto crece con la población
de las ciudades que conecta y decrece con la distancia a ellas.

Variables que se crean (todas en escala log para que no dominen los valores extremos):
- grav_pob_d2:     log(1 + Σ Pob_i / d_i²)     atracción total de las ciudades
- grav_pob_d1:     log(1 + Σ Pob_i / d_i)      versión con menor caída por distancia
- dist_zmvm_km:    distancia a la Zona Metropolitana del Valle de México
- dist_zm500k_km:  distancia a la zona metropolitana de >500 mil hab. más cercana
- pob_zm_cercana:  log de la población de la ciudad más cercana
- grav_paso:       log(1 + Σ_{i<j} Pob_i·Pob_j / d_ij² · peso de "estar en el camino" entre i y j)
                   Alta si el sitio queda sobre la línea entre dos ciudades grandes
                   (tráfico de paso, p. ej. CDMX–Cuernavaca–Acapulco).

No usa el TDPA ni ningún dato del aforo: sólo coordenadas y población de ciudades,
así que se puede calcular igual para los CTN (sin fuga de información).

Poblaciones: zonas metropolitanas y ciudades, Censo 2020 (INEGI/CONAPO), REDONDEADAS.
Verificarlas contra la publicación oficial antes de la entrega final.
"""
import numpy as np
import pandas as pd

# nombre, lat, lon, población aproximada (Censo 2020)
CIUDADES = [
    ("ZM Valle de México", 19.4326, -99.1332, 21_800_000),
    ("ZM Puebla-Tlaxcala", 19.0414, -98.2063, 3_200_000),
    ("ZM Toluca",          19.2826, -99.6557, 2_350_000),
    ("ZM Querétaro",       20.5888, -100.3899, 1_590_000),
    ("ZM Cuernavaca",      18.9242, -99.2216, 1_030_000),
    ("ZM Morelia",         19.7060, -101.1950,  990_000),
    ("ZM Acapulco",        16.8531, -99.8237,   850_000),
    ("ZM Oaxaca",          17.0732, -96.7266,   670_000),
    ("ZM Pachuca",         20.1011, -98.7591,   665_000),
    ("ZM Tlaxcala-Apizaco",19.3182, -98.2375,   570_000),
    ("ZM Cuautla",         18.8121, -98.9548,   480_000),
    ("ZM Orizaba",         18.8510, -97.1000,   470_000),
    ("ZM Tehuacán",        18.4617, -97.3928,   345_000),
    ("ZM Chilpancingo",    17.5515, -99.5006,   285_000),
    ("ZM Tulancingo",      20.0833, -98.3631,   255_000),
    ("ZM Tula",            20.0540, -99.3414,   230_000),
    ("ZM Tianguistenco",   19.1800, -99.4700,   185_000),
    ("Iguala",             18.3448, -99.5397,   155_000),
    ("Atlixco",            18.9087, -98.4364,   140_000),
    ("Zihuatanejo",        17.6417, -101.5519,  125_000),
    ("Taxco",              18.5563, -99.6050,   105_000),
]
D_MIN_KM = 5.0          # evita que un sitio dentro de la ciudad tenga atracción infinita
ANCHO_CORREDOR = 0.10   # desvío relativo tolerado para considerar que el sitio "está en el camino"


def _dist_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def agregar_variables_gravedad(df, lat="lat", lon="lon"):
    c = pd.DataFrame(CIUDADES, columns=["ciudad", "lat", "lon", "pob"])
    la = df[lat].to_numpy(float)[:, None]
    lo = df[lon].to_numpy(float)[:, None]
    d = np.maximum(_dist_km(la, lo, c.lat.to_numpy()[None, :], c.lon.to_numpy()[None, :]), D_MIN_KM)
    pob = c.pob.to_numpy(float)[None, :]

    out = df.copy()
    out["grav_pob_d2"] = np.log1p((pob / d**2).sum(axis=1))
    out["grav_pob_d1"] = np.log1p((pob / d).sum(axis=1))
    out["dist_zmvm_km"] = d[:, 0]
    grandes = c.pob.to_numpy() >= 500_000
    out["dist_zm500k_km"] = d[:, grandes].min(axis=1)
    out["pob_zm_cercana"] = np.log(c.pob.to_numpy()[d.argmin(axis=1)])

    # Tráfico de paso entre pares de ciudades
    paso = np.zeros(len(df))
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            dij = _dist_km(c.lat[i], c.lon[i], c.lat[j], c.lon[j])
            desvio = (d[:, i] + d[:, j] - dij) / dij          # 0 = exactamente en la línea
            peso = np.exp(-desvio / ANCHO_CORREDOR)
            paso += c.pob[i] * c.pob[j] / dij**2 * peso
    out["grav_paso"] = np.log1p(paso / 1e6)
    return out


NUEVAS = ["grav_pob_d2", "grav_pob_d1", "dist_zmvm_km", "dist_zm500k_km", "pob_zm_cercana", "grav_paso"]
