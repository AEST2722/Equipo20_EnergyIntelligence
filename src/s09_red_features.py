"""Paso 9: variables de RED VIAL para sitios de aforo y CTN (Conectividad, PTRF).

Responde a la presentación del sponsor (diap. 6 "distancia de red" y diap. 8 "grafos para
accesibilidad del corredor"). A diferencia de la gravedad en línea recta, aquí todo se mide
sobre la red de la RNC, en TIEMPO de viaje.

Variables (una fila por punto; mismas columnas para sitios y CTN):
- red_snap_dist_m:        distancia del punto al tramo de la red interurbana (calidad)
- red_flujo_paso:         log(1 + flujo gravitacional asignado al tramo / 1e6). Para cada par de
                          ciudades (i, j) se busca la ruta más rápida y se le suma Pob_i·Pob_j/t_ij²
                          a cada tramo de esa ruta ("asignación todo-o-nada" de un modelo de
                          gravedad). Mide el TRÁFICO DE PASO que la carretera canaliza.
- red_grav_t2 / red_grav_t1: log(1 + Σ Pob_c / t_c²) y log(1 + Σ Pob_c / t_c), t_c en minutos por red
- red_tiempo_zmvm_min:    minutos por red a la ZM del Valle de México
- red_tiempo_zm500k_min:  minutos por red a la ZM de >500 mil hab. más cercana

Sin fuga de información: no usa TDPA ni nada del aforo, sólo red, velocidades y población.

Uso (calcula sitios y CTN con el mismo grafo, en una sola corrida):
    python src/s09_red_features.py
Salidas: 03_PROCESSED/features_red_sites.csv y 03_PROCESSED/features_red.csv
"""
from __future__ import annotations

import time

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer

from common import PROCESSED, load_config, load_points, log_decision, register_features
from s01_build_corridor_ctn import detect_road_layer, list_layers, locate_rnc

# Zonas metropolitanas / ciudades (Censo 2020, REDONDEADAS; verificar contra INEGI/CONAPO).
# Misma lista que 04_NOTEBOOKS/features_gravedad.py para que ambas variables sean comparables.
CIUDADES = [
    ("ZM Valle de México", 19.4326, -99.1332, 21_800_000),
    ("ZM Puebla-Tlaxcala", 19.0414, -98.2063, 3_200_000),
    ("ZM Toluca", 19.2826, -99.6557, 2_350_000),
    ("ZM Querétaro", 20.5888, -100.3899, 1_590_000),
    ("ZM Cuernavaca", 18.9242, -99.2216, 1_030_000),
    ("ZM Morelia", 19.7060, -101.1950, 990_000),
    ("ZM Acapulco", 16.8531, -99.8237, 850_000),
    ("ZM Oaxaca", 17.0732, -96.7266, 670_000),
    ("ZM Pachuca", 20.1011, -98.7591, 665_000),
    ("ZM Tlaxcala-Apizaco", 19.3182, -98.2375, 570_000),
    ("ZM Cuautla", 18.8121, -98.9548, 480_000),
    ("ZM Orizaba", 18.8510, -97.1000, 470_000),
    ("ZM Tehuacán", 18.4617, -97.3928, 345_000),
    ("ZM Chilpancingo", 17.5515, -99.5006, 285_000),
    ("ZM Tulancingo", 20.0833, -98.3631, 255_000),
    ("ZM Tula", 20.0540, -99.3414, 230_000),
    ("ZM Tianguistenco", 19.1800, -99.4700, 185_000),
    ("Iguala", 18.3448, -99.5397, 155_000),
    ("Atlixco", 18.9087, -98.4364, 140_000),
    ("Zihuatanejo", 17.6417, -101.5519, 125_000),
    ("Taxco", 18.5563, -99.6050, 105_000),
]

# Vialidades urbanas menores: se excluyen para que el grafo represente la red INTERURBANA
# (y para que un aforo de autopista no "se pegue" a una calle vecina).
TIPOS_EXCLUIDOS = {"Calle", "Privada", "Cerrada", "Andador", "Callejón", "Retorno", "Retorno U",
                   "Vereda", "Peatonal", "Prolongación", "Cerrada privada"}
MARGEN_GRADOS = 0.3     # margen alrededor de puntos y ciudades al leer la RNC
T_MIN = 5.0             # minutos mínimos (evita atracción infinita dentro de una ciudad)


def _campo(cols, clave):
    """Primer campo de la RNC cuyo nombre contiene `clave` (mismo criterio que s01b)."""
    for c in cols:
        if clave in c.upper():
            return c
    return None


def leer_red(path, layer, bbox_ll, crs_geo, crs_m):
    import pyogrio
    kw = {} if path.lower().endswith(".shp") else {"layer": layer}  # igual que read_bbox de s01
    info = pyogrio.read_info(path, **kw)
    t = Transformer.from_crs(crs_geo, info["crs"], always_xy=True)
    x0, y0 = t.transform(bbox_ll[0], bbox_ll[1])
    x1, y1 = t.transform(bbox_ll[2], bbox_ll[3])
    campo_tipo = _campo(info["fields"], "TIPO_VIAL")
    where = None
    if campo_tipo:
        lista = ",".join(f"'{v}'" for v in sorted(TIPOS_EXCLUIDOS))
        where = f"{campo_tipo} NOT IN ({lista})"
    try:
        roads = gpd.read_file(path, bbox=(x0, y0, x1, y1), where=where, **kw)
    except Exception:  # algunos drivers no aceptan `where`: se filtra después
        roads = gpd.read_file(path, bbox=(x0, y0, x1, y1), **kw)
        if campo_tipo:
            roads = roads[~roads[campo_tipo].isin(TIPOS_EXCLUIDOS)]
    roads = roads.to_crs(crs_m).explode(index_parts=False).reset_index(drop=True)
    return roads[roads.geometry.length > 0], campo_tipo


def velocidad_kmh(roads, campo_tipo):
    c_vel, c_peaje = _campo(roads.columns, "VELOC"), _campo(roads.columns, "PEAJE")
    v = pd.to_numeric(roads[c_vel], errors="coerce") if c_vel else pd.Series(np.nan, index=roads.index)
    v = v.where((v > 0) & (v <= 130))
    peaje = roads[c_peaje].astype(str).str.lower().str.startswith("s") if c_peaje else False
    tipo = roads[campo_tipo].astype(str) if campo_tipo else pd.Series("", index=roads.index)
    defecto = np.where(peaje, 100, np.where(tipo.eq("Carretera"), 70, np.where(tipo.eq("Enlace"), 50, 40)))
    return v.fillna(pd.Series(defecto, index=roads.index)), int(v.isna().sum())


def construir_grafo(roads, vel):
    G = nx.Graph()
    for i, (geom, kmh) in enumerate(zip(roads.geometry, vel)):
        c = geom.coords
        a = (round(c[0][0], 1), round(c[0][1], 1))
        b = (round(c[-1][0], 1), round(c[-1][1], 1))
        if a == b:
            continue
        t = geom.length / (kmh / 3.6) / 60.0  # minutos
        if not G.has_edge(a, b) or t < G[a][b]["t"]:  # tramos paralelos: se queda el más rápido
            G.add_edge(a, b, t=t, idx=i)
    comp = max(nx.connected_components(G), key=len)
    return G.subgraph(comp).copy(), len(comp) / max(G.number_of_nodes(), 1)


def main() -> None:
    t0 = time.time()
    cfg = load_config()
    crs_geo, crs_m = cfg["crs"]["geographic"], cfg["crs"]["projected"]
    sets = {k: load_points(k) for k in ("sites", "ctn")}
    ciudades = pd.DataFrame(CIUDADES, columns=["ciudad", "lat", "lon", "pob"])
    lons = pd.concat([s[0].lon for s in sets.values()] + [ciudades.lon])
    lats = pd.concat([s[0].lat for s in sets.values()] + [ciudades.lat])
    bbox = (lons.min() - MARGEN_GRADOS, lats.min() - MARGEN_GRADOS,
            lons.max() + MARGEN_GRADOS, lats.max() + MARGEN_GRADOS)

    path = locate_rnc()
    layer = detect_road_layer(list_layers(path), cfg["rnc"]["road_layer"])
    roads, campo_tipo = leer_red(path, layer, bbox, crs_geo, crs_m)
    vel, sin_vel = velocidad_kmh(roads, campo_tipo)
    print(f"Tramos de red interurbana leídos: {len(roads):,} ({sin_vel:,} sin velocidad válida -> por defecto)")
    G, share = construir_grafo(roads, vel)
    print(f"Grafo: {G.number_of_nodes():,} nodos, {G.number_of_edges():,} tramos "
          f"(componente principal = {share:.1%} de los nodos)")

    # Nodos de ciudades
    nodos = np.array(list(G.nodes))
    to_m = Transformer.from_crs(crs_geo, crs_m, always_xy=True)
    cx, cy = to_m.transform(ciudades.lon.to_numpy(), ciudades.lat.to_numpy())
    nodo_ciudad, snap_ciudad = [], []
    for x, y in zip(cx, cy):
        d = np.hypot(nodos[:, 0] - x, nodos[:, 1] - y)
        k = int(d.argmin())
        nodo_ciudad.append(tuple(nodos[k]))
        snap_ciudad.append(d[k])
    lejos = [c for c, s in zip(ciudades.ciudad, snap_ciudad) if s > 5000]
    if lejos:
        log_decision("s09", f"Ciudades a >5 km de la red: {lejos}", "Revisar bbox o coordenadas")

    # Dijkstra desde cada ciudad (tiempos y predecesores)
    dist, pred = [], []
    for n in nodo_ciudad:
        p, d = nx.dijkstra_predecessor_and_distance(G, n, weight="t")
        pred.append(p)
        dist.append(d)

    # Asignación gravitacional de pares de ciudades a la ruta más rápida
    flujo = {}
    P = ciudades.pob.to_numpy(float)
    for i in range(len(ciudades)):
        for j in range(i + 1, len(ciudades)):
            tij = dist[i].get(nodo_ciudad[j])
            if tij is None:
                continue
            f = P[i] * P[j] / max(tij, T_MIN) ** 2
            v = nodo_ciudad[j]
            while v != nodo_ciudad[i]:
                u = pred[i][v][0]
                e = (u, v) if u < v else (v, u)
                flujo[e] = flujo.get(e, 0.0) + f
                v = u

    # Tramos del grafo como geometrías para pegar los puntos
    aristas = list(G.edges(data=True))
    geom_aristas = gpd.GeoDataFrame(
        {"u": [a for a, _, _ in aristas], "v": [b for _, b, _ in aristas]},
        geometry=roads.geometry.iloc[[d["idx"] for _, _, d in aristas]].to_numpy(), crs=crs_m)
    grandes = ciudades.pob.to_numpy() >= 500_000

    for kind, (pts, idc) in sets.items():
        g = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts.lon, pts.lat), crs=crs_geo).to_crs(crs_m)
        nn = gpd.sjoin_nearest(g, geom_aristas, how="left", distance_col="red_snap_dist_m")
        nn = nn[~nn.index.duplicated(keep="first")]
        filas = []
        for _, r in nn.iterrows():
            u, v = r["u"], r["v"]
            e = (u, v) if u < v else (v, u)
            t = np.array([min(d.get(u, np.inf), d.get(v, np.inf)) for d in dist])
            ok = np.isfinite(t)
            tt = np.maximum(t[ok], T_MIN)
            filas.append({
                idc: r[idc],
                "red_snap_dist_m": r["red_snap_dist_m"],
                "red_flujo_paso": np.log1p(flujo.get(e, 0.0) / 1e6),
                "red_grav_t2": np.log1p((P[ok] / tt**2).sum()),
                "red_grav_t1": np.log1p((P[ok] / tt).sum()),
                "red_tiempo_zmvm_min": t[0] if ok[0] else np.nan,
                "red_tiempo_zm500k_min": t[grandes & ok].min() if (grandes & ok).any() else np.nan,
            })
        out = pd.DataFrame(filas)
        nombre = "features_red.csv" if kind == "ctn" else f"features_red_{kind}.csv"
        out.to_csv(PROCESSED / nombre, index=False)
        lejanos = int((out.red_snap_dist_m > 500).sum())
        print(f"{nombre}: {out.shape} | puntos a >500 m de la red interurbana: {lejanos}")
        if lejanos:
            log_decision("s09", f"{lejanos} puntos ({kind}) a >500 m de la red interurbana",
                         "Pueden estar sobre vialidades urbanas excluidas; revisar")

    log_decision("s09", f"Red interurbana: excluidos tipos {sorted(TIPOS_EXCLUIDOS)}; "
                 f"velocidad RNC o por defecto (cuota 100, carretera 70, enlace 50, otras 40 km/h)",
                 "Tiempo de viaje como impedancia; ruta más rápida entre 21 ciudades")
    comun = dict(source="INEGI RNC 2025 + Censo 2020 (ZM, redondeado)", unit="", **{"class": "continua"},
                 source_period="2025 / 2020", spatial_resolution="tramo RNC",
                 leakage_risk="bajo (no usa aforo)", PTRF_domain="Conectividad")
    register_features([{**comun, "feature_name": n, "definition": d, "transformation": tr,
                        "uncertainty": "velocidades por defecto donde falta VELOCIDAD; población redondeada"}
                       for n, d, tr in [
        ("red_flujo_paso", "Flujo gravitacional entre ciudades asignado a la ruta más rápida", "log1p(Σ PiPj/tij² /1e6)"),
        ("red_grav_t2", "Accesibilidad gravitacional por tiempo de red", "log1p(Σ Pc/tc²), tc en min"),
        ("red_grav_t1", "Accesibilidad gravitacional por tiempo de red (caída lenta)", "log1p(Σ Pc/tc)"),
        ("red_tiempo_zmvm_min", "Minutos por red a la ZM del Valle de México", "Dijkstra"),
        ("red_tiempo_zm500k_min", "Minutos por red a la ZM >500 mil hab. más cercana", "Dijkstra"),
        ("red_snap_dist_m", "Distancia del punto a la red interurbana (calidad)", "sjoin_nearest"),
    ]])
    print(f"Listo en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
