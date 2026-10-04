"""Paso 1 (gate geoespacial): corredor real MEX-095D desde la RNC y CTN por chainage de red.

Uso:
    python src/s01_build_corridor_ctn.py --inspect   # 1) ver capas, campos y valores con "095"
    python src/s01_build_corridor_ctn.py             # 2) construir corredor y CTN

Lógica:
- Se lee la RNC sólo en la caja que rodea origen-destino (más rápido que todo el país).
- Se arma un grafo con la red vial. Los tramos cuyo código coincide con el patrón de ruta
  (p. ej. "095") pesan su longitud real; los demás pesan 5x. Así la ruta más corta sigue la
  MEX-095D por atributo y continuidad topológica, y sólo usa otras vías para cerrar huecos.
- El chainage se mide SOBRE la línea del corredor (no distancia geodésica entre puntos).
- CTN nominales cada `spacing_m` + plazas de cobro como puntos estratégicos.
- Todo CTN sale con geometry_status = RNC_SNAPPED (nunca PRELIMINARY_ROUTE_PROXY).
"""
from __future__ import annotations

import argparse
import sys

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pyogrio
from shapely.geometry import LineString, Point, box
from shapely.ops import transform
from pyproj import Transformer

from common import (INTERIM, PROCESSED, RAW, find_first, load_config, log_decision,
                    log_download, register_features)

PENALTY_NON_ROUTE = 5.0


def locate_rnc() -> str:
    p = find_first(RAW / "road_network", ("*.gpkg", "**/*.gpkg", "*.shp", "**/*.shp"))
    if p is None:
        sys.exit("No encontré la RNC en 01_RAW/road_network/. Descárgala de "
                 "https://www.inegi.org.mx/programas/rnc/ y colócala ahí sin modificar.")
    return str(p)


def list_layers(path: str) -> pd.DataFrame:
    if path.endswith(".shp"):
        info = pyogrio.read_info(path)
        return pd.DataFrame([[path, info["geometry_type"], info["features"]]],
                            columns=["layer", "geometry_type", "features"])
    rows = []
    for name, gtype in pyogrio.list_layers(path):
        info = pyogrio.read_info(path, layer=name)
        rows.append([name, gtype, info["features"]])
    return pd.DataFrame(rows, columns=["layer", "geometry_type", "features"])


def detect_road_layer(layers: pd.DataFrame, forced: str | None) -> str | None:
    if forced:
        return forced
    lines = layers[layers["geometry_type"].str.contains("LineString", case=False, na=False)]
    if lines.empty:
        sys.exit("No hay capas de líneas en el archivo RNC.")
    named = lines[lines["layer"].str.contains("red|vial", case=False)]
    pick = named if not named.empty else lines
    return pick.sort_values("features", ascending=False)["layer"].iloc[0]


def detect_point_layer(layers: pd.DataFrame, keyword: str) -> str | None:
    pts = layers[layers["geometry_type"].str.contains("Point", case=False, na=False)]
    hit = pts[pts["layer"].str.contains(keyword, case=False)]
    return None if hit.empty else hit["layer"].iloc[0]


def read_bbox(path: str, layer: str | None, bbox_ll: tuple, crs_geo: str) -> gpd.GeoDataFrame:
    kw = {} if path.endswith(".shp") else {"layer": layer}
    info = pyogrio.read_info(path, **kw)
    layer_crs = info["crs"]
    t = Transformer.from_crs(crs_geo, layer_crs, always_xy=True)
    bbox_layer = transform(t.transform, box(*bbox_ll)).bounds
    return gpd.read_file(path, bbox=bbox_layer, **kw)


def detect_route_fields(gdf: gpd.GeoDataFrame, pattern: str, forced: str | None) -> list[str]:
    if forced:
        return [forced]
    fields = []
    for c in gdf.columns:
        if c == "geometry" or not (pd.api.types.is_string_dtype(gdf[c]) or gdf[c].dtype == object):
            continue
        if gdf[c].astype(str).str.contains(pattern, case=False, na=False).any():
            fields.append(c)
    return fields


def route_match_mask(gdf: gpd.GeoDataFrame, cfg: dict) -> tuple[np.ndarray, str]:
    """Devuelve (máscara de tramos de la ruta, descripción de la regla).
    Modo preferido: rnc.route_match = {campo: [valores]} con coincidencia EXACTA en todos los campos.
    Modo antiguo: rnc.route_pattern contenido en cualquier campo de texto."""
    rule = cfg["rnc"].get("route_match")
    if rule:
        mask = np.ones(len(gdf), dtype=bool)
        for field, values in rule.items():
            if field not in gdf.columns:
                sys.exit(f"El campo '{field}' de route_match no existe en la RNC. Columnas: {list(gdf.columns)}")
            vals = {str(v).strip().lower() for v in values}
            mask &= gdf[field].astype(str).str.strip().str.lower().isin(vals).to_numpy()
        return mask, " y ".join(f"{k} en {v}" for k, v in rule.items())
    pattern = cfg["rnc"]["route_pattern"]
    fields = detect_route_fields(gdf, pattern, cfg["rnc"]["route_field"])
    if not fields:
        sys.exit(f"Ningún campo contiene '{pattern}'. Corre --inspect y define rnc.route_match en config.yaml.")
    mask = np.zeros(len(gdf), dtype=bool)
    for f in fields:
        mask |= gdf[f].astype(str).str.contains(pattern, case=False, na=False).to_numpy()
    return mask, f"campos {fields} contienen '{pattern}'"


def inspect(cfg: dict) -> None:
    path = locate_rnc()
    print(f"Archivo RNC: {path}\n")
    layers = list_layers(path)
    print(layers.to_string(index=False), "\n")
    road = detect_road_layer(layers, cfg["rnc"]["road_layer"])
    bbox = corridor_bbox(cfg)
    gdf = read_bbox(path, road, bbox, cfg["crs"]["geographic"])
    print(f"Capa de red vial: {road} | tramos en la caja del corredor: {len(gdf)}")
    print("Columnas:", list(gdf.columns), "\n")
    if cfg["rnc"].get("route_match"):
        mask, desc = route_match_mask(gdf, cfg)
        print(f"Regla de ruta: {desc} -> {mask.sum()} tramos")
        name_col = "NOMBRE" if "NOMBRE" in gdf.columns else gdf.columns[0]
        print("Nombres de los tramos que cumplen la regla:")
        print(gdf.loc[mask, name_col].value_counts().head(15).to_string())
    else:
        pat = cfg["rnc"]["route_pattern"]
        for c in detect_route_fields(gdf, pat, None):
            vals = gdf.loc[gdf[c].astype(str).str.contains(pat, case=False, na=False), c]
            print(f"Campo '{c}' contiene '{pat}':", vals.value_counts().head(10).to_dict())
    print("\nSi la regla no es correcta, ajusta rnc.route_match en config.yaml.")


def corridor_bbox(cfg: dict, margin_deg: float = 0.3) -> tuple:
    (x1, y1), (x2, y2) = cfg["rnc"]["origin_lonlat"], cfg["rnc"]["destination_lonlat"]
    return (min(x1, x2) - margin_deg, min(y1, y2) - margin_deg,
            max(x1, x2) + margin_deg, max(y1, y2) + margin_deg)


def build_graph(roads: gpd.GeoDataFrame, match: pd.Series) -> nx.Graph:
    G = nx.Graph()
    for idx, geom, is_route in zip(roads.index, roads.geometry, match):
        coords = list(geom.coords)
        a = (round(coords[0][0], 1), round(coords[0][1], 1))
        b = (round(coords[-1][0], 1), round(coords[-1][1], 1))
        if a == b:
            continue
        length = geom.length
        w = length if is_route else length * PENALTY_NON_ROUTE
        # Si hay tramos paralelos entre los mismos nodos, conservar el de menor peso.
        if G.has_edge(a, b) and G[a][b]["weight"] <= w:
            continue
        G.add_edge(a, b, weight=w, length=length, idx=idx, is_route=bool(is_route))
    return G


def nearest_node(G: nx.Graph, pt: Point) -> tuple[tuple, float]:
    nodes = np.array(list(G.nodes))
    d = np.hypot(nodes[:, 0] - pt.x, nodes[:, 1] - pt.y)
    i = int(d.argmin())
    return tuple(nodes[i]), float(d[i])


def path_to_line(G: nx.Graph, path: list, roads: gpd.GeoDataFrame) -> tuple[LineString, list]:
    coords: list = []
    edge_records = []
    for u, v in zip(path[:-1], path[1:]):
        e = G[u][v]
        geom = roads.geometry.loc[e["idx"]]
        seg = list(geom.coords)
        start = (round(seg[0][0], 1), round(seg[0][1], 1))
        if start != u:
            seg = seg[::-1]
        coords.extend(seg if not coords else seg[1:])
        edge_records.append((e["length"], e["is_route"]))
    return LineString(coords), edge_records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if args.inspect:
        inspect(cfg)
        return

    crs_geo, crs_m = cfg["crs"]["geographic"], cfg["crs"]["projected"]
    path = locate_rnc()
    log_download("INEGI RNC", "https://www.inegi.org.mx/programas/rnc/", path,
                 version="RNC 2025 (verificar en metadatos)",
                 license_="Términos de libre uso de la información del INEGI",
                 notes="Archivo leído sin modificar desde 01_RAW")
    layers = list_layers(path)
    road_layer = detect_road_layer(layers, cfg["rnc"]["road_layer"])
    bbox = corridor_bbox(cfg)
    roads = read_bbox(path, road_layer, bbox, crs_geo).to_crs(crs_m)
    roads = roads.explode(index_parts=False).reset_index(drop=True)
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty]

    match, rule_desc = route_match_mask(roads, cfg)
    pattern = rule_desc
    log_decision("s01", f"Regla de ruta: {rule_desc} ({match.sum()} tramos)",
                 "Corredor definido por atributo + continuidad topológica (guía §3)")

    G = build_graph(roads, pd.Series(match))
    to_m = Transformer.from_crs(crs_geo, crs_m, always_xy=True)
    o = Point(to_m.transform(*cfg["rnc"]["origin_lonlat"]))
    d = Point(to_m.transform(*cfg["rnc"]["destination_lonlat"]))
    on, od = nearest_node(G, o)
    dn, dd = nearest_node(G, d)
    tol = cfg["rnc"]["snap_tolerance_m"]
    for name, dist in (("origen", od), ("destino", dd)):
        log_decision("s01", f"Snap de {name} a la red: {dist:.0f} m",
                     "Distancia del punto configurado al nodo RNC más cercano")
        if dist > tol:
            sys.exit(f"El {name} quedó a {dist:.0f} m de la red (> {tol} m). Revisa config.yaml.")

    try:
        path_nodes = nx.shortest_path(G, on, dn, weight="weight")
    except nx.NetworkXNoPath:
        sys.exit("No hay ruta conectada entre origen y destino en la caja leída. "
                 "Aumenta margin_deg o revisa la topología de la RNC.")
    line, edges = path_to_line(G, path_nodes, roads)
    total = line.length
    route_len = sum(l for l, r in edges if r)
    share = route_len / total
    log_decision("s01", f"Longitud corredor {total/1000:.1f} km; {share:.1%} sobre tramos de la regla ({pattern})",
                 "Porcentaje bajo indica que el patrón/campo no captura bien la ruta")
    print(f"Corredor: {total/1000:.1f} km | {share:.1%} sobre tramos que cumplen la regla ({pattern})")

    INTERIM.mkdir(exist_ok=True)
    gpd.GeoDataFrame({"corridor_id": [cfg["corridor_id"]], "length_km": [total / 1000],
                      "route_match_share": [share]}, geometry=[line], crs=crs_m
                     ).to_file(INTERIM / "corridor_route.gpkg", driver="GPKG")

    # --- CTN nominales por chainage de red ---
    spacing = cfg["ctn"]["spacing_m"]
    chain = list(np.arange(0, total, spacing)) + [total]
    rows = [{"chainage_m": c, "ctn_type": "nominal_10km", "strategic_label": ""} for c in chain]
    rows[0]["ctn_type"], rows[-1]["ctn_type"] = "endpoint_origin", "endpoint_destination"

    # --- Puntos estratégicos: plazas de cobro ---
    toll_layer = detect_point_layer(layers, "plaza|cobro|caseta")
    if cfg["ctn"]["include_toll_plazas"] and toll_layer:
        tolls = read_bbox(path, toll_layer, bbox, crs_geo).to_crs(crs_m)
        tolls = tolls[tolls.distance(line) < 500]
        name_col = next((c for c in tolls.columns if "nom" in c.lower()), None)
        for _, t in tolls.iterrows():
            c = line.project(t.geometry)
            label = str(t[name_col]) if name_col else "plaza_cobro"
            j = int(np.argmin([abs(c - r["chainage_m"]) for r in rows]))
            nearest = abs(c - rows[j]["chainage_m"])
            if nearest >= cfg["ctn"]["min_gap_m"]:
                rows.append({"chainage_m": c, "ctn_type": "toll_plaza", "strategic_label": label})
            else:
                rows[j]["strategic_label"] = f"plaza_cobro:{label}"
                log_decision("s01", f"Plaza '{label}' a {nearest:.0f} m de un CTN; se anota en ese CTN",
                             "min_gap_m: no se duplica el punto")
    elif cfg["ctn"]["include_toll_plazas"]:
        log_decision("s01", "No se encontró capa de plazas de cobro", "Sin puntos estratégicos de peaje")

    ctn = pd.DataFrame(rows).sort_values("chainage_m").reset_index(drop=True)
    ctn["geometry"] = [line.interpolate(c) for c in ctn["chainage_m"]]
    ctn = gpd.GeoDataFrame(ctn, geometry="geometry", crs=crs_m)
    ctn.insert(0, "ctn_id", [f"CTN_{i:03d}" for i in range(len(ctn))])
    ctn["chainage_km"] = (ctn["chainage_m"] / 1000).round(3)
    ctn["geometry_status"] = "RNC_SNAPPED"
    ctn["snap_distance_m"] = 0.0  # interpolado sobre la línea RNC: distancia de snap nula por construcción

    # Feature de red: número de nodos de grado >= 3 (entronques/intersecciones) a <= 1 km
    deg3 = np.array([n for n, k in G.degree() if k >= 3])
    if len(deg3):
        ctn["n_intersections_1km"] = [
            int((np.hypot(deg3[:, 0] - p.x, deg3[:, 1] - p.y) <= 1000).sum()) for p in ctn.geometry]
    geo = ctn.to_crs(crs_geo)
    ctn["lon"], ctn["lat"] = geo.geometry.x.round(6), geo.geometry.y.round(6)

    PROCESSED.mkdir(exist_ok=True)
    ctn.to_file(PROCESSED / "ctn_base.gpkg", driver="GPKG")
    ctn.drop(columns="geometry").to_csv(PROCESSED / "ctn_base.csv", index=False)
    print(f"{len(ctn)} CTN generados -> 03_PROCESSED/ctn_base.gpkg")

    register_features([
        {"feature_name": "chainage_km", "definition": "Distancia acumulada sobre la red desde el origen",
         "unit": "km", "class": "D", "source": "INEGI RNC", "source_version": "2025",
         "source_period": "2025", "spatial_resolution": "tramo vial",
         "transformation": "Ruta más corta ponderada por atributo de ruta; line.project",
         "uncertainty": "Depende de la definición de origen y del campo de ruta",
         "leakage_risk": "bajo", "PTRF_domain": "movilidad"},
        {"feature_name": "ctn_type", "definition": "Nominal cada 10 km o punto estratégico (plaza de cobro)",
         "unit": "categoría", "class": "D", "source": "INEGI RNC", "source_version": "2025",
         "source_period": "2025", "spatial_resolution": "punto",
         "transformation": "Regla de muestreo; no es indicador de idoneidad",
         "uncertainty": "n/a", "leakage_risk": "medio si se usa como variable explicativa",
         "PTRF_domain": "movilidad"},
        {"feature_name": "n_intersections_1km", "definition": "Nodos de grado>=3 de la RNC a <=1 km del CTN",
         "unit": "conteo", "class": "P", "source": "INEGI RNC", "source_version": "2025",
         "source_period": "2025", "spatial_resolution": "buffer 1 km",
         "transformation": "Grado de nodos del grafo vial en la caja del corredor",
         "uncertainty": "Proxy de accesibilidad/entronques; urbano infla el conteo",
         "leakage_risk": "bajo", "PTRF_domain": "movilidad"},
    ])


if __name__ == "__main__":
    main()
