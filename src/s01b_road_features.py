"""Paso 1b: atributos viales de la RNC para cualquier conjunto de puntos (sitios de aforo o CTN).

Para cada punto: atributos del tramo RNC más cercano (carriles, tipo de vialidad, peaje, velocidad,
etc., detectados por nombre de campo) y número de intersecciones (nodos de grado >= 3) a <= 1 km.

Uso:
    python src/s01b_road_features.py --points sites
    python src/s01b_road_features.py --points ctn
"""
from __future__ import annotations

import argparse
from collections import Counter

import geopandas as gpd
import numpy as np
import pandas as pd

from common import (PROCESSED, load_config, load_points, log_decision, out_name, points_arg,
                    register_features)
from s01_build_corridor_ctn import detect_road_layer, list_layers, locate_rnc, read_bbox

ATTR_KEYWORDS = ("CARRIL", "TIPO_VIAL", "PEAJE", "VELOC", "ADMIN", "JURIS", "COND_PAV", "RECUBRI",
                 "ANCHO", "CIRCULA", "NIVEL")


def main() -> None:
    ap = argparse.ArgumentParser(); points_arg(ap); kind = ap.parse_args().points
    cfg = load_config()
    crs_geo, crs_m = cfg["crs"]["geographic"], cfg["crs"]["projected"]
    pts, idc = load_points(kind)
    g = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts.lon, pts.lat), crs=crs_geo)
    m = 0.1
    bbox = (pts.lon.min() - m, pts.lat.min() - m, pts.lon.max() + m, pts.lat.max() + m)

    path = locate_rnc()
    layer = detect_road_layer(list_layers(path), cfg["rnc"]["road_layer"])
    roads = read_bbox(path, layer, bbox, crs_geo).to_crs(crs_m).explode(index_parts=False).reset_index(drop=True)
    attrs = [c for c in roads.columns if c != "geometry" and any(k in c.upper() for k in ATTR_KEYWORDS)]
    log_decision("s01b", f"Atributos RNC usados ({kind}): {attrs}", "Detectados por nombre de campo")

    gm = g.to_crs(crs_m)
    nn = gpd.sjoin_nearest(gm, roads[attrs + ["geometry"]], how="left", distance_col="road_dist_m")
    nn = nn[~nn.index.duplicated(keep="first")]
    out = nn[[idc, "road_dist_m"] + attrs].rename(columns={a: f"road_{a.lower()}" for a in attrs})

    ends = Counter()
    for geom in roads.geometry:
        c = list(geom.coords)
        ends[(round(c[0][0], 1), round(c[0][1], 1))] += 1
        ends[(round(c[-1][0], 1), round(c[-1][1], 1))] += 1
    deg3 = np.array([k for k, v in ends.items() if v >= 3])
    out["n_intersections_1km"] = [int((np.hypot(deg3[:, 0] - p.x, deg3[:, 1] - p.y) <= 1000).sum())
                                  if len(deg3) else 0 for p in gm.geometry]
    out.to_csv(PROCESSED / out_name("features_road", kind), index=False)
    far = (out["road_dist_m"] > 500).sum()
    if far:
        log_decision("s01b", f"{far} puntos a más de 500 m de un tramo RNC ({kind})",
                     "Revisar coordenadas o cobertura de la caja leída")
    print(f"Features viales ({kind}): {out.shape}")

    base = {"source": "INEGI RNC", "source_version": "2025", "source_period": "2025",
            "leakage_risk": "bajo", "PTRF_domain": "movilidad"}
    rows = [{**base, "feature_name": f"road_{a.lower()}", "definition": f"Atributo '{a}' del tramo RNC más cercano",
             "unit": "según diccionario RNC", "class": "O", "spatial_resolution": "tramo vial",
             "transformation": "sjoin_nearest", "uncertainty": "Asignación al tramo más cercano"} for a in attrs]
    rows += [
        {**base, "feature_name": "road_dist_m", "definition": "Distancia al tramo RNC más cercano", "unit": "m",
         "class": "D", "spatial_resolution": "punto", "transformation": "sjoin_nearest",
         "uncertainty": "Control de calidad de coordenadas, no variable explicativa"},
        {**base, "feature_name": "n_intersections_1km", "definition": "Nodos de grado>=3 de la RNC a <=1 km",
         "unit": "conteo", "class": "P", "spatial_resolution": "buffer 1 km",
         "transformation": "Grado de nodos por extremos de tramo", "uncertainty": "Proxy de entronques; urbano infla"},
    ]
    register_features(rows)


if __name__ == "__main__":
    main()
