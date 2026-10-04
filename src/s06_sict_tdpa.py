"""Paso 6: SICT Datos Viales (TDPA) -> TOP (Traffic Observation Points) asignados a CTN.

Recuerda: TOP != CTN. Un TOP es donde se observó el tránsito; el CTN hereda el TDPA del TOP
más representativo POR POSICIÓN DE RED (chainage), no por cercanía euclidiana.

Datos Viales se publica en PDF por estado y año. Flujo:
 1. Descarga los PDF de CDMX, Morelos y Guerrero en 01_RAW/sict_tdpa/ (sin modificarlos):
    https://micrs.sct.gob.mx/infraestructura/direccion-general-de-servicios-tecnicos/datos-viales/
 2. Transcribe las estaciones de la MEX-095D (y libre 95 si se requiere) a 02_INTERIM/top_sict.csv
    usando la plantilla 00_DOCUMENTATION/top_sict_template.csv. Anota archivo y página de origen.
 3. Si la estación no trae coordenadas, llena `km_sict` (kilometraje de la carretera); el script lo
    convierte a chainage con `km_offset` (documentado como proxy en el decision_log).

Uso:  python src/s06_sict_tdpa.py [--km-offset 0]
"""
from __future__ import annotations

import argparse

import geopandas as gpd
import numpy as np
import pandas as pd

from common import DOCS, INTERIM, PROCESSED, load_config, log_decision, register_features

TEMPLATE_COLS = ["top_id", "year", "state", "road", "station_name", "km_sict", "lon", "lat",
                 "tdpa", "pct_A", "pct_B", "pct_C", "direction", "source_file", "source_page"]


def write_template() -> None:
    p = DOCS / "top_sict_template.csv"
    if not p.exists():
        pd.DataFrame(columns=TEMPLATE_COLS).to_csv(p, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--km-offset", type=float, default=0.0,
                    help="chainage_km = km_sict - km_offset (km de SICT en el origen del corredor)")
    args = ap.parse_args()
    write_template()
    cfg = load_config()
    crs_geo, crs_m = cfg["crs"]["geographic"], cfg["crs"]["projected"]
    src = INTERIM / "top_sict.csv"
    if not src.exists():
        raise SystemExit("Falta 02_INTERIM/top_sict.csv (usa la plantilla en 00_DOCUMENTATION).")
    top = pd.read_csv(src)
    line = gpd.read_file(INTERIM / "corridor_route.gpkg").to_crs(crs_m).geometry.iloc[0]

    has_xy = top["lon"].notna() & top["lat"].notna()
    top["top_chainage_km"] = np.nan
    top["top_locating_method"] = ""
    if has_xy.any():
        g = gpd.GeoDataFrame(top[has_xy], geometry=gpd.points_from_xy(top.loc[has_xy, "lon"],
                             top.loc[has_xy, "lat"]), crs=crs_geo).to_crs(crs_m)
        top.loc[has_xy, "top_chainage_km"] = [line.project(p) / 1000 for p in g.geometry]
        top.loc[has_xy, "top_offset_from_route_m"] = g.distance(line).values
        top.loc[has_xy, "top_locating_method"] = "coords_projected_to_route"
    by_km = ~has_xy & top["km_sict"].notna()
    if by_km.any():
        top.loc[by_km, "top_chainage_km"] = top.loc[by_km, "km_sict"] - args.km_offset
        top.loc[by_km, "top_locating_method"] = "km_sict_minus_offset_proxy"
        log_decision("s06", f"{by_km.sum()} TOP ubicados por kilometraje SICT con offset {args.km_offset}",
                     "Sin coordenadas en Datos Viales; el km oficial y el chainage RNC pueden diferir")
    top.to_csv(INTERIM / "top_sict_located.csv", index=False)

    ctn = pd.read_csv(PROCESSED / "ctn_base.csv")
    max_km = cfg["sict"]["max_top_distance_m"] / 1000
    valid = top.dropna(subset=["top_chainage_km", "tdpa"])
    recs = []
    for _, c in ctn.iterrows():
        d = (valid["top_chainage_km"] - c.chainage_km).abs()
        if d.empty or d.min() > max_km:
            recs.append({"ctn_id": c.ctn_id})  # TDPA faltante: no hay TOP representativo
            continue
        t = valid.loc[d.idxmin()]
        recs.append({"ctn_id": c.ctn_id, "tdpa": t.tdpa, "tdpa_year": t.year,
                     "tdpa_pct_heavy": t.pct_C, "tdpa_pct_bus": t.pct_B,
                     "top_id": t.top_id, "top_network_distance_km": round(d.min(), 3),
                     "top_locating_method": t.top_locating_method})
    out = pd.DataFrame(recs)
    out.to_csv(PROCESSED / "features_sict.csv", index=False)
    years = sorted(valid["year"].dropna().unique().tolist())
    if len(years) > 1:
        log_decision("s06", f"TDPA proviene de varios años {years}", "Se conserva tdpa_year por CTN (error #3)")
    print(f"Features SICT: {out.shape}; CTN sin TDPA: {out['tdpa'].isna().sum() if 'tdpa' in out else len(out)}")

    base = {"source": "SICT Datos Viales", "source_version": "PDF por estado (ver source_file)",
            "source_period": ",".join(map(str, years)), "spatial_resolution": "estación de aforo (TOP)",
            "leakage_risk": "bajo", "PTRF_domain": "movilidad"}
    register_features([
        {**base, "feature_name": "tdpa", "definition": "Tránsito Diario Promedio Anual del TOP asignado",
         "unit": "veh/día", "class": "O", "transformation": "Asignación por chainage de red (TOP más cercano en red)",
         "uncertainty": "Observado en el TOP, no en el CTN; años distintos entre TOP"},
        {**base, "feature_name": "tdpa_pct_heavy", "definition": "% vehículos de carga (C)",
         "unit": "%", "class": "O", "transformation": "Igual que tdpa", "uncertainty": "Igual que tdpa"},
        {**base, "feature_name": "tdpa_pct_bus", "definition": "% autobuses (B)", "unit": "%", "class": "O",
         "transformation": "Igual que tdpa", "uncertainty": "Igual que tdpa"},
        {**base, "feature_name": "top_network_distance_km", "definition": "Distancia en red entre CTN y su TOP",
         "unit": "km", "class": "D", "transformation": "|chainage_CTN - chainage_TOP|",
         "uncertainty": "Indicador de confianza del TDPA asignado"},
        {**base, "feature_name": "tdpa_year", "definition": "Año de observación del TDPA", "unit": "año",
         "class": "O", "transformation": "Metadato", "uncertainty": "n/a",
         "leakage_risk": "no usar como variable explicativa"},
    ])


if __name__ == "__main__":
    main()
