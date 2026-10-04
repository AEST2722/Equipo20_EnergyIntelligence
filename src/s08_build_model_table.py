"""Paso 8: tabla de modelado supervisado (sitios de aforo) y tabla de predicción (CTN).

Objetivo (target): log_tdpa = ln(TDPA en ambos sentidos), observado por SICT (clase O).
Entradas: features territoriales calculadas alrededor de cada sitio con las MISMAS funciones que
para los CTN (RNC, NASA POWER, DENUE, Censo, SIAP). Nada derivado del aforo entra como feature.

Columnas de agrupación para validación espacial:
  group_state     estado (entidad) del sitio
  group_route     carretera (ruta SICT): evita que sitios del mismo tramo queden en train y test
  group_block25   celda de 25 km x 25 km en EPSG:6372
  is_corridor_test  True si el sitio está sobre la MEX-095D (a <= 1 km del corredor):
                    se reserva como PRUEBA EXTERNA y nunca entra a entrenamiento.

Uso:  python src/s08_build_model_table.py
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from common import DOCS, INTERIM, PROCESSED, load_config, log_decision, register_features

FEATURE_BASES = ["features_road", "features_nasa_power", "features_denue", "features_censo", "features_dgsiap"]
# Columnas que NUNCA deben usarse como entradas (vienen del aforo, son metadatos o identificadores)
NOT_FEATURES = {"site_id", "ctn_id", "lon", "lat", "tdpa", "log_tdpa", "pct_C", "pct_B", "anio", "KM",
                "lugar_base", "carretera", "ruta", "state_file", "te_list", "n_te", "n_calzadas",
                "cve_mun", "mun_assignment_method", "road_dist_m", "group_state", "group_route",
                "group_block25", "is_corridor_test", "chainage_m", "chainage_km", "ctn_type",
                "strategic_label", "geometry_status", "snap_distance_m", "spatial_block"}


def merge_features(base: pd.DataFrame, idc: str, suffix: str) -> pd.DataFrame:
    for b in FEATURE_BASES:
        p = PROCESSED / (f"{b}.csv" if suffix == "" else f"{b}_{suffix}.csv")
        if p.exists():
            f = pd.read_csv(p, dtype={"cve_mun": str})
            dup = [c for c in f.columns if c in base.columns and c != idc]
            base = base.merge(f.drop(columns=dup), on=idc, how="left")
    return base


def main() -> None:
    cfg = load_config()
    crs_geo, crs_m = cfg["crs"]["geographic"], cfg["crs"]["projected"]
    sites = pd.read_csv(INTERIM / "datos_viales_sites.csv")
    sites["log_tdpa"] = np.log(sites["tdpa"])
    t = merge_features(sites, "site_id", "sites")

    g = gpd.GeoDataFrame(t, geometry=gpd.points_from_xy(t.lon, t.lat), crs=crs_geo).to_crs(crs_m)
    t["group_block25"] = ((g.geometry.x // 25000).astype(int).astype(str) + "_" +
                          (g.geometry.y // 25000).astype(int).astype(str))
    t["group_route"] = t["ruta"].astype(str) + "|" + t["carretera"].astype(str)
    t["group_state"] = t["cve_mun"].str[:2] if "cve_mun" in t else t["state_file"]

    route_file = INTERIM / "corridor_route.gpkg"
    if route_file.exists():
        line = gpd.read_file(route_file).to_crs(crs_m).geometry.iloc[0]
        t["is_corridor_test"] = (g.distance(line).values <= 1000) & t["ruta"].str.contains("095D", na=False)
    else:
        t["is_corridor_test"] = t["ruta"].str.contains("095D", na=False)
        log_decision("s08", "Prueba externa definida sólo por ruta MEX-095D", "No existe corridor_route.gpkg")

    feats = [c for c in t.columns if c not in NOT_FEATURES and c != "geometry"]
    t.to_csv(PROCESSED / "TOP_Model_Table.csv", index=False)
    pd.DataFrame({"feature": feats}).to_csv(PROCESSED / "model_feature_list.csv", index=False)
    n_test = int(t["is_corridor_test"].sum())
    log_decision("s08", f"Tabla de modelado: {len(t)} sitios, {len(feats)} features, {n_test} en prueba externa",
                 "Target log_tdpa; prueba externa = sitios sobre MEX-095D")
    print(f"TOP_Model_Table: {t.shape} | features: {len(feats)} | prueba externa MEX-095D: {n_test}")

    ctn_file = PROCESSED / "ctn_base.csv"
    if ctn_file.exists():
        c = merge_features(pd.read_csv(ctn_file), "ctn_id", "")
        c["toll_road"], c["red"] = True, "federal"  # todos los CTN están sobre la MEX-095D (cuota, federal)
        c.to_csv(PROCESSED / "CTN_Prediction_Table.csv", index=False)
        print(f"CTN_Prediction_Table: {c.shape}")

    register_features([
        {"feature_name": "log_tdpa", "definition": "ln del TDPA en ambos sentidos del sitio de aforo",
         "unit": "ln(veh/día)", "class": "O", "source": "SICT Datos Viales", "source_version": "PDF estatales",
         "source_period": ",".join(map(str, sorted(sites["anio"].unique()))), "spatial_resolution": "sitio de aforo",
         "transformation": "SC1+SC2 (o SC0); calzadas sumadas; mediana entre TE; log natural",
         "uncertainty": "Aforo SICT; mezcla de tipos de estación", "leakage_risk": "es el objetivo",
         "PTRF_domain": "movilidad", "allowed_as_target": "yes"},
        {"feature_name": "toll_road", "definition": "Carretera de cuota (ruta con sufijo D o '(Cuota)')",
         "unit": "booleano", "class": "O", "source": "SICT Datos Viales", "source_version": "PDF estatales",
         "source_period": "2025", "spatial_resolution": "carretera", "transformation": "Regla sobre RUTA",
         "uncertainty": "n/a", "leakage_risk": "bajo (atributo de la vía, no del aforo)", "PTRF_domain": "movilidad"},
        {"feature_name": "red", "definition": "Red federal o estatal", "unit": "categoría", "class": "O",
         "source": "SICT Datos Viales", "source_version": "PDF estatales", "source_period": "2025",
         "spatial_resolution": "carretera", "transformation": "Prefijo de RUTA", "uncertainty": "n/a",
         "leakage_risk": "bajo", "PTRF_domain": "movilidad"},
    ])


if __name__ == "__main__":
    main()
