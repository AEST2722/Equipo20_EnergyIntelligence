"""Paso 4: Censo 2020 (ITER, principales resultados por localidad) -> buffers 5/10/20 km por CTN,
y asignación de municipio a cada CTN.

Descarga manual (una vez): https://www.inegi.org.mx/programas/ccpv/2020/#datos_abiertos
  "Principales resultados por localidad (ITER)" en CSV para CDMX (09), Morelos (17) y Guerrero (12).
  Si el corredor toca otro estado (p. ej. Edomex 15), agrégalo. Descomprime en 01_RAW/censo2020/.
Opcional (recomendado): Marco Geoestadístico 2020, capa municipal (00mun.shp o mun.gpkg) en
  01_RAW/marco_geo/. Si no está, el municipio se asigna por la localidad más cercana (proxy).

Uso:  python src/s04_censo.py
"""
from __future__ import annotations

import argparse
import re

import geopandas as gpd
import numpy as np
import pandas as pd

from common import (INTERIM, PROCESSED, RAW, find_first, load_config, load_points, log_decision,
                    log_download, out_name, points_arg, register_features)

NUM_COLS = ["POBTOT", "GRAPROES", "VIVTOT", "TVIVHAB", "VPH_C_ELEC", "PEA"]


def dms_to_dd(s: str) -> float:
    """Convierte '99°07'59.611\" W' a -99.1332. Acepta también decimales."""
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    try:
        return float(s)
    except ValueError:
        pass
    m = re.match(r"(\d+)°\s*(\d+)'\s*([\d.]+)\"?\s*([NSEWO])", s)
    if not m:
        return np.nan
    deg, mi, se, hemi = m.groups()
    dd = float(deg) + float(mi) / 60 + float(se) / 3600
    return -dd if hemi in ("S", "W", "O") else dd


def load_iter() -> pd.DataFrame:
    files = sorted((RAW / "censo2020").glob("**/ITER_*CSV20*.csv")) or \
        sorted((RAW / "censo2020").glob("**/*.csv"))
    if not files:
        raise SystemExit("No hay archivos ITER en 01_RAW/censo2020/.")
    frames = []
    for f in files:
        df = pd.read_csv(f, dtype=str, encoding="utf-8-sig", low_memory=False)
        df.columns = [c.upper().strip() for c in df.columns]
        frames.append(df)
        log_download("INEGI Censo 2020 ITER", "https://www.inegi.org.mx/programas/ccpv/2020/",
                     str(f.relative_to(RAW.parent)), version="Censo de Población y Vivienda 2020",
                     license_="Términos de libre uso de la información del INEGI",
                     notes="Descarga manual; leído sin modificar")
    df = pd.concat(frames, ignore_index=True)
    # Quitar filas de totales (LOC 0000) y agregados de localidades de 1-2 viviendas (9998/9999)
    df["LOC"] = df["LOC"].astype(int)
    df = df[~df["LOC"].isin([0, 9998, 9999])].copy()
    for c in NUM_COLS:
        if c in df:
            # '*' = dato confidencial; 'N/D' = no disponible -> NaN (faltante explícito)
            df[c] = pd.to_numeric(df[c].replace({"*": np.nan, "N/D": np.nan}), errors="coerce")
    df["lon"] = df["LONGITUD"].map(dms_to_dd)
    df["lat"] = df["LATITUD"].map(dms_to_dd)
    df["cve_mun"] = df["ENTIDAD"].str.zfill(2) + df["MUN"].str.zfill(3)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(); points_arg(ap); kind = ap.parse_args().points
    cfg = load_config()
    crs_geo, crs_m = cfg["crs"]["geographic"], cfg["crs"]["projected"]
    loc = load_iter()
    n_nocoord = loc["lon"].isna().sum()
    if n_nocoord:
        log_decision("s04", f"{n_nocoord} localidades sin coordenada válida excluidas de buffers",
                     "No se pueden ubicar espacialmente")
    loc = loc.dropna(subset=["lon", "lat"])
    gloc = gpd.GeoDataFrame(loc, geometry=gpd.points_from_xy(loc.lon, loc.lat), crs=crs_geo).to_crs(crs_m)

    pts, idc = load_points(kind)
    ctn = gpd.GeoDataFrame(pts.rename(columns={idc: "ctn_id"}),
                           geometry=gpd.points_from_xy(pts.lon, pts.lat), crs=crs_geo).to_crs(crs_m)
    thr = cfg["censo"]["rural_threshold"]
    out = ctn[["ctn_id"]].copy()

    # Localidades dentro de 20 km de algún CTN: tabla intermedia para el EDA
    max_b = max(cfg["censo"]["buffers_m"])
    near = gpd.sjoin(gloc, gpd.GeoDataFrame(geometry=ctn.buffer(max_b), crs=crs_m),
                     predicate="within", how="inner").drop_duplicates(subset=["ENTIDAD", "MUN", "LOC"])
    near.drop(columns=["geometry", "index_right"]).to_csv(
        INTERIM / out_name("censo_localidades_corredor", kind), index=False)

    for b in cfg["censo"]["buffers_m"]:
        km = b // 1000
        buf = gpd.GeoDataFrame(ctn[["ctn_id"]], geometry=ctn.buffer(b), crs=crs_m)
        j = gpd.sjoin(gloc, buf, predicate="within", how="inner")
        g = j.groupby("ctn_id")
        agg = pd.DataFrame({
            f"pop_{km}km": g["POBTOT"].sum(min_count=1),
            f"n_localities_{km}km": g.size(),
            f"rural_pop_share_{km}km": g.apply(
                lambda d: d.loc[d.POBTOT < thr, "POBTOT"].sum() / d.POBTOT.sum()
                if d.POBTOT.sum() > 0 else np.nan, include_groups=False),
        })
        if km == 10:
            agg["schooling_mean_10km"] = g.apply(
                lambda d: np.average(d.GRAPROES.dropna(), weights=d.loc[d.GRAPROES.notna(), "POBTOT"])
                if d.GRAPROES.notna().any() and d.loc[d.GRAPROES.notna(), "POBTOT"].sum() > 0 else np.nan,
                include_groups=False)
            agg["pct_dwellings_electricity_10km"] = g.apply(
                lambda d: 100 * d.VPH_C_ELEC.sum() / d.TVIVHAB.sum() if d.TVIVHAB.sum() > 0 else np.nan,
                include_groups=False)
        out = out.merge(agg.reset_index(), on="ctn_id", how="left")
        # Sin localidades en el buffer -> población 0 (valor real, no faltante)
        out[f"pop_{km}km"] = out[f"pop_{km}km"].fillna(0)
        out[f"n_localities_{km}km"] = out[f"n_localities_{km}km"].fillna(0).astype(int)

    # Distancia a la localidad más cercana y su tamaño
    nn = gpd.sjoin_nearest(ctn[["ctn_id", "geometry"]], gloc[["POBTOT", "NOM_LOC", "cve_mun", "geometry"]],
                           distance_col="dist_nearest_locality_m", how="left").drop_duplicates("ctn_id")
    out = out.merge(nn[["ctn_id", "dist_nearest_locality_m", "POBTOT", "cve_mun"]]
                    .rename(columns={"POBTOT": "pop_nearest_locality", "cve_mun": "cve_mun_nearest_loc"}),
                    on="ctn_id", how="left")

    # Asignación de municipio
    mun_file = find_first(RAW / "marco_geo", ("*mun*.shp", "*mun*.gpkg", "**/*mun*.shp", "**/*mun*.gpkg"))
    if mun_file:
        mun = gpd.read_file(mun_file).to_crs(crs_m)
        mun.columns = [c.upper() if c != "geometry" else c for c in mun.columns]
        key = "CVEGEO" if "CVEGEO" in mun else None
        if key is None:
            mun["CVEGEO"] = mun["CVE_ENT"].astype(str).str.zfill(2) + mun["CVE_MUN"].astype(str).str.zfill(3)
        j = gpd.sjoin(ctn[["ctn_id", "geometry"]], mun[["CVEGEO", "geometry"]], predicate="within", how="left")
        out = out.merge(j[["ctn_id", "CVEGEO"]].rename(columns={"CVEGEO": "cve_mun"}), on="ctn_id", how="left")
        out["mun_assignment_method"] = "polygon_marco_geoestadistico"
        log_download("INEGI Marco Geoestadístico", "https://www.inegi.org.mx/temas/mg/",
                     str(mun_file.relative_to(RAW.parent)), license_="Libre uso INEGI")
    else:
        out["cve_mun"] = out["cve_mun_nearest_loc"]
        out["mun_assignment_method"] = "nearest_locality_proxy"
        log_decision("s04", "Municipio asignado por localidad más cercana",
                     "No se encontró Marco Geoestadístico; puede fallar cerca de límites municipales")
    out = out.drop(columns="cve_mun_nearest_loc")
    out.rename(columns={"ctn_id": idc}).to_csv(PROCESSED / out_name("features_censo", kind), index=False)
    print(f"Features Censo: {out.shape}")

    base = {"source": "INEGI Censo 2020 ITER", "source_version": "CPV 2020", "source_period": "2020",
            "leakage_risk": "bajo", "PTRF_domain": "población/sociedad",
            "uncertainty": "Snapshot 2020; localidades como puntos; '*' confidencial -> NaN; "
                           "población flotante turística no captada"}
    rows = []
    for b in cfg["censo"]["buffers_m"]:
        km = b // 1000
        rows += [
            {**base, "feature_name": f"pop_{km}km", "definition": f"Población en localidades a <= {km} km",
             "unit": "habitantes", "class": "D", "spatial_resolution": f"buffer {km} km",
             "transformation": "Suma de POBTOT de localidades (punto) dentro del buffer, EPSG:6372"},
            {**base, "feature_name": f"n_localities_{km}km", "definition": f"Número de localidades a <= {km} km",
             "unit": "conteo", "class": "D", "spatial_resolution": f"buffer {km} km",
             "transformation": "Conteo de localidades en buffer"},
            {**base, "feature_name": f"rural_pop_share_{km}km",
             "definition": f"Fracción de población en localidades < {thr} hab.", "unit": "fracción",
             "class": "D", "spatial_resolution": f"buffer {km} km", "transformation": "Cociente"},
        ]
    rows += [
        {**base, "feature_name": "schooling_mean_10km", "definition": "Grado promedio de escolaridad ponderado",
         "unit": "años", "class": "D", "spatial_resolution": "buffer 10 km",
         "transformation": "Promedio ponderado por POBTOT"},
        {**base, "feature_name": "pct_dwellings_electricity_10km", "definition": "% viviendas habitadas con electricidad",
         "unit": "%", "class": "P", "spatial_resolution": "buffer 10 km",
         "transformation": "100*VPH_C_ELEC/TVIVHAB", "uncertainty": base["uncertainty"] +
         "; proxy débil de acceso a red, NO de capacidad de red"},
        {**base, "feature_name": "dist_nearest_locality_m", "definition": "Distancia a la localidad más cercana",
         "unit": "m", "class": "D", "spatial_resolution": "punto", "transformation": "sjoin_nearest EPSG:6372"},
        {**base, "feature_name": "pop_nearest_locality", "definition": "Población de la localidad más cercana",
         "unit": "habitantes", "class": "D", "spatial_resolution": "localidad", "transformation": "sjoin_nearest"},
    ]
    register_features(rows)


if __name__ == "__main__":
    main()
