"""Paso 11: uso del suelo y vegetación alrededor de aforos y CTN (dimensión TERRITORIO del PTRF).

Fuente recomendada por el sponsor (diap. 13): INEGI, Conjunto de datos vectoriales de Uso del
Suelo y Vegetación, escala 1:250 000, Serie VII (imágenes base 2018). Archivo esperado en
01_RAW/land_use/*.shp (p. ej. usv250s7gw.shp, redistribuido por CONABIO).

Para cada punto y cada radio (buffers multiescala del sponsor: 1 km = sitio inmediato,
5 km = entorno) se calcula el % del área en cada clase:
    usv_agricola_riego, usv_agricola_temporal, usv_pastizal, usv_bosque, usv_selva,
    usv_matorral, usv_urbano, usv_agua, usv_sin_vegetacion, usv_otros
más usv_diversidad (índice de Shannon de las clases) y usv_clase_sitio (clase en el punto).

La clasificación se hace por palabras clave sobre la DESCRIPCIÓN de INEGI y queda documentada
en este archivo; al final se imprime qué descripciones cayeron en "otros" para revisarlas.

Uso:   python src/s11_uso_suelo.py
Salidas: 03_PROCESSED/features_uso_suelo_sites.csv y 03_PROCESSED/features_uso_suelo.csv
"""
from __future__ import annotations

import time
import unicodedata

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from pyproj import Transformer

from common import PROCESSED, RAW, load_config, load_points, log_decision, register_features

RADIOS_M = [1000, 5000]
# "otros" agrupa vegetación de humedales y costera (manglar, tular, popal, dunas, halófila) y palmares;
# en el corredor representa <1% del área, casi toda cerca de Acapulco.
CLASES = ["agricola_riego", "agricola_temporal", "pastizal", "bosque", "selva", "matorral",
          "urbano", "agua", "sin_vegetacion", "otros"]


def reparar_texto(texto) -> str:
    """El .dbf viene en UTF-8 pero sin .cpg, y se lee como Latin-1 ("VEGETACIÃN"). Se corrige aquí."""
    t = str(texto)
    try:
        return t.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return t  # ya venía bien


def normalizar(texto) -> str:
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return t.upper().strip()


def clasificar(desc: str) -> str:
    """Clase agregada a partir de la descripción de INEGI (incluye vegetación secundaria)."""
    d = normalizar(desc)
    if "URBAN" in d or "ASENTAMIENTO" in d:
        return "urbano"
    if "AGUA" in d or "ACUICOLA" in d:
        return "agua"
    if "AGRICULTURA" in d and "RIEGO" in d:
        return "agricola_riego"
    if "AGRICULTURA" in d:                      # temporal, humedad, nómada
        return "agricola_temporal"
    if "PASTIZAL" in d or "SABANA" in d or "SABANOIDE" in d or "PRADERA" in d:
        return "pastizal"
    if "BOSQUE" in d:
        return "bosque"
    if "SELVA" in d:
        return "selva"
    if "MATORRAL" in d or "MEZQUITAL" in d or "CHAPARRAL" in d:
        return "matorral"
    if "SIN VEGETACION" in d or "DESPROVISTO" in d or "AREA DESPROVISTA" in d:
        return "sin_vegetacion"
    return "otros"


def campo_descripcion(campos) -> str:
    for clave in ("DESCRIP", "DESCRIPCIO", "DES_VEG", "DESVEG", "USO", "TIPO"):
        for c in campos:
            if clave in c.upper():
                return c
    raise SystemExit(f"No encontré un campo de descripción. Campos disponibles: {list(campos)}")


def main() -> None:
    t0 = time.time()
    cfg = load_config()
    crs_geo, crs_m = cfg["crs"]["geographic"], cfg["crs"]["projected"]
    carpeta = RAW / "land_use"
    shp = sorted(carpeta.glob("*.shp"))
    if not shp:
        raise SystemExit(f"No hay ningún .shp en {carpeta}")
    path = str(shp[0])
    info = pyogrio.read_info(path)
    campo = campo_descripcion(info["fields"])
    print(f"Archivo: {shp[0].name} | CRS: {info['crs']} | campo de descripción: {campo}")

    sets = {k: load_points(k) for k in ("sites", "ctn")}
    lons = pd.concat([s[0].lon for s in sets.values()])
    lats = pd.concat([s[0].lat for s in sets.values()])
    margen = 0.07  # ~7 km, cubre el buffer de 5 km
    t = Transformer.from_crs(crs_geo, info["crs"], always_xy=True)
    x0, y0 = t.transform(lons.min() - margen, lats.min() - margen)
    x1, y1 = t.transform(lons.max() + margen, lats.max() + margen)
    usv = gpd.read_file(path, bbox=(x0, y0, x1, y1), columns=[campo]).to_crs(crs_m)
    usv["geometry"] = usv.geometry.make_valid()
    usv[campo] = usv[campo].map(reparar_texto)
    usv["clase"] = usv[campo].map(clasificar)
    print(f"Polígonos leídos en la zona: {len(usv):,}")

    area = usv.geometry.area
    resumen = (pd.DataFrame({"descripcion": usv[campo], "clase": usv["clase"], "km2": area / 1e6})
               .groupby(["clase", "descripcion"]).km2.sum().reset_index().sort_values("km2", ascending=False))
    otros = resumen[resumen.clase == "otros"]
    pct_otros = 100 * otros.km2.sum() / resumen.km2.sum()
    print(f"Área sin clasificar ('otros'): {pct_otros:.1f}% de la zona")
    if len(otros):
        print("Descripciones en 'otros' con más área (revisar si alguna debe reclasificarse):")
        print(otros.head(12).to_string(index=False))
    resumen.to_csv(PROCESSED / "uso_suelo_clasificacion.csv", index=False, encoding="utf-8-sig")

    for kind, (pts, idc) in sets.items():
        g = gpd.GeoDataFrame(pts[[idc]], geometry=gpd.points_from_xy(pts.lon, pts.lat), crs=crs_geo).to_crs(crs_m)
        out = pts[[idc]].copy()
        # Clase en el punto exacto
        en_punto = gpd.sjoin(g, usv[["clase", "geometry"]], how="left", predicate="within")
        out["usv_clase_sitio"] = en_punto[~en_punto.index.duplicated()]["clase"].fillna("otros").to_numpy()
        for r in RADIOS_M:
            sufijo = f"{r // 1000}km"
            buf = g.copy()
            buf["geometry"] = g.buffer(r)
            inter = gpd.overlay(buf, usv[["clase", "geometry"]], how="intersection", keep_geom_type=True)
            inter["a"] = inter.geometry.area
            tabla = inter.pivot_table(index=idc, columns="clase", values="a", aggfunc="sum", fill_value=0.0)
            tabla = tabla.reindex(columns=CLASES, fill_value=0.0)
            total = np.pi * r ** 2
            pct = (100 * tabla / total).clip(upper=100)
            pct.columns = [f"usv_{c}_{sufijo}" for c in pct.columns]
            p = tabla.div(tabla.sum(axis=1).replace(0, np.nan), axis=0)
            pct[f"usv_diversidad_{sufijo}"] = -(p * np.log(p.where(p > 0))).sum(axis=1)
            out = out.merge(pct.reset_index(), on=idc, how="left")
        out = out.fillna({c: 0.0 for c in out.columns if c.startswith("usv_") and c != "usv_clase_sitio"})
        nombre = "features_uso_suelo.csv" if kind == "ctn" else f"features_uso_suelo_{kind}.csv"
        out.to_csv(PROCESSED / nombre, index=False)
        print(f"{nombre}: {out.shape} | clase en el sitio: {out.usv_clase_sitio.value_counts().to_dict()}")

    log_decision("s11", f"Uso de suelo INEGI Serie VII ({shp[0].name}); campo {campo}; "
                 f"clases por palabra clave; {pct_otros:.1f}% del área en 'otros'",
                 "Buffers de 1 y 5 km (multiescala del sponsor); áreas en EPSG proyectado")
    comun = dict(source="INEGI Uso del Suelo y Vegetación 1:250 000 Serie VII (base 2018)", unit="% del área",
                 source_period="2018", spatial_resolution="1:250 000", leakage_risk="bajo (no usa aforo)",
                 PTRF_domain="Compatibilidad", uncertainty="escala 1:250 000 (polígono mínimo ~50 ha); año 2018",
                 transformation="% del área del buffer por clase")
    filas = [{**comun, "feature_name": f"usv_{c}_{r // 1000}km", "class": "continua",
              "definition": f"% del área a {r // 1000} km clasificada como {c}"} for r in RADIOS_M for c in CLASES]
    filas += [{**comun, "feature_name": f"usv_diversidad_{r // 1000}km", "class": "continua", "unit": "índice",
               "definition": f"Diversidad de Shannon de las clases a {r // 1000} km"} for r in RADIOS_M]
    filas += [{**comun, "feature_name": "usv_clase_sitio", "class": "categórica", "unit": "",
               "definition": "Clase de uso de suelo en el punto exacto"}]
    register_features(filas)
    print(f"Listo en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
