"""Paso 5: DGSIAP/SIAP - producción agrícola municipal -> contexto agrícola por CTN.

Descarga manual: https://nube.agricultura.gob.mx/datosAbiertos/Agricola.php
  CSV de cierre agrícola municipal del año configurado (dgsiap.year). Guárdalo en 01_RAW/dgsiap/.

IMPORTANTE (error #2 de la guía): estas variables describen el MUNICIPIO, no el predio del CTN.
Por eso todas llevan prefijo `mun_` y spatial_resolution = municipal. Varios CTN del mismo
municipio comparten valores: en validación espacial conviene bloquear por municipio.

Uso:  python src/s05_dgsiap.py   (requiere haber corrido s04 para tener cve_mun)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import (INTERIM, PROCESSED, RAW, POINT_SETS, find_first, load_config, log_decision, log_download,
                    out_name, points_arg, register_features)


def col(df: pd.DataFrame, *cands: str) -> str:
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    raise KeyError(f"No encontré ninguna de {cands} en {list(df.columns)}")


def load(year: int) -> pd.DataFrame:
    f = find_first(RAW / "dgsiap", (f"*{year}*.csv", "*.csv"))
    if f is None:
        raise SystemExit("No hay CSV en 01_RAW/dgsiap/.")
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(f, encoding=enc, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    log_download("SIAP/DGSIAP cierre agrícola municipal", "https://nube.agricultura.gob.mx/datosAbiertos/Agricola.php",
                 str(f.relative_to(RAW.parent)), version=f"Cierre {year}",
                 license_="Datos abiertos Gobierno de México (Libre uso MX)", notes="Descarga manual")
    c_year = col(df, "Anio", "Año", "anio")
    df = df[df[c_year] == year].copy()
    if df.empty:
        raise SystemExit(f"El CSV no contiene registros del año {year}. Ajusta dgsiap.year.")
    ren = {
        col(df, "Idestado"): "id_ent", col(df, "Idmunicipio"): "id_mun",
        col(df, "Nommunicipio"): "nom_mun", col(df, "Nomcultivo"): "cultivo",
        col(df, "Nommodalidad"): "modalidad", col(df, "Nomcicloproductivo", "Nomciclo"): "ciclo",
        col(df, "Sembrada"): "sembrada_ha", col(df, "Cosechada"): "cosechada_ha",
        col(df, "Siniestrada"): "siniestrada_ha", col(df, "Valorproduccion"): "valor_mxn",
    }
    df = df.rename(columns=ren)
    for c in ("sembrada_ha", "cosechada_ha", "siniestrada_ha", "valor_mxn"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["cve_mun"] = df["id_ent"].astype(int).astype(str).str.zfill(2) + \
        df["id_mun"].astype(int).astype(str).str.zfill(3)
    df["cultivo"] = df["cultivo"].astype(str).str.strip().str.upper()
    return df


def shannon(w: pd.Series) -> float:
    p = w[w > 0] / w[w > 0].sum()
    return float(-(p * np.log(p)).sum()) if len(p) else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(); points_arg(ap); kind = ap.parse_args().points
    idc = POINT_SETS[kind]["id"]
    cfg = load_config()
    year = cfg["dgsiap"]["year"]
    agri = load(year)
    censo = pd.read_csv(PROCESSED / out_name("features_censo", kind), dtype={"cve_mun": str})
    censo = censo.rename(columns={idc: "ctn_id"})
    muns = censo["cve_mun"].dropna().unique()
    sub = agri[agri["cve_mun"].isin(muns)].copy()
    sub.to_csv(INTERIM / out_name("dgsiap_cultivos_municipios_corredor", kind), index=False)  # para EDA (cardinalidad)

    def agg(g: pd.DataFrame) -> pd.Series:
        by_crop = g.groupby("cultivo")["sembrada_ha"].sum().sort_values(ascending=False)
        sown = g["sembrada_ha"].sum()
        return pd.Series({
            "mun_agri_sown_ha": sown,
            "mun_agri_harvested_ha": g["cosechada_ha"].sum(),
            "mun_agri_value_mxn": g["valor_mxn"].sum(),
            "mun_agri_n_crops": by_crop.index.nunique(),
            "mun_agri_crop_shannon": shannon(by_crop),
            "mun_agri_top_crop": by_crop.index[0] if len(by_crop) else np.nan,
            "mun_agri_top_crop_share": by_crop.iloc[0] / sown if sown > 0 else np.nan,
            "mun_agri_irrigated_share": g.loc[g["modalidad"].str.upper().str.contains("RIEGO", na=False),
                                              "sembrada_ha"].sum() / sown if sown > 0 else np.nan,
            "mun_agri_loss_share": g["siniestrada_ha"].sum() / sown if sown > 0 else np.nan,
        })

    mun_feats = sub.groupby("cve_mun").apply(agg, include_groups=False).reset_index()
    out = censo[["ctn_id", "cve_mun"]].merge(mun_feats, on="cve_mun", how="left")
    missing = out.loc[out["mun_agri_sown_ha"].isna(), "cve_mun"].unique()
    if len(missing):
        log_decision("s05", f"Municipios sin registro agrícola {year}: {list(missing)}",
                     "Puede ser municipio urbano (p. ej. alcaldías CDMX) -> faltante estructural, no aleatorio")
    out.drop(columns="cve_mun").rename(columns={"ctn_id": idc}).to_csv(
        PROCESSED / out_name("features_dgsiap", kind), index=False)
    print(f"Features DGSIAP: {out.shape}; municipios sin datos: {len(missing)}")

    base = {"class": "D", "source": "SIAP/DGSIAP", "source_version": f"Cierre agrícola {year}",
            "source_period": str(year), "spatial_resolution": "municipal",
            "uncertainty": "Contexto MUNICIPAL, no predial; CTN del mismo municipio comparten valor",
            "leakage_risk": "medio: agrupar por municipio en validación espacial", "PTRF_domain": "agricultura"}
    defs = {"mun_agri_sown_ha": ("Superficie sembrada total del municipio", "ha"),
            "mun_agri_harvested_ha": ("Superficie cosechada total", "ha"),
            "mun_agri_value_mxn": ("Valor de la producción agrícola", "MXN"),
            "mun_agri_n_crops": ("Número de cultivos distintos", "conteo"),
            "mun_agri_crop_shannon": ("Diversidad de Shannon por superficie sembrada", "índice"),
            "mun_agri_top_crop": ("Cultivo con mayor superficie", "categoría"),
            "mun_agri_top_crop_share": ("Fracción de superficie del cultivo principal", "fracción"),
            "mun_agri_irrigated_share": ("Fracción de superficie en riego", "fracción"),
            "mun_agri_loss_share": ("Fracción siniestrada/sembrada", "fracción")}
    register_features([{**base, "feature_name": k, "definition": d, "unit": u,
                        "transformation": "Agregación municipal de registros por cultivo/ciclo/modalidad"}
                       for k, (d, u) in defs.items()])


if __name__ == "__main__":
    main()
