"""Paso 7: integra todas las features en CTN_Master y genera el Data Quality Report.

Uso:  python src/s07_build_master.py
Salidas:
  03_PROCESSED/CTN_Master.csv / .gpkg
  06_RESULTS/data_quality_report.md + data_quality_completeness.csv
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd

from common import DOCS, PROCESSED, RESULTS, load_config, now_iso

FEATURE_FILES = ["features_road.csv", "features_nasa_power.csv", "features_denue.csv", "features_censo.csv",
                 "features_dgsiap.csv", "features_sict.csv"]
BLOCK_KM = 50  # bloques espaciales a lo largo del corredor para validación futura


def main() -> None:
    cfg = load_config()
    base = gpd.read_file(PROCESSED / "ctn_base.gpkg")
    master = base.copy()
    used = []
    for f in FEATURE_FILES:
        p = PROCESSED / f
        if p.exists():
            df = pd.read_csv(p, dtype={"cve_mun": str})
            dup = [c for c in df.columns if c in master.columns and c != "ctn_id"]
            master = master.merge(df.drop(columns=dup), on="ctn_id", how="left")
            used.append(f)
    master["spatial_block"] = (master["chainage_km"] // BLOCK_KM).astype(int)
    master.to_file(PROCESSED / "CTN_Master.gpkg", driver="GPKG")
    master.drop(columns="geometry").to_csv(PROCESSED / "CTN_Master.csv", index=False)

    # ---------------- Data Quality Report ----------------
    df = master.drop(columns="geometry")
    prov = pd.read_csv(DOCS / "feature_provenance.csv") if (DOCS / "feature_provenance.csv").exists() \
        else pd.DataFrame(columns=["feature_name"])
    comp = pd.DataFrame({
        "feature": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "completeness_pct": (df.notna().mean() * 100).round(1).values,
        "n_unique": df.nunique().values,
    }).merge(prov[["feature_name", "class", "source", "source_period", "spatial_resolution",
                   "uncertainty", "leakage_risk"]].rename(columns={"feature_name": "feature"}),
             on="feature", how="left")
    RESULTS.mkdir(exist_ok=True)
    comp.to_csv(RESULTS / "data_quality_completeness.csv", index=False)
    per_ctn = (df.notna().mean(axis=1) * 100).round(1)
    periods = prov[prov["feature_name"].isin(df.columns)].groupby("source")["source_period"].first()

    lines = [
        "# Data Quality Report – CTN_Master", "",
        f"Generado: {now_iso()} | Corredor: {cfg['corridor_id']} | CRS de cálculo: {cfg['crs']['projected']}", "",
        f"- CTN: {len(df)} | Features: {df.shape[1]} | Fuentes integradas: {', '.join(used)}",
        f"- CTN con geometría RNC_SNAPPED: {(df['geometry_status'] == 'RNC_SNAPPED').sum()} de {len(df)}",
        f"- Duplicados de ctn_id: {df['ctn_id'].duplicated().sum()}", "",
        "## Completeness por CTN", f"- Mínima: {per_ctn.min()}% | Mediana: {per_ctn.median()}%", "",
        "## Periodos por fuente (Consistency: no mezclar años sin registrarlo)",
        *[f"- {s}: {p}" for s, p in periods.items()], "",
        "## Features con completeness < 100%",
        *[f"- {r.feature}: {r.completeness_pct}%" for r in comp.itertuples() if r.completeness_pct < 100], "",
        "## Escalas espaciales (Resolution)",
        *[f"- {r.feature}: {r.spatial_resolution}" for r in comp.dropna(subset=['spatial_resolution']).itertuples()], "",
        "## Riesgos de leakage señalados",
        *[f"- {r.feature}: {r.leakage_risk}" for r in comp.dropna(subset=['leakage_risk']).itertuples()
          if str(r.leakage_risk).lower() != "bajo"], "",
        "Notas: variables `mun_*` describen el municipio, no el predio del CTN. Ninguna feature está",
        "autorizada como etiqueta de 'sitio ideal' (allowed_as_target = no).",
    ]
    (RESULTS / "data_quality_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"CTN_Master: {df.shape} -> 03_PROCESSED/ | Reporte -> 06_RESULTS/data_quality_report.md")


if __name__ == "__main__":
    main()
