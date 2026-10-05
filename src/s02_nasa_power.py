"""Paso 2: NASA POWER (radiación y clima) para todos los CTN, mismo periodo para todos.

Uso:  python src/s02_nasa_power.py
Salidas:
  01_RAW/nasa_power/<ctn_id>.json        respuesta cruda (no se modifica)
  02_INTERIM/nasa_power_monthly.csv      serie mensual larga (ctn, parámetro, año, mes, valor)
  03_PROCESSED/features_nasa_power.csv   estadísticos por CTN

No requiere token. La resolución es de malla regional (~0.5°): varios CTN vecinos pueden
compartir la misma celda. Se registra como clase D con advertencia "no es medición de sitio".
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import requests

from common import (INTERIM, PROCESSED, RAW, load_config, load_points, log_download, out_name,
                    points_arg, register_features)

URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
FILL = -999.0


def fetch(lat: float, lon: float, cfg: dict) -> dict:
    p = cfg["nasa_power"]
    params = {"parameters": ",".join(p["parameters"]), "community": p["community"],
              "latitude": lat, "longitude": lon, "start": p["start_year"],
              "end": p["end_year"], "format": "JSON"}
    for attempt in range(4):
        r = requests.get(URL, params=params, timeout=120)
        if r.status_code == 200:
            return r.json()
        time.sleep(5 * (attempt + 1))
    r.raise_for_status()
    return {}


def to_long(ctn_id: str, payload: dict) -> pd.DataFrame:
    rows = []
    for param, series in payload["properties"]["parameter"].items():
        for key, val in series.items():
            year, month = int(key[:4]), int(key[4:])
            if month == 13:  # NASA incluye el promedio anual como mes 13
                continue
            rows.append([ctn_id, param, year, month, np.nan if val == FILL else val])
    return pd.DataFrame(rows, columns=["ctn_id", "parameter", "year", "month", "value"])


def summarize(long: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (ctn, param), g in long.groupby(["ctn_id", "parameter"]):
        clim = g.groupby("month")["value"].mean()  # climatología mensual
        out.append({
            "ctn_id": ctn, "parameter": param,
            "mean": g["value"].mean(), "std": g["value"].std(),
            "p10": g["value"].quantile(0.10), "p90": g["value"].quantile(0.90),
            "seasonal_amplitude": clim.max() - clim.min(),
            "min_month": int(clim.idxmin()), "missing_pct": g["value"].isna().mean() * 100,
        })
    wide = pd.DataFrame(out).pivot(index="ctn_id", columns="parameter")
    wide.columns = [f"{p.lower()}_{s}" for s, p in wide.columns]
    return wide.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(); points_arg(ap); kind = ap.parse_args().points
    cfg = load_config()
    pts, idc = load_points(kind)
    out_dir = RAW / "nasa_power"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for _, r in pts.iterrows():
        pid = r[idc]
        # El nombre incluye las coordenadas: si los IDs se reasignan (p. ej. al agregar estados),
        # nunca se reutiliza la respuesta de otro lugar.
        f = out_dir / f"{pid}_{r.lat:.5f}_{r.lon:.5f}.json"
        if not f.exists():  # no re-descargar: RAW es inmutable
            payload = fetch(r.lat, r.lon, cfg)
            f.write_text(json.dumps(payload), encoding="utf-8")
            log_download("NASA POWER", URL, str(f.relative_to(out_dir.parents[1])),
                         version=payload.get("header", {}).get("api", {}).get("version", ""),
                         license_="NASA open data (sin restricciones de uso)",
                         notes=f"lat={r.lat}, lon={r.lon}, {cfg['nasa_power']['start_year']}-"
                               f"{cfg['nasa_power']['end_year']}")
            time.sleep(1)
        frames.append(to_long(pid, json.loads(f.read_text(encoding="utf-8"))))
        print(f"{pid} ok")

    long = pd.concat(frames, ignore_index=True)
    long.to_csv(INTERIM / out_name("nasa_power_monthly", kind), index=False)
    feats = summarize(long).rename(columns={"ctn_id": idc})
    feats.to_csv(PROCESSED / out_name("features_nasa_power", kind), index=False)
    print(f"Features NASA POWER: {feats.shape}")

    period = f"{cfg['nasa_power']['start_year']}-{cfg['nasa_power']['end_year']}"
    units = {"allsky_sfc_sw_dwn": "kWh/m²/día", "t2m": "°C", "t2m_max": "°C",
             "prectotcorr": "mm/día", "ws10m": "m/s"}
    rows = []
    for col in feats.columns[1:]:
        base = next((u for u in units if col.startswith(u)), "")
        rows.append({"feature_name": col, "definition": f"Estadístico mensual {col}",
                     "unit": units.get(base, "") if not col.endswith(("min_month", "missing_pct")) else "",
                     "class": "D", "source": "NASA POWER", "source_version": "API v2 (ver download_log)",
                     "source_period": period, "spatial_resolution": "malla ~0.5° x 0.625°",
                     "transformation": "Serie mensual -> media/std/p10/p90/amplitud estacional",
                     "uncertainty": "Reanálisis/satélite; no es medición en sitio; CTN vecinos comparten celda",
                     "leakage_risk": "bajo", "PTRF_domain": "energía"})
    register_features(rows)


if __name__ == "__main__":
    main()
