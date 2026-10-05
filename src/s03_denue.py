"""Paso 3: DENUE (INEGI) - establecimientos a <= 5 km de cada CTN (método Buscar).

Token: regístrate gratis en https://www.inegi.org.mx/servicios/api_denue.html
y guárdalo en un archivo .env (excluido de git) como:  DENUE_TOKEN=xxxxxxxx
o como variable de entorno. NUNCA en el código ni en config.yaml.

Uso:  python src/s03_denue.py
Salidas:
  01_RAW/denue/<ctn_id>.json            respuesta cruda
  02_INTERIM/denue_establecimientos.csv  establecimientos únicos con CTN asociado
  03_PROCESSED/features_denue.csv        conteos y diversidad por CTN
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import requests

from common import (INTERIM, PROCESSED, RAW, ROOT, load_config, load_points, log_download, out_name,
                    points_arg, register_features)

URL_TMPL = "https://www.inegi.org.mx/app/api/denue/v1/consulta/Buscar/todos/{lat},{lon}/{r}/{token}"


def get_token(var: str) -> str:
    tok = os.environ.get(var)
    env = ROOT / ".env"
    if not tok and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(f"{var}="):
                tok = line.split("=", 1)[1].strip()
    if not tok or "pega_aqui" in tok:
        raise SystemExit(f"Falta el token. Crea .env con {var}=tu_token (ver .env.example).")
    return tok


def fetch_denue(lat: float, lon: float, radius: int, token: str, attempts: int = 4):
    """Consulta Buscar con reintentos. Devuelve (lista, estado) o (None, estado) si falla.
    - HTTP 200 con lista JSON -> éxito (lista vacía = 0 establecimientos reales).
    - HTTP 200 con el texto "No hay resultados." -> 0 establecimientos reales.
    - HTTP 200 con cuerpo vacío -> INEGI no devolvió registros; se trata como 0 tras confirmar con reintentos.
    - Cualquier otro caso (error HTTP, HTML, texto) -> fallo; no se guarda para reintentar después."""
    url = URL_TMPL.format(lat=lat, lon=lon, r=radius, token=token)
    status = "sin respuesta"
    empty_bodies = 0
    for k in range(attempts):
        try:
            resp = requests.get(url, timeout=180)
            status = f"HTTP {resp.status_code}"
            if resp.status_code == 200:
                body = resp.text.strip()
                # INEGI responde en texto plano "No hay resultados." cuando no hay establecimientos en el radio:
                # es un cero real (verificado el 2026-10-04 en tramos de montaña de la MEX-095D).
                if "no hay resultados" in body.lower():
                    return [], "HTTP 200 'No hay resultados' (cero real)"
                if body == "":
                    empty_bodies += 1
                    if empty_bodies >= 2:          # vacío confirmado dos veces -> sin establecimientos
                        return [], "HTTP 200 vacío (confirmado)"
                else:
                    data = resp.json()
                    if isinstance(data, list):
                        return data, status
                    status = f"HTTP 200 no es lista: {str(data)[:80]}"
        except (requests.RequestException, ValueError) as e:
            status = f"{type(e).__name__}"
        time.sleep(5 * (k + 1))
    return None, status


def scian_from_clee(clee: str) -> str:
    """La CLEE codifica entidad(2) + municipio(3) + SCIAN(6) + ...
    Si el formato no cuadra, se devuelve vacío y queda como faltante."""
    s = str(clee)
    code = s[5:11]
    return code if code.isdigit() else ""


def shannon(counts: pd.Series) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum()) if len(p) else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(); points_arg(ap); kind = ap.parse_args().points
    cfg = load_config()
    token = get_token(cfg["denue"]["token_env_var"])
    radius = cfg["denue"]["radius_m"]
    ctn, idc = load_points(kind)
    ctn = ctn.rename(columns={idc: "ctn_id"})  # nombre interno; se restaura al guardar
    out_dir = RAW / "denue"
    out_dir.mkdir(parents=True, exist_ok=True)

    recs, failed = [], []
    for _, r in ctn.iterrows():
        # El nombre incluye las coordenadas para no reutilizar respuestas de otro lugar si cambian los IDs
        f = out_dir / f"{r.ctn_id}_{r.lat:.5f}_{r.lon:.5f}.json"
        if not f.exists():
            data, status = fetch_denue(r.lat, r.lon, radius, token)
            if data is None:
                # Fallo del servidor: NO se guarda nada, para reintentar en la siguiente corrida
                failed.append(r.ctn_id)
                print(f"{r.ctn_id}: FALLÓ la consulta ({status}); se reintentará al volver a correr")
                continue
            f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            # El token NO se registra: se guarda la URL sin él.
            log_download("INEGI DENUE API Buscar",
                         URL_TMPL.format(lat=r.lat, lon=r.lon, r=radius, token="<TOKEN>"),
                         str(f.relative_to(ROOT)), version="DENUE vigente a la fecha de consulta",
                         license_="Términos de libre uso de la información del INEGI",
                         notes=f"{status}; {len(data)} registros")
            time.sleep(1)
        data = json.loads(f.read_text(encoding="utf-8"))
        for e in data:
            recs.append({"ctn_id": r.ctn_id, **e})
        print(f"{r.ctn_id}: {len(data)} establecimientos")

    if failed:
        print(f"\n{len(failed)} puntos fallaron: {failed}. Vuelve a correr el mismo comando para reintentarlos "
              "(los ya descargados no se repiten).")

    est = pd.DataFrame(recs)
    if est.empty:
        raise SystemExit("DENUE no devolvió datos. Revisa token y conexión.")
    if "Clase_actividad" not in est:
        est["Clase_actividad"] = np.nan
    est["scian"] = est["CLEE"].map(scian_from_clee) if "CLEE" in est else ""
    est["sector_2d"] = est["scian"].str[:2]
    est.to_csv(INTERIM / out_name("denue_establecimientos", kind), index=False)

    def feats(g: pd.DataFrame) -> pd.Series:
        sc = g["scian"].fillna("")
        estr = g.get("Estrato", pd.Series(dtype=str)).astype(str)
        return pd.Series({
            "denue_n_est_5km": len(g),
            "denue_n_food_lodging_5km": int(sc.str.startswith("72").sum()),
            "denue_n_retail_5km": int(sc.str[:2].isin(["46"]).sum()),
            "denue_n_gas_stations_5km": int(sc.str.startswith("468411").sum()),
            "denue_n_auto_repair_5km": int(sc.str.startswith("8111").sum()),
            "denue_n_large_est_5km": int(estr.str.contains("51|101|251", regex=True).sum()),
            "denue_sector_shannon_5km": shannon(g["sector_2d"].value_counts()),
            "denue_n_distinct_activities_5km": g["Clase_actividad"].nunique()
            if "Clase_actividad" in g else np.nan,
        })

    out = est.groupby("ctn_id").apply(feats, include_groups=False).reset_index()
    out = ctn[["ctn_id"]].merge(out, on="ctn_id", how="left")
    # Puntos consultados con éxito pero sin establecimientos -> 0 (valor real).
    # Puntos que fallaron -> se quedan como faltantes (NaN), nunca como 0.
    ok_zero = [c for c in ctn["ctn_id"] if c not in failed and c not in set(est["ctn_id"])]
    count_cols = [c for c in out.columns if c.startswith("denue_n_")]
    out.loc[out["ctn_id"].isin(ok_zero), count_cols] = 0
    # CTN sin registros: 0 establecimientos es un valor real, no faltante, SI la consulta fue exitosa.
    out.rename(columns={"ctn_id": idc}).to_csv(PROCESSED / out_name("features_denue", kind), index=False)
    print(f"Features DENUE: {out.shape}")

    base = {"class": "D", "source": "INEGI DENUE", "source_version": "consulta API (ver download_log)",
            "source_period": "fecha de consulta", "spatial_resolution": "buffer 5 km (radio API)",
            "uncertainty": "Registro administrativo; informalidad subregistrada; geocodificación variable",
            "leakage_risk": "bajo", "PTRF_domain": "economía/servicios"}
    defs = {
        "denue_n_est_5km": "Total de establecimientos", "denue_n_food_lodging_5km": "SCIAN 72 (alojamiento y alimentos)",
        "denue_n_retail_5km": "SCIAN 46 (comercio al por menor)", "denue_n_gas_stations_5km": "SCIAN 468411 (gasolineras)",
        "denue_n_auto_repair_5km": "SCIAN 8111 (reparación automotriz)",
        "denue_n_large_est_5km": "Establecimientos con 51+ personas ocupadas",
        "denue_sector_shannon_5km": "Diversidad de Shannon de sectores SCIAN 2 dígitos",
        "denue_n_distinct_activities_5km": "Número de clases de actividad distintas",
    }
    register_features([{**base, "feature_name": k, "definition": v,
                        "unit": "índice" if "shannon" in k else "conteo",
                        "transformation": "Conteo por CTN; SCIAN extraído de CLEE (pos. 6-11)"}
                       for k, v in defs.items()])


if __name__ == "__main__":
    main()
