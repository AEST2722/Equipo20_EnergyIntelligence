"""Utilidades comunes ML-SEDP: configuración, rutas, bitácoras y provenance.

Principios que este módulo hace cumplir:
- 01_RAW nunca se modifica: sólo se escribe ahí la respuesta cruda de una descarga.
- Toda descarga queda en 00_DOCUMENTATION/download_log.csv (URL, fecha, versión, licencia).
- Toda decisión manual queda en 07_TEAM_LOG/decision_log.csv.
- Toda feature se registra en 00_DOCUMENTATION/feature_provenance.csv con su clase O/D/P/M,
  periodo, escala espacial y riesgo de leakage.
"""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "01_RAW"
INTERIM = ROOT / "02_INTERIM"
PROCESSED = ROOT / "03_PROCESSED"
DOCS = ROOT / "00_DOCUMENTATION"
RESULTS = ROOT / "06_RESULTS"
TEAM_LOG = ROOT / "07_TEAM_LOG"


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def log_download(source: str, url: str, local_path: str, version: str = "",
                 license_: str = "", notes: str = "") -> None:
    """Registra cada descarga o consulta (manual o por API)."""
    _append_row(DOCS / "download_log.csv", {
        "source": source, "url": url, "local_path": local_path,
        "extraction_date": now_iso(), "version": version,
        "license": license_, "notes": notes,
    })


def log_decision(step: str, decision: str, reason: str) -> None:
    """Registra decisiones manuales, errores de snap, cambios de ruta, etc."""
    _append_row(TEAM_LOG / "decision_log.csv", {
        "timestamp": now_iso(), "step": step, "decision": decision, "reason": reason,
    })


PROVENANCE_FIELDS = [
    "feature_name", "definition", "unit", "class", "source", "source_version",
    "source_period", "spatial_resolution", "extraction_date", "transformation",
    "uncertainty", "leakage_risk", "PTRF_domain", "allowed_as_target",
]


def register_features(rows: list[dict]) -> None:
    """Añade/actualiza filas del registro de provenance (clave: feature_name)."""
    path = DOCS / "feature_provenance.csv"
    existing: dict[str, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing[r["feature_name"]] = r
    for r in rows:
        full = {k: r.get(k, "") for k in PROVENANCE_FIELDS}
        full["extraction_date"] = full["extraction_date"] or now_iso()
        # Regla del proyecto: ninguna feature se usa como etiqueta de "sitio ideal".
        full["allowed_as_target"] = full["allowed_as_target"] or "no"
        existing[full["feature_name"]] = full
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PROVENANCE_FIELDS)
        w.writeheader()
        for r in existing.values():
            w.writerow(r)


def find_first(folder: Path, patterns: tuple[str, ...]) -> Path | None:
    for pat in patterns:
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    return None


# ---------------------------------------------------------------------------
# Conjuntos de puntos: el mismo cálculo de features sirve para dos usos
#   - "sites": sitios de aforo SICT con TDPA observado -> ENTRENAMIENTO/VALIDACIÓN del modelo
#   - "ctn":   nodos del corredor MEX-095D -> PREDICCIÓN (y caracterización)
# ---------------------------------------------------------------------------
POINT_SETS = {
    "ctn": {"file": "03_PROCESSED/ctn_base.csv", "id": "ctn_id"},
    "sites": {"file": "02_INTERIM/datos_viales_sites.csv", "id": "site_id"},
}


def load_points(kind: str):
    """Devuelve (DataFrame con columnas id, lon, lat, nombre de la columna id)."""
    import pandas as pd
    spec = POINT_SETS[kind]
    df = pd.read_csv(ROOT / spec["file"])
    return df[[spec["id"], "lon", "lat"]].copy(), spec["id"]


def points_arg(parser) -> None:
    parser.add_argument("--points", choices=list(POINT_SETS), default="ctn",
                        help="ctn = nodos del corredor (predicción); sites = sitios de aforo (entrenamiento)")


def out_name(base: str, kind: str) -> str:
    """features_x.csv para CTN (compatibilidad) y features_x_sites.csv para sitios."""
    return f"{base}.csv" if kind == "ctn" else f"{base}_{kind}.csv"
