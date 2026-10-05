"""Actualiza el texto del notebook de entrega (Avance1.20.ipynb) cuando NASA POWER y DENUE ya están
integrados a TOP_Model_Table. Calcula los números directamente de la tabla y reemplaza las frases que
decían que esas fuentes estaban pendientes. Sólo modifica celdas de texto (markdown): no toca código ni
salidas, así que la ejecución secuencial del notebook se conserva.

Uso (después de s08, s07 y de ejecutar el notebook con nbconvert):
    python src/update_eda_text.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "04_NOTEBOOKS" / "Avance1.20.ipynb"


def fmt(x: float, d: int = 2) -> str:
    return f"{x:,.{d}f}"


def main() -> None:
    t = pd.read_csv(ROOT / "03_PROCESSED" / "TOP_Model_Table.csv")
    feats = pd.read_csv(ROOT / "03_PROCESSED" / "model_feature_list.csv")["feature"].tolist()
    n_feats, n_sites = len(feats), len(t)
    if "denue_n_est_5km" not in t or "allsky_sfc_sw_dwn_mean" not in t:
        raise SystemExit("TOP_Model_Table aún no tiene NASA o DENUE. Corre s08 después de s02 y s03 para los sitios.")

    rad = t["allsky_sfc_sw_dwn_mean"]
    den = t["denue_n_est_5km"]
    rho_den = stats.spearmanr(den, t["log_tdpa"], nan_policy="omit")[0]
    zero_pct = 100 * (den == 0).mean()
    miss_pct = 100 * den.isna().mean()

    num = [c for c in feats if c in t and pd.api.types.is_numeric_dtype(t[c]) and t[c].dtype != bool
           and t[c].nunique() > 2]
    rel = sorted(((c, stats.spearmanr(t[c], t["log_tdpa"], nan_policy="omit")[0]) for c in num),
                 key=lambda x: -abs(x[1]) if not np.isnan(x[1]) else 0)[:5]
    top = ", ".join(f"`{c}` (ρ={r:+.2f})" for c, r in rel)

    repl = {
        # Celda 0: párrafo inicial
        "Se integraron 36 features de RNC 2025, ITER 2020 y SIAP 2024. NASA POWER para TOP y DENUE todavía no "
        "están integrados; los análisis y conclusiones se limitan a las fuentes presentes.":
        f"Se integraron {n_feats} features de RNC 2025, ITER 2020, SIAP 2024, NASA POWER (2015–2024) y DENUE "
        f"(establecimientos a 5 km, consulta de octubre de 2026).",
        # Celda 14: introducción de features
        "NASA POWER para TOP y DENUE están pendientes; su ausencia limita el alcance de estas asociaciones.":
        "También se incluyen clima y radiación solar (NASA POWER, malla regional de ~0.5°) y establecimientos "
        "económicos a 5 km (DENUE), agregados por sector SCIAN.",
        # Celda 40: conclusiones (dos frases)
        "- La tabla tiene 36 features territoriales de RNC, Censo e SIAP; NASA POWER y DENUE para los sitios de "
        "entrenamiento aún faltan. Las correlaciones reportadas son exploratorias y no sustituyen validación espacial.":
        f"- La tabla tiene {n_feats} features territoriales de RNC, Censo, SIAP, NASA POWER y DENUE. Las features "
        f"más asociadas con `ln(TDPA)` son {top}. Estas correlaciones son exploratorias y no sustituyen la "
        f"validación espacial.\n"
        f"- **Recurso solar (NASA POWER):** la radiación media es alta y casi uniforme ({fmt(rad.min())}–"
        f"{fmt(rad.max())} kWh/m²/día, sólo {rad.nunique()} valores distintos en {n_sites:,} sitios por la malla "
        f"regional). En esta escala el recurso solar no diferencia sitios: es contexto regional, no medición "
        f"en sitio, y sus variables deben reducirse a unas pocas para no actuar como identificador de zona.\n"
        f"- **Actividad económica (DENUE):** el número de establecimientos a 5 km tiene correlación de Spearman "
        f"{rho_den:+.2f} con `ln(TDPA)`. El {fmt(zero_pct, 1)}% de los sitios no tiene ningún establecimiento en "
        f"5 km (INEGI responde \"No hay resultados\"; es un cero real, no un faltante)"
        + (f" y el {fmt(miss_pct, 1)}% quedó sin dato por fallas de consulta" if miss_pct > 0 else "")
        + ". En el corredor, los tramos de montaña entre los km 140 y 200 no tienen servicios a 5 km, lo que "
        "es relevante para la caracterización de nodos Green Stop.",
        "- Antes del modelado, incorporar NASA/DENUE si están disponibles, revisar faltantes, confirmar la "
        "asignación del corredor y ejecutar la selección dentro de cada fold espacial.":
        "- Antes del modelado: reducir las variables redundantes de NASA POWER, revisar faltantes, confirmar la "
        "asignación del corredor y ejecutar la selección de features dentro de cada fold espacial.",
    }

    nb = json.loads(NB.read_text(encoding="utf-8"))
    done = set()
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        src = "".join(cell["source"])
        new = src
        for old, rep in repl.items():
            if old in new:
                new = new.replace(old, rep)
                done.add(old)
        if new != src:
            cell["source"] = new.splitlines(keepends=True)
    missing = [k[:70] for k in repl if k not in done]
    if missing:
        print("AVISO: no encontré estas frases (¿ya se actualizaron o cambió el texto?):")
        for m in missing:
            print("  -", m, "...")
    if done:
        shutil.copy(NB, NB.with_suffix(".ipynb.bak"))
        NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"Actualizadas {len(done)} de {len(repl)} frases en {NB.name} (respaldo: {NB.name}.bak)")
    print(f"\nResumen: {n_sites:,} sitios | {n_feats} features | radiación {fmt(rad.min())}–{fmt(rad.max())} "
          f"({rad.nunique()} valores) | DENUE ρ={rho_den:+.2f}, ceros={fmt(zero_pct, 1)}%, faltantes={fmt(miss_pct, 1)}%")
    print("Top 5 vs ln(TDPA):", top)


if __name__ == "__main__":
    main()
