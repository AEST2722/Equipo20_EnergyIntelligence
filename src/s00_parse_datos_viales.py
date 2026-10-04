"""Paso 0: extrae las tablas de estaciones de los PDF de SICT "Datos Viales" a CSV.

Los PDF traen, por carretera: CLAVE, RUTA, AÑO y por estación: LUGAR, KM, TE (tipo de estación),
SC (sentido de circulación), TDPA, composición vehicular (%), K', D y coordenadas (lat/lon).

El texto del PDF viene desordenado, por eso se usa la POSICIÓN x de cada palabra: cada número se
asigna a la columna cuyo encabezado está más cerca. Se valida cada fila (coordenadas en México,
composición que suma ~100 %) y se marcan las dudosas en vez de descartarlas en silencio.

Uso:
    python src/s00_parse_datos_viales.py            # procesa todos los PDF de 01_RAW/sict_tdpa/
Salidas:
    02_INTERIM/datos_viales_rows.csv    una fila por estación-sentido (tal como viene en el PDF)
    02_INTERIM/datos_viales_sites.csv   una fila por sitio (TDPA en ambos sentidos)  <- objetivo del modelo
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pdfplumber

from common import INTERIM, RAW, log_decision, log_download

COLS = ["KM", "TE", "SC", "TDPA", "M", "A", "B", "C2", "C3", "T3S2", "T3S3", "T3S2R4", "OTROS",
        "A_TOT", "B_TOT", "C_TOT", "K", "D", "LATITUD", "LONGITUD"]
HEADER_TOKENS = {"KM", "TE", "SC", "TDPA", "M", "A", "B", "C2", "C3", "T3S2", "T3S3", "T3S2R4",
                 "OTROS", "C", "K'", "D", "LATITUD", "LONGITUD"}
NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")
CARR_RE = re.compile(r"^(\d+)\s+C\s?A\s?R\s?R\s?:\s*(.+)$")


def group_lines(words: list[dict], tol: float = 2.5) -> list[list[dict]]:
    lines: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if lines and abs(lines[-1][0]["top"] - w["top"]) <= tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [sorted(l, key=lambda w: w["x0"]) for l in lines]


def header_centers(line: list[dict]) -> dict[str, float] | None:
    toks = [w["text"] for w in line]
    if "TDPA" not in toks or "LATITUD" not in toks:
        return None
    centers, seen = {}, {}
    for w in line:
        t = w["text"]
        if t not in HEADER_TOKENS:
            continue
        seen[t] = seen.get(t, 0) + 1
        # 'A' y 'B' aparecen dos veces: composición detallada y luego agregada (A, B, C)
        if t in ("A", "B"):
            name = t if seen[t] == 1 else f"{t}_TOT"
        else:
            name = {"C": "C_TOT", "K'": "K"}.get(t, t)
        centers[name] = (w["x0"] + w["x1"]) / 2
    return centers if len(centers) >= 18 else None


def parse_pdf(path: Path) -> pd.DataFrame:
    state = path.stem
    rows = []
    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            words = page.extract_words()
            lines = group_lines(words)
            centers, meta = None, {}
            for line in lines:
                text = " ".join(w["text"] for w in line)
                if "CLAVE:" in text and "RUTA" in text:
                    m = re.match(r"^(\d+)\s+CARR:\s*(.*?)\s+CLAVE:\s*(\S*)\s*RUTA\s*:\s*(\S*)\s*AÑO\s*:\s*(\d{4})", text)
                    if m:
                        meta = {"carr_idx": int(m.group(1)), "carretera": m.group(2).strip(),
                                "clave": m.group(3),
                                "ruta": m.group(4), "anio": int(m.group(5))}
                    continue
                c = header_centers(line)
                if c:
                    centers = c
                    continue
                if not centers or not meta:
                    continue
                x_name_end = centers["KM"] - 25
                name = " ".join(w["text"] for w in line if w["x1"] < x_name_end).strip()
                nums = [w for w in line if w["x1"] >= x_name_end and NUM_RE.match(w["text"])]
                if not name or not nums:
                    continue
                rec = {**meta, "lugar": name, "source_file": path.name, "page": pno, "state_file": state}
                for w in nums:
                    xc = (w["x0"] + w["x1"]) / 2
                    col = min(centers, key=lambda k: abs(centers[k] - xc))
                    rec[col] = float(w["text"])
                rows.append(rec)
    df = pd.DataFrame(rows)
    for c in COLS:
        if c not in df:
            df[c] = np.nan
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    comp = df[["M", "A", "B", "C2", "C3", "T3S2", "T3S3", "T3S2R4", "OTROS"]].sum(axis=1, min_count=1)
    df["comp_sum"] = comp.round(1)
    df["has_tdpa"] = df["TDPA"].notna()
    flags = []
    for r in df.itertuples():
        f = []
        if r.has_tdpa:
            if not (14 <= (r.LATITUD or 0) <= 33 and -118 <= (r.LONGITUD or 0) <= -86):
                f.append("coord_fuera_de_mexico")
            if pd.notna(r.comp_sum) and abs(r.comp_sum - 100) > 2:
                f.append("composicion_no_suma_100")
            if r.TE not in (1, 2, 3) or r.SC not in (0, 1, 2):
                f.append("TE_SC_inesperado")
        flags.append(";".join(f))
    df["qa_flag"] = flags
    return df


def to_sites(rows: pd.DataFrame) -> pd.DataFrame:
    """Una fila por sitio con TDPA total en ambos sentidos.
    Reglas (registradas en decision_log):
      1. SC=0 ya es ambos sentidos; SC=1 + SC=2 se suman.
      2. Plazas de cobro (TE=2) traen SC 1 y 2 con el mismo valor: es el aforo de la plaza repartido
         por sentido, y también se suma. Verificado: en Paso Morelos TE=2 suma 9,192 y la estación
         TE=1 del mismo sitio suma 9,260 (contarlo una vez subestimaría el TDPA a la mitad).
      5. Estaciones repetidas en PDF de estados vecinos (mismo tramo en dos archivos) se eliminan
         por ruta + km + coordenadas redondeadas.
      3. Calzadas '(Lateral)'/'(Central)' en el mismo km se suman (son flujos distintos).
      4. Si un mismo sitio tiene varios tipos de estación (TE), se usa la mediana y se guarda n_te.
    """
    d = rows[rows["has_tdpa"] & (rows["qa_flag"].fillna("") == "")].copy()
    n0 = len(d)
    d["_lat4"], d["_lon4"] = d["LATITUD"].round(4), d["LONGITUD"].round(4)
    d = d.drop_duplicates(subset=["ruta", "KM", "TE", "SC", "_lat4", "_lon4"])
    if len(d) < n0:
        log_decision("s00", f"{n0 - len(d)} filas duplicadas entre PDF de distintos estados eliminadas",
                     "Mismo tramo publicado en dos archivos estatales")
    d["calzada"] = d["lugar"].str.extract(r"\((Lateral|Central)\)", expand=False).fillna("")
    d["lugar_base"] = d["lugar"].str.replace(r"\s*\((Lateral|Central)\)", "", regex=True).str.strip()
    key = ["state_file", "ruta", "carretera", "KM", "lugar_base"]

    def two_way(g: pd.DataFrame) -> pd.Series:
        if (g["SC"] == 0).any():
            t = g.loc[g["SC"] == 0, "TDPA"].iloc[0]
        else:
            t = g["TDPA"].sum()
        w = g["TDPA"] / g["TDPA"].sum()
        return pd.Series({"tdpa_two_way": t, "pct_C": (g["C_TOT"] * w).sum(), "pct_B": (g["B_TOT"] * w).sum(),
                          "lat": g["LATITUD"].mean(), "lon": g["LONGITUD"].mean(), "n_rows": len(g)})

    per_te = d.groupby(key + ["calzada", "TE"]).apply(two_way, include_groups=False).reset_index()
    per_cal = per_te.groupby(key + ["calzada"]).agg(
        tdpa_two_way=("tdpa_two_way", "median"), pct_C=("pct_C", "mean"), pct_B=("pct_B", "mean"),
        lat=("lat", "mean"), lon=("lon", "mean"), n_te=("TE", "nunique"), te_list=("TE", lambda s: ",".join(map(str, sorted(set(s.astype(int))))))
    ).reset_index()
    sites = per_cal.groupby(key).agg(
        tdpa=("tdpa_two_way", "sum"), pct_C=("pct_C", "mean"), pct_B=("pct_B", "mean"),
        lat=("lat", "mean"), lon=("lon", "mean"), n_te=("n_te", "max"), te_list=("te_list", "first"),
        n_calzadas=("calzada", lambda s: (s != "").sum() or 1)).reset_index()
    anio = d.groupby(key)["anio"].first().reset_index()
    sites = sites.merge(anio, on=key)
    sites["toll_road"] = sites["ruta"].str.endswith("D") | sites["carretera"].str.contains("Cuota")
    sites["red"] = np.where(sites["ruta"].str.startswith("MEX"), "federal", "estatal")
    sites.insert(0, "site_id", [f"S{i:04d}" for i in range(len(sites))])
    return sites


def main() -> None:
    pdfs = sorted((RAW / "sict_tdpa").glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No hay PDF de Datos Viales en 01_RAW/sict_tdpa/.")
    frames = []
    for p in pdfs:
        df = parse_pdf(p)
        frames.append(df)
        log_download("SICT Datos Viales", "https://micrs.sct.gob.mx/infraestructura/direccion-general-de-servicios-tecnicos/datos-viales/",
                     str(p.relative_to(RAW.parent)), version=f"Datos Viales, año(s) {sorted(df['anio'].dropna().unique().astype(int).tolist())}",
                     license_="Información pública SICT", notes=f"{len(df)} filas extraídas con pdfplumber")
        print(f"{p.name}: {len(df)} filas, {df['TDPA'].notna().sum()} con TDPA")
    rows = validate(pd.concat(frames, ignore_index=True))
    INTERIM.mkdir(exist_ok=True)
    rows.to_csv(INTERIM / "datos_viales_rows.csv", index=False)
    bad = rows[rows["qa_flag"].fillna("") != ""]
    if len(bad):
        log_decision("s00", f"{len(bad)} filas con TDPA marcadas por QA y excluidas de sitios",
                     f"Motivos: {bad['qa_flag'].value_counts().to_dict()} (revisar en datos_viales_rows.csv)")
    sites = to_sites(rows)
    sites.to_csv(INTERIM / "datos_viales_sites.csv", index=False)
    log_decision("s00", "Agregación a sitio: SC1+SC2 (incluye plazas TE=2); SC0 directo; "
                 "Lateral+Central sumadas; mediana entre TE", "Ver docstring de to_sites()")
    print(f"Sitios con TDPA en ambos sentidos: {len(sites)}")


if __name__ == "__main__":
    main()
