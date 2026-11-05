"""Paso 12: tablas _v3 = tablas _v2 + uso del suelo (s11). No modifica las tablas anteriores.

    python src/s12_tabla_v3.py

Crea 03_PROCESSED/FE_TOP_Model_Table_v3.csv y FE_CTN_Prediction_Table_v3.csv.
Para regresar a la versión anterior basta con leer las tablas _v2.
"""
import pandas as pd

from common import PROCESSED, log_decision

PARES = [("FE_TOP_Model_Table_v2.csv", "features_uso_suelo_sites.csv", "site_id"),
         ("FE_CTN_Prediction_Table_v2.csv", "features_uso_suelo.csv", "ctn_id")]


def main() -> None:
    for tabla, feats, idc in PARES:
        t = pd.read_csv(PROCESSED / tabla, low_memory=False)
        f = pd.read_csv(PROCESSED / feats)
        if not f[idc].is_unique:
            raise ValueError(f"{feats}: {idc} repetido")
        t = t.drop(columns=[c for c in f.columns if c != idc and c in t.columns])
        out = t.merge(f, on=idc, how="left", validate="one_to_one")
        if len(out) != len(t):
            raise ValueError(f"{tabla}: cambió el número de filas al unir")
        faltan = out[[c for c in f.columns if c != idc]].isna().any(axis=1).sum()
        destino = tabla.replace("_v2.csv", "_v3.csv")
        out.to_csv(PROCESSED / destino, index=False)
        print(f"{destino}: {out.shape} | filas sin uso de suelo: {faltan}")
        log_decision("s12", f"{destino} = {tabla} + {feats}", "Tabla nueva; la anterior no se modifica")


if __name__ == "__main__":
    main()
