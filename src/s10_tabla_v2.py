"""Paso 10: tablas de modelado _v2 con las variables de red, SIN modificar las originales.

    python src/s10_tabla_v2.py

Lee   03_PROCESSED/FE_TOP_Model_Table.csv      + features_red_sites.csv  (por site_id)
      03_PROCESSED/FE_CTN_Prediction_Table.csv + features_red.csv        (por ctn_id)
Crea  03_PROCESSED/FE_TOP_Model_Table_v2.csv y FE_CTN_Prediction_Table_v2.csv

red_snap_dist_m es una variable de CALIDAD (como road_dist_m): se queda en features_red*.csv
pero no entra a la tabla de modelado, para que el modelo no la use como predictor.
Para regresar a la versión anterior basta con volver a leer las tablas sin _v2.
"""
import pandas as pd

from common import PROCESSED, log_decision

CALIDAD = ["red_snap_dist_m"]
PARES = [("FE_TOP_Model_Table.csv", "features_red_sites.csv", "site_id"),
         ("FE_CTN_Prediction_Table.csv", "features_red.csv", "ctn_id")]


def main() -> None:
    for tabla, feats, idc in PARES:
        t = pd.read_csv(PROCESSED / tabla, low_memory=False)
        f = pd.read_csv(PROCESSED / feats).drop(columns=CALIDAD, errors="ignore")
        if not f[idc].is_unique:
            raise ValueError(f"{feats}: {idc} repetido")
        repetidas = [c for c in f.columns if c != idc and c in t.columns]
        if repetidas:
            t = t.drop(columns=repetidas)  # se reemplazan por la versión nueva
        out = t.merge(f, on=idc, how="left", validate="one_to_one")
        if len(out) != len(t):
            raise ValueError(f"{tabla}: cambió el número de filas al unir")
        faltan = out[[c for c in f.columns if c != idc]].isna().any(axis=1).sum()
        destino = tabla.replace(".csv", "_v2.csv")
        out.to_csv(PROCESSED / destino, index=False)
        print(f"{destino}: {out.shape} | filas sin variables de red: {faltan}")
        log_decision("s10", f"{destino} = {tabla} + {feats}", "Tabla nueva; la original no se modifica")


if __name__ == "__main__":
    main()
