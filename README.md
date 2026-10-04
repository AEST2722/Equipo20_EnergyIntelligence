# ML–SEDP · Corredor CDMX–Acapulco (MEX-095D)

Pipeline reproducible de adquisición, integración y calidad de datos territoriales para caracterizar
Candidate Territorial Nodes (CTN) de infraestructura energética distribuida (Green Stop), y EDA del Avance 1.

> **ML inference ≠ engineering decision; territorial suitability ≠ project feasibility.**
> Este repositorio describe y caracteriza el territorio. No decide dónde construir.

## 1. Instalación

```bash
pip install -r requirements.txt
cp .env.example .env      # y pega tu token de DENUE dentro de .env (no se sube a GitHub)
```

## 2. Descargas manuales (una sola vez, sin modificar los archivos)

| Fuente | Qué descargar | Dónde guardarlo |
|---|---|---|
| INEGI RNC 2025 | Red Nacional de Caminos completa (GeoPackage o SHP): https://www.inegi.org.mx/programas/rnc/ | `01_RAW/road_network/` |
| INEGI Censo 2020 | "Principales resultados por localidad (ITER)" en CSV para CDMX (09), Guerrero (12), Edomex (15), Morelos (17), Puebla (21): https://www.inegi.org.mx/programas/ccpv/2020/ | `01_RAW/censo2020/` |
| INEGI Marco Geoestadístico (opcional, recomendado) | Capa municipal (`00mun.shp`): https://www.inegi.org.mx/temas/mg/ | `01_RAW/marco_geo/` |
| SIAP/DGSIAP | CSV de cierre agrícola municipal del año en `config.yaml`: https://nube.agricultura.gob.mx/datosAbiertos/Agricola.php | `01_RAW/dgsiap/` |
| SICT Datos Viales | PDF por estado (Morelos, CDMX, Edomex, Guerrero, Puebla): https://micrs.sct.gob.mx/infraestructura/direccion-general-de-servicios-tecnicos/datos-viales/ | `01_RAW/sict_tdpa/` (se extraen solos con `s00`) |
| INEGI DENUE | Token gratuito: https://www.inegi.org.mx/servicios/api_denue.html | `.env` |

NASA POWER no requiere descarga ni token. Copernicus (NDVI) y Uso de Suelo y Vegetación quedan para el Avance 2.

## 3. Planteamiento supervisado

| Elemento | Definición |
|---|---|
| Unidad de entrenamiento | Sitio de aforo SICT (TOP), extraído de los PDF de Datos Viales |
| Variable objetivo | `log_tdpa` = ln(TDPA en ambos sentidos), observado por SICT (no sale de las entradas) |
| Entradas | Features territoriales alrededor de cada sitio: RNC, NASA POWER, DENUE, Censo, SIAP |
| Conjunto de predicción | CTN del corredor MEX-095D (donde casi no hay aforos) |
| Validación | GroupKFold espacial (bloques de 25 km / carretera) + prueba externa con los sitios de la MEX-095D |
| Baseline y alternativos | Mediana por tipo de vía, Ridge, Random Forest, Gradient Boosting |

Con los datos de Morelos, la correlación del TDPA entre sitios cae de 0.62 (< 2 km) a ~0 (> 25 km):
de ahí el tamaño de bloque de 25 km.

## 4. Orden de ejecución

```bash
python src/s00_parse_datos_viales.py             # PDF SICT -> sitios con TDPA (objetivo)
python src/s01_build_corridor_ctn.py --inspect   # revisar capas y el campo con el código de la carretera
python src/s01_build_corridor_ctn.py             # corredor real + CTN por chainage de red

# Features: una vez para los sitios (entrenamiento) y otra para los CTN (predicción)
for P in sites ctn; do
  python src/s01b_road_features.py --points $P
  python src/s02_nasa_power.py     --points $P
  python src/s03_denue.py          --points $P
  python src/s04_censo.py          --points $P
  python src/s05_dgsiap.py         --points $P
done

python src/s08_build_model_table.py              # TOP_Model_Table + CTN_Prediction_Table
python src/s07_build_master.py                   # CTN_Master + Data Quality Report
jupyter notebook 04_NOTEBOOKS/02_EDA_TOP_Model_Table.ipynb   # EDA principal (Avance 1)
jupyter notebook 04_NOTEBOOKS/01_EDA_CTN_Master.ipynb        # EDA complementario del corredor
```

PDF de Datos Viales: pongan en `01_RAW/sict_tdpa/` los de Morelos, CDMX, Estado de México, Guerrero y Puebla
(más estados = más sitios de entrenamiento). Censo ITER: descarguen los mismos estados (09, 12, 15, 17, 21),
porque las features se calculan alrededor de cada sitio. `s06_sict_tdpa.py` queda como alternativa manual.

## 5. Estructura

```
00_DOCUMENTATION/  download_log.csv, feature_provenance.csv (data dictionary), plantilla TOP
01_RAW/            originales inmutables por fuente (los pesados no se versionan; ver download_log)
02_INTERIM/        corredor, TOP ubicados, series NASA, tablas fuente recortadas al corredor
03_PROCESSED/      ctn_base, features_*.csv, CTN_Master, CTN_Master_preprocessed
04_NOTEBOOKS/      01_EDA_CTN_Master.ipynb
06_RESULTS/        data_quality_report.md, figuras del EDA, selección de features
07_TEAM_LOG/       decision_log.csv (snaps, cambios de ruta, decisiones manuales)
src/               scripts s01–s07 y utilidades
```

## 6. Salvaguardas metodológicas (guía §9)

| Error a evitar | Cómo lo evita el pipeline |
|---|---|
| Etiqueta de "sitio ideal" construida con los mismos criterios | El objetivo es el TDPA observado por SICT; sólo `log_tdpa` tiene `allowed_as_target = yes`; nada derivado del aforo entra como feature |
| Variable municipal tratada como predial | Prefijo `mun_*`, resolución "municipal" en provenance; validación debe bloquear por municipio |
| Mezclar años sin registrar | `source_period` por feature; `tdpa_year` por CTN; mismo periodo NASA para todos los CTN |
| División aleatoria con R² alto | GroupKFold por bloques de 25 km y por carretera; prueba externa en la MEX-095D; comparación random vs. espacial |
| Confundir predicción con decisión | Lenguaje de caracterización en todos los productos |

Todo CTN sale con `geometry_status = RNC_SNAPPED` (criterio de salida de la guía §3).
