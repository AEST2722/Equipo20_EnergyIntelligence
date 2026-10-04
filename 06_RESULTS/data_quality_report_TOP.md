# Control de calidad — TOP SICT 2025

**Fuente:** Datos Viales SICT, edición 2026 (aforos observados en 2025). PDF originales descargables en el portal oficial: https://micrs.sct.gob.mx/index.php/infraestructura/direccion-general-de-servicios-tecnicos/datos-viales/2026. Los PDF están en `01_RAW/sict_tdpa/` localmente y se excluyen de Git.

## Cobertura

| Estado | Sitios con TDPA | TDPA válido | Marcados prueba MEX-095D |
|---|---:|---:|---:|
| Ciudad de México | 104 | 104 | 6 |
| Guerrero | 205 | 205 | 17 |
| Estado de México | 429 | 429 | 0 |
| Morelos | 118 | 118 | 0 |
| Puebla | 219 | 219 | 0 |
| **Total** | **1075** | **1075** | **23** |

Todos los sitios de la tabla tienen `anio = 2025`. La tabla contiene 36 variables explicativas, calculadas con RNC 2025, Censo ITER 2020 y SIAP municipal 2024. Se marcaron como prueba externa preliminar 23 sitios que cumplen la regla actual de ruta/distancia en MEX-095D; la selección se debe revisar antes de modelar.

## Distribución del objetivo

| Estadístico | TDPA (veh/día) |
|---|---:|
| Media | 20,804.5 |
| Desviación estándar | 27,004.4 |
| Mínimo | 275 |
| Percentil 5 | 1,843 |
| Mediana | 10,924 |
| Percentil 95 | 73,583 |
| Máximo | 225,051 |
| Asimetría TDPA | 3.054 |
| Asimetría ln(TDPA) | -0.017 |

La transformación logarítmica reduce la asimetría del objetivo; no se concluye normalidad sólo con este resumen.

## Faltantes en features

| Variable | Sitios faltantes |
|---|---:|
| `road_cond_pav` | 25.3% |
| `road_recubri` | 25.3% |
| `road_administra` | 25.3% |
| `road_jurisdi` | 25.3% |
| `rural_pop_share_5km` | 13.1% |
| `pct_dwellings_electricity_10km` | 10.1% |
| `schooling_mean_10km` | 10.1% |
| `rural_pop_share_10km` | 10.1% |
| `rural_pop_share_20km` | 5.7% |
| `road_carriles` | 1.3% |

NASA POWER y DENUE para TOP aún no están integrados. Este conjunto basta para el EDA preliminar del objetivo y las features disponibles, pero las relaciones observadas no sustituyen la validación espacial ni deben presentarse como resultados de un modelo entrenado.
