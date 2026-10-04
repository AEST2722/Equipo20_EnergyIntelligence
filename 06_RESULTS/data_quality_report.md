# Data Quality Report – CTN_Master

Generado: 2026-10-04T14:28:06 | Corredor: MEX-095D_CDMX_ACAPULCO | CRS de cálculo: EPSG:6372

- CTN: 46 | Features: 82 | Fuentes integradas: features_road.csv, features_nasa_power.csv, features_censo.csv, features_dgsiap.csv
- CTN con geometría RNC_SNAPPED: 46 de 46
- Duplicados de ctn_id: 0

## Completeness por CTN
- Mínima: 95.1% | Mediana: 100.0%

## Periodos por fuente (Consistency: no mezclar años sin registrarlo)
- INEGI Censo 2020 ITER: 2020
- INEGI RNC: 2025
- NASA POWER: 2015-2024
- SIAP/DGSIAP: 2024

## Features con completeness < 100%
- road_cond_pav: 89.1%
- road_recubri: 89.1%
- road_administra: 89.1%
- road_jurisdi: 89.1%
- rural_pop_share_5km: 95.7%

## Escalas espaciales (Resolution)
- ctn_type: punto
- chainage_km: tramo vial
- n_intersections_1km: buffer 1 km
- road_dist_m: punto
- road_tipo_vial: tramo vial
- road_cond_pav: tramo vial
- road_recubri: tramo vial
- road_carriles: tramo vial
- road_nivel: tramo vial
- road_peaje: tramo vial
- road_administra: tramo vial
- road_jurisdi: tramo vial
- road_circula: tramo vial
- road_velocidad: tramo vial
- road_ancho: tramo vial
- allsky_sfc_sw_dwn_mean: malla ~0.5° x 0.625°
- prectotcorr_mean: malla ~0.5° x 0.625°
- t2m_mean: malla ~0.5° x 0.625°
- t2m_max_mean: malla ~0.5° x 0.625°
- ws10m_mean: malla ~0.5° x 0.625°
- allsky_sfc_sw_dwn_std: malla ~0.5° x 0.625°
- prectotcorr_std: malla ~0.5° x 0.625°
- t2m_std: malla ~0.5° x 0.625°
- t2m_max_std: malla ~0.5° x 0.625°
- ws10m_std: malla ~0.5° x 0.625°
- allsky_sfc_sw_dwn_p10: malla ~0.5° x 0.625°
- prectotcorr_p10: malla ~0.5° x 0.625°
- t2m_p10: malla ~0.5° x 0.625°
- t2m_max_p10: malla ~0.5° x 0.625°
- ws10m_p10: malla ~0.5° x 0.625°
- allsky_sfc_sw_dwn_p90: malla ~0.5° x 0.625°
- prectotcorr_p90: malla ~0.5° x 0.625°
- t2m_p90: malla ~0.5° x 0.625°
- t2m_max_p90: malla ~0.5° x 0.625°
- ws10m_p90: malla ~0.5° x 0.625°
- allsky_sfc_sw_dwn_seasonal_amplitude: malla ~0.5° x 0.625°
- prectotcorr_seasonal_amplitude: malla ~0.5° x 0.625°
- t2m_seasonal_amplitude: malla ~0.5° x 0.625°
- t2m_max_seasonal_amplitude: malla ~0.5° x 0.625°
- ws10m_seasonal_amplitude: malla ~0.5° x 0.625°
- allsky_sfc_sw_dwn_min_month: malla ~0.5° x 0.625°
- prectotcorr_min_month: malla ~0.5° x 0.625°
- t2m_min_month: malla ~0.5° x 0.625°
- t2m_max_min_month: malla ~0.5° x 0.625°
- ws10m_min_month: malla ~0.5° x 0.625°
- allsky_sfc_sw_dwn_missing_pct: malla ~0.5° x 0.625°
- prectotcorr_missing_pct: malla ~0.5° x 0.625°
- t2m_missing_pct: malla ~0.5° x 0.625°
- t2m_max_missing_pct: malla ~0.5° x 0.625°
- ws10m_missing_pct: malla ~0.5° x 0.625°
- pop_5km: buffer 5 km
- n_localities_5km: buffer 5 km
- rural_pop_share_5km: buffer 5 km
- pop_10km: buffer 10 km
- n_localities_10km: buffer 10 km
- rural_pop_share_10km: buffer 10 km
- schooling_mean_10km: buffer 10 km
- pct_dwellings_electricity_10km: buffer 10 km
- pop_20km: buffer 20 km
- n_localities_20km: buffer 20 km
- rural_pop_share_20km: buffer 20 km
- dist_nearest_locality_m: punto
- pop_nearest_locality: localidad
- mun_agri_sown_ha: municipal
- mun_agri_harvested_ha: municipal
- mun_agri_value_mxn: municipal
- mun_agri_n_crops: municipal
- mun_agri_crop_shannon: municipal
- mun_agri_top_crop: municipal
- mun_agri_top_crop_share: municipal
- mun_agri_irrigated_share: municipal
- mun_agri_loss_share: municipal

## Riesgos de leakage señalados
- ctn_type: medio si se usa como variable explicativa
- mun_agri_sown_ha: medio: agrupar por municipio en validación espacial
- mun_agri_harvested_ha: medio: agrupar por municipio en validación espacial
- mun_agri_value_mxn: medio: agrupar por municipio en validación espacial
- mun_agri_n_crops: medio: agrupar por municipio en validación espacial
- mun_agri_crop_shannon: medio: agrupar por municipio en validación espacial
- mun_agri_top_crop: medio: agrupar por municipio en validación espacial
- mun_agri_top_crop_share: medio: agrupar por municipio en validación espacial
- mun_agri_irrigated_share: medio: agrupar por municipio en validación espacial
- mun_agri_loss_share: medio: agrupar por municipio en validación espacial

Notas: variables `mun_*` describen el municipio, no el predio del CTN. Ninguna feature está
autorizada como etiqueta de 'sitio ideal' (allowed_as_target = no).