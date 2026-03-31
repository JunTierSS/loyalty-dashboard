# Loyalty Intelligence System — Documentacion Tecnica Completa

## CMR Puntos Falabella | Chile | 2025

---

## 1. Descripcion General

Sistema end-to-end de inteligencia para el programa de fidelizacion CMR Puntos de Falabella Chile.
Predice el comportamiento de canje de 12M+ clientes y genera recomendaciones personalizadas de contacto.

### 1.1 Objetivo de negocio
- Predecir que clientes canjearan el proximo mes
- Estimar el impacto causal (incrementalidad) del programa de canje sobre el gasto
- Segmentar clientes para campanas personalizadas
- Priorizar contactos para maximizar ROI de marketing

### 1.2 Arquitectura de 4 fases

| Fase | Descripcion | Tecnologias |
|------|-------------|-------------|
| Fase 1 | Data Foundation — extraccion, limpieza, feature engineering | BigQuery, SQL, Python |
| Fase 2 | ML & Analytics — modelo predictivo, clustering, incrementalidad | XGBoost, K-Means, PSM |
| Fase 3 | Produccion — scoring pipeline, API, dashboard | Python, Streamlit |
| Fase 4 | Feedback Loop — monitoreo, reentrenamiento | Pendiente |

---

## 2. Fase 1: Data Foundation

### 2.1 Fuentes de datos

Todas las tablas residen en BigQuery bajo el proyecto `fif-loy-cl-cg-consumption`:

| Tabla | Descripcion | Particion | Tamano aprox |
|-------|-------------|-----------|--------------|
| `svw_clients_entity` | Registro mensual de clientes (tier, puntos, flags) | `partition_date` | ~748MB (500K) |
| `frozen_transaction_entity` | Transacciones de compra | `tran_date` | ~15GB (500K, 5 anos) |
| `svw_redemptions_entity` | Canjes de puntos | `redemption_date` | ~48MB (500K) |
| `base_colaboradores` | Lista de colaboradores (excluidos) | - | Pequena |

### 2.2 Exclusiones (01_excluded_customers.sql)

Se excluyen 4 tipos de clientes para evitar sesgo:

1. **FANTASMA**: 0 transacciones Y 0 canjes en 36 meses
2. **MONTO_EXTREMO**: Gasto total > percentil 99.9 (~$500M CLP)
3. **FRAUDE**: Flag `fraud_flag = TRUE` en alguna transaccion
4. **COLABORADOR**: Presente en `base_colaboradores` (opcional — fallback con UNNEST de array vacio si tabla no existe)

### 2.3 Muestreo estratificado (02_sample_customers.sql)

- **Tamano**: 500,000 clientes de ~12M totales
- **Metodo**: FARM_FINGERPRINT sobre `cust_id` para reproducibilidad
- **Estratificacion**: Proporcional por tier (NORMAL/FAN/PREMIUM/ELITE)
- **Sesgo conocido**: Enrollment bias — clientes mas nuevos tienen menos historial. Documentado pero no corregido.

### 2.4 Funnel Markov (03_markov_transition_matrix.sql)

#### 6 estados del funnel:
1. **Inscrito**: Solo registrado, sin actividad transaccional
2. **Participante**: Al menos 1 compra pero < 1000 puntos acumulados
3. **Posibilidad Canje**: >= 1000 puntos, nunca canjeo
4. **Canjeador**: Realizo al menos 1 canje
5. **Recurrente**: 2+ canjes en los ultimos 12 meses
6. **Fuga**: Fue Canjeador o Recurrente, pero > 12 meses sin canjear

#### Umbral de 1000 puntos
Se usa 1000 puntos como umbral minimo para canjear porque es el minimo requerido en el catalogo CMR Puntos.

#### Matriz de transicion Markov
Se calcula la probabilidad de transicion entre estados mes a mes:
```
P(estado_t+1 | estado_t) = COUNT(transiciones) / COUNT(clientes en estado_t)
```
Esto permite predecir la evolucion del funnel y detectar cuellos de botella.

### 2.5 Customer Snapshot (04_customer_snapshot.sql)

Query monolitica de ~74 features por cliente-mes. Genera un "snapshot" mensual para cada t0.

#### Diseno temporal (Panel de snapshots)
- **27 t0s mensuales**: Enero 2023 a Marzo 2025
- **Ventana pre (features)**: [t0 - 12 meses, t0) — ultimos 12 meses de historial
- **Ventana post 12m (target original)**: [t0, t0 + 12 meses)
- **Ventana post 1m (target actual)**: [t0, t0 + 1 mes)
- **Total filas**: ~13.5M (500K x 27 t0s)

#### Features principales (48+)

**RFM (Recency, Frequency, Monetary):**
- `recency_days`: Dias desde la ultima compra
- `frequency_total`: Total compras en ventana pre
- `frequency_monthly_avg`: Promedio mensual
- `monetary_total`: Gasto total en ventana pre
- `monetary_avg_ticket`: Ticket promedio
- `monetary_monthly_avg`: Gasto mensual promedio

**Puntos:**
- `stock_points_at_t0`: Stock de puntos al momento t0
- `exp_points_current_at_t0`: Puntos que vencen en el periodo actual
- `exp_points_next_at_t0`: Puntos que vencen en el siguiente periodo
- `points_earned_total`: Total puntos ganados en ventana pre
- `earn_velocity_30/90`: Velocidad de acumulacion (ultimos 30/90 dias)
- `redeem_count_pre`: Canjes en ventana pre
- `redeem_count_12m_pre`: Canjes en los ultimos 12 meses
- `redeem_rate`: Puntos canjeados / puntos ganados (clipped [0,1])
- `redeem_capacity`: stock_points / avg_redeem_points (cuantos canjes puede hacer)
- `points_above_threshold`: stock_points - 1000
- `points_pressure`: exp_points_current / stock_points (urgencia de canjear)

**Retail cross-shopping:**
- `freq_falabella`, `freq_sodimac`, `freq_tottus`, `freq_fcom`, `freq_ikea`: Frecuencia por retailer
- `retailer_count`: Numero de retailers donde compra
- `dominant_retailer`: Retailer con mayor gasto

**Canal/Digital:**
- `pct_cmr_payments`: % pagos con tarjeta CMR
- `pct_debit_payments`: % pagos con debito
- `pct_redeem_digital`: % canjes digitales (vs presencial)
- `contact_email_flg`, `contact_phone_flg`, `contact_push_flg`: Flags de contacto

**Demograficos:**
- `tier`: NORMAL/FAN/PREMIUM/ELITE
- `gender`, `age`, `city`
- `tenure_months`: Antiguedad en el programa

**Funnel:**
- `funnel_state_at_t0`: Estado del funnel al momento t0
- `months_in_current_state`: Meses **consecutivos** en el estado actual

**Estacionalidad:**
- `month_of_t0`: Mes del snapshot
- `is_cyber_month`: Flag Cyber Day/Monday (mayo, octubre)
- `is_holiday_month`: Flag Navidad/Fiestas Patrias

**Targets:**
- `y`: Target ternario original (0=no canjea, 1=primera vez, 2=recurrente) — ventana 12m
- `y_1m`: Target binario 1 mes (0=no canjea, 1=canjea) — ventana 1m
- `canjea_post`: Flag booleano de canje en post
- `n_canjes_post`: Numero de canjes en post
- `revenue_post_12m`: Gasto del cliente en los 12m post-t0

#### Bug fix: months_in_current_state
El calculo original contaba TODOS los meses con el mismo estado, no solo los consecutivos.
Se implemento un CTE `funnel_consecutive` con `state_group` (tecnica de islas) para contar solo los consecutivos.

#### Bug fix: redeem_rate > 1
Ocurria cuando puntos canjeados > puntos ganados (por ejemplo, puntos de bienvenida). Se corrigio con `LEAST(x, 1.0)`.

### 2.6 Extraccion de datos

#### Metodo de extraccion (extract_500k_v3.py)
1. Subir 500K `cust_id` a tabla temporal `discovery.temporal.sample_500k`
2. Ejecutar queries de Fase 1 en BigQuery con server-side JOINs
3. Descargar solo el snapshot resultante (912MB)
4. Las transacciones (15GB) se procesan en BigQuery y solo se bajan las metricas

**Razon**: No es viable descargar 15.5GB de transacciones a una MacBook Air de 16GB. Todas las agregaciones se hacen en BigQuery.

---

## 3. Fase 2: ML & Analytics

### 3.1 Modelo Predictivo (XGBoost)

#### Cambio de modelo
- **Original**: Ternario (y=0/1/2) con ventana 12 meses
- **Actual**: Binario (canjea/no canjea) con ventana 1 mes
- **Post-clasificacion**: Si canjea y `has_redeemed_before_t0 == False` → Canjeador Nuevo; si True → Recurrente

#### Configuracion XGBoost
```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    scale_pos_weight=auto,  # compensar desbalance
    eval_metric='auc',
    early_stopping_rounds=20,
    random_state=42
)
```

#### Train/Test split temporal
- **Train**: t0s de Ene-2023 a Sep-2024 (21 t0s)
- **Test**: t0s de Ene-2025 a Mar-2025 (3 t0s)
- **Razon**: Split temporal evita data leakage. No se usa cross-validation aleatoria.

#### Encoding
- `OrdinalEncoder` para categoricas (tier, funnel, gender, city, dominant_retailer)
- Las originales se guardan ANTES del encoding y se restauran despues del scoring

#### Metricas (5K clientes, test set)
| Metrica | Valor |
|---------|-------|
| AUC | 0.925 |
| Tasa base | ~3.4% |
| Accuracy | ~97% (inflado por clase mayoritaria) |
| F1 | Variable segun threshold |
| Precision | Variable |
| Recall | Variable |

**Nota**: La tasa base de canje a 1 mes es ~3.4%, por lo que el accuracy es enganoso. El AUC es la metrica principal.

### 3.2 Revenue Estimation (Two-stage)

1. **Stage 1**: Clasificacion binaria (canjea/no canjea) — XGBoost
2. **Stage 2**: Regresion de revenue solo para los que canjean — XGBRegressor

```python
revenue_esperado_1m = P(canje) x revenue_mensual_estimado
```

### 3.3 Clustering (K-Means)

#### Features de clustering (5, normalizadas con StandardScaler)
1. `frequency_total` (log-transform)
2. `monetary_total` (log-transform)
3. `recency_days`
4. `stock_points_at_t0` (log-transform)
5. `redeem_rate`

#### Numero de clusters
- Se prueba K=3..8 y se selecciona por Silhouette Score
- K optimo: 5 (silhouette ~0.35)

#### Asignacion de nombres (Hungarian Algorithm)
Los centroides se comparan contra arquetipos predefinidos:
- **Heavy Users**: Alta frecuencia, alto gasto, baja recency
- **Exploradores**: Multiples retailers, frecuencia media
- **Cazadores de Canje**: Alto redeem_rate, muchos puntos
- **Digitales**: Alto pct_redeem_digital
- **Dormidos**: Alta recency, baja frecuencia, bajo gasto

Se usa el Hungarian Algorithm (scipy.optimize.linear_sum_assignment) para asignacion optima cluster-arquetipo.

#### Imputation para K-Means
Los outliers (>P99) se imputan al P99 ANTES de clustering para evitar que 1-2 clientes distorsionen los centroides. Esto es diferente de imputar con 999 (que se usaba antes y era incorrecto).

### 3.4 RFM Segmentation

Scoring independiente 1-5 para cada dimension:
- **R** (Recency): 5=mas reciente, 1=mas antiguo (qcut en quintiles)
- **F** (Frequency): 5=mas frecuente
- **M** (Monetary): 5=mayor gasto

Segmentos RFM (basados en combinaciones de R, F, M):
| Segmento | Regla |
|----------|-------|
| Champions | R>=4, F>=4, M>=4 |
| Loyal | F>=3, M>=3 |
| Nuevos | R>=4, F<=2 |
| En Riesgo | R<=2, F>=3, M>=3 (fueron buenos) |
| Perdidos | R<=2, F<=2, M<=2 |
| Otros | Todo lo demas |

### 3.5 Propensity Score Matching (PSM)

#### Objetivo
Estimar el efecto causal del canje sobre el gasto futuro, controlando por autoseleccion.

#### Metodologia
1. **Tratamiento**: Canjeo en periodo de seleccion (SEL: Mar-2023 a Mar-2024)
2. **Control**: Potencial (>=1000 puntos) que no canjeo
3. **Propensity Score**: LogisticRegression cross-fitted (5 folds) sobre features pre-tratamiento
4. **Matching**: Nearest-neighbor 1:1 con caliper=0.1 sigma
5. **Estimacion**: Diferencia de medias de gasto POST entre matched pairs

```python
LogisticRegression(
    max_iter=100,  # Reducido de 1000 para eficiencia
    solver='lbfgs',
    C=1.0,
    random_state=42
)
```

#### Limitacion importante
El tratamiento es **observacional** (no experimental). El PSM controla por observables pero no por variables no observadas (motivacion intrinseca, etc.). Los resultados son "mejor estimacion causal" pero no un RCT.

### 3.6 T-Learner / Uplift

#### Objetivo
Estimar el efecto heterogeneo del tratamiento (CATE) para cada cliente individual.

#### Metodologia (T-Learner)
1. Entrenar modelo M0 solo con controles: Y ~ X
2. Entrenar modelo M1 solo con tratados: Y ~ X
3. CATE(x) = M1(x) - M0(x)

Esto da un `uplift_x` por cliente: cuanto gasto ADICIONAL genera contactar a ese cliente especifico.

### 3.7 Incrementalidad GitLab (3 periodos)

#### Metodologia exacta replicada de GitLab

**3 periodos de 12 meses cada uno:**
- **PREV** (Previo): Mar-2022 a Mar-2023 — gasto historico
- **SEL** (Seleccion): Mar-2023 a Mar-2024 — quien canjeo aqui define si es Canjeador
- **POST** (Posterior): Mar-2024 a Mar-2025 — gasto que se compara

**Clasificacion de clientes:**
- **Canjeador (CJ)**: Canjeo al menos 1 vez en periodo SEL
- **Potencial (PT)**: >= 1000 puntos en alguna snapshot de SEL, pero NO canjeo
- **Acumulador (AC)**: < 1000 puntos, no canjeo (excluido del analisis de lift)

**Exclusion**: Clientes cuyo primer canje fue DESPUES del periodo SEL se excluyen para evitar contaminacion.

**Calculo del Lift por decil:**
1. Ordenar CJ y PT por gasto PREVIO
2. Dividir en 10 deciles de gasto previo
3. Para cada decil d:
   ```
   lift_d = (mean_gasto_post_CJ_d - mean_gasto_post_PT_d) / mean_gasto_post_PT_d
   ```
4. Lift ponderado total:
   ```
   Lift_total = SUM(lift_d x gasto_CJ_d) / SUM(gasto_CJ_d) x 100
   ```

**Resultados (5K clientes):**
| Retail | GitLab Lift % | PSM Lift % |
|--------|---------------|------------|
| FALABELLA | +28.1% | +21.2% |
| SODIMAC | +25.0% | +3.9% |
| TOTTUS | +37.0% | +23.5% |
| FCOM | +19.3% | +28.9% |
| **TOTAL** | **+35.3%** | **+16.2%** |

**Interpretacion**: GitLab sobreestima ~2.2x porque no controla por autoseleccion. Los canjeadores ya eran clientes mas activos.

### 3.8 Decision Engine (Motor de Decision)

5 criterios de priorizacion:
1. **P(canje)**: Probabilidad de canjear — para campanas de "accelerate"
2. **Expected Value**: P(canje) x revenue estimado — maximizar revenue
3. **Uplift (CATE)**: Efecto causal — contactar a quien realmente cambia
4. **Riesgo de fuga**: CLV alto + estado Fuga — reactivar antes de perder
5. **Activacion**: Nunca canjeo + tiene puntos + prob > 20% — primer canje

Cada criterio genera una lista rankeada diferente. El criterio optimo depende del objetivo de la campana.

### 3.9 CLV (Customer Lifetime Value)

```
CLV = revenue_post_12m x 1.5
```

El factor 1.5 es un proxy de retencion futura. En un modelo mas sofisticado se usaria un modelo BG/NBD o similar.

**Diferencial de CLV:**
- CLV canjeador >> CLV no canjeador (tipicamente 3-5x)
- Esto justifica invertir en campanas de activacion

---

## 4. Fase 3: Produccion

### 4.1 Scoring Pipeline (scoring_pipeline.py)

**Flujo:**
1. Cargar modelos entrenados (XGBoost clasificador, regresor, scaler, encoder)
2. Para cada chunk de datos:
   a. Imputar NAs
   b. Encoding categoricas
   c. Predict P(canje)
   d. Predict revenue (si P > threshold)
   e. Clustering
   f. Calcular uplift
   g. Asignar prioridad
   h. Asignar canal, timing, accion

### 4.2 Prioridad

```python
if uplift > 0 and prob > 0.3: prioridad = 'Alta'
elif prob > 0.5: prioridad = 'Media'
elif funnel == 'Fuga' and clv > median_clv: prioridad = 'Urgente'
elif prob < 0.1: prioridad = 'No contactar'
else: prioridad = 'Baja'
```

### 4.3 Dashboard (Streamlit)

**Deployment**: Streamlit Community Cloud desde repo privado GitHub.

**Vistas:**
1. **Resumen Ejecutivo** — KPIs principales, oportunidades, temporal
2. **CLV & Revenue** — Valor por tipo de cliente y categoria
3. **RFM** — Segmentacion clasica con scatter y tablas
4. **Funnel** — Area chart temporal, distribucion actual, tasa por estado
5. **Segmentos** — Clusters K-Means con acciones recomendadas
6. **Modelo** — AUC, F1, Precision, Recall, Confusion Matrix, Lift chart, ROC
7. **Incrementalidad** — GitLab vs PSM, tablas por decil y retail
8. **Prediccion Mes** — Cuantos canjearan, revenue esperado, tablas Quintil x Categoria
9. **Motor Decision** — 5 criterios de priorizacion con top 20 clientes
10. **Simulador** — Impacto de campana con sliders (Revenue - Costo = Ganancia neta)
11. **Ficha Cliente** — Perfil individual
12. **Aperturas** — Heatmaps de cruces (Quintil x Tier, Cluster x Funnel, etc.)
13. **Exportar** — Download de listas por prioridad, cluster, RFM

**Limitaciones de memoria**: Streamlit Cloud tiene 1GB RAM. Los datos se reducen a:
- `data_scored.csv.gz`: 5,000 clientes deduplicados (~1.9MB)
- `data_temporal.csv`: Series temporales pre-agregadas
- `data_funnel_evo.csv`: Evolucion funnel pre-agregada
- `data_metrics.json`: Metricas resumen

### 4.4 Nomenclatura

| Original (BQ) | Dashboard |
|----------------|-----------|
| NORMAL | Entrada |
| FAN | Fan |
| PREMIUM | Premium |
| ELITE | Elite |
| INSCRITO | Inscrito |
| PARTICIPANTE | Participante |
| POSIBILIDAD_CANJE | Posibilidad Canje |
| CANJEADOR | Canjeador |
| RECURRENTE | Recurrente |
| FUGA | Fuga |

---

## 5. Supuestos y Parametros Hardcodeados

| Parametro | Valor | Justificacion |
|-----------|-------|---------------|
| Umbral canje | 1000 puntos | Minimo del catalogo CMR Puntos |
| Valor por punto | 6.5 CLP | Promedio ponderado del catalogo |
| CLV factor | 1.5 x revenue_12m | Proxy de retencion, simplificado |
| Revenue reactivacion | 70% de un canjeador nuevo | Engagement menor post-fuga |
| Revenue participante | 50% de un canjeador nuevo | Sin habito de canje aun |
| PSM caliper | 0.1 sigma del logit | Estandar en la literatura |
| Fuga threshold | 12 meses sin canjear | Definicion operacional del programa |
| Tasa base canje 1m | ~3.4% | Observada en datos reales |
| P(canje) threshold alto | 0.5 | Para prediccion de "canjeara" |
| P(canje) threshold activacion | 0.2 | Para capturar primeros canjes |
| XGBoost n_estimators | 300 | Con early stopping a 20 rounds |
| LogReg max_iter | 100 | Suficiente para convergencia |
| K-Means K | 5 | Seleccionado por Silhouette Score |
| GitLab periodos | 12 meses cada uno | Replica de metodologia productiva |
| Percentil outliers | P99 | Para imputacion y winsorize |

---

## 6. Limitaciones Conocidas

### 6.1 Sesgo de seleccion
El PSM controla por variables observadas pero no por no observadas (motivacion, acceso a tiendas, etc.). Los resultados de incrementalidad son estimaciones, no causal puro.

### 6.2 Data leakage potencial
- `redeem_rate` usa informacion de todo el periodo pre, lo cual esta bien
- `funnel_state_at_t0` se calcula al momento t0, sin leakage
- `months_in_current_state` fue corregido para contar solo consecutivos
- El split temporal (train hasta Sep-2024, test desde Ene-2025) previene leakage temporal

### 6.3 Desbalance de clases
Solo ~3.4% de clientes canjea en 1 mes. Se usa `scale_pos_weight` en XGBoost para compensar. F1 y AUC son las metricas relevantes (no accuracy).

### 6.4 Muestra para presentacion
Los resultados se basan en 5,000 clientes (de 500K extraidos, de 12M totales). Los intervalos de confianza son amplios. Para produccion se usarian los 500K o los 12M.

### 6.5 Estacionalidad
Cyber Day (mayo, octubre) y Navidad (diciembre) distorsionan las tasas de canje. Los features `is_cyber_month` e `is_holiday_month` intentan capturar esto, pero el modelo podria beneficiarse de features mas granulares.

### 6.6 Ventana asimetrica
Para los t0s mas recientes (post Sep-2024), la ventana post de 12 meses esta truncada. El target `y_1m` (1 mes) mitiga esto pero no lo elimina completamente para los t0s de Feb-Mar 2025.

### 6.7 Breakage
El breakage (% puntos no canjeados) se estima como `1 - redeem_rate` por cliente. Esto es una aproximacion — el breakage real requiere informacion de vencimiento de puntos que es parcial en nuestros datos.

---

## 7. Estructura de archivos

```
loyalty/
  fase1/
    sql/
      01_excluded_customers.sql
      02_sample_customers.sql
      03_markov_transition_matrix.sql
      04_customer_snapshot.sql
    data_real_500k/          # Datos reales (no en git)
      02_clients_entity.parquet
      03_txn_20XX.parquet (x 5 anos)
      04_redemptions.parquet
      05_customer_snapshot.parquet
    data_real_5k/            # Muestra para presentacion
      scored.csv.gz
      temporal.csv
      funnel_evo.csv
      incrementalidad.csv
      metrics.json
    test_mock_local.py
    extract_500k_v3.py
    run_500k_bigquery.py
  fase3/
    scoring_pipeline.py
  run_5k_presentacion.py
  run_50k_all.py
  dashboard_presentacion.py     # Dashboard local
  DOCUMENTACION_TECNICA_COMPLETA.md
```

Dashboard deployado (repo GitHub privado):
```
/tmp/loyalty-dashboard/
  app.py
  data_scored.csv.gz
  data_temporal.csv
  data_funnel_evo.csv
  data_metrics.json
  requirements.txt
  .streamlit/config.toml
```

---

## 8. Glosario

| Termino | Definicion |
|---------|-----------|
| t0 | Fecha de corte del snapshot. Features miran hacia atras, targets hacia adelante |
| Ventana pre | [t0-12m, t0) — periodo de observacion para construir features |
| Ventana post | [t0, t0+1m) o [t0, t0+12m) — periodo de prediccion |
| Propensity Score | P(tratamiento dado X) — probabilidad de canjear dado el perfil |
| PSM | Propensity Score Matching — emparejar tratados y controles por propensity |
| CATE | Conditional Average Treatment Effect — efecto individual del tratamiento |
| T-Learner | Metodo de uplift: 2 modelos separados (tratados y controles) |
| Lift | Ratio de tasa de canje del grupo vs la tasa base |
| Breakage | % de puntos que se acumulan pero nunca se canjean |
| CLV | Customer Lifetime Value — valor futuro esperado del cliente |
| RFM | Recency-Frequency-Monetary — segmentacion clasica de CRM |
| Funnel | Secuencia de estados del cliente en el programa de fidelizacion |
| Scale_pos_weight | Peso asignado a la clase positiva para compensar desbalance |
| Caliper | Distancia maxima permitida entre matched pairs en PSM |
| Hungarian Algorithm | Metodo de asignacion optima cluster-arquetipo |
| Silhouette Score | Metrica de calidad de clustering (-1 a 1, mayor=mejor) |
| FARM_FINGERPRINT | Funcion hash deterministica de BigQuery para sampleo reproducible |
