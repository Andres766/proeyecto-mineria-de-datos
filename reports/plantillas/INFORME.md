# Segmentación avanzada de clientes con clustering y reducción de dimensionalidad

Proyecto final · Inteligencia de Negocios y Minería de Datos · Ingeniería de Software

## Resumen

Este trabajo diseña un sistema de segmentación de clientes para un minorista online británico de artículos de regalo a
partir del dataset *Online Retail* (UCI). Las transacciones originales ({filas_originales} líneas) se limpiaron hasta
{filas_limpias} líneas de venta de {clientes} clientes, con un tratamiento explícito de las cancelaciones que anulan
compras. Por cada cliente se construyeron variables de comportamiento (recency, frecuencia, ticket promedio y variedad
de productos), se transformaron con log1p, se recortaron a los percentiles 1–99 y se estandarizaron. Se compararon tres
familias de algoritmos (K-Means, mezclas gaussianas y DBSCAN/HDBSCAN) y se usaron PCA y t-SNE para interpretar y
visualizar. La validación combinó métricas internas, estabilidad por remuestreo, robustez a los empates de la variable
frecuencia, concordancia con un baseline RFM por quintiles, pruebas de exactitud con datos de entrenamiento y de prueba,
y un criterio de negocio.

El modelo final es **K-Means con k = {k_final}** (silhouette {final_silhouette}, estabilidad mediana
{final_estabilidad_mediana}). Al clasificar clientes que no vio al entrenar, coincide con el segmento final en el
{val_reasignacion} de los casos, y un árbol de decisión reproduce los segmentos con un {val_arbol_prueba} de exactitud
en prueba. Da cuatro segmentos accionables: **{s3_nombre}** ({s3_pct_clientes} de los clientes y
{s3_pct_ingresos} de los ingresos), **{s2_nombre}**, **{s1_nombre}** y **{s0_nombre}**. Dentro de VIP, el cruce con los
atípicos de DBSCAN identifica un subgrupo de **{ck_nombre_subgrupo}**: {ck_clientes} clientes ({ck_pct_clientes}) que
generan el {ck_pct_ingresos} de los ingresos. Cada segmento tiene estrategias de marketing con acción, momento y KPI.
Los resultados se publican en un aplicativo web en Streamlit.

## 1. Introducción

Segmentar clientes permite dejar de tratar a todos por igual: concentrar recursos en quienes generan más valor,
recuperar a quienes se están yendo y no gastar en acciones caras con quien no va a responder. El enfoque clásico es el
análisis RFM (Recency, Frequency, Monetary), que puntúa a cada cliente por quintiles. Es sencillo y explicable, pero
sus cortes son arbitrarios y no usan la estructura conjunta de los datos.

**Objetivo.** Diseñar un sistema de segmentación basado en clustering y reducción de dimensionalidad que identifique
perfiles de clientes útiles para estrategias de marketing, validarlo con rigor y desplegarlo en un aplicativo web.

**Alcance.** El proyecto cubre todo el ciclo:

1. Análisis exploratorio y limpieza documentada.
2. Ingeniería de características a nivel de cliente.
3. Comparación de K-Means, GMM y DBSCAN (y HDBSCAN), con PCA y t-SNE.
4. Validación interna, de estabilidad y de negocio.
5. Perfilamiento con estrategias.
6. Un dashboard desplegable.

Todas las cifras de este informe salen de ejecutar el código del repositorio: el documento se genera desde una
plantilla con `python -m segmentacion.reportes`.

## 2. Datos

El dataset *Online Retail* (Chen, Sahinoglu y Guo, 2012) contiene las transacciones de una tienda online británica
entre el {fecha_inicio} y el {fecha_fin} (último registro). Cada fila es una línea de factura con ocho columnas:

- número de factura (`InvoiceNo`; las que empiezan por "C" son cancelaciones);
- código y descripción del producto;
- cantidad, fecha y hora, y precio unitario;
- identificador de cliente (`CustomerID`) y país.

Muchos de sus clientes son mayoristas, lo que produce distribuciones muy asimétricas.

## 3. Metodología

El trabajo sigue las fases de un proyecto de minería de datos: comprensión y preparación de los datos, modelado,
evaluación y despliegue.

- **Herramientas:** Python 3.11, pandas, scikit-learn, Plotly y Streamlit.
- **Organización del código:** la lógica está en un paquete (`src/segmentacion`: carga, limpieza, características,
  modelado, evaluación, perfilamiento y predicción) y los notebooks solo la invocan y narran los resultados.
- **Reproducibilidad:** todos los procesos aleatorios usan `random_state = 42`, las dependencias tienen versiones fijas
  y el proyecto incluye {n_pruebas} pruebas automáticas (limpieza, características, modelado, medidas de exactitud,
  clasificación de clientes nuevos y la propia aplicación).

Las decisiones se tomaron comparando alternativas con datos y no por convención. Las más relevantes:

- el emparejamiento de compras anuladas;
- la exclusión de variables redundantes o no definidas;
- el tratamiento de los clientes extremos;
- el `reg_covar` de GMM;
- el `eps` de DBSCAN;
- el número de segmentos.

## 4. Análisis exploratorio y limpieza

### 4.1 Calidad de los datos

- **Clientes sin identificar.** El {pct_lineas_sin_cliente} de las líneas no tiene `CustomerID`. Suponen el
  {pct_importe_sin_cliente} del importe vendido y faltan por factura completa (compras sin cliente registrado). No se
  pueden imputar porque no hay forma de saber de quién son, así que se excluyen. La segmentación describe, por tanto, a
  los clientes registrados.
- **Duplicados.** Hay {duplicados} filas duplicadas exactas.
- **Cancelaciones.** Las facturas canceladas restan el equivalente al {pct_importe_cancelado} de lo vendido.
- **Códigos que no son producto.** Se identificaron revisando sus descripciones: portes, comisiones, descuentos,
  ajustes manuales y bonos regalo.

### 4.2 Cancelaciones y compras anuladas

Los dos mayores valores de cantidad del dataset (hasta {max_cantidad_anulada} unidades) son compras que el mismo
cliente canceló minutos después. Si solo se eliminaran las facturas "C", esas compras quedarían como ventas reales e
inflarían el gasto. Por eso la limpieza **empareja cada cancelación con la compra que anula**: mismo cliente, producto,
cantidad y precio, la compra más reciente anterior a la cancelación, con emparejamiento 1 a 1.

Además, algunas anulaciones se registran con el código manual `M` por el importe total. Se aplica una regla
conservadora: misma persona, mismo importe exacto y menos de 24 horas entre la compra y la anulación. En total se
anularon {compras_anuladas} compras ({anuladas_exactas} por cancelación exacta y {anuladas_manuales} por anulación
manual).

Las pruebas automáticas detectaron un error en la primera versión del emparejamiento. Cuando un cliente tenía dos
compras idénticas y dos cancelaciones idénticas, solo se anulaba una compra, porque ambas cancelaciones apuntaban a la
misma. Se corrigió con un emparejamiento iterativo realmente 1 a 1 y se regeneraron todos los resultados.

Tabla: Bitácora de limpieza (filas antes y después de cada paso).

{tabla_bitacora}

![Filas restantes tras cada paso de limpieza](figures/01_bitacora_limpieza.png)

El resultado son {filas_limpias} líneas, {facturas} facturas, {clientes} clientes, {productos} productos de {paises}
países y {ingresos} en ventas.

### 4.3 Patrones de venta

Las ventas tienen una fuerte estacionalidad de fin de año: el mes de mayor facturación es {mes_pico}
({ingresos_mes_pico}). Diciembre de 2011 solo llega hasta el día 9: se conserva para calcular la recency, pero se marca
como incompleto en los gráficos.

No hay ventas en sábado ({facturas_sabado} facturas) y la hora con más pedidos son las {hora_pico} h. Este horario
laboral, sumado a pedidos de gran tamaño, indica una clientela en buena parte mayorista (B2B). Reino Unido concentra el
{pct_clientes_uk} de los clientes y el {pct_ingresos_uk} de los ingresos.

![Ingresos mensuales (diciembre de 2011, en gris, está incompleto)](figures/01_ventas_mensuales.png)

![Facturas por día de la semana y por hora](figures/01_dia_hora.png)

### 4.4 Distribuciones y concentración

Todas las variables son muy asimétricas a la derecha. El 1 % de los clientes genera el {pct_ingresos_top1} de los
ingresos. En escala logarítmica las distribuciones se vuelven aproximadamente simétricas, lo que justifica transformar
con log1p antes de usar algoritmos basados en distancias.

![Distribuciones en escala logarítmica](figures/01_distribuciones_log.png)

## 5. Ingeniería de características

### 5.1 Variables por cliente

Se calcularon, para cada cliente:

- **Recency:** días entre la fecha de referencia ({fecha_referencia}, el día siguiente al último registro) y la última
  compra.
- **Frequency:** número de facturas distintas.
- **Monetary:** gasto total.
- **Ticket promedio**, **productos distintos**, **antigüedad**, **días promedio entre compras**, **unidades por
  factura** y **tasa de devoluciones**.

### 5.2 Días entre compras

La variable no está definida para los {clientes_una_compra} clientes de una sola factura ({pct_una_compra}). Se
compararon cuatro alternativas:

| Alternativa | Problema |
|---|---|
| Imputar con la mediana ({mediana_dias_entre_compras} días) | Crea un pico artificial en un tercio de la base |
| Imputar con la recency | Duplica una variable que ya está en el modelo |
| Variable indicadora | Coincide exactamente con frequency = 1 |
| Excluirla del clustering | Entre los clientes recurrentes correlaciona {corr_dias_frequency} con frequency |

Se excluyó del clustering y se usa solo para perfilar.

### 5.3 Tasa de devoluciones

Se define de forma coherente con la limpieza: importe devuelto computable dividido entre el importe comprado.

- No cuentan las cancelaciones emparejadas, cuyas compras ya se eliminaron.
- Del resto solo cuentan las unidades que caben en lo que el cliente compró antes de ese producto dentro del periodo.
  Las devoluciones de compras anteriores al inicio de los datos computan 0.
- La tasa se recorta a [0, 1] por robustez. El máximo observado es {tasa_maxima}.

El {pct_tasa_cero} de los clientes tiene tasa 0. Incluirla en el clustering crea un grupo pequeño
({cluster_devolvedores_k4} clientes, {pct_cluster_devolvedores_k4}) formado al 100 % por clientes que devuelven, y sube
el silhouette de forma engañosa (de {silhouette_sin_tasa_k3} a {silhouette_con_tasa_k3} con k = 3). Por eso se usa
solo para perfilar.

### 5.4 Selección de variables

El gasto total es exactamente frecuencia × ticket promedio, así que en escala logarítmica es una suma de las otras dos.
Con las tres juntas el factor de inflación de la varianza (VIF) se dispara: {vif_monetary} en monetary, {vif_frequency}
en frequency y {vif_ticket} en el ticket. Además, frequency y monetary correlacionan {corr_frequency_monetary}, y el
ticket y las unidades por factura, {corr_ticket_unidades}.

El conjunto elegido es **recency, frequency, ticket promedio y productos distintos**: VIF máximo {vif_max_final} y
correlación máxima entre pares de {corr_max_final}. El gasto total no se pierde, porque queda recogido en frecuencia ×
ticket y se usa para perfilar. La antigüedad se excluyó porque es idéntica a la recency en todos los clientes de una
sola compra.

![Correlación de Spearman entre variables de cliente](figures/02_correlaciones.png)

### 5.5 Transformación, escalado y clientes extremos

- **Transformación:** log1p deja las cuatro variables aproximadamente simétricas.
- **Escalado:** se eligió StandardScaler porque da a todas las variables el mismo peso. RobustScaler obtenía un
  silhouette como mucho {dif_max_silhouette_escaladores} mayor, pero a costa de dar más peso al ticket.
- **Clientes extremos:** los {clientes_extremos} clientes con |z| > 3 en alguna variable generan el
  {pct_ingresos_extremos} de los ingresos, así que excluirlos dejaría fuera a los clientes más valiosos. Se recortó cada
  variable (en escala log) a sus percentiles 1 y 99. Afecta a {clientes_recortados} clientes ({pct_recortados}) y solo
  modifica la entrada del modelo, no sus valores reales.

![Variables antes y después de log1p](figures/02_log1p.png)

### 5.6 Baseline RFM

Como referencia se construyó la segmentación RFM clásica: puntuaciones de 1 a 5 por quintiles y diez segmentos según
el mapa R × F (Campeones, Leales, En riesgo, Hibernando, etc.). Tiene una limitación: como un tercio de los clientes
tiene frequency = 1, los quintiles de F se cortan en parte por orden de aparición.

![Baseline RFM: peso de cada segmento en clientes e ingresos](figures/02_baseline_rfm.png)

## 6. Modelado

### 6.1 Reducción de dimensionalidad con PCA

PCA sobre las cuatro variables estandarizadas: la primera componente explica el {pca_pc1} de la varianza y la segunda
el {pca_pc2} (juntas, el {pca_2}). Sus cargas permiten interpretarlas:

- **PC1 = actividad / valor:** más facturas y más productos distintos, compras más recientes.
- **PC2 = tamaño del pedido:** ticket alto y compras más espaciadas.

Clusterizar sobre 2 o 3 componentes no mejoró la calidad medida en el espacio original. El silhouette calculado en el
espacio PCA sale inflado porque ignora la varianza descartada. Se clusterizó sobre las cuatro variables y PCA se usa
para interpretar y visualizar.

![PCA: varianza explicada](figures/03_pca_varianza.png)

![Cargas de las componentes principales](figures/03_pca_cargas.png)

### 6.2 K-Means

Se evaluó k = 2 a 10 con la inercia (método del codo), el silhouette, Davies-Bouldin y Calinski-Harabasz. Las métricas
internas favorecen k = 2 (silhouette {k2_silhouette}) y empeoran poco a poco al subir k. La estabilidad se midió con 50
submuestras del 80 %, reajustando el modelo y comparando con el ARI.

![Métricas de K-Means para k = 2 a 10](figures/03_kmeans_metricas.png)

![Estabilidad de K-Means por k](figures/03_kmeans_estabilidad.png)

### 6.3 Mezclas gaussianas (GMM) y componentes degenerados

Con la regularización por defecto de scikit-learn (`reg_covar` = 1e-6), GMM degenera. Coloca componentes exactamente
sobre el valor frequency = 1, compartido por un tercio de los clientes, con varianza casi nula en esa dimensión. La
verosimilitud crece sin límite y el BIC baja artificialmente hasta {bic_degenerado}.

Se definió un criterio de degeneración (varianza empírica de un componente en frequency menor que 0,01) y se buscó el
`reg_covar` mínimo que lo evita: **{reg_covar}**. Con ese valor, el menor BIC ({bic_elegido}) corresponde a covarianza
{gmm_tipo} con {gmm_k} componentes. Aun así, el modelo resultante tiene un silhouette de {gmm_silhouette}, una
estabilidad mediana de {gmm_estabilidad_mediana} y un componente con solo el {gmm_pct_menor} de los clientes.

![Varianza empírica mínima frente a reg_covar](figures/03_gmm_reg_covar.png)

### 6.4 DBSCAN y HDBSCAN

- **Parámetros de DBSCAN:** `min_samples` = {min_samples} (dos veces el número de variables) y `eps` = {eps}, tomado del
  codo del gráfico de k-distancias.
- **Falsas zonas densas por los empates.** Los clientes con frequency = 1 tienen una distancia mediana a su octavo
  vecino de {kdist_f1}, frente a {kdist_resto} en el resto. Parecen más densos solo porque comparten coordenada. Con eps
  menor que el hueco entre frequency = 1 y 2 ({hueco_frequency}), los "clusters" de DBSCAN son rebanadas de un único
  valor de frecuencia.
- **Resultado de DBSCAN:** con el eps del codo queda un único grupo denso y {atipicos} clientes atípicos
  ({pct_atipicos}), que generan el {pct_ingresos_atipicos} de los ingresos.
- **HDBSCAN:** separa exactamente a los clientes de una factura del resto. Al romper los empates repartiendo la
  frecuencia dentro de su intervalo de redondeo, esa estructura desaparece (ARI {hdbscan_ari_descuantizado}): es un
  artefacto.

Conclusión: no hay grupos separados por densidad. DBSCAN se usa para marcar atípicos, no para segmentar.

![Gráfico de k-distancias de DBSCAN](figures/03_dbscan_kdistancias.png)

### 6.5 t-SNE

t-SNE se usó solo para visualizar, con perplexities de 5, 30, 50 y 100. Conserva vecindarios locales, pero no las
distancias entre grupos ni sus tamaños, y el dibujo cambia con la perplexity. Se comprobó con los datos: el segmento
más disperso en realidad (segmento {tsne_mas_disperso_real}) aparece como el más compacto en el mapa (segmento
{tsne_mas_compacto_mapa}). Por eso no se usó para validar.

![t-SNE con distintas perplexities, coloreado con los segmentos de K-Means](figures/03_tsne_perplexities.png)

## 7. Validación y elección del modelo

Tabla: Comparación de los modelos candidatos (métricas en el mismo espacio; en DBSCAN y HDBSCAN el ruido se excluye).

{tabla_modelos}

### 7.1 Estabilidad: k = 4 frente a k = 5

k = 5 tiene la mejor estabilidad mediana ({k5_estabilidad_mediana}), pero su percentil 10 cae a
{k5_estabilidad_p10}. Tiene dos soluciones casi igual de buenas: las alternativas solo tienen entre un {k5_dif_min} y
un {k5_dif_max} más de inercia que la mejor. En una parte de las submuestras sus segmentos se reorganizan, y lo que
cambia es precisamente su segmento distintivo (ticket alto). k = 4 mantiene {final_estabilidad_p10} también en el peor
caso.

![Distribución de la estabilidad con k = 4 y k = 5](figures/03_estabilidad_k4_k5.png)

### 7.2 Concordancia

El ARI del modelo final con el baseline RFM es {final_ari_baseline_rfm}. Es bajo en parte porque se comparan cuatro
segmentos con diez, pero la tabla cruzada (sección 8) muestra que encajan.

![Concordancia entre modelos y con el baseline RFM (ARI)](figures/03_matriz_ari.png)

### 7.3 Exactitud con datos de entrenamiento y de prueba

En un problema supervisado la exactitud compara la predicción con una etiqueta real. La segmentación es no supervisada:
no existe un «segmento verdadero» de cada cliente, así que la exactitud no se puede calcular sobre los clusters. Lo que sí
se puede medir, separando datos de **entrenamiento** y de **prueba**, es si los segmentos se reproducen con clientes que
el modelo no ha visto (notebook `05_validacion`). Se hicieron dos pruebas:

1. **Exactitud de reasignación.** Con validación cruzada estratificada de {val_n_pliegues} pliegues se reentrena el
   proceso completo (log1p, límites de winsorizado, escalado y K-Means) sin los clientes de un pliegue y se les asigna
   segmento. Tras alinear la numeración de los clusters con el algoritmo húngaro (Kuhn, 1955), coinciden con el modelo
   final en el **{val_reasignacion}** de los casos (entre {val_reasignacion_min} y {val_reasignacion_max} según el
   pliegue). Es la operación que hace el aplicativo con un cliente nuevo, y es la idea de la validación por predicción
   de Lange et al. (2004).
2. **Clasificadores supervisados.** Con los segmentos como etiqueta, los clientes se dividieron en {val_pct_entrenamiento}
   para entrenamiento ({val_n_entrenamiento}) y {val_pct_prueba} para prueba ({val_n_prueba}), de forma estratificada.
   Los hiperparámetros se eligieron solo con validación cruzada dentro del entrenamiento (Kohavi, 1995), y el conjunto
   de prueba se usó una única vez al final.

Tabla: Exactitud de los clasificadores que aprenden los segmentos (validación cruzada de 5 pliegues en el entrenamiento y conjunto de prueba de {val_n_prueba} clientes).

{tabla_clasificadores}

El árbol de decisión (Breiman et al., 1984) se ajustó con una búsqueda en rejilla de {val_n_combinaciones}
combinaciones de profundidad máxima y mínimo de clientes por hoja. La mejor es profundidad {val_arbol_profundidad} con
mínimo {val_arbol_min_hoja} por hoja. La curva de la figura muestra el compromiso entre sencillez y exactitud: con
profundidad 3 el árbol acierta el {val_arbol_cv_prof3} en validación cruzada y con 5 el {val_arbol_cv_prof5}. A partir de
ahí la exactitud de validación se estanca, mientras la de entrenamiento sigue subiendo hasta el
{val_arbol_entrenamiento_max}: más profundidad solo sobreajusta. El bosque aleatorio (Breiman, 2001) llega al
{val_bosque_prueba} en prueba. En todos los casos la exactitud en prueba es parecida a la de validación cruzada, señal de
que la estimación es fiable.

![Exactitud del árbol de decisión según su profundidad](figures/05_exactitud_profundidad.png)

Tabla: Precisión, sensibilidad y F1 del árbol ajustado por segmento (conjunto de prueba).

{tabla_reporte_segmentos}

Los errores se concentran en **{val_peor_segmento}** (precisión {val_peor_precision}, sensibilidad {val_peor_recall}),
el segmento intermedio que limita con los otros tres. Ningún VIP se confunde con un pequeño ocasional ni al revés: los
errores son siempre entre segmentos vecinos.

![Matriz de confusión del árbol ajustado en el conjunto de prueba](figures/05_matriz_confusion.png)

Un árbol de solo profundidad 3 acierta el {val_arbol3_prueba} en prueba con reglas que se leen en unidades de negocio:

{reglas_arbol3}

### 7.4 Ajustes de hiperparámetros

Tabla: Hiperparámetros y decisiones de modelado probados, valor elegido y criterio.

{tabla_ajustes}

### 7.5 Decisión

Se eligió **K-Means con k = {k_final}**:

- **Métricas:** silhouette {final_silhouette}, Davies-Bouldin {final_davies_bouldin} y Calinski-Harabasz
  {final_calinski_harabasz}.
- **Estabilidad:** mediana {final_estabilidad_mediana} y percentil 10 {final_estabilidad_p10}.
- **Robustez a los empates:** ARI {final_ari_descuantizado}.
- **Exactitud con clientes no vistos:** {val_reasignacion} de reasignación; un árbol de decisión reproduce los
  segmentos con un {val_arbol_prueba} de exactitud en prueba.
- **Negocio:** segmentos accionables de tamaño útil.

Las métricas y el negocio no coinciden: las métricas premian k = 2. Es lo esperable en una nube continua, donde el
silhouette siempre favorece el k mínimo, y k = 2 solo separa a los clientes activos del resto (su segmento menor tiene
el {k2_pct_menor} de los clientes), sin permitir estrategias diferenciadas.

**¿Es aceptable un silhouette de {final_silhouette}?** Sí, en este contexto. El silhouette mide lo separados que están
los grupos. En la escala de Kaufman y Rousseeuw (1990), un valor por debajo de 0,25 indica que no hay una estructura de
grupos sustancial, y eso coincide con lo que muestran DBSCAN y HDBSCAN: los clientes forman un continuo. Aquí el objetivo no es
descubrir grupos naturales, sino **particionar ese continuo** en segmentos útiles. Para eso importa que la partición
sea estable, robusta e interpretable, y lo es.

## 8. Resultados: perfiles de los segmentos

Tabla: Perfil de los segmentos (medianas).

{tabla_segmentos}

- **{s3_nombre}:** compran a menudo ({s3_frequency} facturas de mediana), hace poco ({s3_recency} días) y mucha
  variedad ({s3_productos} productos). Generan el {s3_pct_ingresos} de los ingresos con el {s3_pct_clientes} de los
  clientes. El {s3_rfm_valor} son "Campeones" o "Leales" según RFM.
- **{s2_nombre}:** clientes recientes ({s2_pct_compra_90d} compró en los últimos 90 días) que ya repiten. Una parte entró
  en la campaña de Navidad.
- **{s1_nombre}:** el ticket más alto ({s1_ticket}), pero el {s1_pct_una_compra} compró una sola vez. El
  {s1_una_compra_temporada} de sus clientes de una compra lo hizo en septiembre–noviembre, un patrón de
  aprovisionamiento para Navidad.
- **{s0_nombre}:** pedidos pequeños (ticket de {s0_ticket}, {s0_productos} productos) y de vez en cuando. Aportan el
  {s0_pct_ingresos} de los ingresos.

![Snake plot: media estandarizada de las variables por segmento](figures/04_snake_plot.png)

![Radar: percentil medio de cada segmento](figures/04_radar.png)

**Valor añadido frente al baseline RFM.** Para RFM, {s0_nombre} y {s1_nombre} son casi lo mismo: el {s0_rfm_riesgo} y el
{s1_rfm_riesgo} son clientes hibernando o en riesgo. K-Means los separa por el tamaño del pedido y su valor es muy
distinto: el {s0_pct_ingresos} frente al {s1_pct_ingresos} de los ingresos. Por eso merecen inversiones de reactivación
distintas.

![Composición de cada segmento según el baseline RFM](figures/04_contingencia_rfm.png)

**Cuentas clave.** Los atípicos de DBSCAN se concentran en VIP. El subgrupo de cuentas clave reúne a los VIP atípicos
que gastan al menos la mediana del resto de VIP ({ck_umbral}). Son {ck_clientes} clientes ({ck_pct_clientes}) que
generan el {ck_pct_ingresos} de los ingresos: compran {ck_frequency} veces de mediana, con un ticket de {ck_ticket}.

![Segmentos en el plano PCA y cuentas clave (rombos)](figures/04_cuentas_clave_pca.png)

**Estacionalidad.** El {s2_pct_temporada} de las facturas de {s2_nombre} son de septiembre–noviembre (el más
estacional), y el {s2_captados_temporada} del segmento hizo su primera compra desde octubre de 2011. El mes de compra no
es independiente de la recency, así que estas cifras se interpretan con cautela.

![Estacionalidad de cada segmento](figures/04_estacionalidad.png)

## 9. Estrategias de marketing

Cada estrategia tiene una acción concreta, el momento de aplicarla y un KPI. La línea base es el valor actual del
indicador, calculado con los datos; no se fijan metas inventadas.

{lista_estrategias}

## 10. Aplicativo web

El aplicativo, desarrollado en Streamlit y Plotly, tiene seis páginas:

1. **Inicio:** objetivo, indicadores clave y hallazgos principales.
2. **Exploración:** limpieza y patrones de venta.
3. **Modelos y validación:** comparación de los algoritmos, justificación del modelo final, exactitud con datos de
   prueba, reglas del árbol de decisión y tabla de ajustes.
4. **Clusters:** PCA 2D y 3D y t-SNE, con selector de algoritmo y la advertencia sobre t-SNE.
5. **Perfiles y estrategias.**
6. **Explorador:** filtros, búsqueda por cliente, descarga en CSV y clasificación de un cliente nuevo.

Para clasificar un cliente nuevo se aplica exactamente el preprocesamiento del entrenamiento, incluido el recorte p1–p99,
y se reproduce la regla de cuenta clave. El atípico se decide por cercanía a los puntos núcleo de DBSCAN. Las pruebas
comprueban que el formulario reproduce el segmento de los {clientes} clientes. La aplicación se probó en un entorno con
solo las dependencias de despliegue y en un contenedor Docker.

![Página de inicio del aplicativo](figures/app_inicio.png)

![Clasificación de un cliente nuevo sobre el mapa PCA](figures/app_explorador.png)

## 11. Conclusiones

1. Una limpieza cuidadosa cambia el resultado. Sin el emparejamiento de compras anuladas aparecerían clientes con
   decenas de miles de libras de gasto que en realidad no compraron.
2. Las variables de clustering se eligieron por redundancia (VIF y correlación), por si estaban definidas para todos los
   clientes y por su efecto en el modelo. Así se descartaron el gasto total, la antigüedad, los días entre compras y la
   tasa de devoluciones, que se usan para perfilar.
3. K-Means con k = {k_final} es el mejor equilibrio entre métricas, estabilidad y utilidad. Las métricas internas solas
   habrían llevado a k = 2.
4. Los métodos de densidad confirmaron que los clientes forman un continuo. Su aporte fue detectar atípicos, de donde
   sale el subgrupo de cuentas clave ({ck_pct_clientes} de los clientes y {ck_pct_ingresos} de los ingresos).
5. Frente al baseline RFM, el clustering distingue a los ocasionales de pedido grande de los de pedido pequeño, una
   diferencia con impacto directo en cómo invertir en reactivación.
6. Aunque el silhouette es moderado, los segmentos son reproducibles: con clientes que el modelo no vio, la
   reasignación coincide en el {val_reasignacion} de los casos y un árbol de decisión los reconoce con un
   {val_arbol_prueba} de exactitud, con reglas sencillas que se pueden explicar a negocio.

## 12. Limitaciones y trabajo futuro

- Se excluyó el {pct_lineas_sin_cliente} de las líneas, sin cliente identificado; la segmentación no describe a esos
  compradores.
- Con un solo año de datos no se puede separar del todo la estacionalidad del crecimiento ni confirmar que los
  compradores de temporada repitan cada Navidad.
- La segmentación es una foto fija: con datos nuevos habría que recalcular las variables y vigilar la estabilidad.
- Las estrategias no se han probado. El siguiente paso es medirlas con grupos de control contra las líneas base
  indicadas.
- Como trabajo futuro: incorporar información de producto (categorías) y modelos de valor de vida del cliente (CLV).

## Referencias

- Arthur, D. y Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. *Proceedings of the 18th Annual
  ACM-SIAM Symposium on Discrete Algorithms*, 1027–1035.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Breiman, L., Friedman, J. H., Olshen, R. A. y Stone, C. J. (1984). *Classification and Regression Trees*. Wadsworth.
- Caliński, T. y Harabasz, J. (1974). A dendrite method for cluster analysis. *Communications in Statistics*, 3(1), 1–27.
- Campello, R. J. G. B., Moulavi, D. y Sander, J. (2013). Density-based clustering based on hierarchical density
  estimates. *Advances in Knowledge Discovery and Data Mining (PAKDD)*, 160–172.
- Chen, D., Sahinoglu, Z. y Guo, Y. (2012). Data mining for the online retail industry: A case study of RFM model-based
  customer segmentation using data mining. *Journal of Database Marketing & Customer Strategy Management*, 19(3),
  197–208.
- Davies, D. L. y Bouldin, D. W. (1979). A cluster separation measure. *IEEE Transactions on Pattern Analysis and
  Machine Intelligence*, 1(2), 224–227.
- Ester, M., Kriegel, H.-P., Sander, J. y Xu, X. (1996). A density-based algorithm for discovering clusters in large
  spatial databases with noise. *Proceedings of the 2nd International Conference on Knowledge Discovery and Data
  Mining (KDD)*, 226–231.
- Hennig, C. (2007). Cluster-wise assessment of cluster stability. *Computational Statistics & Data Analysis*, 52(1),
  258–271.
- Hubert, L. y Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2(1), 193–218.
- Hughes, A. M. (1994). *Strategic Database Marketing*. Probus Publishing.
- Kaufman, L. y Rousseeuw, P. J. (1990). *Finding Groups in Data: An Introduction to Cluster Analysis*. John Wiley &
  Sons.
- Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection.
  *Proceedings of the 14th International Joint Conference on Artificial Intelligence (IJCAI)*, 2, 1137–1143.
- Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly*, 2(1–2),
  83–97.
- Lange, T., Roth, V., Braun, M. L. y Buhmann, J. M. (2004). Stability-based validation of clustering solutions.
  *Neural Computation*, 16(6), 1299–1323.
- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the
  Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281–297.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12,
  2825–2830.
- Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis.
  *Journal of Computational and Applied Mathematics*, 20, 53–65.
- Schubert, E., Sander, J., Ester, M., Kriegel, H.-P. y Xu, X. (2017). DBSCAN revisited, revisited: Why and how you
  should (still) use DBSCAN. *ACM Transactions on Database Systems*, 42(3), 1–21.
- Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics*, 6(2), 461–464.
- UCI Machine Learning Repository. Online Retail [Dataset]. https://archive.ics.uci.edu/dataset/352/online+retail
- van der Maaten, L. y Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9,
  2579–2605.
- Wattenberg, M., Viégas, F. y Johnson, I. (2016). How to use t-SNE effectively. *Distill*.
  https://doi.org/10.23915/distill.00002
