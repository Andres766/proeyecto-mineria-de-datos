# Segmentación de clientes: resumen para la exposición

Proyecto final · Inteligencia de Negocios y Minería de Datos · Ingeniería de Software

## Cómo usar este resumen

Es la versión corta del informe, pensada para preparar la exposición. Cada apartado trae **lo que hay que decir** y las
cifras que lo respaldan; todas salen del código del proyecto (`python -m segmentacion.reportes`). Al final hay
preguntas probables con respuestas cortas. Para profundizar: informe completo y guía de sustentación
({n_preguntas_guia} preguntas).

**La idea en una frase:** agrupamos a los {clientes} clientes de una tienda online según cómo compran, comparamos tres
familias de algoritmos y elegimos **K-Means con {k_final} segmentos**, que son estables, se reproducen con clientes nuevos
({val_reasignacion} de coincidencia) y permiten estrategias de marketing distintas para cada grupo.

**App publicada:** https://segmentacion-retail-ucc.streamlit.app

## 1. Orden sugerido de la exposición

1. **Problema y datos:** qué es el dataset y por qué segmentar.
2. **Limpieza:** el hallazgo de las compras anuladas.
3. **Variables:** las cuatro variables y su preparación.
4. **Modelos:** K-Means, GMM y DBSCAN; por qué k = {k_final}.
5. **Validación:** pruebas, exactitud con datos de prueba y ajustes.
6. **Resultados:** los segmentos, las cuentas clave y las estrategias.
7. **Demostración de la app** y cierre.

## 2. Problema y datos

- **Dataset:** *Online Retail* (UCI), un minorista online británico de artículos de regalo. Transacciones del
  {fecha_inicio} al {fecha_fin}: {filas_originales} líneas de factura, {paises} países y {productos} productos.
- **Problema:** tratar a todos los clientes por igual desperdicia recursos. El análisis RFM clásico (recencia,
  frecuencia, gasto) usa cortes arbitrarios por quintiles; el clustering busca los grupos a partir de los datos.
- **Objetivo:** segmentar con clustering y reducción de dimensionalidad, validar los segmentos, perfilarlos, proponer
  estrategias y publicar un dashboard.

## 3. Limpieza: lo importante

- Se quitaron duplicados, líneas sin cliente ({pct_lineas_sin_cliente} de las líneas), códigos que no son productos y
  cantidades o precios no positivos.
- **El hallazgo clave:** las cancelaciones anulan compras anteriores. Borrar solo las facturas «C» deja la compra
  original como si se hubiera hecho; el mayor pedido del dataset ({max_cantidad_anulada} unidades) se canceló minutos
  después. Se emparejó cada cancelación con su compra (1 a 1) y se quitaron ambas: **{compras_anuladas} compras
  anuladas**.
- **Resultado:** {filas_limpias} líneas de venta de **{clientes} clientes**, con {ingresos} de ingresos.

## 4. Variables y preparación

- **Variables del clustering** (una fila por cliente): días desde la última compra, número de facturas, ticket promedio
  y productos distintos.
- **Fuera del clustering, solo para perfilar:** el gasto total (es facturas × ticket, redundante), la tasa de
  devoluciones (creaba un grupo solo de devolvedores), los días entre compras (no existen para el {pct_una_compra} de
  clientes con una sola compra) y el país.
- **Preparación:** log1p (las variables tienen colas muy largas), recorte a los percentiles 1–99 (los extremos no
  arrastran los centroides) y estandarización (todas pesan igual).

## 5. Modelos y elección

Tabla: Modelos comparados (silhouette: separación de los grupos; estabilidad: ARI en el peor 10 % de 50 submuestras).

| Modelo | Silhouette | Estabilidad (peor 10 %) | Veredicto |
|---|---|---|---|
| K-Means k = 2 | {k2_silhouette} | {k2_estabilidad_p10} | Mejor métrica, pero solo separa activos de inactivos |
| **K-Means k = {k_final}** | **{final_silhouette}** | **{final_estabilidad_p10}** | **Elegido:** estable y accionable |
| K-Means k = 5 | {k5_silhouette} | {k5_estabilidad_p10} | Inestable: dos soluciones casi empatadas |
| GMM ({gmm_tipo}, k = {gmm_k}) | {gmm_silhouette} | {gmm_estabilidad_p10} | Peor en todo; necesitó `reg_covar` = {reg_covar} |
| DBSCAN | — | — | Un solo grupo y {atipicos} atípicos: no hay grupos por densidad |

- **Por qué k = {k_final} y no k = 2:** el silhouette baja con cada k porque los clientes forman una nube continua; en
  ese caso siempre gana el k más pequeño. Con k = 2 no hay estrategias distintas que proponer.
- **PCA** resume los datos en dos ejes que explican el {pca_2}: PC1 = actividad / valor y PC2 = tamaño del pedido. Se
  usa para interpretar y dibujar. **t-SNE** solo sirve para visualizar: no conserva distancias ni tamaños.

![Segmentos sobre el mapa PCA, con las cuentas clave marcadas (0 = {s0_nombre}, 1 = {s1_nombre}, 2 = {s2_nombre}, 3 = {s3_nombre})](figures/04_cuentas_clave_pca.png)

## 6. Validación: pruebas, exactitud y ajustes

**Pruebas de validación del clustering** (no hay etiqueta real, así que se valida su calidad y su consistencia):

- **Métricas internas:** silhouette {final_silhouette}, Davies-Bouldin {final_davies_bouldin}, Calinski-Harabasz
  {final_calinski_harabasz}.
- **Estabilidad:** el modelo se repitió con 50 submuestras del 80 %. ARI mediano {final_estabilidad_mediana} (1 =
  mismos grupos) y {final_estabilidad_p10} en el peor 10 %.
- **Robustez:** ARI {final_ari_descuantizado} al repartir al azar los empates de la frecuencia.

**Exactitud con datos de entrenamiento y de prueba.** La exactitud clásica necesita una respuesta correcta, y en
clustering no hay «segmento verdadero». Por eso se mide con clientes que el modelo **no vio al entrenar**:

- **Reasignación ({val_n_pliegues} pliegues):** se reentrena todo sin un 20 % de los clientes y se les asigna segmento.
  Coincide con el modelo final en el **{val_reasignacion}** de los casos (entre {val_reasignacion_min} y
  {val_reasignacion_max}).
- **Clasificadores** con partición {val_pct_entrenamiento} entrenamiento / {val_pct_prueba} prueba ({val_n_prueba}
  clientes de prueba):

Tabla: Exactitud de los clasificadores que aprenden los segmentos.

{tabla_clasificadores}

- **Lectura:** la exactitud en prueba es casi igual a la de validación cruzada, así que el modelo no está
  sobreajustado. Los errores se concentran en {val_peor_segmento}, el segmento intermedio, y son siempre entre segmentos
  vecinos.

![Matriz de confusión del árbol ajustado en el conjunto de prueba](figures/05_matriz_confusion.png)

**Ajustes de hiperparámetros.** Todo lo que se probó y el criterio de elección:

Tabla: Ajustes del proyecto.

{tabla_ajustes}

## 7. Resultados: los segmentos

Tabla: Los cuatro segmentos (medianas por cliente).

| Segmento | % clientes | % ingresos | Facturas | Ticket | Días sin comprar |
|---|---|---|---|---|---|
| {s3_nombre} | {s3_pct_clientes} | {s3_pct_ingresos} | {s3_frequency} | {s3_ticket} | {s3_recency} |
| {s2_nombre} | {s2_pct_clientes} | {s2_pct_ingresos} | {s2_frequency} | {s2_ticket} | {s2_recency} |
| {s1_nombre} | {s1_pct_clientes} | {s1_pct_ingresos} | {s1_frequency} | {s1_ticket} | {s1_recency} |
| {s0_nombre} | {s0_pct_clientes} | {s0_pct_ingresos} | {s0_frequency} | {s0_ticket} | {s0_recency} |

- **Cuentas clave:** dentro de VIP, {ck_clientes} clientes que DBSCAN marca como atípicos y que gastan al menos la
  mediana del resto de VIP. Son el {ck_pct_clientes} de los clientes y generan el **{ck_pct_ingresos}** de los ingresos.
- **Frente al RFM:** RFM ve casi iguales a los dos grupos de ocasionales ({s0_rfm_riesgo} y {s1_rfm_riesgo} hibernando
  o en riesgo); K-Means los separa por el tamaño del pedido, y su valor es muy distinto ({s0_pct_ingresos} frente a
  {s1_pct_ingresos} de los ingresos).

**Estrategia principal de cada segmento** (cada una tiene momento y KPI; el resto está en el informe):

- **{s3_nombre}:** {s3_estrategia} *KPI:* {s3_estrategia_kpi}
- **Cuentas clave:** {ck_estrategia} *KPI:* {ck_estrategia_kpi}
- **{s2_nombre}:** {s2_estrategia} *KPI:* {s2_estrategia_kpi}
- **{s1_nombre}:** {s1_estrategia} *KPI:* {s1_estrategia_kpi}
- **{s0_nombre}:** {s0_estrategia} *KPI:* {s0_estrategia_kpi}

## 8. La aplicación

Streamlit + Plotly, seis páginas: Inicio (indicadores y hallazgos), Exploración, Modelos y validación (incluye la
exactitud y los ajustes), Clusters (PCA 2D/3D y t-SNE), Perfiles y estrategias, y Explorador (filtros, búsqueda,
descarga en CSV y **clasificación de un cliente nuevo**). Para clasificar se usa exactamente el mismo preprocesamiento
del entrenamiento.

**Para la demo:** abrir la app unos minutos antes (Streamlit «duerme» las apps sin uso) y tener a mano un cliente de
ejemplo, como el 14646 (VIP · cuenta clave).

## 9. Preguntas probables y respuestas cortas

**1. ¿Cuál es la exactitud del modelo?**
En clustering no hay una respuesta correcta con la que comparar, así que se mide con clientes no vistos: el modelo les
asigna el mismo segmento en el **{val_reasignacion}** de los casos. Un árbol de decisión reconoce los segmentos con un
**{val_arbol_prueba}** en prueba y un bosque aleatorio con un **{val_bosque_prueba}**.

**2. ¿Por qué no se puede comparar con la exactitud de un modelo supervisado, como un 79 %?**
Porque miden cosas distintas. En un problema supervisado la exactitud depende de lo predecible que sea una etiqueta
real. Aquí mide que los segmentos son reproducibles con datos nuevos y explicables con reglas.

**3. ¿Qué pruebas hicieron para validar?**
Métricas internas, estabilidad en 50 submuestras (ARI {final_estabilidad_mediana}), robustez a los empates
({final_ari_descuantizado}), concordancia con RFM, exactitud con entrenamiento y prueba, y {n_pruebas} pruebas
automáticas del código con `pytest`.

**4. ¿Qué ajustes hicieron?**
{n_ajustes} en total: k de 2 a 10 en K-Means; tipo de covarianza, número de componentes y `reg_covar` en GMM; `eps` y
`min_samples` en DBSCAN; perplexity en t-SNE; escalado y recorte de extremos en el preprocesado; y una búsqueda en
rejilla de {val_n_combinaciones} combinaciones para el árbol de decisión.

**5. ¿Cómo evitaron el sobreajuste?**
Los hiperparámetros se eligieron con validación cruzada dentro del entrenamiento y el conjunto de prueba se usó una sola
vez. El árbol llega al {val_arbol_entrenamiento_max} en entrenamiento pero se estanca en {val_arbol_cv} en validación:
más profundidad solo memoriza.

**6. ¿No es circular entrenar un árbol con etiquetas de K-Means?**
El árbol no prueba que los segmentos sean «verdaderos», sino que son reproducibles con reglas sencillas en clientes no
vistos. Si fueran ruido, fallaría en prueba. La prueba más directa es la reasignación, que no usa otro modelo.

**7. ¿Están balanceadas las clases? ¿No infla eso la exactitud?**
Están balanceadas en número de clientes: el segmento mayor ({seg_mayor_nombre}, {seg_mayor_pct_clientes}) tiene solo
{seg_ratio_mayor_menor} veces más clientes que el menor ({seg_menor_nombre}, {seg_menor_pct_clientes}), y la partición es
estratificada. La exactitud balanceada del árbol es {val_exactitud_balanceada} y el F1 macro {val_f1_macro}, casi igual a su
exactitud ({val_arbol_prueba}). El desbalance está en los ingresos ({s3_nombre} aporta el {s3_pct_ingresos}): es un
hallazgo de negocio, no un problema del modelo.

**8. Las métricas dicen k = 2. ¿Por qué k = {k_final}?**
El silhouette baja con cada k porque los datos son una nube continua, y ahí siempre gana el k mínimo. k = 2 solo separa
activos de inactivos; k = {k_final} es muy estable (peor caso {final_estabilidad_p10}) y da segmentos accionables.

**9. ¿Por qué no k = 5?**
En el peor 10 % de submuestras su estabilidad cae a {k5_estabilidad_p10}: tiene dos soluciones casi empatadas y salta
de una a otra.

**10. Un silhouette de {final_silhouette} es bajo. ¿Es aceptable?**
Sí. Mide la separación entre grupos, y los clientes forman un continuo (lo confirmaron DBSCAN y HDBSCAN). El objetivo es
dividir ese continuo en segmentos útiles, y para eso importan la estabilidad, la robustez y la interpretabilidad, que
son altas.

**11. ¿Por qué esas cuatro variables y no el gasto total?**
El gasto es facturas × ticket: incluirlo contaría la misma información dos veces. Frecuencia y ticket juntos ya lo
contienen y además separan dos comportamientos distintos.

**12. ¿Por qué log1p, recorte p1–p99 y estandarización?**
Las variables tienen colas muy largas: sin log1p unos pocos mayoristas definirían toda la geometría. El recorte evita que
los extremos arrastren los centroides sin sacar a nadie, y la estandarización hace que todas pesen igual.

**13. ¿Qué pasó con GMM?**
Un tercio de los clientes tiene una sola factura, y GMM ponía una campana de varianza casi cero sobre ese empate. Se
corrigió con `reg_covar` = {reg_covar}, pero quedó peor que K-Means (silhouette {gmm_silhouette}).

**14. ¿Y DBSCAN?**
Con `eps` = {eps} y `min_samples` = {min_samples} encuentra un solo grupo y {atipicos} atípicos: no hay grupos separados
por densidad. Se usa para detectar atípicos, de donde salen las cuentas clave.

**15. ¿Para qué PCA y t-SNE?**
PCA para interpretar y dibujar (dos ejes explican el {pca_2}). t-SNE solo para visualizar: no conserva distancias ni
tamaños, así que no sirve para validar.

**16. ¿Qué son las cuentas clave?**
{ck_clientes} clientes VIP atípicos para DBSCAN y con gasto alto: el {ck_pct_clientes} de los clientes y el
{ck_pct_ingresos} de los ingresos. Merecen un gestor de cuenta dedicado.

**17. ¿Qué aporta frente al RFM?**
RFM no distingue a los dos grupos de ocasionales; K-Means los separa por el tamaño del pedido ({s0_pct_ingresos} frente
a {s1_pct_ingresos} de los ingresos), lo que cambia cuánto invertir en reactivar a cada uno.

**18. ¿Cómo clasifica la app a un cliente nuevo?**
No reentrena: carga el mismo preprocesador (con los límites del entrenamiento) y el mismo K-Means. Las pruebas
comprueban que reproduce el segmento de los {clientes} clientes y las {ck_clientes} cuentas clave.

**19. ¿Por qué quitaron las líneas sin cliente?**
Porque se segmentan clientes y esas líneas ({pct_lineas_sin_cliente}) no se pueden asignar a nadie. Está declarado como
limitación.

**20. ¿Cómo sabrían si las estrategias funcionan?**
Cada estrategia tiene un KPI con su línea base actual. Se aplicaría a una parte del segmento, con otra como grupo de
control, y se compararía el KPI.

**21. ¿Cuáles son las limitaciones?**
Un solo año de datos (no separa estacionalidad de crecimiento), clientes sin identificar excluidos, segmentación como
foto fija que hay que recalcular, y estrategias aún sin probar.

## 10. Para cerrar

- Una buena limpieza cambia el resultado: sin emparejar las compras anuladas habría clientes con gastos que nunca
  existieron.
- Las métricas internas solas habrían elegido k = 2; la estabilidad y el negocio llevaron a k = {k_final}.
- Los segmentos son consistentes y reproducibles ({val_reasignacion} con clientes nuevos) y cada uno tiene una
  estrategia medible.
