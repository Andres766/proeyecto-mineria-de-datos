# Guía de sustentación

Guía para explicar el proyecto con palabras sencillas y responder a las preguntas del docente. Todas las cifras se
generan desde el código (`python -m segmentacion.reportes`), igual que el informe.

## 1. El proyecto en 30 segundos

Agrupamos a los 4.324 clientes de una tienda online según **cómo compran**: cuándo fue su última compra, cuántas
veces compran, cuánto gastan por pedido y cuánta variedad de productos llevan. Comparamos tres familias de algoritmos y
nos quedamos con **K-Means con 4 segmentos**: VIP, En crecimiento, Grandes ocasionales y Pequeños ocasionales.

Los VIP son el 20,5 % de los clientes y generan el 70,2 % de los ingresos. Dentro de ellos hay
66 **cuentas clave** que por sí solas suman el 32,3 % de los ingresos. Cada segmento tiene
estrategias con acción, momento y KPI. Todo está en una app web.

## 2. Conceptos explicados con sencillez

**RFM.** Recency (cuántos días hace de la última compra), Frequency (cuántas facturas) y Monetary (cuánto gastó en
total). Es la forma clásica de describir a un cliente. El RFM "por quintiles" ordena a los clientes en cada variable, los
divide en cinco grupos iguales y les da una nota de 1 a 5; según las notas, cada cliente recibe un nombre (Campeones,
Hibernando…). Lo usamos como punto de comparación (baseline).

**log1p.** Es log(1 + x). El gasto, la frecuencia y el resto de variables tienen colas larguísimas: unos pocos clientes
compran muchísimo. El logaritmo "comprime" esas colas y deja distribuciones más simétricas. Así un cliente que gasta
200.000 £ no aplasta a todos los demás en el cálculo de distancias. Se usa log(1 + x) porque está definido en 0.

**Escalado (StandardScaler).** Resta la media y divide por la desviación típica de cada variable. Sin esto, la variable
con números más grandes dominaría las distancias. Tras escalar, todas pesan igual.

**Winsorización.** Recortar cada variable a sus percentiles 1 y 99: lo que pasa del límite se queda en el límite. El
cliente sigue en el análisis, pero no puede tirar de los centroides con un valor extremo.

**VIF (factor de inflación de la varianza).** Mide cuánto se puede predecir una variable a partir de las demás. Un VIF
alto (por encima de 5 o 10) significa que la variable es redundante.

**K-Means.** Elige k centros, asigna cada cliente al centro más cercano, mueve cada centro al promedio de sus clientes y
repite hasta que nada cambia. Produce grupos compactos y del mismo "tamaño de radio".

**GMM (mezcla de gaussianas).** Supone que los datos son una suma de "campanas" gaussianas, cada una con su forma. En
vez de asignar al cliente a un grupo, le da una **probabilidad** de pertenecer a cada uno. El número de campanas se
elige con el BIC.

**BIC / AIC.** Criterios para elegir modelos probabilísticos: premian lo bien que el modelo explica los datos y castigan
que tenga muchos parámetros. Cuanto más bajo, mejor. El BIC castiga más la complejidad que el AIC.

**DBSCAN.** Agrupa por **densidad**: un punto es "núcleo" si tiene al menos `min_samples` vecinos a menos de `eps`.
Los núcleos cercanos forman un cluster, y los puntos que no están cerca de ningún núcleo son **ruido** (atípicos). No
hay que fijar el número de clusters.

**HDBSCAN.** Es DBSCAN probando todos los `eps` a la vez. Se queda con los grupos que aparecen de forma más estable.

**PCA.** Combina las variables en ejes nuevos (componentes) ordenados por cuánta variación de los datos explican.
Sirve para resumir e interpretar. Aquí, PC1 es "actividad / valor" y PC2 "tamaño del pedido".

**t-SNE.** Dibuja los datos en 2D intentando que los clientes parecidos queden cerca. Es muy bueno para **ver**, pero
**no conserva distancias entre grupos ni tamaños**: no sirve para medir nada.

**Silhouette.** Para cada cliente compara lo cerca que está de su propio grupo con lo cerca que está del grupo vecino.
Va de −1 a 1; cerca de 1 significa grupos muy separados.

**Davies-Bouldin.** Compara lo disperso de cada grupo con la distancia al grupo más parecido. Más bajo es mejor.

**Calinski-Harabasz.** Razón entre la separación entre grupos y la dispersión dentro de ellos. Más alto es mejor.

**ARI (índice de Rand ajustado).** Mide cuánto se parecen dos agrupaciones: 1 = idénticas, 0 = lo que daría el azar.
No importa cómo se llamen los grupos, solo quién va con quién.

**Estabilidad por remuestreo.** Repetimos el clustering 50 veces con el 80 % de los clientes elegidos al azar y medimos
con el ARI si salen los mismos grupos. Si cambian mucho, los segmentos no son fiables.

## 3. Decisiones clave del proyecto (resumen)

| Decisión | Qué se hizo | Por qué |
|---|---|---|
| Compras anuladas | Emparejar cada cancelación con la compra que anula y quitar ambas | Si no, quedan "clientes" con gastos enormes que nunca se concretaron |
| Regla `M` | Anular una compra si hay una cancelación manual del mismo importe en menos de 24 h | Algunas anulaciones se registran con un código manual por el total |
| Variables | recency, frequency, ticket promedio y productos distintos | Sin redundancia (VIF máximo 2,018) |
| Gasto total | Fuera del clustering, usado para perfilar | Es frequency × ticket: redundante (VIF 199) |
| Tasa de devoluciones | Solo para perfilar | Creaba un grupo marginal de "devolvedores" y un silhouette engañoso |
| Mayoristas | Winsorizar p1–p99 sin excluir a nadie | Los extremos generan el 29,7 % de los ingresos |
| Número de segmentos | k = 4 | Equilibrio entre métricas, estabilidad y utilidad |
| GMM | `reg_covar` = 0,05 | Con el valor por defecto los componentes degeneraban |
| DBSCAN | Solo para marcar atípicos | No hay grupos separados por densidad |
| t-SNE | Solo para visualizar | No conserva distancias ni tamaños |

## 4. Preguntas probables y respuestas

**1. ¿Por qué eliminaron las líneas sin CustomerID si eran tantas?**
Porque segmentamos clientes y esas líneas no se pueden asignar a nadie. Son el 24,93 % de las líneas y
el 16,46 % del importe, y les falta el cliente en facturas completas. No hay ninguna variable que
permita imputarlo. Lo declaramos como limitación: el modelo describe a los clientes registrados.

**2. ¿Qué son las "compras anuladas" y por qué no basta con borrar las facturas que empiezan por C?**
Una factura "C" cancela una compra anterior. Si solo borras la "C", la compra original queda como si se hubiera hecho.
Los dos mayores valores del dataset (80.995 unidades) son pedidos cancelados minutos después: sin
emparejarlos, esos clientes aparecerían con miles de libras de gasto ficticio. Por eso emparejamos cada cancelación con
su compra (mismo cliente, producto, cantidad y precio, la compra más reciente anterior) y quitamos ambas:
2.726 compras.

**3. ¿Qué es la regla M y por qué 24 horas?**
Algunas anulaciones no se registran con el producto, sino con una línea manual `M` por el importe total. Por ejemplo,
un pedido de 60 cestas a un precio erróneo, anulado a los 3 minutos con una `M` por el importe exacto. Las coincidencias
por importe lejanas en el tiempo eran ambiguas (una misma `M` coincidía con muchas compras de meses atrás), así que la
regla es conservadora: mismo cliente, mismo importe exacto, menos de 24 h y emparejamiento 1 a 1. Anuló
5 compras.

**4. ¿Qué error encontraron gracias a las pruebas?**
Al escribir pruebas con casos pequeños de resultado conocido, una falló: con dos compras idénticas y dos cancelaciones
idénticas solo se anulaba una compra. La función usaba `merge_asof`, que mandaba las dos cancelaciones a la misma
compra (la más reciente), y la segunda se descartaba. Lo corregimos con un emparejamiento iterativo realmente 1 a 1 y
regeneramos datos, notebook y resultados. En el historial de git quedan la prueba y la corrección en commits separados.

**5. ¿Cómo definieron la tasa de devoluciones?**
Importe devuelto computable dividido entre importe comprado:
- No cuentan las cancelaciones de compras anuladas, cuyas compras ya quitamos.
- Del resto solo cuenta lo que el cliente pudo devolver: no más unidades de las que compró antes de ese producto dentro
  del periodo.
- Las devoluciones de compras anteriores al inicio de los datos computan 0, porque su compra no está en el denominador.
- La recortamos a [0, 1] por robustez (el máximo real es 0,6451).

**6. ¿Por qué la tasa de devoluciones no entra en el clustering?**
Porque el 73,5 % de los clientes tiene tasa 0. Al incluirla, aparecía un grupo pequeño
(160 clientes) formado al 100 % por clientes que devuelven, y el silhouette subía de
0,258 a 0,316 solo porque ese grupo quedaba aislado. Es una separación
"devuelve / no devuelve" sin valor para segmentar, así que la usamos para perfilar.

**7. Si el RFM usa el gasto total (M), ¿por qué lo sacaron del clustering?**
Porque el gasto total es exactamente frecuencia × ticket promedio, y en logaritmos eso es una suma. Con las tres
variables juntas el VIF del gasto era 199: información repetida que cuenta doble en las distancias. No se
pierde nada, porque frecuencia y ticket juntos contienen el gasto, y además separan dos comportamientos que el total
mezcla (comprar a menudo frente a comprar mucho de una vez). El gasto total se usa en el perfilamiento y en el baseline.

**8. ¿Qué hicieron con los días entre compras?**
No existe para los 1.503 clientes de una sola compra (34,8 %). Imputar con la mediana
inventaba un valor para un tercio de los clientes; imputar con la recency duplicaba otra variable; una indicadora
"una compra" era idéntica a frequency = 1. La dejamos para perfilar.

**9. ¿Por qué transformaron con logaritmo?**
Porque las distribuciones tienen colas larguísimas y K-Means trabaja con distancias: sin el logaritmo, unos pocos
mayoristas definirían toda la geometría. En escala log las variables quedan casi simétricas.

**10. ¿Por qué StandardScaler y no RobustScaler, si hay outliers?**
Tras el logaritmo los extremos ya están contenidos y además los tratamos con la winsorización. RobustScaler dejaba a las
variables con pesos distintos (más peso al ticket) y su ventaja en silhouette era como mucho de
0,015. Preferimos pesos iguales y explícitos.

**11. ¿Por qué no eliminaron a los mayoristas?**
Porque son los clientes más valiosos: los 91 extremos generan el 29,7 % de los
ingresos. Excluirlos cambiaba la segmentación del resto y dejaba fuera a quien más importa al negocio. Con el recorte
p1–p99 siguen en su segmento sin dominar los centroides.

**12. ¿Cómo eligieron el número de segmentos? ¿Por qué 4 y no 2, si las métricas dicen 2?**
Miramos codo, silhouette, Davies-Bouldin, Calinski-Harabasz, estabilidad, robustez a los empates, tamaño de los grupos
e interpretabilidad. Las métricas internas premian k = 2 (silhouette 0,326), pero el silhouette baja con cada
k: es la señal de una nube continua, donde siempre gana el k más pequeño. Con k = 2 solo separamos "activos" de "el
resto", y eso no permite estrategias distintas. k = 4 pierde poco en métricas, es muy estable (peor caso
0,929) y da cuatro segmentos accionables.

**13. ¿Por qué no k = 5, si tiene la mejor estabilidad mediana?**
Porque su peor caso es malo (percentil 10 de 0,744). k = 5 tiene dos soluciones casi igual de buenas
(las alternativas solo tienen entre un 0,12 % y un 0,20 % más de inercia) y, según la submuestra, cae en una
u otra. Lo que cambia es justo su segmento nuevo (ticket alto). Una segmentación de marketing no puede cambiar cada vez
que se actualizan los datos.

**14. Un silhouette de 0,240 es bajo. ¿Por qué es aceptable?**
- **Qué mide:** el silhouette mide lo **separados** que están los grupos. Un valor bajo dice que no hay fronteras
  nítidas, y eso es una propiedad de los datos, no un error del modelo. Los clientes forman un continuo, y lo
  confirmaron por separado DBSCAN (un único grupo denso) y HDBSCAN (sin estructura al romper los empates).
- **Cuál es el objetivo:** no buscamos "tipos naturales" de clientes, sino **particionar ese continuo** en segmentos
  útiles. Para eso los criterios correctos son que la partición sea:
  - **estable**: ARI mediano 0,958 en 50 submuestras;
  - **robusta**: 0,941 al romper los empates;
  - **interpretable y de tamaño útil**: cuatro perfiles claros; el más pequeño reúne el 20,5 % de los
    clientes.
- **Por qué no buscar un silhouette más alto:** se puede, pero con trampas. k = 2 daba 0,326 y no sirve para
  marketing; añadir la tasa de devoluciones lo subía a 0,316 creando un grupo marginal; calcularlo en
  el espacio de PCA lo inflaba ignorando la varianza descartada. Un silhouette alto obtenido así sería peor resultado.
- **Contexto:** en la escala de interpretación de Kaufman y Rousseeuw (1990), un silhouette por debajo de 0,25 indica
  que no hay una estructura de grupos sustancial. Eso es justo lo que concluimos con DBSCAN y HDBSCAN: no estamos
  descubriendo grupos naturales, sino dividiendo un continuo. Por eso la calidad se demuestra con la estabilidad, la
  robustez y la utilidad, y no con el silhouette solo.

**15. ¿Cómo midieron la estabilidad y qué es el ARI?**
Repetimos el clustering 50 veces, cada vez con el 80 % de los clientes elegidos al azar, y comparamos con el ARI la
partición de esos clientes con la del modelo completo. El ARI vale 1 si las particiones son idénticas y 0 si se parecen
lo mismo que por azar. Usamos submuestras sin reemplazo y no bootstrap con reemplazo porque los duplicados alteran la
densidad y sesgan a DBSCAN; así todos los algoritmos se evalúan igual.

**16. ¿Qué pasó con GMM y qué es `reg_covar`?**
Un tercio de los clientes tiene exactamente frequency = 1, así que en esa dimensión están todos en el mismo punto. GMM
ponía una campana justo ahí con varianza casi cero, su densidad se disparaba y el BIC bajaba artificialmente hasta
6.257: el modelo describía el empate, no a los clientes. `reg_covar` es una constante que se suma a la
varianza para que no llegue a 0. Buscamos el mínimo valor que elimina la degeneración (0,05). Aun así GMM quedó
peor que K-Means: silhouette 0,145 y estabilidad mediana 0,753.

**17. ¿Por qué DBSCAN no sirvió para segmentar? ¿Cómo eligieron sus parámetros?**
`min_samples` = 8 (dos veces el número de variables, la regla habitual) y `eps` = 0,717 (el codo del gráfico
de distancias al vecino número 8). Con eps pequeños parecía haber clusters, pero eran **falsas zonas
densas**: rebanadas de clientes con el mismo número de facturas, que parecen más densos porque comparten coordenada
(distancia mediana 0,276 frente a 0,415). Con el eps del codo, por encima del hueco entre niveles, queda un
único grupo y 163 atípicos. Por eso DBSCAN aporta la marca de atípicos y no una segmentación.

**18. ¿Qué es HDBSCAN y por qué lo descartaron?**
Es DBSCAN sin `eps` fijo: se queda con los grupos más persistentes. Siempre devolvía dos grupos, que eran exactamente
"una compra" frente a "más de una". Al repartir la frecuencia al azar dentro de su intervalo de redondeo esa estructura
desaparecía (ARI 0,041): era un artefacto de los empates.

**19. ¿Para qué usaron PCA y por qué no clusterizaron sobre las componentes?**
Para interpretar (PC1 = actividad / valor, PC2 = tamaño del pedido; juntas explican el 80,2 %) y para dibujar los
mapas 2D y 3D. Con solo cuatro variables no hace falta reducir dimensiones, y quedarse con 2 o 3 componentes no mejoraba
el clustering. Su silhouette solo "mejoraba" si se medía en el espacio reducido, lo que es engañoso.

**20. ¿Por qué dicen que t-SNE no sirve para validar?**
Porque t-SNE solo intenta conservar quién es vecino de quién; las distancias entre grupos y el tamaño aparente de cada
grupo no significan nada, y cambian con la perplexity. Lo comprobamos: el segmento más disperso en realidad aparece como
el más compacto en el mapa. Lo usamos solo para mostrar que los segmentos ocupan zonas coherentes, y la app lo advierte.

**21. ¿Qué aporta el clustering frente al RFM por quintiles?**
Coinciden en lo evidente: el 94,7 % de los VIP son Campeones o Leales en RFM. Pero RFM ve casi iguales a los dos
grupos de ocasionales (88 % y 90 % hibernando o en riesgo), y K-Means los separa por el tamaño del
pedido. Esa diferencia importa: Grandes ocasionales genera el 15,8 % de los ingresos y Pequeños ocasionales el
2,7 %, así que merecen inversiones de reactivación muy distintas. Además, el clustering no depende de cortes
arbitrarios en una variable con muchos empates.

**22. ¿Qué son las cuentas clave?**
Clientes VIP que DBSCAN marca como atípicos y que gastan al menos la mediana del resto de VIP (2.936,20 £). Son
66 clientes (1,5 %) que generan el 32,3 % de los ingresos: compran 26 veces
con un ticket de 1.078,19 £. Exigimos ese gasto mínimo porque DBSCAN marca rarezas en cualquier dirección y la
estrategia de gestor dedicado solo se justifica por volumen.

**23. ¿Cómo garantizan que la app clasifica a un cliente nuevo igual que en el entrenamiento?**
La app no reentrena: carga el mismo preprocesador (log1p, recorte p1–p99 con los límites del entrenamiento y
escalado) y el mismo K-Means. Para la marca de cuenta clave, un cliente nuevo es atípico si no tiene ningún punto núcleo
de DBSCAN a distancia ≤ eps, la misma regla con la que DBSCAN etiqueta a los puntos frontera. Las pruebas comprueban que
este proceso reproduce el segmento de los 4.324 clientes y exactamente las 66 cuentas clave, que los
valores extremos se recortan sin errores y que los inválidos dan un mensaje claro. La app se probó también en un entorno
con solo las dependencias de despliegue.

**24. ¿Cómo sabrían si las estrategias funcionan? ¿Por qué no pusieron metas numéricas?**
Cada estrategia tiene un KPI y su **línea base** actual calculada con los datos (por ejemplo, el porcentaje de VIP sin
compras en 90 días). La forma correcta de medir es aplicar la acción a una parte del segmento, dejar otra parte como
grupo de control y comparar el KPI contra la línea base. No pusimos metas ("+10 %") porque serían cifras inventadas.

**25. ¿Cuáles son las principales limitaciones?**
- Excluimos las compras sin cliente identificado.
- Con un solo año de datos no podemos separar del todo estacionalidad y crecimiento, ni confirmar que los compradores de
  temporada repitan cada Navidad.
- La segmentación es una foto fija y habría que recalcularla con datos nuevos.
- Las estrategias están sin probar.

Todo ello está declarado en el informe.
