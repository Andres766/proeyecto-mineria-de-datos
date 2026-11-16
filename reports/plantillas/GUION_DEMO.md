# Guion de la demostración (5–7 minutos)

Orden recomendado de las páginas, qué hacer en cada una y qué decir. Los tiempos son orientativos (total ≈ 6 min 30 s).
Antes de empezar, abre la app y visita una vez cada página para que los datos queden en caché y todo cargue al
instante.

## 0:00 – 0:45 · Inicio

**Mostrar:** los cuatro indicadores y las tres tarjetas de hallazgos clave.

**Decir:** "Analizamos {clientes} clientes de una tienda online británica de regalos. Con K-Means encontramos {k_final}
segmentos. El resultado en tres ideas:
- RFM no distingue a dos grupos de clientes ocasionales que K-Means sí separa, y que valen muy distinto:
  {s0_pct_ingresos} frente a {s1_pct_ingresos} de los ingresos.
- Hay {ck_clientes} cuentas clave, el {ck_pct_clientes} de los clientes, que generan el {ck_pct_ingresos} de los
  ingresos.
- Elegimos 4 segmentos aunque las métricas preferían 2, y lo vamos a justificar."

## 0:45 – 1:30 · Exploración

**Mostrar:** el gráfico de la limpieza y el de ingresos mensuales.

**Decir:** "Partimos de {filas_originales} líneas. La decisión clave de la limpieza fue emparejar cada cancelación con la
compra que anula: los mayores pedidos del dataset se cancelaron minutos después y, sin esto, contarían como ventas.
También se ve la estacionalidad: el pico es {mes_pico}. Diciembre de 2011 sale en gris porque solo llega hasta el día 9."

## 1:30 – 2:45 · Modelos y validación

**Mostrar:** la tabla comparativa (fila azul = modelo final), la pestaña K-Means (las cuatro métricas y la gráfica de
estabilidad) y, de pasada, la pestaña GMM.

**Decir:** "Comparamos K-Means, GMM y DBSCAN. Las métricas internas premian k = 2, pero bajan con cada k: los clientes
forman un continuo, y en ese caso siempre gana el k más pequeño. Por eso también medimos la estabilidad: repetimos el
modelo en 50 submuestras. k = 5 parecía mejor, pero en el peor caso cae a {k5_estabilidad_p10}; k = 4 se mantiene en
{final_estabilidad_p10}. GMM degeneraba por los empates en la frecuencia y lo corregimos con `reg_covar`, pero aun así
quedó peor. DBSCAN no encuentra grupos: lo usamos para detectar atípicos."

## 2:45 – 3:45 · Clusters

**Mostrar:** el mapa PCA 2D con K-Means y cuentas clave resaltadas. Pasa el cursor por un rombo para enseñar sus datos.
Cambia a PCA 3D y gíralo un momento. Después cambia a t-SNE y señala la advertencia.

**Decir:** "El eje horizontal es la actividad del cliente y el vertical el tamaño del pedido; juntos explican el
{pca_2} de la información. Los rombos son las cuentas clave, en el extremo de los VIP. t-SNE se ve bonito, pero solo sirve
para visualizar: no conserva distancias ni tamaños, y lo comprobamos con los datos."

## 3:45 – 5:00 · Perfiles y estrategias

**Mostrar:** las cuatro fichas, el radar, la pestaña de estrategias de VIP y el mapa de composición RFM.

**Decir:** "Los VIP son el {s3_pct_clientes} de los clientes y generan el {s3_pct_ingresos} de los ingresos.
{s1_nombre} tiene el ticket más alto pero compra una sola vez, y muchos de ellos lo hacen antes de Navidad, así que su
estrategia es una campaña de pre-temporada en agosto. Cada estrategia tiene acción, momento y KPI con su línea base
actual. En el mapa de RFM se ve el valor añadido: RFM confunde a los dos grupos de ocasionales."

## 5:00 – 6:15 · Explorador

**Mostrar y hacer:**
1. En «Buscar», escribe el cliente **14646**. Es una cuenta clave; muestra su comparación con la mediana.
2. Escribe **abc** para enseñar que la app valida la entrada.
3. En «Clasificar un cliente nuevo», introduce 5 días, 60 facturas, 1500 £ de ticket y 150 productos, y pulsa
   «Clasificar cliente».

**Decir:** "Este cliente no existía: la app aplica exactamente el mismo preprocesamiento del entrenamiento, incluido el
recorte de extremos (aquí avisa de que 60 facturas se acotó al máximo de entrenamiento), y lo sitúa como estrella en el
mapa. Sale VIP y cuenta clave, con sus estrategias. Las pruebas automáticas verifican que este formulario clasifica
igual que el modelo a los {clientes} clientes."

**Si hay tiempo:** clasifica 200 días, 1 factura, 90 £ y 6 productos, que sale en {s0_nombre}, para mostrar el
contraste.

## 6:15 – 6:45 · Cierre

**Decir:** "En resumen: cuatro segmentos estables y accionables, un subgrupo de cuentas clave que sostiene un tercio de
la facturación, y una herramienta que clasifica clientes nuevos sin reentrenar. El silhouette es moderado porque los
clientes forman un continuo, y por eso lo validamos con estabilidad y criterio de negocio."

## Plan B

Si la app publicada tarda en despertar (Streamlit Cloud duerme las apps sin uso), ábrela 5 minutos antes. Si no hay
conexión, ejecútala en local con `streamlit run app/Inicio.py` o con Docker (ver README).
