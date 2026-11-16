"""Catálogo de segmentos: nombres, descripciones y estrategias de marketing.

Los nombres y textos se definieron DESPUÉS de perfilar los segmentos con datos
(notebooks/04_perfiles.ipynb). Los textos son cualitativos a propósito: las cifras
(tamaños, ingresos, líneas base de cada KPI) se calculan con el código y se guardan en
``models/segmentos.json``; nunca se escriben a mano aquí.

Cada estrategia indica una acción concreta, el momento de aplicarla, el KPI que la mide e
``indicador``: la clave de la métrica (``perfilamiento.indicadores_segmento``) cuyo valor
actual sirve de línea base del KPI.
"""

from __future__ import annotations

NOMBRE_CUENTAS_CLAVE: str = "Cuentas clave"

SEGMENTOS: dict[int, dict] = {
    0: {
        "nombre": "Dormidos",
        "descripcion": (
            "Compraron una vez (o muy pocas) hace meses, con pedidos pequeños y pocos productos. "
            "Son muchos clientes, pero aportan muy poco a los ingresos y casi no devuelven."
        ),
        "estrategias": [
            {
                "accion": "Reactivación de bajo coste: un único email automatizado con cupón de un solo uso y una "
                          "selección de los productos más vendidos.",
                "momento": "Octubre, al empezar la demanda de regalos de Navidad.",
                "kpi": "Tasa de reactivación (% que compra en los 60 días siguientes) y coste por cliente reactivado.",
                "indicador": "pct_compra_90d",
            },
            {
                "accion": "Higiene de la base: tras dos campañas sin respuesta, pasar a comunicación trimestral y "
                          "excluirlos de acciones con coste (descuentos, envíos de catálogo).",
                "momento": "Al cerrar la campaña de octubre–noviembre.",
                "kpi": "Coste de marketing por libra de ingresos del segmento.",
                "indicador": "pct_ingresos",
            },
        ],
    },
    1: {
        "nombre": "Grandes ocasionales",
        "descripcion": (
            "Hacen pedidos grandes (ticket alto y muchas unidades), pero casi siempre una sola vez, y no han "
            "vuelto en meses. Muchos de sus pedidos únicos son de septiembre–noviembre: un patrón de "
            "aprovisionamiento para Navidad."
        ),
        "estrategias": [
            {
                "accion": "Campaña de pre-temporada para quienes compraron en septiembre–noviembre: catálogo de "
                          "Navidad anticipado con condiciones de mayorista.",
                "momento": "Finales de agosto e inicio de septiembre, antes de su ventana habitual de compra.",
                "kpi": "% de estos clientes que vuelve a comprar entre septiembre y noviembre.",
                "indicador": "pct_una_compra_en_temporada",
            },
            {
                "accion": "Llamada comercial a los clientes de pedidos grandes puntuales (marcados como atípicos) "
                          "para proponer un acuerdo de reposición.",
                "momento": "Primer trimestre, fuera de la temporada alta.",
                "kpi": "% que hace una segunda compra en los 90 días siguientes al contacto.",
                "indicador": "pct_una_compra",
            },
            {
                "accion": "Encuesta breve sobre por qué no repitieron, con un incentivo por responder.",
                "momento": "60 días después de su compra si no han repetido.",
                "kpi": "Tasa de respuesta y % de encuestados que hace una segunda compra.",
                "indicador": "pct_una_compra",
            },
        ],
    },
    2: {
        "nombre": "En crecimiento",
        "descripcion": (
            "Clientes recientes que ya repiten (varias facturas) con pedidos de tamaño medio. Una parte "
            "entró en la campaña de Navidad. Su valor futuro depende de que sigan comprando."
        ),
        "estrategias": [
            {
                "accion": "Secuencia de recompra: email con productos complementarios a lo que ya compraron.",
                "momento": "21 días después de cada pedido (automático).",
                "kpi": "Tasa de recompra a 60 días.",
                "indicador": "pct_intervalo_60d",
            },
            {
                "accion": "Retener a los captados en Navidad: oferta para la primera compra del año.",
                "momento": "Enero–febrero, los meses de menor actividad.",
                "kpi": "% de los captados en octubre–diciembre que vuelve a comprar en enero–febrero.",
                "indicador": "pct_captados_temporada",
            },
            {
                "accion": "Subir el ticket: envío gratis a partir de un pedido mínimo por encima de su ticket actual "
                          "y packs de productos.",
                "momento": "Desde su siguiente pedido.",
                "kpi": "Ticket promedio del segmento.",
                "indicador": "ticket_mediano",
            },
        ],
    },
    3: {
        "nombre": "VIP",
        "descripcion": (
            "Los mejores clientes: compran a menudo, hace poco, mucha variedad y con pedidos grandes. Son una "
            "quinta parte de la base y generan la mayor parte de los ingresos. También son los que más devuelven."
        ),
        "estrategias": [
            {
                "accion": "Programa VIP: acceso anticipado a la colección de Navidad y descuento escalonado por "
                          "volumen anual.",
                "momento": "Agosto, antes del pico de pedidos de septiembre–noviembre.",
                "kpi": "% de VIP que compra en septiembre–noviembre.",
                "indicador": "pct_clientes_compran_temporada",
            },
            {
                "accion": "Alerta de abandono: si un VIP supera 1,5 veces su intervalo habitual entre compras sin "
                          "pedir, contacto personal del equipo comercial.",
                "momento": "Revisión semanal automática durante todo el año.",
                "kpi": "% de VIP sin compras en los últimos 90 días.",
                "indicador": "pct_sin_compra_90d",
            },
            {
                "accion": "Revisión de las causas de devolución con los VIP que más devuelven (producto, embalaje, "
                          "transporte) y gestión prioritaria de sus incidencias.",
                "momento": "Trimestral.",
                "kpi": "% de VIP con devoluciones y tasa media de devolución.",
                "indicador": "pct_devuelve",
            },
        ],
    },
}

CUENTAS_CLAVE: dict = {
    "nombre": NOMBRE_CUENTAS_CLAVE,
    "descripcion": (
        "Subgrupo de VIP que DBSCAN marca como atípico: volúmenes muy superiores al resto (muchas más "
        "facturas y tickets mucho mayores). Son pocos clientes que, por sí solos, sostienen una parte muy "
        "grande de la facturación: perder uno tiene un impacto directo."
    ),
    "estrategias": [
        {
            "accion": "Gestor de cuenta dedicado con condiciones negociadas: precios por volumen, plazos de pago y "
                      "reposición programada.",
            "momento": "Revisión trimestral; negociación anual en julio–agosto, antes de la temporada.",
            "kpi": "% de cuentas clave con compra en los últimos 90 días y % de los ingresos totales que aportan.",
            "indicador": "pct_compra_90d",
        },
        {
            "accion": "Planificación conjunta del pedido de Navidad con previsión de stock por cliente.",
            "momento": "Julio–agosto.",
            "kpi": "Ingresos del subgrupo en septiembre–noviembre frente al año anterior.",
            "indicador": "ingresos_temporada",
        },
    ],
}
