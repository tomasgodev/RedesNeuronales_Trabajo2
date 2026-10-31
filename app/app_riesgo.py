"""¿Me prestas? Aplicación web de riesgo de crédito con los modelos reales del Trabajo 2.

Para correrla localmente con un solo comando (desde la raíz del repositorio):
    streamlit run app/app_riesgo.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
from motor_riesgo import MotorRiesgo  # noqa: E402
from scorecard_engine import CreditScorecard  # noqa: E402
from estilos import CIRUELA, aplicar_estilos, encabezado, moneda_puntaje, titulo_seccion  # noqa: E402

# [PENDIENTE] Enlaces definitivos
ENLACE_REPORTE = ""
ENLACE_VIDEO = ""
ENLACE_REPOSITORIO = "https://github.com/tomasgodev/RedesNeuronales_Trabajo2"

NOMBRES = {
    "loan_amnt": "Monto solicitado (USD)", "term": "Plazo", "purpose": "Propósito del crédito",
    "sub_grade": "Subcalificación asignada", "int_rate": "Tasa de interés asignada (%)",
    "annual_inc": "Ingreso anual (USD)", "home_ownership": "Vivienda", "verification_status": "Verificación del ingreso",
    "addr_state": "Estado de residencia (EE. UU.)", "dti": "Relación deuda/ingreso (%)",
    "delinq_2yrs": "Moras de 30+ días en los últimos 2 años", "inq_last_6mths": "Consultas de crédito en los últimos 6 meses",
    "open_acc": "Líneas de crédito abiertas", "total_acc": "Líneas de crédito en total",
    "revol_bal": "Saldo en crédito rotativo (USD)", "revol_util": "Uso del cupo rotativo (%)",
    "collections_12_mths_ex_med": "Cobros en los últimos 12 meses (sin médicos)",
    "acc_now_delinq": "Cuentas en mora hoy", "tot_cur_bal": "Saldo total de todas sus cuentas (USD)",
}
AYUDAS = {
    "dti": "Pagos mensuales de deudas (sin hipoteca ni este crédito) divididos por el ingreso mensual.",
    "sub_grade": "La asigna la plataforma al evaluar la solicitud: A1 es la mejor y G5 la peor.",
    "int_rate": "La asigna la plataforma junto con la subcalificación.",
    "revol_util": "Qué parte del cupo de tarjetas y líneas rotativas está usando.",
}
ETIQUETAS = {
    "term": {" 36 months": "36 meses", " 60 months": "60 meses"},
    "home_ownership": {"MORTGAGE": "Hipotecada", "RENT": "Arrendada", "OWN": "Propia", "OTHER": "Otra",
                       "NONE": "Ninguna", "ANY": "Cualquiera"},
    "verification_status": {"Not Verified": "No verificado", "Source Verified": "Fuente verificada", "Verified": "Verificado"},
    "purpose": {"car": "Carro", "credit_card": "Tarjeta de crédito", "debt_consolidation": "Consolidar deudas",
                "educational": "Educación", "home_improvement": "Mejoras del hogar", "house": "Vivienda",
                "major_purchase": "Compra grande", "medical": "Gastos médicos", "moving": "Mudanza", "other": "Otro",
                "renewable_energy": "Energía renovable", "small_business": "Pequeño negocio",
                "vacation": "Vacaciones", "wedding": "Boda"},
}
GRUPOS_FORMULARIO = [
    ("El crédito", ["loan_amnt", "term", "purpose"]),
    ("Condiciones asignadas por la plataforma", ["sub_grade", "int_rate"]),
    ("El solicitante", ["annual_inc", "dti", "home_ownership", "verification_status", "addr_state"]),
    ("Historial de crédito", ["delinq_2yrs", "inq_last_6mths", "open_acc", "total_acc", "revol_bal", "revol_util",
                              "collections_12_mths_ex_med", "acc_now_delinq", "tot_cur_bal"]),
]


def etiqueta(variable, valor):
    return ETIQUETAS.get(variable, {}).get(valor, valor)


@st.cache_resource(show_spinner="Cargando los modelos…")
def cargar_motor():
    return MotorRiesgo()


def campo(motor, variable, valor, clave, contenedor):
    """Dibuja el campo de una variable y devuelve el valor elegido."""
    cats = motor.opciones["categoricas"]
    if variable in cats:
        opciones = cats[variable]["opciones"]
        return contenedor.selectbox(NOMBRES[variable], opciones, index=opciones.index(valor), key=clave,
                                    format_func=lambda v: etiqueta(variable, v), help=AYUDAS.get(variable))
    info = motor.opciones["numericas"][variable]
    if info["entero"]:
        return float(contenedor.number_input(NOMBRES[variable], min_value=0, value=int(valor), step=1, key=clave,
                                             help=AYUDAS.get(variable)))
    paso = 0.1 if info["max"] < 200 else 500.0
    return float(contenedor.number_input(NOMBRES[variable], min_value=0.0, value=float(valor), step=paso, key=clave,
                                         help=AYUDAS.get(variable)))


def grafica_poblacion(puntajes, puntaje_cliente):
    fig, ax = plt.subplots(figsize=(7, 2.6))
    bins = np.arange(puntajes.min() - 0.5, puntajes.max() + 1.5, 3)
    ax.hist(puntajes, bins=bins, color="#E7DDFB")
    ax.hist(puntajes[puntajes <= puntaje_cliente], bins=bins, color="#CDB4F6")
    ax.axvline(puntaje_cliente, color="#E36A78", linewidth=2.5)
    ax.set(xlabel="Puntaje", yticks=[], xlim=(np.percentile(puntajes, 0.5) - 10, np.percentile(puntajes, 99.5) + 10))
    return estilo_grafica(fig, ax)


def grafica_puntos(base, puntos):
    top = puntos.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([NOMBRES.get(v, v) for v in top.index], top.values, height=0.62,
            color=["#7FCFAF" if v > 0 else "#EE7C87" for v in top.values])
    ax.axvline(0, color=CIRUELA, linewidth=0.8)
    ax.set(xlabel=f"Puntos frente al cliente promedio ({base:.0f} puntos)")
    ax.grid(axis="x", color="#E7DDFB")
    return estilo_grafica(fig, ax)


def estilo_grafica(fig, ax):
    """Fondo transparente, texto ciruela y sin marcos, para que la gráfica se integre al diseño."""
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.set_axisbelow(True)
    for lado in ["top", "right", "left"]:
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color("#CDB4F6")
    ax.tick_params(colors="#6E5C85", length=0)
    ax.xaxis.label.set_color("#6E5C85")
    fig.tight_layout()
    return fig


def enlace(texto, url):
    return f"[{texto}]({url})" if url else f"{texto} (enlace pendiente)"


def moneda(valor):
    """Cifra en dólares que cabe en una tarjeta: exacta hasta 999,999 y en millones (M) desde un millón."""
    if abs(round(valor)) < 1e6:
        return f"${valor:,.0f}"
    decimales = 2 if abs(valor) < 1e7 else 1
    return f"${valor / 1e6:,.{decimales}f} M"


def main():
    st.set_page_config(page_title="¿Me prestas? · Riesgo de crédito", page_icon=":material/savings:", layout="wide")
    aplicar_estilos()
    motor = cargar_motor()

    with st.sidebar:
        st.markdown("## ¿Me prestas?")
        st.write("Estima la probabilidad de que un crédito de consumo no se pague, la convierte en un puntaje "
                 "y calcula cuánto podría perder el prestamista. Usa una red neuronal entrenada con créditos "
                 "de Lending Club originados entre 2012 y 2014.")
        st.markdown("### Enlaces")
        st.markdown(f"- {enlace('Reporte técnico', ENLACE_REPORTE)}\n"
                    f"- {enlace('Video promocional', ENLACE_VIDEO)}\n"
                    f"- {enlace('Repositorio', ENLACE_REPOSITORIO)}")
        st.markdown("### Parámetros de la scorecard")
        puntaje_base = st.number_input("Puntaje base", min_value=300, max_value=900, value=600, step=10, key="sc_base")
        odds_base = st.number_input("Odds de referencia (buenos por cada malo)", min_value=0.5, max_value=500.0,
                                    value=round(motor.odds_cartera, 3), step=0.5, format="%.3f", key="sc_odds")
        pdo = st.number_input("Puntos para duplicar las odds (PDO)", min_value=5.0, max_value=100.0, value=20.0, step=5.0, key="sc_pdo")
        st.caption(f"Por defecto, las odds de referencia son las de la cartera de entrenamiento "
                   f"({motor.odds_cartera:.2f} buenos por cada malo): un puntaje igual al puntaje base es el de un "
                   f"cliente con el riesgo promedio. Cada vez que las odds de pagar se duplican, el puntaje sube PDO puntos.")
    sc = CreditScorecard(score_base=puntaje_base, odds_base=odds_base, pdo=pdo)

    encabezado("¿Me prestas?",
               "Calcula qué tan probable es que un crédito de consumo no se pague, cuántos puntos saca en la "
               "scorecard y cuánto podría perder quien presta. Cambia los datos de la solicitud y el resultado se "
               "actualiza al instante; los valores iniciales son los de un solicitante típico.")
    defecto = motor.cliente_por_defecto()
    col_form, col_res = st.columns([1, 1.15], gap="large")

    cliente = {}
    with col_form:
        titulo_seccion(1, "La solicitud")
        for titulo, variables in GRUPOS_FORMULARIO:
            contenedor = st.expander(titulo, expanded=titulo != "Historial de crédito")
            columnas = contenedor.columns(2)
            for i, v in enumerate(variables):
                cliente[v] = campo(motor, v, defecto[v], f"f_{v}", columnas[i % 2])
        fuera = [NOMBRES[v] for v, info in motor.opciones["numericas"].items()
                 if not info["min"] <= cliente[v] <= info["max"]]
        if fuera:
            st.warning("Estos valores están fuera del rango típico de los datos de entrenamiento (percentiles 1 a 99): "
                       + ", ".join(fuera) + ". La estimación es menos confiable.")

    r = motor.evaluar(cliente, sc)
    with col_res:
        titulo_seccion(2, "El resultado")
        moneda_puntaje(r["puntaje"], r["pd"], r["percentil"])
        st.caption("Créditos de prueba (originados entre noviembre de 2014 y diciembre de 2015): en lavanda, los que "
                   "tienen un puntaje menor o igual; la línea rosa es esta solicitud.")
        st.pyplot(grafica_poblacion(motor.puntajes_poblacion(sc), r["puntaje"]), clear_figure=True)
        d, e, f = st.columns(3)
        d.metric("Pérdida si incumple (LGD)", f"{r['lgd']:.0%}")
        e.metric("Exposición estimada (EAD)", moneda(r["ead"]))
        f.metric("Pérdida esperada", moneda(r["perdida_esperada"]))
        # La pérdida esperada nunca supera la exposición, así que basta revisar la exposición.
        # Los $ van escapados: en Markdown, dos $ en la misma línea se interpretan como una fórmula.
        exactos = (f" Valores exactos (M = millones): exposición de \\${r['ead']:,.0f} y pérdida esperada de "
                   f"\\${r['perdida_esperada']:,.0f}." if abs(round(r["ead"])) >= 1e6 else "")
        st.caption("Pérdida esperada = probabilidad de incumplimiento × pérdida si incumple × exposición. "
                   "La exposición es el monto por la fracción que en promedio queda pendiente al incumplir, "
                   "según el plazo." + exactos)

    st.divider()
    titulo_seccion(3, "¿Por qué ese puntaje?")
    base, puntos = motor.puntos_por_variable(cliente, sc)
    st.write(f"Un cliente promedio tendría unos **{base:.0f} puntos**. Las barras muestran cuántos puntos suma "
             "(menta) o resta (rosa) cada característica de esta solicitud frente a ese cliente promedio.")
    g1, g2 = st.columns([1.3, 1])
    g1.pyplot(grafica_puntos(base, puntos), clear_figure=True)
    g2.caption("Los puntos se calculan con gradientes integrados (Sundararajan et al., 2017): se reparte entre "
               "las variables la diferencia entre el logit de esta solicitud y el del cliente promedio, y se "
               "multiplica por el Factor de la scorecard. Su suma da el puntaje antes de redondear.")

    st.divider()
    titulo_seccion(4, "¿Qué pasaría si cambia algo?")
    w1, w2 = st.columns([1, 1.3], gap="large")
    with w1:
        variable = st.selectbox("Característica a cambiar", list(NOMBRES), format_func=NOMBRES.get, key="wi_variable")
        nuevo = campo(motor, variable, cliente[variable], f"wi_{variable}", st)
    r2 = motor.evaluar(cliente | {variable: nuevo}, sc)
    with w2:
        a, b, c = st.columns(3)
        a.metric("Nuevo puntaje", r2["puntaje"], delta=r2["puntaje"] - r["puntaje"])
        b.metric("Nueva probabilidad", f"{r2['pd']:.1%}", delta=f"{(r2['pd'] - r['pd']) * 100:+.1f} pp", delta_color="inverse")
        c.metric("Nuevo percentil", f"{r2['percentil']:.0f}", delta=f"{r2['percentil'] - r['percentil']:+.0f}")

    escenarios = [
        ("Pedirlo a 36 meses", {"term": " 36 months"}),
        ("Pedir 20 % menos de monto", {"loan_amnt": cliente["loan_amnt"] * 0.8}),
        ("Bajar la relación deuda/ingreso 5 puntos", {"dti": max(0.0, cliente["dti"] - 5)}),
        ("Ganar 20 % más", {"annual_inc": cliente["annual_inc"] * 1.2}),
        ("Usar 20 puntos menos del cupo rotativo", {"revol_util": max(0.0, cliente["revol_util"] - 20)}),
    ]
    filas = []
    for texto, cambio in escenarios:
        if all(cliente[k] == v for k, v in cambio.items()):
            continue
        rc = motor.evaluar(cliente | cambio, sc)
        filas.append({"Cambio": texto, "Puntaje": rc["puntaje"], "Diferencia": rc["puntaje"] - r["puntaje"],
                      "Probabilidad": f"{rc['pd']:.1%}"})
    st.markdown("**Algunos cambios que dependen del solicitante**")
    st.dataframe(filas, hide_index=True, width="stretch")

    st.divider()
    st.caption("Trabajo 2 · Redes neuronales y algoritmos bioinspirados · Universidad Nacional de Colombia, "
               "Facultad de Minas · 2026-02. Herramienta académica: no es una oferta ni una decisión de crédito. "
               "Modelos: red neuronal de clasificación (AUC 0.70 en prueba) y red de LGD, entrenadas con datos "
               "públicos de Lending Club.")


main()
