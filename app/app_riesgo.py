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
    ax.hist(puntajes, bins=bins, color="#9aa5b1", alpha=0.8)
    ax.hist(puntajes[puntajes <= puntaje_cliente], bins=bins, color="#2a6fdb", alpha=0.8)
    ax.axvline(puntaje_cliente, color="#c0392b", linewidth=2)
    ax.set(xlabel="Puntaje", yticks=[], xlim=(np.percentile(puntajes, 0.5) - 10, np.percentile(puntajes, 99.5) + 10))
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    return fig


def grafica_puntos(base, puntos):
    top = puntos.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([NOMBRES.get(v, v) for v in top.index], top.values,
            color=["#2a9d8f" if v > 0 else "#c0392b" for v in top.values])
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set(xlabel=f"Puntos frente al cliente promedio ({base:.0f} puntos)")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def enlace(texto, url):
    return f"[{texto}]({url})" if url else f"{texto} (enlace pendiente)"


def main():
    st.set_page_config(page_title="¿Me prestas? · Riesgo de crédito", page_icon="🏦", layout="wide")
    motor = cargar_motor()

    with st.sidebar:
        st.markdown("## 🏦 ¿Me prestas?")
        st.write("Estima la probabilidad de que un crédito de consumo no se pague, la convierte en un puntaje "
                 "y calcula cuánto podría perder el prestamista. Usa una red neuronal entrenada con créditos "
                 "de Lending Club originados entre 2012 y 2014.")
        st.markdown("### Enlaces")
        st.markdown(f"- 📝 {enlace('Reporte técnico', ENLACE_REPORTE)}\n"
                    f"- 🎥 {enlace('Video promocional', ENLACE_VIDEO)}\n"
                    f"- 💻 {enlace('Repositorio', ENLACE_REPOSITORIO)}")
        st.markdown("### Parámetros de la scorecard")
        puntaje_base = st.number_input("Puntaje base", min_value=300, max_value=900, value=600, step=10, key="sc_base")
        odds_base = st.number_input("Odds de referencia (buenos por cada malo)", min_value=1.0, max_value=500.0,
                                    value=50.0, step=5.0, key="sc_odds")
        pdo = st.number_input("Puntos para duplicar las odds (PDO)", min_value=5.0, max_value=100.0, value=20.0, step=5.0, key="sc_pdo")
        st.caption("Con odds de 50:1, un puntaje de 600 equivale a una probabilidad de incumplimiento de 1.96 %. "
                   "Cada vez que las odds de pagar se duplican, el puntaje sube PDO puntos.")
    sc = CreditScorecard(score_base=puntaje_base, odds_base=odds_base, pdo=pdo)

    st.title("¿Me prestas?")
    st.write("Ingresa los datos de la solicitud. El resultado se actualiza solo. Los valores iniciales son los de "
             "un solicitante típico.")
    defecto = motor.cliente_por_defecto()
    col_form, col_res = st.columns([1, 1.15], gap="large")

    cliente = {}
    with col_form:
        st.subheader("1. La solicitud")
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
        st.subheader("2. El resultado")
        a, b, c = st.columns(3)
        a.metric("Puntaje", r["puntaje"])
        b.metric("Probabilidad de incumplimiento", f"{r['pd']:.1%}")
        c.metric("Percentil", f"{r['percentil']:.0f}")
        st.caption(f"Este puntaje es mayor o igual que el del {r['percentil']:.0f} % de los créditos de prueba "
                   "(originados entre noviembre de 2014 y diciembre de 2015). En azul, los créditos con un puntaje "
                   "menor o igual; en rojo, esta solicitud.")
        st.pyplot(grafica_poblacion(motor.puntajes_poblacion(sc), r["puntaje"]), clear_figure=True)
        d, e, f = st.columns(3)
        d.metric("Pérdida si incumple (LGD)", f"{r['lgd']:.0%}")
        e.metric("Exposición estimada (EAD)", f"${r['ead']:,.0f}")
        f.metric("Pérdida esperada", f"${r['perdida_esperada']:,.0f}")
        st.caption("Pérdida esperada = probabilidad de incumplimiento × pérdida si incumple × exposición. "
                   "La exposición es el monto por la fracción que en promedio queda pendiente al incumplir, "
                   "según el plazo.")

    st.divider()
    st.subheader("3. ¿Por qué ese puntaje?")
    base, puntos = motor.puntos_por_variable(cliente, sc)
    st.write(f"Un cliente promedio tendría unos **{base:.0f} puntos**. Las barras muestran cuántos puntos suma "
             "(verde) o resta (rojo) cada característica de esta solicitud frente a ese cliente promedio.")
    g1, g2 = st.columns([1.3, 1])
    g1.pyplot(grafica_puntos(base, puntos), clear_figure=True)
    g2.caption("Los puntos se calculan con gradientes integrados (Sundararajan et al., 2017): se reparte entre "
               "las variables la diferencia entre el logit de esta solicitud y el del cliente promedio, y se "
               "multiplica por el Factor de la scorecard. Su suma da el puntaje antes de redondear.")

    st.divider()
    st.subheader("4. ¿Qué pasaría si cambia algo?")
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
