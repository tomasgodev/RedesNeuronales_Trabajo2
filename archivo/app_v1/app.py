"""
app/app.py
==========
Aplicación Web Interactiva de Riesgo Crediticio y Pérdida Esperada.
Curso: Redes Neuronales y Algoritmos Bioinspirados (2026-02)
Universidad Nacional de Colombia - Facultad de Minas

Para ejecutar localmente con un solo comando:
    streamlit run app/app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Importar motor local
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_app import ScoringEngine

# Configuración de página
st.set_page_config(
    page_title="Credit Scoring & Expected Loss Engine | Lending Club",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 18px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .score-badge {
        font-size: 2.2rem;
        font-weight: 800;
        color: #047857;
    }
</style>
""", unsafe_allow_html=True)

# Título y encabezado
st.markdown('<div class="main-title">🏦 Credit Scoring & Expected Loss Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Herramienta analítica de evaluación de riesgo crediticio de consumo '
    'basada en Redes Neuronales y Modelos Lineales sobre datos de <b>Lending Club</b>.</div>',
    unsafe_allow_html=True
)

# Barra lateral con documentación y parámetros de la Scorecard
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/Escudo_UNAL.svg/300px-Escudo_UNAL.svg.png", width=90)
    st.markdown("### 🎓 Proyecto Académico")
    st.markdown("**Universidad Nacional de Colombia**  \nFacultad de Minas  \nCurso: Redes Neuronales")
    
    st.markdown("---")
    st.markdown("### 🔗 Enlaces Oficiales")
    st.markdown("""
    - 📝 [Reporte Técnico (Blog Post)](https://github.com/)
    - 💻 [Repositorio GitHub](https://github.com/)
    - 🎥 [Video Promocional (Pitch)](https://youtube.com/)
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Calibración de Scorecard")
    st.caption("Ajusta los parámetros regulatorios de la transformación afín:")
    score_base = st.number_input("Puntaje Base (Score₀)", min_value=300, max_value=850, value=600, step=10)
    odds_base = st.number_input("Odds Base (Buenos : Malos)", min_value=1.0, max_value=200.0, value=50.0, step=5.0)
    pdo = st.number_input("PDO (Puntos para duplicar odds)", min_value=5.0, max_value=100.0, value=20.0, step=5.0)

# Inicializar motor
engine = ScoringEngine(score_base=score_base, odds_base=odds_base, pdo=pdo)

# Layout principal: 2 columnas
col_izq, col_der = st.columns([1.1, 1.2], gap="large")

with col_izq:
    st.markdown("### 📋 Formulario de Solicitud de Crédito")
    st.caption("Ingresa la información financiera disponible en el momento de la originación (t₀):")
    
    c1, c2 = st.columns(2)
    with c1:
        loan_amnt = st.number_input("Monto Solicitado (USD)", min_value=1000, max_value=40000, value=15000, step=500)
        int_rate = st.slider("Tasa de Interés (%)", min_value=5.0, max_value=30.0, value=12.5, step=0.1)
        term = st.selectbox("Plazo del Crédito", ["36 meses", "60 meses"])
    with c2:
        annual_inc = st.number_input("Ingreso Anual Declarado (USD)", min_value=10000, max_value=400000, value=75000, step=2500)
        dti = st.slider("Relación Deuda/Ingreso (DTI %)", min_value=0.0, max_value=50.0, value=16.5, step=0.5)
        emp_length = st.selectbox(
            "Antigüedad Laboral",
            ["< 1 año", "1 año", "2 años", "3 años", "4 años", "5 años", "6 años", "7 años", "8 años", "9 años", "10+ años"],
            index=5
        )

    home_ownership = st.radio("Tenencia de Vivienda", ["MORTGAGE (Hipoteca)", "RENT (Arrendada)", "OWN (Propia)"], horizontal=True)
    home_clean = "RENT" if "RENT" in home_ownership else ("OWN" if "OWN" in home_ownership else "MORTGAGE")

    term_num = 60 if "60" in term else 36
    emp_years = 10 if "10+" in emp_length else (0 if "<" in emp_length else int(emp_length.split()[0]))

# Cálculos centrales
pd_cliente = engine.calcular_pd(int_rate, dti, annual_inc, term_num, emp_years, home_clean)
score_cliente = engine.pd_a_score(pd_cliente)
lgd_cliente = engine.calcular_lgd(int_rate, annual_inc, term_num)
ead_cliente = engine.calcular_ead(loan_amnt, term_num)
el_cliente = engine.calcular_perdida_esperada(pd_cliente, lgd_cliente, ead_cliente)
percentil_cliente = engine.calcular_percentil(score_cliente)

with col_der:
    st.markdown("### 🎯 Resultados de la Evaluación de Riesgo")
    
    # Tarjeta de Scorecard
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.95rem; color: #64748B; font-weight: 600;">PUNTAJE DE SCORECARD</div>
        <div class="score-badge">{score_cliente} <span style="font-size: 1.2rem; color: #64748B; font-weight: 500;">/ 850 pts</span></div>
        <div style="margin-top: 5px; font-size: 0.95rem;">
            Probabilidad de Incumplimiento (PD): <b>{pd_cliente * 100:.2f} %</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    m1, m2, m3 = st.columns(3)
    m1.metric("LGD Estimada", f"{lgd_cliente * 100:.1f} %", help="Pérdida dado incumplimiento estimada por el MLP")
    m2.metric("EAD en Originación", f"${ead_cliente:,.0f}", help="Exposición al incumplimiento esperada según amortización")
    m3.metric("Pérdida Esperada E[L]", f"${el_cliente:,.2f}", help="E[L] = PD * LGD * EAD")

    # Gráfica de distribución poblacional
    st.markdown("#### 📊 Ubicación Frente a la Población de Créditos")
    fig, ax = plt.subplots(figsize=(7, 2.5), facecolor="#F8FAFC")
    x_axis = np.linspace(300, 850, 300)
    y_densidad = np.exp(-0.5 * ((x_axis - 625) / 68)**2) / (68 * np.sqrt(2 * np.pi))
    
    ax.plot(x_axis, y_densidad, color="#1E3A8A", lw=2, label="Distribución Histórica")
    ax.fill_between(x_axis, y_densidad, where=(x_axis <= score_cliente), color="#3B82F6", alpha=0.35, label=f"Percentil {percentil_cliente}%")
    ax.axvline(score_cliente, color="#DC2626", lw=2.5, linestyle="--", label=f"Cliente ({score_cliente} pts)")
    
    ax.set_xlim(300, 850)
    ax.set_yticks([])
    ax.set_xlabel("Puntaje en Scorecard (pts)", fontsize=9)
    ax.legend(loc="upper left", frameon=True, fontsize=8)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    st.pyplot(fig)
    
    st.info(f"💡 **Posición Competitiva:** El cliente se ubica en el **percentil {percentil_cliente}%** de solvencia crediticia. "
            f"Supera en perfil de riesgo al {percentil_cliente}% de los solicitantes históricos de la plataforma.")

# Sección de Análisis What-If (Simulación de Sensibilidad)
st.markdown("---")
st.markdown("### 🔬 Simulador de Sensibilidad (*What-If Analysis*)")
st.caption("Comprueba cómo cambiaría el puntaje y la probabilidad de incumplimiento si el solicitante mejora sus condiciones:")

w1, w2, w3 = st.columns(3)
with w1:
    sim_inc = st.slider("Simular Nuevo Ingreso Anual (USD)", 10000, 250000, int(annual_inc), 5000)
with w2:
    sim_dti = st.slider("Simular Nuevo DTI (%)", 0.0, 50.0, float(dti), 1.0)
with w3:
    sim_tasa = st.slider("Simular Nueva Tasa de Interés (%)", 5.0, 30.0, float(int_rate), 0.5)

pd_sim = engine.calcular_pd(sim_tasa, sim_dti, sim_inc, term_num, emp_years, home_clean)
score_sim = engine.pd_a_score(pd_sim)
delta_score = score_sim - score_cliente
delta_pd = (pd_sim - pd_cliente) * 100.0

st.write("")
res_w1, res_w2, res_w3 = st.columns(3)
res_w1.metric("Nuevo Score Simulado", f"{score_sim} pts", delta=f"{delta_score:+d} pts")
res_w2.metric("Nueva Probabilidad (PD)", f"{pd_sim * 100:.2f} %", delta=f"{delta_pd:+.2f} %", delta_color="inverse")
res_w3.metric("Nuevo Percentil Poblacional", f"{engine.calcular_percentil(score_sim)} %", delta=f"{engine.calcular_percentil(score_sim) - percentil_cliente:+d} %")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #9CA3AF; font-size: 0.85rem;">
    Trabajo 2: Redes Neuronales para Datos Tabulares | Universidad Nacional de Colombia — Sede Medellín | Semestre 2026-02
</div>
""", unsafe_allow_html=True)


"""
cd RedesNeuronales_Trabajo2
ls
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

pip install streamlit torch scikit-learn pandas numpy matplotlib

streamlit run app/app.py
python -m streamlit run app/app.py
"""