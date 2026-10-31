"""Estilo visual de la app: superficies claras tipo arcilla, fondo que se mueve despacio y una moneda 3D
con el puntaje que gira y cuenta cada vez que cambia el resultado.

Paleta "tinta de billete": verde de billete sobre papel gris verdoso, y una escala de riesgo verde (bajo),
ámbar (medio) y ladrillo (alto) que solo usan la moneda del puntaje, sus cifras y la gráfica de puntos.
"""
import streamlit as st

TINTA, TINTA_SUAVE, VERDE = "#16302B", "#566A63", "#1F6A52"
SALVIA, SALVIA_CLARA, PAPEL = "#8FB5A3", "#DCE5DF", "#F3F5F2"
RIESGO_BAJO, RIESGO_MEDIO, RIESGO_ALTO = "#3E8E6A", "#B27A1C", "#C4513F"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,700&family=Fredoka:wght@500;600;700&display=swap');

:root {
  --tinta: #16302B; --tinta-suave: #566A63; --verde: #1F6A52; --salvia: #8FB5A3; --salvia-clara: #DCE5DF;
  --verde-tenue: #E3EEE8; --oro-tenue: #EFE5CC; --borde: #D3DED8; --fondo: #F3F5F2;
  --sombra: rgba(22,48,43,.08);
  --titulo: 'Fredoka', 'Nunito', 'Trebuchet MS', sans-serif;
  --texto: 'DM Sans', 'Helvetica Neue', Arial, sans-serif;
}

/* ---------- Fondo: manchas tenues que se mueven muy despacio ---------- */
.stApp {
  background: var(--fondo);
  font-family: var(--texto);
  color: var(--tinta);
}
.stApp::before, .stApp::after {
  content: ""; position: fixed; z-index: 0; pointer-events: none;
  width: 46vw; height: 46vw; border-radius: 50%; filter: blur(70px); opacity: .5;
}
.stApp::before {
  background: radial-gradient(circle at 30% 30%, var(--salvia-clara), transparent 65%);
  top: -12vw; right: -8vw; animation: deriva-a 26s ease-in-out infinite alternate;
}
.stApp::after {
  background: radial-gradient(circle at 60% 60%, var(--verde-tenue), transparent 60%),
              radial-gradient(circle at 20% 80%, var(--oro-tenue), transparent 55%);
  bottom: -16vw; left: -10vw; animation: deriva-b 32s ease-in-out infinite alternate;
}
@keyframes deriva-a { to { transform: translate(-8vw, 10vh) scale(1.12); } }
@keyframes deriva-b { to { transform: translate(10vw, -8vh) scale(1.08); } }
[data-testid="stAppViewContainer"] > .main, [data-testid="stMain"] { position: relative; z-index: 1; }
[data-testid="stHeader"] { background: transparent; }

/* ---------- Tipografía ---------- */
h1, h2, h3, h4 { font-family: var(--titulo) !important; color: var(--tinta) !important; letter-spacing: .2px; }
p, li, label, .stMarkdown, [data-testid="stCaptionContainer"] { font-family: var(--texto); }
[data-testid="stCaptionContainer"] p { color: var(--tinta-suave); }
a { color: var(--verde) !important; }

/* ---------- Encabezado ---------- */
.mp-encabezado { margin: .2rem 0 1.6rem; max-width: 46rem; }
.mp-encabezado h1 {
  font-size: clamp(2.6rem, 6vw, 4.2rem); line-height: 1; margin: 0 0 .7rem; font-weight: 700;
  color: var(--verde) !important;
  text-shadow: 0 3px 0 var(--salvia-clara), 0 12px 24px rgba(31,106,82,.16);
}
.mp-encabezado p { font-size: 1.08rem; line-height: 1.55; color: var(--tinta-suave); margin: 0; }

/* ---------- Títulos de sección: el número es el paso del recorrido ---------- */
.mp-seccion { display: flex; align-items: center; gap: .75rem; margin: 1.6rem 0 .9rem; }
.mp-seccion .paso {
  flex: none; width: 2.3rem; height: 2.3rem; border-radius: 50%; display: grid; place-items: center;
  font-family: var(--titulo); font-weight: 600; color: #fff; background: var(--verde);
  box-shadow: inset 0 -3px 0 rgba(0,0,0,.18), inset 0 2px 0 rgba(255,255,255,.2), 0 6px 14px rgba(31,106,82,.25);
}
.mp-seccion h3 { margin: 0 !important; padding: 0 !important; font-size: 1.45rem; font-weight: 600; }

/* ---------- Superficies tipo arcilla ---------- */
[data-testid="stExpander"] details {
  border: none !important; border-radius: 22px; background: rgba(255,255,255,.88);
  box-shadow: inset 0 2px 0 #fff, inset 0 -3px 0 rgba(31,106,82,.07), 0 12px 28px var(--sombra);
}
[data-testid="stExpander"] summary { font-family: var(--titulo); font-weight: 600; color: var(--tinta); }
[data-baseweb="input"], [data-baseweb="select"] > div {
  border-radius: 14px !important; border-color: var(--borde) !important; background: #FFFFFF !important;
}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {
  border-color: var(--verde) !important; box-shadow: 0 0 0 3px rgba(31,106,82,.2) !important;
}

/* Tarjetas de cifras: inclinación 3D al pasar el cursor */
[data-testid="stMetric"] {
  background: rgba(255,255,255,.9); border-radius: 18px; padding: .9rem 1rem;
  box-shadow: inset 0 2px 0 #fff, inset 0 -4px 0 var(--salvia-clara), 0 10px 22px var(--sombra);
  transform: perspective(700px) rotateX(0deg); transition: transform .35s cubic-bezier(.2,.8,.2,1), box-shadow .35s;
}
[data-testid="stMetric"]:hover {
  transform: perspective(700px) rotateX(8deg) translateY(-3px);
  box-shadow: inset 0 2px 0 #fff, inset 0 -4px 0 var(--salvia-clara), 0 18px 30px rgba(22,48,43,.14);
}
[data-testid="stMetricLabel"] p { color: var(--tinta-suave) !important; font-weight: 500; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important; line-height: 1.25;
}
/* La cifra se achica con el ancho de la tarjeta en vez de cortarse con "…" */
[data-testid="stMetric"] { container-type: inline-size; }
[data-testid="stMetricValue"] {
  font-family: var(--titulo); color: var(--tinta);
  font-size: clamp(1.35rem, 16cqi, 2.25rem) !important; font-variant-numeric: tabular-nums;
}

/* ---------- Barra lateral ---------- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #E7EFEA 0%, var(--fondo) 100%);
  border-right: 1px solid var(--borde);
}
[data-testid="stSidebar"] h2 { font-size: 1.9rem; }

/* ---------- Moneda 3D del puntaje ---------- */
@property --puntos { syntax: '<integer>'; initial-value: 0; inherits: false; }
.mp-resultado {
  display: grid; grid-template-columns: auto 1fr; gap: 1.6rem; align-items: center;
  padding: 1.4rem 1.6rem; border-radius: 28px; margin-bottom: 1rem;
  background: linear-gradient(140deg, rgba(255,255,255,.92), rgba(255,255,255,.6));
  box-shadow: inset 0 3px 0 #fff, inset 0 -5px 0 var(--tono-suave), 0 18px 36px rgba(22,48,43,.10);
}
.mp-escena { perspective: 900px; }
.mp-moneda {
  width: 168px; height: 168px; border-radius: 50%; display: grid; place-items: center; position: relative;
  background: radial-gradient(circle at 32% 28%, rgba(255,255,255,.5) 0 8%, transparent 30%),
              radial-gradient(circle at 50% 50%, var(--tono) 0 58%, var(--tono-borde) 59% 64%, var(--tono) 65%);
  box-shadow: 0 2px 0 var(--tono-borde), 0 4px 0 var(--tono-borde), 0 6px 0 var(--tono-borde),
              0 8px 0 var(--tono-borde), 0 10px 0 var(--tono-borde), 0 26px 30px var(--tono-sombra),
              inset 0 -8px 14px rgba(0,0,0,.16), inset 0 8px 14px rgba(255,255,255,.28);
  transform-style: preserve-3d; animation: girar 1.1s cubic-bezier(.2,.9,.25,1.15) both;
}
.mp-moneda .numero {
  font-family: var(--titulo); font-weight: 700; font-size: 3.1rem; color: #fff; line-height: 1;
  counter-reset: puntos var(--puntos); animation: contar 1.1s cubic-bezier(.2,.8,.2,1) both;
  text-shadow: 0 2px 0 rgba(0,0,0,.2);
}
.mp-moneda .numero::after { content: counter(puntos); }
.mp-moneda .unidad {
  position: absolute; bottom: 30px; font-size: .78rem; color: rgba(255,255,255,.88); font-weight: 500;
}
@keyframes girar {
  0% { transform: rotateY(-200deg) translateY(-14px); }
  70% { transform: rotateY(12deg) translateY(2px); }
  100% { transform: rotateY(0deg) translateY(0); }
}
@keyframes contar { from { --puntos: var(--desde); } to { --puntos: var(--hasta); } }
.mp-detalle h4 { font-size: 1.35rem; margin: 0 0 .35rem; font-weight: 600; }
.mp-detalle p { margin: 0 0 .9rem; color: var(--tinta-suave); line-height: 1.5; }
.mp-cifras { display: flex; gap: .7rem; flex-wrap: wrap; }
.mp-cifra {
  background: var(--tono-suave); border-radius: 14px; padding: .55rem .85rem; flex: 1 1 6.5rem;
  box-shadow: inset 0 2px 0 rgba(255,255,255,.75), inset 0 -3px 0 rgba(22,48,43,.08);
}
.mp-cifra span { display: block; font-size: .8rem; color: var(--tinta-suave); }
.mp-cifra strong { font-family: var(--titulo); font-size: 1.35rem; font-weight: 600; color: var(--tinta); }

@media (max-width: 720px) {
  .mp-resultado { grid-template-columns: 1fr; justify-items: center; text-align: center; }
  .mp-cifras { justify-content: center; }
}
@media (prefers-reduced-motion: reduce) {
  .stApp::before, .stApp::after, .mp-moneda, .mp-moneda .numero { animation: none !important; }
  [data-testid="stMetric"], [data-testid="stMetric"]:hover { transition: none; transform: none; }
}
</style>
"""

TONOS = {  # (color de la cara, borde, sombra, fondo suave)
    "bajo": (RIESGO_BAJO, "#2A6B4F", "rgba(42,107,79,.35)", "#E2EFE7"),
    "medio": (RIESGO_MEDIO, "#87590F", "rgba(135,89,15,.32)", "#F5E9D2"),
    "alto": (RIESGO_ALTO, "#94382A", "rgba(148,56,42,.32)", "#F6E0DB"),
}


def aplicar_estilos():
    st.markdown(CSS, unsafe_allow_html=True)


def encabezado(titulo, texto):
    st.markdown(f'<div class="mp-encabezado"><h1>{titulo}</h1><p>{texto}</p></div>', unsafe_allow_html=True)


def titulo_seccion(paso, texto):
    st.markdown(f'<div class="mp-seccion"><span class="paso">{paso}</span><h3>{texto}</h3></div>',
                unsafe_allow_html=True)


def moneda_puntaje(puntaje, pd_cliente, percentil):
    """Moneda 3D con el puntaje; el color depende de cómo se ubica frente a la población."""
    nivel = "bajo" if percentil >= 66 else ("medio" if percentil >= 33 else "alto")
    tono, borde, sombra, suave = TONOS[nivel]
    titular = {
        "bajo": "Riesgo bajo frente a la cartera",
        "medio": "Riesgo intermedio frente a la cartera",
        "alto": "Riesgo alto frente a la cartera",
    }[nivel]
    desde = max(0, puntaje - 120)
    st.markdown(f"""
<div class="mp-resultado" style="--tono:{tono};--tono-borde:{borde};--tono-sombra:{sombra};--tono-suave:{suave};">
  <div class="mp-escena">
    <div class="mp-moneda" role="img" aria-label="Puntaje: {puntaje}">
      <span class="numero" style="--desde:{desde};--hasta:{puntaje};--puntos:{puntaje};"></span>
      <span class="unidad">puntos</span>
    </div>
  </div>
  <div class="mp-detalle">
    <h4>{titular}</h4>
    <p>Este puntaje es mayor o igual que el del {percentil:.0f} % de los créditos de prueba.</p>
    <div class="mp-cifras">
      <div class="mp-cifra"><span>Probabilidad de no pagar</span><strong>{pd_cliente:.1%}</strong></div>
      <div class="mp-cifra"><span>Percentil</span><strong>{percentil:.0f}</strong></div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)
