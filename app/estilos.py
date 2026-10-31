"""Estilo visual de la app: superficies pastel tipo arcilla, fondo que se mueve despacio y una moneda 3D
con el puntaje que gira y cuenta cada vez que cambia el resultado.

Paleta: lavanda #CDB4F6, menta #A8E6CF, durazno #FFD3B6, rosa #FF9AA2, cielo #B5D8F7; texto ciruela #3B2A4F.
"""
import streamlit as st

LAVANDA, MENTA, DURAZNO, ROSA, CIELO, CIRUELA = "#CDB4F6", "#A8E6CF", "#FFD3B6", "#FF9AA2", "#B5D8F7", "#3B2A4F"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,700&family=Fredoka:wght@500;600;700&display=swap');

:root {
  --lavanda: #CDB4F6; --lavanda-fuerte: #9B7FD9; --menta: #A8E6CF; --menta-fuerte: #4FB58E;
  --durazno: #FFD3B6; --durazno-fuerte: #F0A273; --rosa: #FF9AA2; --rosa-fuerte: #E36A78;
  --cielo: #B5D8F7; --ciruela: #3B2A4F; --ciruela-suave: #6E5C85; --fondo: #FBF8FF;
  --titulo: 'Fredoka', 'Nunito', 'Trebuchet MS', sans-serif;
  --texto: 'DM Sans', 'Helvetica Neue', Arial, sans-serif;
}

/* ---------- Fondo: manchas pastel que se mueven muy despacio ---------- */
.stApp {
  background: var(--fondo);
  font-family: var(--texto);
  color: var(--ciruela);
}
.stApp::before, .stApp::after {
  content: ""; position: fixed; z-index: 0; pointer-events: none;
  width: 46vw; height: 46vw; border-radius: 50%; filter: blur(70px); opacity: .55;
}
.stApp::before {
  background: radial-gradient(circle at 30% 30%, var(--lavanda), transparent 65%);
  top: -12vw; right: -8vw; animation: deriva-a 26s ease-in-out infinite alternate;
}
.stApp::after {
  background: radial-gradient(circle at 60% 60%, var(--menta), transparent 60%),
              radial-gradient(circle at 20% 80%, var(--durazno), transparent 55%);
  bottom: -16vw; left: -10vw; animation: deriva-b 32s ease-in-out infinite alternate;
}
@keyframes deriva-a { to { transform: translate(-8vw, 10vh) scale(1.12); } }
@keyframes deriva-b { to { transform: translate(10vw, -8vh) scale(1.08); } }
[data-testid="stAppViewContainer"] > .main, [data-testid="stMain"] { position: relative; z-index: 1; }
[data-testid="stHeader"] { background: transparent; }

/* ---------- Tipografía ---------- */
h1, h2, h3, h4 { font-family: var(--titulo) !important; color: var(--ciruela) !important; letter-spacing: .2px; }
p, li, label, .stMarkdown, [data-testid="stCaptionContainer"] { font-family: var(--texto); }
[data-testid="stCaptionContainer"] p { color: var(--ciruela-suave); }
a { color: var(--lavanda-fuerte) !important; }

/* ---------- Encabezado ---------- */
.mp-encabezado { margin: .2rem 0 1.6rem; max-width: 46rem; }
.mp-encabezado h1 {
  font-size: clamp(2.6rem, 6vw, 4.2rem); line-height: 1; margin: 0 0 .7rem; font-weight: 700;
  color: #fff !important;
  text-shadow: 0 1px 0 #E9DEFC, 0 2px 0 #DCCBFA, 0 3px 0 var(--lavanda), 0 4px 0 #BDA0EE,
               0 5px 0 #AE8FE6, 0 10px 18px rgba(155,127,217,.45);
  -webkit-text-stroke: 1.5px var(--lavanda-fuerte);
}
.mp-encabezado p { font-size: 1.08rem; line-height: 1.55; color: var(--ciruela-suave); margin: 0; }

/* ---------- Títulos de sección: el número es el paso del recorrido ---------- */
.mp-seccion { display: flex; align-items: center; gap: .75rem; margin: 1.6rem 0 .9rem; }
.mp-seccion .paso {
  flex: none; width: 2.3rem; height: 2.3rem; border-radius: 50%; display: grid; place-items: center;
  font-family: var(--titulo); font-weight: 600; color: var(--ciruela); background: var(--cielo);
  box-shadow: inset 0 -4px 0 rgba(59,42,79,.12), inset 0 3px 0 rgba(255,255,255,.7), 0 6px 14px rgba(110,160,220,.35);
}
.mp-seccion h3 { margin: 0 !important; padding: 0 !important; font-size: 1.45rem; font-weight: 600; }

/* ---------- Superficies tipo arcilla ---------- */
[data-testid="stExpander"] details {
  border: none !important; border-radius: 22px; background: rgba(255,255,255,.72);
  box-shadow: inset 0 2px 0 rgba(255,255,255,.9), inset 0 -3px 0 rgba(155,127,217,.12),
              0 12px 28px rgba(155,127,217,.18);
}
[data-testid="stExpander"] summary { font-family: var(--titulo); font-weight: 600; color: var(--ciruela); }
[data-baseweb="input"], [data-baseweb="select"] > div {
  border-radius: 14px !important; border-color: #E6DAFB !important; background: #FFFFFF !important;
}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {
  border-color: var(--lavanda-fuerte) !important; box-shadow: 0 0 0 3px rgba(155,127,217,.25) !important;
}

/* Tarjetas de cifras: inclinación 3D al pasar el cursor */
[data-testid="stMetric"] {
  background: rgba(255,255,255,.8); border-radius: 18px; padding: .9rem 1rem;
  box-shadow: inset 0 2px 0 #fff, inset 0 -4px 0 rgba(181,216,247,.45), 0 10px 22px rgba(110,160,220,.22);
  transform: perspective(700px) rotateX(0deg); transition: transform .35s cubic-bezier(.2,.8,.2,1), box-shadow .35s;
}
[data-testid="stMetric"]:hover {
  transform: perspective(700px) rotateX(8deg) translateY(-3px);
  box-shadow: inset 0 2px 0 #fff, inset 0 -4px 0 rgba(181,216,247,.45), 0 18px 30px rgba(110,160,220,.3);
}
[data-testid="stMetricLabel"] p { color: var(--ciruela-suave) !important; font-weight: 500; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important; line-height: 1.25;
}
/* La cifra se achica con el ancho de la tarjeta en vez de cortarse con "…" */
[data-testid="stMetric"] { container-type: inline-size; }
[data-testid="stMetricValue"] {
  font-family: var(--titulo); color: var(--ciruela);
  font-size: clamp(1.35rem, 16cqi, 2.25rem) !important; font-variant-numeric: tabular-nums;
}

/* ---------- Barra lateral ---------- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #F1E9FF 0%, #E8F4FE 100%);
  border-right: 1px solid rgba(205,180,246,.5);
}
[data-testid="stSidebar"] h2 { font-size: 1.9rem; }

/* ---------- Moneda 3D del puntaje ---------- */
@property --puntos { syntax: '<integer>'; initial-value: 0; inherits: false; }
.mp-resultado {
  display: grid; grid-template-columns: auto 1fr; gap: 1.6rem; align-items: center;
  padding: 1.4rem 1.6rem; border-radius: 28px; margin-bottom: 1rem;
  background: linear-gradient(140deg, rgba(255,255,255,.92), rgba(255,255,255,.6));
  box-shadow: inset 0 3px 0 #fff, inset 0 -5px 0 var(--tono-suave), 0 22px 40px var(--tono-sombra);
}
.mp-escena { perspective: 900px; }
.mp-moneda {
  width: 168px; height: 168px; border-radius: 50%; display: grid; place-items: center; position: relative;
  background: radial-gradient(circle at 32% 28%, #fff 0 8%, transparent 30%),
              radial-gradient(circle at 50% 50%, var(--tono) 0 58%, var(--tono-borde) 59% 64%, var(--tono) 65%);
  box-shadow: 0 2px 0 var(--tono-borde), 0 4px 0 var(--tono-borde), 0 6px 0 var(--tono-borde),
              0 8px 0 var(--tono-borde), 0 10px 0 var(--tono-borde), 0 26px 30px var(--tono-sombra),
              inset 0 -8px 14px rgba(59,42,79,.10), inset 0 8px 14px rgba(255,255,255,.7);
  transform-style: preserve-3d; animation: girar 1.1s cubic-bezier(.2,.9,.25,1.15) both;
}
.mp-moneda .numero {
  font-family: var(--titulo); font-weight: 700; font-size: 3.1rem; color: var(--ciruela); line-height: 1;
  counter-reset: puntos var(--puntos); animation: contar 1.1s cubic-bezier(.2,.8,.2,1) both;
  text-shadow: 0 2px 0 rgba(255,255,255,.8);
}
.mp-moneda .numero::after { content: counter(puntos); }
.mp-moneda .unidad {
  position: absolute; bottom: 30px; font-size: .78rem; color: var(--ciruela-suave); font-weight: 500;
}
@keyframes girar {
  0% { transform: rotateY(-200deg) translateY(-14px); }
  70% { transform: rotateY(12deg) translateY(2px); }
  100% { transform: rotateY(0deg) translateY(0); }
}
@keyframes contar { from { --puntos: var(--desde); } to { --puntos: var(--hasta); } }
.mp-detalle h4 { font-size: 1.35rem; margin: 0 0 .35rem; font-weight: 600; }
.mp-detalle p { margin: 0 0 .9rem; color: var(--ciruela-suave); line-height: 1.5; }
.mp-cifras { display: flex; gap: .7rem; flex-wrap: wrap; }
.mp-cifra {
  background: var(--tono-suave); border-radius: 14px; padding: .55rem .85rem; flex: 1 1 6.5rem;
  box-shadow: inset 0 2px 0 rgba(255,255,255,.75), inset 0 -3px 0 rgba(59,42,79,.08);
}
.mp-cifra span { display: block; font-size: .8rem; color: var(--ciruela-suave); }
.mp-cifra strong { font-family: var(--titulo); font-size: 1.35rem; font-weight: 600; color: var(--ciruela); }

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
    "bajo": ("#A8E6CF", "#7FCFAF", "rgba(79,181,142,.35)", "rgba(168,230,207,.45)"),
    "medio": ("#FFD3B6", "#F2B48C", "rgba(240,162,115,.35)", "rgba(255,211,182,.5)"),
    "alto": ("#FF9AA2", "#EE7C87", "rgba(227,106,120,.35)", "rgba(255,154,162,.4)"),
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
