"""Pruebas de robustez: cada comando con parámetros cambiados, extremos e inválidos.

Uso:
    python scripts/probar_robustez.py            # todos los casos (≈ 10 min)
    python scripts/probar_robustez.py demo       # solo los casos cuyo nombre contiene "demo"

Un caso PASA si:
  - se espera que funcione ("ok"): termina sin error y sin traceback;
  - se espera que se rechace ("rechazo"): termina con un mensaje de error claro, sin traceback de Python.
Cualquier traceback cuenta como CAÍDA.

Corre sobre una copia temporal del repositorio, así que no modifica resultados ni figuras.
"""
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PY = sys.executable
D, P1, P1B, P2, EL = ("scripts/demo_sustentacion.py", "scripts/parte1_evaluacion.py", "scripts/parte1_busqueda.py",
                      "scripts/parte2_lgd.py", "scripts/perdida_esperada.py")
R = ["--muestra", "20000", "--epocas", "3"]   # demo corto cuando el caso no prueba esos parámetros
R2 = ["--epocas", "5"]

CASOS = [
    # ---------------------------------------------------------------- demo de la sustentación
    ("demo: por defecto", D, [], "ok"),
    ("demo: lr 0.5", D, ["--lr", "0.5"] + R, "ok"),
    ("demo: lr 5 (muy alta)", D, ["--lr", "5"] + R, "ok"),
    ("demo: lr 1e-6", D, ["--lr", "1e-6"] + R, "ok"),
    ("demo: lr 0", D, ["--lr", "0"] + R, "ok"),
    ("demo: dropout 0", D, ["--dropout", "0"] + R, "ok"),
    ("demo: dropout 0.95", D, ["--dropout", "0.95"] + R, "ok"),
    ("demo: dropout 1", D, ["--dropout", "1"] + R, "ok"),
    ("demo: una neurona", D, ["--capas", "1"] + R, "ok"),
    ("demo: red grande", D, ["--capas", "256", "128", "64", "32"] + R, "ok"),
    ("demo: tanh", D, ["--activacion", "tanh"] + R, "ok"),
    ("demo: leaky_relu con capas", D, ["--activacion", "leaky_relu", "--capas", "32"] + R, "ok"),
    ("demo: último lote de 1 crédito", D, ["--muestra", "20001", "--batch_size", "1000", "--epocas", "2"], "ok"),
    ("demo: lote más grande que los datos", D, ["--batch_size", "500000"] + R, "ok"),
    ("demo: lote de 2", D, ["--batch_size", "2", "--muestra", "2000", "--epocas", "1"], "ok"),
    ("demo: 1 época", D, ["--epocas", "1", "--muestra", "20000"], "ok"),
    ("demo: paciencia 0", D, ["--paciencia", "0"] + R, "ok"),
    ("demo: semilla 0", D, ["--semilla", "0"] + R, "ok"),
    ("demo: semilla grande", D, ["--semilla", "987654321"] + R, "ok"),
    ("demo: muestra de 500", D, ["--muestra", "500", "--epocas", "3"], "ok"),
    ("demo: muestra de 30", D, ["--muestra", "30", "--epocas", "3"], "ok"),
    ("demo: sin peso de clase", D, ["--sin_peso_clase"] + R, "ok"),
    ("demo: weight_decay 0", D, ["--weight_decay", "0"] + R, "ok"),
    ("demo: weight_decay 0.5", D, ["--weight_decay", "0.5"] + R, "ok"),
    ("demo: PDO 1", D, ["--pdo", "1"] + R, "ok"),
    ("demo: PDO 100", D, ["--pdo", "100"] + R, "ok"),
    ("demo: puntaje base 0", D, ["--puntaje_base", "0"] + R, "ok"),
    ("demo: puntaje base 1000", D, ["--puntaje_base", "1000"] + R, "ok"),
    ("demo: odds 0.5", D, ["--odds_base", "0.5"] + R, "ok"),
    ("demo: odds 1000", D, ["--odds_base", "1000"] + R, "ok"),
    ("demo: todo cambiado a la vez", D, ["--lr", "0.003", "--dropout", "0.3", "--capas", "64", "32", "--activacion",
                                         "tanh", "--batch_size", "256", "--weight_decay", "0.001", "--pdo", "30",
                                         "--puntaje_base", "700", "--odds_base", "10", "--semilla", "3"] + R, "ok"),
    ("demo: dropout 1.5", D, ["--dropout", "1.5"] + R, "rechazo"),
    ("demo: dropout negativo", D, ["--dropout", "-0.1"] + R, "rechazo"),
    ("demo: lr negativa", D, ["--lr", "-0.01"] + R, "rechazo"),
    ("demo: 0 épocas", D, ["--epocas", "0"], "rechazo"),
    ("demo: lote de 0", D, ["--batch_size", "0"] + R, "rechazo"),
    ("demo: lote de 1", D, ["--batch_size", "1"] + R, "rechazo"),
    ("demo: capa de 0 neuronas", D, ["--capas", "0"] + R, "rechazo"),
    ("demo: muestra 0", D, ["--muestra", "0"], "rechazo"),
    ("demo: muestra mayor que los datos", D, ["--muestra", "10000000"], "rechazo"),
    ("demo: semilla negativa", D, ["--semilla", "-1"] + R, "rechazo"),
    ("demo: paciencia negativa", D, ["--paciencia", "-1"] + R, "rechazo"),
    ("demo: weight_decay negativo", D, ["--weight_decay", "-0.1"] + R, "rechazo"),
    ("demo: PDO 0", D, ["--pdo", "0"] + R, "rechazo"),
    ("demo: PDO negativo", D, ["--pdo", "-20"] + R, "rechazo"),
    ("demo: odds 0", D, ["--odds_base", "0"] + R, "rechazo"),
    ("demo: odds negativas", D, ["--odds_base", "-3"] + R, "rechazo"),
    ("demo: activación que no existe", D, ["--activacion", "sigmoid"], "rechazo"),
    ("demo: lr con texto", D, ["--lr", "abc"], "rechazo"),
    # ---------------------------------------------------------------- evaluación de la Parte 1
    ("evaluación: por defecto", P1, [], "ok"),
    ("evaluación: rápida", P1, ["--rapido"], "ok"),
    ("evaluación: PDO 1", P1, ["--pdo", "1", "--rapido"], "ok"),
    ("evaluación: PDO 200", P1, ["--pdo", "200", "--rapido"], "ok"),
    ("evaluación: puntaje base 0", P1, ["--puntaje_base", "0", "--rapido"], "ok"),
    ("evaluación: odds 1", P1, ["--odds_base", "1", "--rapido"], "ok"),
    ("evaluación: odds 5000", P1, ["--odds_base", "5000", "--rapido"], "ok"),
    ("evaluación: PDO 0", P1, ["--pdo", "0"], "rechazo"),
    ("evaluación: odds 0", P1, ["--odds_base", "0"], "rechazo"),
    ("evaluación: odds negativas", P1, ["--odds_base", "-1"], "rechazo"),
    # ---------------------------------------------------------------- búsqueda de la Parte 1
    ("búsqueda: solo semillas sin búsqueda previa", P1B, ["--demo", "--solo_semillas"], "rechazo"),
    ("búsqueda: demo", P1B, ["--demo"], "ok"),
    ("búsqueda: demo sin configuraciones aleatorias", P1B, ["--demo", "--n_configuraciones", "0"], "ok"),
    ("búsqueda: demo con 1 semilla", P1B, ["--demo", "--n_semillas", "1"], "ok"),
    ("búsqueda: demo con 1 época", P1B, ["--demo", "--epocas", "1"], "ok"),
    ("búsqueda: solo semillas en demo", P1B, ["--demo", "--solo_semillas"], "ok"),
    ("búsqueda: 0 semillas", P1B, ["--demo", "--n_semillas", "0"], "rechazo"),
    ("búsqueda: configuraciones negativas", P1B, ["--demo", "--n_configuraciones", "-1"], "rechazo"),
    ("búsqueda: 0 épocas", P1B, ["--demo", "--epocas", "0"], "rechazo"),
    # ---------------------------------------------------------------- Parte 2
    ("parte 2: por defecto", P2, [], "ok"),
    ("parte 2: demo", P2, ["--demo"], "ok"),
    ("parte 2: lr 0.5", P2, ["--lr", "0.5"] + R2, "ok"),
    ("parte 2: lr 5 (muy alta)", P2, ["--lr", "5"] + R2, "ok"),
    ("parte 2: pérdida bce", P2, ["--perdida", "bce"] + R2, "ok"),
    ("parte 2: una neurona", P2, ["--capas", "1"] + R2, "ok"),
    ("parte 2: red grande", P2, ["--capas", "256", "128", "64"] + R2, "ok"),
    ("parte 2: relu", P2, ["--activacion", "relu"] + R2, "ok"),
    ("parte 2: dropout 0.9", P2, ["--dropout", "0.9"] + R2, "ok"),
    ("parte 2: último lote de 1 crédito", P2, ["--batch_size", "3"] + R2, "ok"),
    ("parte 2: 1 época", P2, ["--epocas", "1"], "ok"),
    ("parte 2: paciencia 0", P2, ["--paciencia", "0"] + R2, "ok"),
    ("parte 2: semilla 0", P2, ["--semilla", "0"] + R2, "ok"),
    ("parte 2: dropout 1.5", P2, ["--dropout", "1.5"] + R2, "rechazo"),
    ("parte 2: lr negativa", P2, ["--lr", "-1"] + R2, "rechazo"),
    ("parte 2: 0 épocas", P2, ["--epocas", "0"], "rechazo"),
    ("parte 2: lote de 1", P2, ["--batch_size", "1"] + R2, "rechazo"),
    ("parte 2: capa de 0 neuronas", P2, ["--capas", "0"] + R2, "rechazo"),
    ("parte 2: pérdida que no existe", P2, ["--perdida", "huber"], "rechazo"),
    # ---------------------------------------------------------------- archivos de configuración modificados
    ("config: JSON con una coma de menos", P1, ["--config", "configs/prueba_roto.json"], "rechazo"),
    ("config: PDO 25 en el archivo", P1, ["--config", "configs/prueba_pdo25.json", "--rapido"], "ok"),
    ("config: odds mal escritas en el archivo", P1, ["--config", "configs/prueba_odds_texto.json"], "rechazo"),
    ("config: 0 épocas en el demo del archivo", P1B, ["--demo", "--config", "configs/prueba_epocas0.json"], "rechazo"),
    ("config: dropout 1.5 en el espacio de búsqueda", P1B, ["--demo", "--config", "configs/prueba_dropout.json"], "rechazo"),
    ("config: lote de 1 en el modelo base", P1B, ["--demo", "--config", "configs/prueba_lote1.json"], "rechazo"),
    ("config: espacio de búsqueda vacío", P1B, ["--demo", "--config", "configs/prueba_espacio_vacio.json"], "rechazo"),
    ("config parte 2: lr negativa en el espacio", P2, ["--demo", "--config", "configs/prueba2_lr.json"], "rechazo"),
    ("config parte 2: 3 épocas en el archivo", P2, ["--config", "configs/prueba2_epocas3.json"], "ok"),
    ("config: archivo que no existe", P1, ["--config", "configs/no_existe.json"], "rechazo"),
    # ---------------------------------------------------------------- pérdida esperada y app
    ("pérdida esperada: red", EL, [], "ok"),
    ("pérdida esperada: logística", EL, ["--modelo_pd", "logistica"], "ok"),
    ("app: casos de prueba", "scripts/probar_app.py", [], "ok"),
]


def preparar_configs(copia):
    """Variantes de los archivos de configuración, como las que podría pedir el profesor (o un error al editarlas)."""
    import json
    c1 = json.loads((copia / "configs/parte1.json").read_text())
    c2 = json.loads((copia / "configs/parte2.json").read_text())
    texto = (copia / "configs/parte1.json").read_text()
    (copia / "configs/prueba_roto.json").write_text(texto.replace('"pdo": 20', '"pdo": 20,', 1).replace('"puntaje_base": 600,', '"puntaje_base": 600', 1))
    variantes1 = {
        "prueba_pdo25": lambda c: c["scorecard"].update(pdo=25),
        "prueba_odds_texto": lambda c: c["scorecard"].update(odds_base="carteras"),
        "prueba_epocas0": lambda c: c["demo"].update(epocas=0),
        "prueba_dropout": lambda c: c["busqueda"]["espacio"].update(dropout=[0.2, 1.5]),
        "prueba_lote1": lambda c: c["modelo_base"].update(batch_size=1),
        "prueba_espacio_vacio": lambda c: c["busqueda"]["espacio"].update(activacion=[]),
    }
    for nombre, cambio in variantes1.items():
        c = json.loads(json.dumps(c1)); cambio(c)
        (copia / f"configs/{nombre}.json").write_text(json.dumps(c, indent=2))
    for nombre, cambio in {"prueba2_lr": lambda c: c["espacio_busqueda"].update(lr=[0.001, -0.5]),
                           "prueba2_epocas3": lambda c: c.update(epocas=3)}.items():
        c = json.loads(json.dumps(c2)); cambio(c)
        (copia / f"configs/{nombre}.json").write_text(json.dumps(c, indent=2))


def correr(caso, carpeta):
    nombre, script, args, espera = caso
    t = time.perf_counter()
    proc = subprocess.run([PY, script, *args], cwd=carpeta, capture_output=True, text=True, timeout=900)
    seg = time.perf_counter() - t
    salida = proc.stdout + proc.stderr
    traceback = "Traceback (most recent call last)" in salida
    if traceback:
        estado = "CAÍDA"
    elif espera == "ok":
        estado = "PASA" if proc.returncode == 0 else "FALLA"
    else:
        estado = "PASA" if proc.returncode != 0 else "FALLA (no se rechazó)"
    lineas = [l for l in salida.strip().splitlines() if l.strip() and "Warning" not in l]
    ultima = lineas[-1][:150] if lineas else ""
    return estado, seg, ultima


def main():
    filtro = sys.argv[1] if len(sys.argv) > 1 else ""
    casos = [c for c in CASOS if filtro in c[0]]
    with tempfile.TemporaryDirectory() as tmp:
        copia = Path(tmp) / "repo"
        shutil.copytree(RAIZ, copia, ignore=shutil.ignore_patterns(".venv", ".git", "__pycache__"))
        preparar_configs(copia)
        fallos = []
        for i, caso in enumerate(casos, 1):
            estado, seg, ultima = correr(caso, copia)
            print(f"{i:3d}/{len(casos)} [{estado:^22}] {caso[0]:<45} {seg:6.1f} s  | {ultima}", flush=True)
            if not estado.startswith("PASA"):
                fallos.append(caso[0])
    print(f"\n{len(casos) - len(fallos)} de {len(casos)} casos pasan.")
    if fallos:
        print("No pasan:", "; ".join(fallos))
    sys.exit(0 if not fallos else 1)


if __name__ == "__main__":
    main()
