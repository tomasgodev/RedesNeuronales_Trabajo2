"""Validación de parámetros para los scripts: valores fuera de rango o archivos de configuración mal
escritos terminan con un mensaje claro en español, no con un error de Python."""
import argparse
import json
import sys
from pathlib import Path


def _tipo(convertir, condicion, explicacion):
    def validar(texto):
        try:
            valor = convertir(texto)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(f"'{texto}' no es un número válido")
        if not condicion(valor):
            raise argparse.ArgumentTypeError(f"{texto} no es válido: {explicacion}")
        return valor
    return validar


fraccion = _tipo(float, lambda v: 0 <= v <= 1, "debe estar entre 0 y 1")
no_negativo = _tipo(float, lambda v: v >= 0, "no puede ser negativo")
positivo = _tipo(float, lambda v: v > 0, "debe ser mayor que 0")
entero_positivo = _tipo(int, lambda v: v >= 1, "debe ser un entero mayor o igual que 1")
entero_no_negativo = _tipo(int, lambda v: v >= 0, "debe ser un entero mayor o igual que 0")
tamano_lote = _tipo(int, lambda v: v >= 2, "debe ser al menos 2, porque la normalización por lotes "
                                            "(batch normalization) necesita dos o más créditos por lote")


def salir(mensaje):
    print(f"error: {mensaje}", file=sys.stderr)
    sys.exit(2)


def leer_config(ruta):
    """Lee un archivo JSON de configuración; si está mal escrito, dice dónde."""
    ruta = Path(ruta)
    if not ruta.exists():
        salir(f"no existe el archivo de configuración {ruta}")
    try:
        return json.loads(ruta.read_text())
    except json.JSONDecodeError as e:
        salir(f"el archivo {ruta.name} no es un JSON válido (línea {e.lineno}, columna {e.colno}): {e.msg}. "
              "Revise comas, comillas y corchetes cerca de esa posición.")


def revisar(valores, nombre_archivo):
    """valores: lista de (nombre, valor, validador). Valida valores que vienen del archivo de configuración."""
    for nombre, valor, validador in valores:
        try:
            validador(str(valor))
        except argparse.ArgumentTypeError as e:
            salir(f"en {nombre_archivo}, '{nombre}' = {valor}: {e}")


def revisar_espacio(espacio, nombre_archivo):
    """Valida cada lista del espacio de búsqueda de hiperparámetros."""
    reglas = {"dropout": fraccion, "weight_decay": no_negativo, "lr": no_negativo, "batch_size": tamano_lote}
    for clave, opciones in espacio.items():
        if not isinstance(opciones, list) or not opciones:
            salir(f"en {nombre_archivo}, '{clave}' del espacio de búsqueda debe ser una lista con al menos un valor")
        if clave in reglas:
            revisar([(f"espacio.{clave}", v, reglas[clave]) for v in opciones], nombre_archivo)
        if clave == "capas_ocultas":
            for capas in opciones:
                if not isinstance(capas, list) or not capas:
                    salir(f"en {nombre_archivo}, cada opción de 'capas_ocultas' debe ser una lista de neuronas, p. ej. [64, 32]")
                revisar([("espacio.capas_ocultas", n, entero_positivo) for n in capas], nombre_archivo)
