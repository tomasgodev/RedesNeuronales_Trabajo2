"""Modelos y utilidades compartidas por los scripts y la aplicación.

Los notebooks no se pueden importar desde un script, así que las clases y funciones
marcadas como COPIA están copiadas sin cambios desde los notebooks del equipo:

- CreditRiskNN y predecir_proba: notebooks/clasificacion.ipynb (celdas 6 y 9).
- ACTIVACIONES, LGDNN, fijar_semilla y predecir: notebooks/regresion.ipynb (celdas 15 y 16).

entrenar_clasificador convierte en función el bucle de entrenamiento de
notebooks/clasificacion.ipynb (celdas 7 y 8), con la misma lógica, para poder
lanzarlo con distintas configuraciones y semillas.
"""
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ))
sys.path.append(str(RAIZ / "scripts"))

from src.preprocesamiento import cargar_datos_procesados, columnas_entrada, particion_temporal  # noqa: E402

RUTA_CLASIFICADOR = RAIZ / "weights" / "modelo_clasificador.pth"
RUTA_PREPRO_CLASIFICADOR = RAIZ / "weights" / "preprocesamiento_clasificador.joblib"
RUTA_LGD = RAIZ / "weights" / "modelo_lgd.pth"
RUTA_LINEAL_LGD = RAIZ / "weights" / "regresion_lineal_lgd.joblib"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# COPIA de notebooks/clasificacion.ipynb, celda 6
# ---------------------------------------------------------------------------
class CreditRiskNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            # Capa 1
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Capa 2
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Capa 3
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            #nn.Dropout(0.2),
            # Salida — probabilidad de default
            nn.Linear(16, 1),
            #nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


# ---------------------------------------------------------------------------
# COPIA de notebooks/clasificacion.ipynb, celda 9
# ---------------------------------------------------------------------------
def predecir_proba(model, X_t, pos_weight):
    model.eval()
    with torch.no_grad():
        logits = model(X_t) - torch.log(pos_weight)   # deshace el efecto de pos_weight
        return torch.sigmoid(logits).cpu().numpy()


# ---------------------------------------------------------------------------
# COPIA de notebooks/regresion.ipynb, celdas 15 y 16
# ---------------------------------------------------------------------------
ACTIVACIONES = {"relu": nn.ReLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}


class LGDNN(nn.Module):
    def __init__(self, input_dim, capas_ocultas, activacion="relu", dropout=0.2):
        super().__init__()
        capas, entrada = [], input_dim
        for neuronas in capas_ocultas:
            capas += [
                nn.Linear(entrada, neuronas),
                nn.BatchNorm1d(neuronas),
                ACTIVACIONES[activacion](),
                nn.Dropout(dropout),
            ]
            entrada = neuronas
        capas.append(nn.Linear(entrada, 1))
        self.network = nn.Sequential(*capas)

    def forward(self, x):
        # Devuelve el logit; la LGD es sigmoid(logit)
        return self.network(x).squeeze(1)


def fijar_semilla(semilla):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)


def predecir(modelo, X_t):
    modelo.eval()
    with torch.no_grad():
        return torch.sigmoid(modelo(X_t)).cpu().numpy()


def contar_parametros(modelo):
    # COPIA de notebooks/regresion.ipynb, celda 17
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Utilidades nuevas (no estaban en los notebooks)
# ---------------------------------------------------------------------------
def a_tensor(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32).to(device)


def calcular_pos_weight(y_train):
    """Igual que notebooks/clasificacion.ipynb, celda 7: negativos / positivos en entrenamiento."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    return torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)


def datos_parte1(df=None):
    """Datos procesados y partición temporal de la Parte 1, sin escalar."""
    if df is None:
        df = cargar_datos_procesados()
    particion = particion_temporal(df["issue_d"])
    features = columnas_entrada(df)
    datos = {"df": df, "particion": particion, "features": features}
    for clave in ["train", "val", "test"]:
        datos[f"X_{clave}"] = df.loc[particion == clave, features]
        datos[f"y_{clave}"] = df.loc[particion == clave, "incumple"].to_numpy()
    return datos


def cargar_clasificador(ruta_pesos=RUTA_CLASIFICADOR, ruta_prepro=RUTA_PREPRO_CLASIFICADOR):
    """Red de la Parte 1 guardada por notebooks/clasificacion.ipynb (celda 16), con su escalador y columnas."""
    prepro = joblib.load(ruta_prepro)
    modelo = CreditRiskNN(input_dim=len(prepro["columnas"])).to(device)
    modelo.load_state_dict(torch.load(ruta_pesos, map_location=device))
    modelo.eval()
    return modelo, prepro["scaler"], list(prepro["columnas"])


def cargar_modelo_lgd(ruta=RUTA_LGD):
    """Red de la Parte 2 guardada por notebooks/regresion.ipynb (celda 35).

    El archivo guarda también el escalador (un objeto de scikit-learn), por eso necesita weights_only=False.
    """
    guardado = torch.load(ruta, map_location=device, weights_only=False)
    config = guardado["config"]
    modelo = LGDNN(len(guardado["columnas"]), config["capas_ocultas"], config["activacion"], config["dropout"]).to(device)
    modelo.load_state_dict(guardado["state_dict"])
    modelo.eval()
    return modelo, guardado["scaler"], list(guardado["columnas"]), config


def cargar_factor_ead(ruta=RUTA_LINEAL_LGD):
    """Fracción promedio pendiente al castigo por plazo (notebooks/regresion.ipynb, celda 33)."""
    return joblib.load(ruta)["factor_ead"]


def estimar_ead(loan_amnt, plazo_60, factor_ead):
    """EAD al originar: monto por la fracción promedio pendiente al castigo (docs/parte2_lgd.md, sección 8)."""
    return np.asarray(loan_amnt) * np.where(np.asarray(plazo_60) == 1, factor_ead["60 meses"], factor_ead["36 meses"])


def construir_clasificador(config, input_dim):
    """Arquitectura de Tomás (CreditRiskNN) o una arquitectura parametrizada con la clase LGDNN de Sebastián."""
    if config.get("modelo", "CreditRiskNN") == "CreditRiskNN":
        return CreditRiskNN(input_dim).to(device)
    return LGDNN(input_dim, config["capas_ocultas"], config["activacion"], config["dropout"]).to(device)


def entrenar_clasificador(config, X_train_t, y_train_t, X_val_t, y_val, pos_weight, semilla=None, verbose=False):
    """Bucle de notebooks/clasificacion.ipynb (celda 8) convertido en función.

    Misma lógica: BCEWithLogitsLoss con pos_weight, Adam, ReduceLROnPlateau sobre el AUC de validación
    (modo max, paciencia 3, factor 0.5) y parada temprana con paciencia sobre el AUC de validación,
    restaurando los mejores pesos. A diferencia del notebook, fija la semilla y guarda los mejores
    pesos en memoria en lugar de en best_model.pth.
    """
    if semilla is not None:
        fijar_semilla(semilla)
    model = construir_clasificador(config, X_train_t.shape[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=config["batch_size"], shuffle=True)

    history = {"train_loss": [], "val_auc": []}
    best_val_auc, best_state, patience_counter = 0.0, None, 0
    for epoch in range(1, config["epocas"] + 1):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            y_prob_val = model(X_val_t).cpu().numpy()
        val_auc = roc_auc_score(y_val, y_prob_val)
        scheduler.step(val_auc)
        history["train_loss"].append(avg_loss)
        history["val_auc"].append(val_auc)
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"Época {epoch:02d}/{config['epocas']} | pérdida: {avg_loss:.4f} | AUC val: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc, patience_counter = val_auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config["paciencia"]:
                if verbose:
                    print(f"Parada temprana en la época {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, history
